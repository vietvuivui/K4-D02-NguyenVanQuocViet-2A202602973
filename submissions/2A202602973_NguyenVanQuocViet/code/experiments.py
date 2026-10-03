"""experiments.py - điều phối Bước 1-5: chạy (hoặc nạp lại) thí nghiệm, quy tắc chọn trên VAL,
suy luận, chung kết trên test, xuất results.xlsx và hình cho báo cáo.

Mọi lựa chọn (backbone, công thức, phương pháp suy luận) chỉ dùng số liệu trên VAL theo các quy tắc
ghi ở đầu file, và in ra lý do. Test chỉ được chạm ở `final_predict` (Bước 4), mỗi seed đúng một lần.

Tiết kiệm GPU: `run_or_load` bỏ qua lần chạy đã xong (Colab bị ngắt thì chạy lại ô là tiếp tục), và
dùng lại một lần chạy cũ có CÙNG cấu hình (chỉ khác exp_id), ví dụ T00 = backbone đã chọn ở Bước 1,
F01 seed 0 = công thức tốt nhất ở Bước 2. Lần dùng lại được ghi rõ trong summary ("reused_from").
"""
from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark as Bm
import dataset as D
import inference as I
import model as M
from train import Config, curve_path, plot_curves, pred_path, run, run_dir, softmax_np  # thêm gốc repo vào sys.path
from eval import compute_metrics, mean_std, save_predictions  # noqa: E402

# ----------------------------------------------------------------------------- quy tắc chọn trên VAL
NOISE_F1 = 0.005          # chênh lệch macro-F1 val nhỏ hơn mức này (1 seed) coi là "không phân biệt được"
INFER_MIN_GAIN = 0.002    # phương pháp suy luận phải hơn I00 ít nhất mức này mới được chọn cho chung kết
LATENCY_BUDGET_MS = 100.0  # ngân sách thời gian thực, p95 batch 1 (slide trang 61)
RARE = {"Chinee Apple": 0, "Snake Weed": 7}

# Ghim tag trọng số (timm "kiến_trúc.tag"): tag mặc định đổi theo phiên bản timm, và mặc định của convnext_tiny
# là in12k_ft_in1k (tiền huấn luyện ImageNet-12k) -> không công bằng. Tất cả dưới đây chỉ dùng ImageNet-1k.
BACKBONES = [
    ("B01", "resnet50.a1_in1k"),                   # ResNet (mốc)
    ("B02", "convnext_tiny.fb_in1k"),              # ConvNeXt
    ("B03", "deit_small_patch16_224.fb_in1k"),     # transformer
    ("B04", "efficientnet_b0.ra_in1k"),            # nhẹ
    ("B05", "mobilenetv3_large_100.ra_in1k"),      # nhẹ
]

# (exp_id, trục, mô tả "khác T00 ở điểm nào", ghi đè Config). Mỗi dòng khác T00 đúng MỘT yếu tố.
ABLATIONS = [
    ("T01", "A", "init=frozen (chỉ train head)", dict(init="frozen")),
    ("T02", "B", "aug=trivial (TrivialAugmentWide)", dict(aug="trivial")),
    ("T03", "B", "aug=flip_rot (lật dọc + xoay 90°)", dict(aug="flip_rot")),
    ("T04", "B", "CutMix alpha=1.0", dict(mix="cutmix", mix_alpha=1.0)),
    ("T05", "C", "loss=label smoothing 0.1", dict(loss="ls", label_smoothing=0.1)),
    ("T06", "C", "loss=focal gamma=2", dict(loss="focal", focal_gamma=2.0)),
    ("T07", "C", "loss=CE trọng số 1/n_c (train)", dict(loss="ce_weighted", class_weight_beta=0.0)),
    ("T08", "F", "EMA decay=0.995", dict(ema_decay=0.995)),  # ~2000 bước ở 12 epoch: 0.995^2000 ≈ 0,004% trọng số khởi tạo còn lại
]
COMBO_ID = "T09"

# các trường không ảnh hưởng kết quả huấn luyện: bỏ qua khi so "cùng cấu hình"
_NON_RECIPE = {"exp_id", "desc", "save_test_predictions", "num_workers", "save_checkpoint", "images_dir",
               "labels_dir", "out_dir", "pred_dir", "curves_dir", "cache_images", "cache_dir"}


def _slug(text: str) -> str:
    """Tên file ASCII: bỏ dấu tiếng Việt, ký tự lạ thành '_'."""
    import unicodedata
    text = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    keep = "".join(c if (c.isascii() and c.isalnum()) or c in "._-" else "_" for c in text)
    return "_".join(p for p in keep.split("_") if p)[:40]


def _read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_json(p: Path, d: dict) -> None:
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


# ============================================================================= chạy / nạp lại
def _recipe(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _NON_RECIPE}


def find_equivalent(cfg: Config) -> Path | None:
    """Tìm lần chạy đã xong có cùng công thức + seed (khác exp_id)."""
    want = _recipe(dataclasses.asdict(cfg))
    for cj in sorted(Path(cfg.out_dir).glob("*/seed*/config.json")):
        if not (cj.parent / "summary.json").exists():
            continue
        d = _read_json(cj)
        if d.get("exp_id") != cfg.exp_id and _recipe(d) == want:
            return cj.parent
    return None


def _alias_run(src: Path, cfg: Config) -> dict:
    """Dùng lại lần chạy `src` dưới exp_id mới: chép log/logit (không chép checkpoint), vẽ lại đường cong."""
    dst = run_dir(cfg)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("best.pt"))
    old = _read_json(src / "summary.json")
    s = {**old, "exp_id": cfg.exp_id, "run_dir": str(dst), "reused_from": f"{old['exp_id']} seed{old['seed']}"}
    _write_json(dst / "config.json", dataclasses.asdict(cfg))
    z = np.load(dst / "val_logits.npz")
    save_predictions(pred_path(cfg, "val"), z["filenames"], z["y_true"], softmax_np(z["logits"]))
    hist = pd.read_csv(dst / "history.csv").to_dict("records")
    lrs = np.load(dst / "lr_steps.npy").tolist() if (dst / "lr_steps.npy").exists() else None
    title = (f"{cfg.exp_id} · {cfg.backbone} · seed {cfg.seed} · best epoch {s['best_epoch']} "
             f"(val macro-F1 {s['val_macro_f1']:.4f}) · cùng lần chạy với {s['reused_from']}")
    plot_curves(hist, curve_path(cfg), title, lrs)
    _write_json(dst / "summary.json", s)
    print(f"[{cfg.exp_id} seed{cfg.seed}] dùng lại {s['reused_from']} (cùng công thức và seed)")
    return s


def run_or_load(cfg: Config, reuse: bool = True) -> dict:
    """Đã chạy xong -> đọc summary; có lần chạy cùng cấu hình -> dùng lại; còn lại -> train.run(cfg)."""
    summ = run_dir(cfg) / "summary.json"
    if summ.exists():
        return _read_json(summ)
    src = find_equivalent(cfg) if reuse else None
    return _alias_run(src, cfg) if src is not None else run(cfg)


def load_config(summary: dict) -> Config:
    return Config(**_read_json(Path(summary["run_dir"]) / "config.json"))


def load_trained(summary: dict, device) -> torch.nn.Module:
    """Dựng lại kiến trúc (không tải trọng số ImageNet) rồi nạp checkpoint tốt nhất của lần chạy."""
    cfg = load_config(summary)
    m = M.build_model(cfg.backbone, pretrained=False, num_classes=D.NUM_CLASSES, drop_rate=cfg.drop_rate,
                      init="scratch")
    ckpt = summary.get("checkpoint") or str(Path(summary["run_dir"]) / "best.pt")
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()


def _norm_of(model) -> tuple:
    pc = getattr(model, "pretrained_cfg", {}) or {}
    return pc.get("mean", D.IMAGENET_MEAN), pc.get("std", D.IMAGENET_STD)


def _val_logits(summary: dict):
    z = np.load(Path(summary["run_dir"]) / "val_logits.npz")
    return z["filenames"], z["y_true"], z["logits"]


def _metrics(y, probs) -> dict:
    return compute_metrics(np.asarray(y), probs.argmax(1), probs)


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================= Bước 1
def step1_backbones(paths: dict, device, backbones=BACKBONES, **overrides) -> pd.DataFrame:
    """Huấn luyện mỗi backbone bằng công thức nền T00 (cùng seed 0) + đo độ trễ sơ bộ batch 1 FP32."""
    rows = []
    for exp_id, bb in backbones:
        cfg = Config(exp_id=exp_id, backbone=bb, desc=bb, seed=0, **overrides, **paths)
        s = run_or_load(cfg)
        if "latency_b1_p50_ms" not in s:
            m = load_trained(s, device)
            lat = Bm.latency_report(m, 1, cfg.img_size, "fp32", device.type, warmup=10, iters=100)
            s.update(latency_b1_p50_ms=lat["p50"], latency_b1_p95_ms=lat["p95"], latency_gpu=lat["gpu"])
            _write_json(Path(s["run_dir"]) / "summary.json", s)
            del m
            _empty_cache()
        rows.append({**s, "img_size": cfg.img_size, "epochs": cfg.epochs})
    df = pd.DataFrame(rows)
    df["note"] = ("công thức nền T00, seed 0, 1 seed; độ trễ sơ bộ FP32 batch 1 (warmup 10, 100 lần); "
                  "GMAC đếm bằng torch FlopCounterMode (FLOPs/2)")
    return df[["exp_id", "backbone", "weights_tag", "params_M", "gmacs", "img_size", "epochs", "seed",
               "best_epoch", "val_macro_f1", "val_top1", "val_balanced_acc", "train_time_per_epoch_s",
               "latency_b1_p50_ms", "latency_b1_p95_ms", "latency_gpu", "note"]]


def backbone_analysis(bb_df: pd.DataFrame, paths: dict, overfit_tol: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Thêm cột phân tích từ history.csv của mỗi B0x và tính tương quan FLOPs với thời gian/độ trễ.

    - epochs_to_99pct: epoch đầu tiên đạt >= 99% macro-F1 val tốt nhất (hội tụ nhanh hay chậm)
    - val_loss_min_epoch, val_loss_rise: val loss ở epoch cuối trừ val loss nhỏ nhất
    - overfit: val loss tăng > overfit_tol sau điểm thấp nhất trong khi train loss vẫn giảm
    Tương quan Spearman (thứ hạng, 5 điểm nên chỉ mang tính tham khảo).
    """
    df = bb_df.copy()
    extra = []
    for r in df.itertuples():
        h = pd.read_csv(Path(paths["out_dir"]) / r.exp_id / "seed0" / "history.csv")
        best = h["val_macro_f1"].max()
        i_min = int(h["val_loss"].idxmin())
        rise = float(h["val_loss"].iloc[-1] - h["val_loss"].iloc[i_min])
        train_falls = bool(h["train_loss"].iloc[-1] < h["train_loss"].iloc[i_min])
        extra.append({"epochs_to_99pct": int(h.loc[h["val_macro_f1"] >= 0.99 * best, "epoch"].iloc[0]),
                      "val_loss_min_epoch": int(h["epoch"].iloc[i_min]), "val_loss_rise": rise,
                      "overfit": bool(rise > overfit_tol and train_falls),
                      "train_val_gap_last": float(h["val_loss"].iloc[-1] - h["train_loss"].iloc[-1])})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    corr = {f"spearman(GMAC, {c})": float(df["gmacs"].corr(df[c], method="spearman"))
            for c in ("latency_b1_p50_ms", "train_time_per_epoch_s")}
    corr["spearman(params, latency_b1_p50_ms)"] = float(df["params_M"].corr(df["latency_b1_p50_ms"], method="spearman"))
    return df, corr


def plot_backbone_curves(bb_df: pd.DataFrame, paths: dict, path: str | Path) -> None:
    """Chồng macro-F1 val và val loss theo epoch của mọi backbone: so tốc độ hội tụ và quá khớp."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r in bb_df.itertuples():
        h = pd.read_csv(Path(paths["out_dir"]) / r.exp_id / "seed0" / "history.csv")
        ax[0].plot(h["epoch"], h["val_macro_f1"], "o-", ms=3, label=f"{r.exp_id} {r.backbone}")
        ax[1].plot(h["epoch"], h["val_loss"], "o-", ms=3, label=f"{r.exp_id}")
    ax[0].set(xlabel="epoch", ylabel="macro-F1 val", title="Hội tụ: macro-F1 val theo epoch")
    ax[1].set(xlabel="epoch", ylabel="val loss (CE)", title="Quá khớp: val loss theo epoch")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def select_backbone(bb_df: pd.DataFrame, override: str | None = None) -> tuple[str, str]:
    """Quy tắc: lấy các backbone có macro-F1 val trong khoảng NOISE_F1 của mức cao nhất (không phân biệt
    được với 1 seed), rồi chọn cái có độ trễ batch 1 thấp nhất. Trả về (backbone, lý do)."""
    if override:
        return override, f"chọn thủ công: {override}"
    best = bb_df["val_macro_f1"].max()
    cands = bb_df[bb_df["val_macro_f1"] >= best - NOISE_F1].sort_values("latency_b1_p50_ms")
    pick = cands.iloc[0]
    top = bb_df.loc[bb_df["val_macro_f1"].idxmax()]
    reason = (f"macro-F1 val cao nhất: {top.backbone} {top.val_macro_f1:.4f}. Các backbone trong khoảng "
              f"{NOISE_F1} (1 seed, không phân biệt được): {', '.join(cands.backbone)}. "
              f"Chọn {pick.backbone} (macro-F1 {pick.val_macro_f1:.4f}, p50 batch-1 {pick.latency_b1_p50_ms:.1f} ms, "
              f"{pick.params_M:.1f}M tham số) vì nhanh nhất trong nhóm đó.")
    return pick.backbone, reason


# ============================================================================= Bước 2
def _rare_f1(y, logits) -> dict:
    m = _metrics(y, softmax_np(logits))
    return {f"f1_{k.split()[0].lower()}": float(m["f1"][i]) for k, i in RARE.items()}


def _training_row(s: dict, exp_id: str, axis: str, change: str, base_f1: float | None) -> dict:
    _, y, lg = _val_logits(s)
    return {"exp_id": exp_id, "backbone": s["backbone"], "axis": axis, "change_vs_T00": change, "seed": s["seed"],
            "val_macro_f1": s["val_macro_f1"], "val_top1": s["val_top1"],
            "delta_vs_T00": s["val_macro_f1"] - base_f1 if base_f1 is not None else 0.0,
            **_rare_f1(y, lg), "best_epoch": s["best_epoch"], "reused_from": s.get("reused_from", "")}


def step2_training(paths: dict, backbone: str, ablations=ABLATIONS, **overrides) -> pd.DataFrame:
    """T00 (nền) + mỗi ablation khác T00 đúng một yếu tố, cùng backbone, seed 0."""
    s0 = run_or_load(Config(exp_id="T00", backbone=backbone, desc=f"{backbone}_baseline", seed=0,
                            **overrides, **paths))
    rows = [_training_row(s0, "T00", "-", "công thức nền", None)]
    for exp_id, axis, change, ov in ablations:
        s = run_or_load(Config(exp_id=exp_id, backbone=backbone, desc=_slug(change), seed=0,
                               **{**overrides, **ov}, **paths))
        rows.append(_training_row(s, exp_id, axis, change, s0["val_macro_f1"]))
    return pd.DataFrame(rows)


def select_combination(tr_df: pd.DataFrame, ablations=ABLATIONS) -> tuple[dict, str]:
    """Quy tắc kết hợp (tham lam theo trục): mỗi trục lấy biến thể có Δ lớn nhất; giữ các trục có Δ > 0.
    Nếu ít hơn 2 trục có Δ > 0, vẫn ghép 2 biến thể Δ lớn nhất ở 2 trục khác nhau để kiểm tra
    tính cộng dồn (GUIDE mục 3.1, ý 4). Trả về (ghi đè Config, lý do)."""
    ov_of = {e: ov for e, _, _, ov in ablations}
    abl = tr_df[tr_df.exp_id != "T00"].sort_values("delta_vs_T00", ascending=False)
    per_axis = abl.groupby("axis", sort=False).head(1)
    chosen = per_axis[per_axis.delta_vs_T00 > 0]
    if len(chosen) < 2:
        chosen = per_axis.head(2)
    combo = {}
    for e in chosen.exp_id:
        combo.update(ov_of[e])
    reason = ("Ghép " + " + ".join(f"{r.exp_id} ({r.change_vs_T00}, Δ={r.delta_vs_T00:+.4f})"
                                   for r in chosen.itertuples()) + f" thành {COMBO_ID}.")
    return combo, reason


def step2_combo(paths: dict, backbone: str, combo: dict, **overrides) -> dict:
    return run_or_load(Config(exp_id=COMBO_ID, backbone=backbone, desc="combo", seed=0,
                              **{**overrides, **combo}, **paths))


def select_recipe(tr_df: pd.DataFrame, ablations=ABLATIONS, combo: dict | None = None) -> tuple[str, dict, str]:
    """Công thức chung kết = dòng có macro-F1 val cao nhất trong T00..T09. Trả về (exp_id, ghi đè, lý do)."""
    ov_of = {"T00": {}, **{e: ov for e, _, _, ov in ablations}, COMBO_ID: combo or {}}
    best = tr_df.loc[tr_df["val_macro_f1"].idxmax()]
    base = tr_df.loc[tr_df.exp_id == "T00", "val_macro_f1"].item()
    reason = (f"{best.exp_id} có macro-F1 val cao nhất {best.val_macro_f1:.4f} (T00 {base:.4f}, "
              f"Δ={best.val_macro_f1 - base:+.4f}; 1 seed -> {'vượt' if best.val_macro_f1 - base > NOISE_F1 else 'chưa vượt'} "
              f"ngưỡng nhiễu {NOISE_F1}).")
    return best.exp_id, ov_of[best.exp_id], reason


# ============================================================================= Bước 3
METHODS = {
    "I00": dict(kind="single", img_size=224, k=1, label="1 view (resize 256 + center crop 224)"),
    "I01": dict(kind="hflip", space="prob", img_size=224, k=2, label="TTA lật ngang, gộp xác suất"),
    "I03a": dict(kind="hflip", space="logit", img_size=224, k=2, label="TTA lật ngang, gộp logit"),
    "I02": dict(kind="multicrop", space="prob", img_size=224, k=10, label="TTA 5 crop + lật (K=10), gộp xác suất"),
    "I03b": dict(kind="multicrop", space="logit", img_size=224, k=10, label="TTA 5 crop + lật (K=10), gộp logit"),
    "I04_256": dict(kind="single", img_size=256, k=1, label="độ phân giải kiểm tra 256"),
    "I04_288": dict(kind="single", img_size=288, k=1, label="độ phân giải kiểm tra 288"),
    "I04_320": dict(kind="single", img_size=320, k=1, label="độ phân giải kiểm tra 320"),
}


def method_logits(model, cfg: Config, df: pd.DataFrame, method: dict, device):
    """Logit theo từng view của một phương pháp suy luận. Trả về (filenames, y, [logits_view]*K)."""
    mean, std = _norm_of(model)
    if method["kind"] == "multicrop":   # ảnh 256 nguyên vẹn, cắt 5 crop 224 trên GPU
        tf = D.build_transforms(False, 256, mean=mean, std=std, eval_resize=256)
        views_fn = lambda x: I.views_multicrop(x, method["img_size"], flip=True)  # noqa: E731
    else:
        tf = D.build_transforms(False, method["img_size"], mean=mean, std=std)
        views_fn = I.views_hflip if method["kind"] == "hflip" else (lambda x: [x])  # noqa: E731
    loader = D.make_loader(df, cfg.images_dir, tf, batch_size=64, train=False,
                           num_workers=cfg.num_workers, cache=cfg.cache_dir or False)
    return I.predict_views(model, loader, device, views_fn, amp=cfg.amp)


def combine(view_logits: list, method: dict) -> tuple[np.ndarray, np.ndarray]:
    """Gộp view -> (probs, pseudo-logit dùng cho temperature scaling)."""
    if len(view_logits) == 1:
        lg = view_logits[0]
        return softmax_np(lg), lg
    probs = I.aggregate_views(view_logits, method.get("space", "prob"))
    pseudo = np.mean(view_logits, 0) if method.get("space") == "logit" else np.log(np.clip(probs, 1e-12, None))
    return probs, pseudo


def _ece_crossfit(logits, y, seed: int = 0) -> tuple[float, float]:
    """ECE sau TS ước lượng trung thực hơn: khớp T trên một nửa val, đo ECE trên nửa còn lại (2 chiều)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    halves = np.array_split(idx, 2)
    eces, temps = [], []
    for a, b in ((0, 1), (1, 0)):
        T = I.fit_temperature(logits[halves[a]], y[halves[a]])
        p = I.apply_temperature(logits[halves[b]], T)
        eces.append(compute_metrics(y[halves[b]], p.argmax(1), p)["ece"])
        temps.append(T)
    return float(np.mean(eces)), float(np.mean(temps))


def _lat_row(cfg_name, lat, fused: bool) -> dict:
    return {"config": cfg_name, "gpu": lat["gpu"], "dtype": lat["dtype"], "batch": lat["batch"],
            "img_size": lat["img_size"], "k_views": lat["k_views"], "bn_fused": fused,
            "p50_ms": lat["p50"], "p95_ms": lat["p95"], "p99_ms": lat["p99"],
            "images_per_s": lat["images_per_s"], "torch": lat["torch"], "preprocessing_included": False}


def step3_inference(recipe_summary: dict, candidates: list[dict], device, cache_csv: str | Path,
                    ema_pair: tuple[dict, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """So sánh phương pháp suy luận trên VAL cho model của công thức đã chọn (seed 0).

    candidates: các lần chạy (summary) để ensemble (I05). ema_pair: (T00, T08) cho dòng I06.
    Trả về (bảng Inference, bảng Latency); lưu cache CSV để chạy lại không tốn GPU.
    """
    cache_csv = Path(cache_csv)
    lat_csv = cache_csv.with_name(cache_csv.stem + "_latency.csv")
    if cache_csv.exists() and lat_csv.exists():
        return pd.read_csv(cache_csv), pd.read_csv(lat_csv)

    cfg = load_config(recipe_summary)
    model = load_trained(recipe_summary, device)
    _, val_df, _ = D.load_split(cfg.labels_dir, cfg.fold)
    who = f"{recipe_summary['exp_id']} seed{recipe_summary['seed']} ({cfg.backbone})"
    rows, lat_rows = [], []
    dev = device.type

    def lat_of(img_size, k, dtype="fp32", m=None, batch=1):
        return Bm.latency_report(m or model, batch, img_size, dtype, dev, warmup=10, iters=100, k_views=k)

    base_lat = lat_of(224, 1)
    for code, meth in METHODS.items():
        try:
            names, y, views = method_logits(model, cfg, val_df, meth, device)
        except Exception as e:  # vd ViT không nhận độ phân giải khác 224
            rows.append({"exp_id": code, "method": meth["label"], "model": who, "K": meth["k"],
                         "note": f"không áp dụng: {type(e).__name__}: {str(e)[:80]}"})
            continue
        probs, _ = combine(views, meth)
        m = _metrics(y, probs)
        lat = base_lat if code == "I00" else lat_of(meth["img_size"], meth["k"])
        rows.append({"exp_id": code, "method": meth["label"], "model": who, "K": meth["k"],
                     "img_size": meth["img_size"], "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
                     "val_ece": m["ece"], "p50_ms": lat["p50"], "p95_ms": lat["p95"], "p99_ms": lat["p99"],
                     "images_per_s": lat["images_per_s"]})
        if code == "I00":
            y0, logits0 = y, views[0]

    # I05: ensemble các lần chạy tốt nhất (logit val 1-view đã lưu, cùng thứ tự file)
    if len(candidates) >= 2:
        probs_list, lats, members = [], [], []
        for s in candidates:
            n_, y_, lg_ = _val_logits(s)
            probs_list.append(softmax_np(lg_))
            mm = load_trained(s, device)
            lats.append(lat_of(224, 1, m=mm))
            members.append(f"{s['exp_id']}({s['backbone']})")
            del mm
            _empty_cache()
        p = I.ensemble_probs(probs_list)
        m = _metrics(y_, p)
        rows.append({"exp_id": "I05", "method": f"ensemble {len(candidates)} mô hình (trung bình xác suất)",
                     "model": " + ".join(members), "K": len(candidates), "img_size": 224,
                     "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"],
                     "p50_ms": sum(l["p50"] for l in lats), "p95_ms": sum(l["p95"] for l in lats),
                     "p99_ms": sum(l["p99"] for l in lats),
                     "images_per_s": 1000.0 / sum(l["p50"] for l in lats),
                     "note": "độ trễ = tổng độ trễ từng mô hình (chạy tuần tự)"})

    # I06: trọng số EMA (T08) so với không EMA (T00): cùng chi phí suy luận
    if ema_pair is not None:
        for s, tag in ((ema_pair[1], "có EMA"), (ema_pair[0], "không EMA")):
            rows.append({"exp_id": "I06" if tag == "có EMA" else "I06_ref", "method": f"trọng số {tag}",
                         "model": f"{s['exp_id']} seed{s['seed']}", "K": 1, "img_size": 224,
                         "val_macro_f1": s["val_macro_f1"], "val_top1": s["val_top1"], "val_ece": s["val_ece"],
                         "p50_ms": base_lat["p50"], "p95_ms": base_lat["p95"], "p99_ms": base_lat["p99"],
                         "images_per_s": base_lat["images_per_s"],
                         "note": "chi phí suy luận như I00 (khác ở lúc train)"})

    # I07: temperature scaling trên logit I00
    T = I.fit_temperature(logits0, y0)
    p_ts = I.apply_temperature(logits0, T)
    m_ts = _metrics(y0, p_ts)
    ece_cf, _ = _ece_crossfit(logits0, y0)
    ece_before = _metrics(y0, softmax_np(logits0))["ece"]
    rows.append({"exp_id": "I07", "method": f"temperature scaling (T={T:.3f}, khớp trên val)", "model": who,
                 "K": 1, "img_size": 224, "val_macro_f1": m_ts["macro_f1"], "val_top1": m_ts["top1"],
                 "val_ece": m_ts["ece"], "val_ece_before": ece_before, "val_ece_crossfit": ece_cf,
                 "temperature": T, "p50_ms": base_lat["p50"], "p95_ms": base_lat["p95"],
                 "p99_ms": base_lat["p99"], "images_per_s": base_lat["images_per_s"],
                 "note": "ECE sau TS đo trên chính val (in-sample); val_ece_crossfit: khớp T ở nửa val, đo nửa kia"})

    # I08: gộp BN + FP16/AMP: độ chính xác và độ trễ
    x_chk = torch.randn(2, 3, 224, 224, device=device)
    fused = I.fuse_conv_bn(model, check_input=x_chk)
    variants = [("I08_fused_fp32", "gộp BN, FP32", fused, "fp32", True)]
    if dev == "cuda":
        variants += [("I08_amp", "AMP (autocast FP16)", model, "amp", False),
                     ("I08_fp16", "FP16 (model.half())", model, "fp16", False),
                     ("I08_fused_fp16", "gộp BN + FP16", fused, "fp16", True)]
    mean, std = _norm_of(model)
    tf = D.build_transforms(False, 224, mean=mean, std=std)
    loader = D.make_loader(val_df, cfg.images_dir, tf, batch_size=64, train=False,
                           num_workers=cfg.num_workers, cache=cfg.cache_dir or False)
    for code, label, mm, dtype, is_fused in variants:
        if dtype == "fp16":
            mh = copy.deepcopy(mm).half()
            _, y8, lg8 = I.predict_logits(mh, loader, device, view=lambda t: t.half())
            del mh
        else:
            _, y8, lg8 = I.predict_logits(mm, loader, device, amp=(dtype == "amp"))
        m8 = _metrics(y8, softmax_np(lg8.astype(np.float64)))
        lat = lat_of(224, 1, dtype=dtype, m=mm)
        rows.append({"exp_id": code, "method": label, "model": who, "K": 1, "img_size": 224,
                     "val_macro_f1": m8["macro_f1"], "val_top1": m8["top1"], "val_ece": m8["ece"],
                     "p50_ms": lat["p50"], "p95_ms": lat["p95"], "p99_ms": lat["p99"],
                     "images_per_s": lat["images_per_s"],
                     "note": f"sai số gộp BN {getattr(fused, 'fuse_max_abs_diff', float('nan')):.1e}" if is_fused else ""})

    # Bảng Latency: batch 1 và batch 32, các dtype, có/không gộp BN
    for dtype in (["fp32", "amp", "fp16"] if dev == "cuda" else ["fp32"]):
        for batch in (1, 32):
            lat_rows.append(_lat_row(f"{who} 1-view", lat_of(224, 1, dtype, batch=batch), False))
            lat_rows.append(_lat_row(f"{who} 1-view", lat_of(224, 1, dtype, m=fused, batch=batch), True))
    lat_rows.append(_lat_row(f"{who} TTA lật K=2", lat_of(224, 2), False))
    lat_rows.append(_lat_row(f"{who} TTA 10 crop", lat_of(224, 10), False))

    inf = pd.DataFrame(rows)
    inf["rel_cost_vs_I00"] = inf["p50_ms"] / base_lat["p50"]
    lat_df = pd.DataFrame(lat_rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    inf.to_csv(cache_csv, index=False)
    lat_df.to_csv(lat_csv, index=False)
    del model, fused
    _empty_cache()
    return inf, lat_df


def select_inference(inf_df: pd.DataFrame, override: str | None = None) -> tuple[str, str]:
    """Quy tắc: trong các phương pháp MỘT mô hình (I00-I04), lấy macro-F1 val cao nhất; chỉ dùng nếu hơn
    I00 ít nhất INFER_MIN_GAIN, ngược lại giữ I00. Ensemble không chọn vì chung kết phải train lại ≥ 3 seed
    cho mọi thành viên (vượt ngân sách GPU). Temperature scaling luôn áp dụng thêm (không đổi accuracy)."""
    if override:
        return override, f"chọn thủ công: {override}"
    single = inf_df[inf_df.exp_id.isin(METHODS.keys()) & inf_df.val_macro_f1.notna()]
    i00 = single.loc[single.exp_id == "I00", "val_macro_f1"].item()
    best = single.loc[single.val_macro_f1.idxmax()]
    if best.val_macro_f1 - i00 >= INFER_MIN_GAIN:
        return best.exp_id, (f"{best.exp_id} ({best.method}) macro-F1 val {best.val_macro_f1:.4f} > I00 {i00:.4f} "
                             f"(Δ={best.val_macro_f1 - i00:+.4f} ≥ {INFER_MIN_GAIN}); chi phí x{best.rel_cost_vs_I00:.1f}.")
    return "I00", (f"Không phương pháp một mô hình nào hơn I00 ({i00:.4f}) từ {INFER_MIN_GAIN} trở lên "
                   f"(tốt nhất {best.exp_id} {best.val_macro_f1:.4f}); giữ 1 view.")


# ============================================================================= Bước 4
def final_predict(summary: dict, method_code: str, exp_id: str, pred_dir: str | Path, device,
                  calibrate: bool = True, realtime_id: str | None = None) -> dict:
    """Chạy VAL và TEST đúng một lần cho một seed, ghi file dự đoán đúng định dạng eval.py.

    - <exp_id>_seed<k>_val.csv / _test.csv : phương pháp `method_code` (+ temperature scaling nếu calibrate,
      T khớp trên VAL của chính seed đó)
    - <exp_id>uncal_seed<k>_test.csv        : cùng phương pháp, chưa temperature scaling (I4a)
    - <realtime_id>_seed<k>_test.csv        : 1 view + TS (cấu hình thời gian thực, I5)
    Nếu file test đã tồn tại thì KHÔNG chạy lại (quy tắc: test một lần mỗi seed).
    """
    pred_dir = Path(pred_dir)
    seed = summary["seed"]
    out_test = pred_dir / f"{exp_id}_seed{seed}_test.csv"
    info_path = Path(summary["run_dir"]) / f"final_{exp_id}.json"
    if out_test.exists():
        print(f"{out_test.name} đã có -> không chạy test lần nữa")
        return _read_json(info_path) if info_path.exists() else {}

    cfg = load_config(summary)
    model = load_trained(summary, device)
    _, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    meth = METHODS[method_code]
    info = {"exp_id": exp_id, "seed": seed, "source_run": f"{summary['exp_id']} seed{seed}",
            "method": method_code, "calibrate": calibrate}

    # --- VAL: khớp T (chỉ dùng val)
    nv, yv, views_v = method_logits(model, cfg, val_df, meth, device)
    pv, zv = combine(views_v, meth)
    T = I.fit_temperature(zv, yv) if calibrate else 1.0
    save_predictions(pred_dir / f"{exp_id}_seed{seed}_val.csv", nv, yv, I.apply_temperature(zv, T) if calibrate else pv)
    info["T"] = T
    if realtime_id:
        nv1, yv1, v1 = method_logits(model, cfg, val_df, METHODS["I00"], device)
        T1 = I.fit_temperature(v1[0], yv1)
        info["T_realtime"] = T1

    # --- TEST: đúng một lần
    nt, yt, views_t = method_logits(model, cfg, test_df, meth, device)
    pt, zt = combine(views_t, meth)
    if calibrate:
        save_predictions(pred_dir / f"{exp_id}uncal_seed{seed}_test.csv", nt, yt, pt)
        save_predictions(out_test, nt, yt, I.apply_temperature(zt, T))
    else:
        save_predictions(out_test, nt, yt, pt)
    if realtime_id:
        if method_code == "I00":
            zt1 = views_t[0]
        else:
            _, _, v1t = method_logits(model, cfg, test_df, METHODS["I00"], device)
            zt1 = v1t[0]
        save_predictions(pred_dir / f"{realtime_id}_seed{seed}_test.csv", nt, yt, I.apply_temperature(zt1, T1))
    _write_json(info_path, info)
    del model
    _empty_cache()
    print(f"[{exp_id} seed{seed}] đã ghi dự đoán val/test ({method_code}, T={T:.3f})")
    return info


def step4_final(paths: dict, backbone: str, recipe: dict, method_code: str, pred_dir: str | Path, device,
                seeds=(0, 1, 2), **overrides) -> dict:
    """Train (hoặc dùng lại) F01 và mốc T00 với các seed, rồi ghi dự đoán test mỗi seed một lần."""
    out = {"F01": [], "T00": []}
    for seed in seeds:
        fs = run_or_load(Config(exp_id="F01", backbone=backbone, desc="final", seed=seed,
                                **{**overrides, **recipe}, **paths))
        bs = run_or_load(Config(exp_id="T00", backbone=backbone, desc=f"{backbone}_baseline", seed=seed,
                                **overrides, **paths))
        out["F01"].append(fs)
        out["T00"].append(bs)
    for fs, bs in zip(out["F01"], out["T00"]):
        final_predict(fs, method_code, "F01", pred_dir, device, calibrate=True, realtime_id="F01rt")
        final_predict(bs, "I00", "T00", pred_dir, device, calibrate=False)
    return out


def final_configs(backbone: str, recipe_id: str, recipe: dict, method: str) -> dict:
    """Mô tả cấu hình (backbone + công thức + suy luận) của từng nhóm trong sheet Final."""
    rec = f"{recipe_id} {recipe}" if recipe else "T00 (công thức nền)"
    meth = METHODS[method]["label"]
    return {"F01": f"{backbone} + {rec} + {method} ({meth}) + temperature scaling",
            "F01uncal": f"{backbone} + {rec} + {method} ({meth}), chưa temperature scaling",
            "F01rt": f"{backbone} + {rec} + I00 (1 view) + temperature scaling [thời gian thực]",
            "T00": f"{backbone} + T00 (công thức nền) + I00 (1 view) [mốc]"}


def final_table(pred_dir: str | Path, groups=("F01", "T00", "F01rt", "F01uncal"),
                configs: dict | None = None) -> pd.DataFrame:
    """Bảng Final: mỗi seed một dòng + dòng mean ± std, tính từ file dự đoán bằng eval.compute_metrics.
    configs: {nhóm: mô tả cấu hình} (xem final_configs)."""
    configs = configs or {}
    import eval as ev
    pred_dir = Path(pred_dir)
    rows = []
    for g in groups:
        files = sorted(pred_dir.glob(f"{g}_seed*_test.csv"))
        if not files:
            continue
        per = []
        for f in files:
            p = ev.read_pred(str(f))
            mt = compute_metrics(p.y_true, p.y_pred, p.probs)
            vf = pred_dir / f.name.replace("_test.csv", "_val.csv")
            mv = None
            if vf.exists():
                pv = ev.read_pred(str(vf))
                mv = compute_metrics(pv.y_true, pv.y_pred, pv.probs)
            r = {"exp_id": g, "config": configs.get(g, ""), "seed": p.seed, "val_macro_f1": mv["macro_f1"] if mv else np.nan,
                 "test_macro_f1": mt["macro_f1"], "test_top1": mt["top1"], "test_balanced_acc": mt["balanced_acc"],
                 "test_ece": mt["ece"], **{f"test_recall_{k.split()[0].lower()}": mt["recall"][i] for k, i in RARE.items()}}
            per.append(r)
        rows += per
        agg = {"exp_id": g, "config": configs.get(g, ""), "seed": f"mean ± std ({len(per)} seed)"}
        for k in per[0]:
            if k in ("exp_id", "config", "seed"):
                continue
            mu, sd = mean_std([r[k] for r in per])
            agg[k] = f"{mu:.4f} ± {sd:.4f}"
        rows.append(agg)
    return pd.DataFrame(rows)


# ============================================================================= Bước 5: hình và xlsx
def plot_backbones(bb_df: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for _, r in bb_df.iterrows():
        ax[0].scatter(r.latency_b1_p50_ms, r.val_macro_f1, s=30 + 8 * r.params_M)
        ax[0].annotate(r.backbone, (r.latency_b1_p50_ms, r.val_macro_f1), fontsize=8, xytext=(4, 4),
                       textcoords="offset points")
        ax[1].scatter(r.gmacs, r.latency_b1_p50_ms)
        ax[1].annotate(r.backbone, (r.gmacs, r.latency_b1_p50_ms), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[0].set(xlabel="độ trễ p50 batch 1 (ms)", ylabel="macro-F1 val", title="Backbone: chất lượng vs độ trễ (cỡ = #params)")
    ax[1].set(xlabel="GMAC", ylabel="độ trễ p50 batch 1 (ms)", title="FLOPs có dự đoán được độ trễ?")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tradeoff(inf_df: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    d = inf_df[inf_df.val_macro_f1.notna() & inf_df.p50_ms.notna()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d.p95_ms, d.val_macro_f1)
    for _, r in d.iterrows():
        ax.annotate(r.exp_id, (r.p95_ms, r.val_macro_f1), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axvline(LATENCY_BUDGET_MS, color="red", ls="--", lw=1, label=f"ngân sách {LATENCY_BUDGET_MS:.0f} ms")
    ax.set_xscale("log")
    ax.set(xlabel="độ trễ p95 batch 1 (ms, thang log)", ylabel="macro-F1 val",
           title="Đánh đổi độ chính xác – độ trễ của các phương pháp suy luận")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(pred_glob_dir: str | Path, group: str, path: str | Path) -> np.ndarray:
    """Ma trận nhầm lẫn TỔNG qua các seed (số ảnh), hàng = nhãn thật, cột = dự đoán."""
    import matplotlib.pyplot as plt
    import eval as ev
    cms = []
    for f in sorted(Path(pred_glob_dir).glob(f"{group}_seed*_test.csv")):
        p = ev.read_pred(str(f))
        cms.append(ev.confusion_matrix(p.y_true, p.y_pred))
    cm = np.sum(cms, 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(D.CLASS_NAMES)), D.CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(D.CLASS_NAMES)), D.CLASS_NAMES, fontsize=8)
    ax.set(xlabel="dự đoán", ylabel="nhãn thật", title=f"Ma trận nhầm lẫn test – {group} (tổng {len(cms)} seed)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm


def plot_errors(pred_file: str | Path, images_dir: str | Path, path: str | Path,
                pairs=((0, 7), (7, 0)), n: int = 6) -> None:
    """Ảnh test bị đoán sai cho các cặp (thật -> dự đoán), mặc định Chinee Apple <-> Snake Weed."""
    import matplotlib.pyplot as plt
    from PIL import Image
    import eval as ev
    p = ev.read_pred(str(pred_file))
    fig, axes = plt.subplots(len(pairs), n, figsize=(2.2 * n, 2.6 * len(pairs)), squeeze=False)
    for r, (t, q) in enumerate(pairs):
        idx = np.where((p.y_true == t) & (p.y_pred == q))[0][:n]
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(idx):
                i = idx[c]
                ax.imshow(Image.open(Path(images_dir) / p.filenames[i]))
                ax.set_title(f"conf {p.probs[i].max():.2f}", fontsize=7)
        axes[r, 0].text(-0.1, 0.5, f"{D.CLASS_NAMES[t]}\n→ {D.CLASS_NAMES[q]}\n({len(np.where((p.y_true == t) & (p.y_pred == q))[0])} ảnh)",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def per_class_table(eval_out: str | Path, tags=("F01", "T00", "F01rt")) -> pd.DataFrame:
    """Sheet PerClass từ file <tag>_per_class.csv do `eval.py score --out` ghi ra."""
    frames = []
    for t in tags:
        f = Path(eval_out) / f"{t}_per_class.csv"
        if f.exists():
            d = pd.read_csv(f)
            d.insert(0, "config", t)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary_table(bb_df, tr_df, inf_df, final_df) -> pd.DataFrame:
    """Top 10 cấu hình theo macro-F1 val + dòng chung kết và mốc trên test."""
    rows = []
    for r in bb_df.itertuples():
        rows.append({"exp_id": r.exp_id, "loại": "backbone", "mô tả": r.backbone, "val_macro_f1": r.val_macro_f1,
                     "val_top1": r.val_top1, "p95_b1_ms": r.latency_b1_p95_ms})
    for r in tr_df.itertuples():
        if r.exp_id == "T00":
            continue
        rows.append({"exp_id": r.exp_id, "loại": "huấn luyện", "mô tả": r.change_vs_T00,
                     "val_macro_f1": r.val_macro_f1, "val_top1": r.val_top1, "p95_b1_ms": np.nan})
    for r in inf_df[inf_df.val_macro_f1.notna()].itertuples():
        rows.append({"exp_id": r.exp_id, "loại": "suy luận", "mô tả": r.method, "val_macro_f1": r.val_macro_f1,
                     "val_top1": r.val_top1, "p95_b1_ms": r.p95_ms})
    top = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).head(10)
    fin = final_df[final_df.seed.astype(str).str.startswith("mean")][
        ["exp_id", "seed", "val_macro_f1", "test_macro_f1", "test_top1", "test_ece"]]
    fin = fin.rename(columns={"seed": "mô tả"}).assign(loại="chung kết (test)")
    return pd.concat([top, fin], ignore_index=True)


def write_results_xlsx(path: str | Path, sheets: dict[str, pd.DataFrame], highlight: dict[str, str] | None = None) -> None:
    """Ghi results.xlsx: cố định hàng tiêu đề, 4 chữ số thập phân, tự giãn cột, tô đậm dòng tốt nhất
    (highlight = {sheet: tên cột để lấy max})."""
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    highlight = highlight or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for j, col in enumerate(df.columns, 1):
                width = max([len(str(col))] + [len(f"{v:.4f}" if isinstance(v, float) else str(v)) for v in df[col]])
                ws.column_dimensions[get_column_letter(j)].width = min(60, width + 2)
                if pd.api.types.is_float_dtype(df[col]):
                    for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                        row[0].number_format = "0.0000"
            col = highlight.get(name)
            if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
                best = int(pd.to_numeric(df[col], errors="coerce").idxmax()) + 2
                for cell in ws[best]:
                    cell.fill = PatternFill("solid", fgColor="FFF2CC")
                    cell.font = Font(bold=True)


# ============================================================================= tiện ích cho notebook
def run_summary(paths: dict, exp_id: str, seed: int = 0) -> dict:
    return _read_json(Path(paths["out_dir"]) / exp_id / f"seed{seed}" / "summary.json")


def add_combo_row(tr_df: pd.DataFrame, combo_summary: dict, reason: str) -> pd.DataFrame:
    base = tr_df.loc[tr_df.exp_id == "T00", "val_macro_f1"].item()
    row = _training_row(combo_summary, COMBO_ID, "kết hợp", reason, base)
    return pd.concat([tr_df[tr_df.exp_id != COMBO_ID], pd.DataFrame([row])], ignore_index=True)


def ensemble_candidates(paths: dict, *dfs: pd.DataFrame, k: int = 3) -> list[dict]:
    """k lần chạy có macro-F1 val cao nhất (Bước 1-2), bỏ các lần 'dùng lại' để không trùng mô hình."""
    allr = pd.concat([d[["exp_id", "val_macro_f1"]] for d in dfs]).drop_duplicates("exp_id")
    out = []
    for e in allr.sort_values("val_macro_f1", ascending=False).exp_id:
        s = run_summary(paths, e)
        if not s.get("reused_from"):
            out.append(s)
        if len(out) == k:
            break
    return out


def realtime_p95(inf_df: pd.DataFrame) -> tuple[float, str]:
    """p95 batch 1 của cấu hình thời gian thực F01rt (1 view). Dự đoán chung kết chạy với AMP trên GPU,
    nên lấy dòng I08_amp nếu có (cùng dtype với file dự đoán), ngược lại lấy I00 (FP32)."""
    for code in ("I08_amp", "I00"):
        r = inf_df[inf_df.exp_id == code]
        if len(r) and pd.notna(r.p95_ms.iloc[0]):
            return float(r.p95_ms.iloc[0]), code
    raise ValueError("chưa có số đo độ trễ I00")


def env_info() -> dict:
    import platform

    import timm
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda": torch.version.cuda}


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v).replace("|", "/")


def md_table(df: pd.DataFrame, cols: list[str] | None = None, digits: int = 4) -> str:
    """Bảng Markdown (không cần thư viện tabulate)."""
    d = df[[c for c in cols if c in df.columns]] if cols else df
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(_fmt(v, digits) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


REPORT_TEMPLATE = """# Báo cáo Lab Day 2 — DeepWeeds · Nguyễn Văn Quốc Việt · 2A202602973

> Bản nháp sinh tự động từ kết quả chạy thật (bảng số liệu, lý do chọn cấu hình). Các mục **TODO** là phần
> nhận xét, phân tích bạn tự viết.

## 1. Tóm tắt
- Cấu hình tốt nhất F01: backbone `{backbone}` + công thức `{recipe_id}` + suy luận `{method}` + temperature scaling.
- Test (mean ± std, {n_seed}): macro-F1 **{f_mf1}**, top-1 **{f_top1}**, ECE {f_ece}.
- Mốc T00 + I00: macro-F1 {b_mf1}, top-1 {b_top1}.
- TODO: 3–5 dòng kết luận chính (yếu tố nào đóng góp nhiều nhất, có vượt nhiễu không).

## 2. Dữ liệu và thiết lập
- DeepWeeds, fold 0 chia sẵn (train_subset0 / val_subset0 / test_subset0), không sửa, không lọc, không chia lại.
{split_md}- Chỉ số chính: macro-F1 (9 lớp, `eval.compute_metrics`); phụ: top-1, balanced accuracy, F1/recall từng lớp, ECE 15 bin.
- Công thức nền T00: ImageNet pretrained, tinh chỉnh toàn bộ, RandomResizedCrop 224 + lật ngang, AdamW (LR backbone 1e-4,
  head 1e-3, weight decay 0,05 trừ norm/bias), warmup 1 epoch + cosine theo bước, CE, batch 64, {epochs} epoch, AMP,
  channels_last, chọn checkpoint theo macro-F1 val (hòa lấy epoch sớm hơn).
- Val/test: resize 256 → center crop 224, chuẩn hoá theo mean/std của trọng số. Seed 0 cho quét sàng; 0, 1, 2 cho chung kết.
- Môi trường: {env}.
{pipeline_md}- TODO: chèn biểu đồ phân bố lớp và ảnh mẫu từ notebook (Bước 0); nhận xét mất cân bằng và cặp loài dễ nhầm.

## 3. So sánh backbone (Bước 1, 1 seed)
{bb_table}

Phân tích tự động từ `history.csv` (epochs_to_99pct: epoch đầu đạt 99% macro-F1 tốt nhất; overfit: val loss
tăng > 0,05 sau điểm thấp nhất trong khi train loss vẫn giảm):

{bb_analysis}

Tương quan thứ hạng (5 backbone, chỉ tham khảo): {bb_corr}

![backbones](figures/backbones.png)

![backbone curves](figures/backbones_curves.png)

**Lựa chọn (quy tắc trên val):** {why_bb}

TODO: backbone nào hội tụ nhanh nhất / có overfit không (bảng trên + `curves/B0x_*.png`); thứ hạng trên DeepWeeds so
với ImageNet; FLOPs có dự đoán được thời gian train và độ trễ không (tương quan trên; slide trang 43). Ghi rõ đây là
kết quả 1 seed: chênh lệch < {noise} macro-F1 là không phân biệt được.

## 4. Công thức huấn luyện (Bước 2, 1 seed, ngưỡng nhiễu {noise})
{tr_table}

Mỗi T0x khác T00 đúng một yếu tố. Kết hợp theo kiểu tham lam theo trục: {why_combo}

**Công thức chung kết:** {why_recipe}

TODO: yếu tố nào giúp / không giúp và vì sao (liên hệ slide); chênh lệch nhỏ hơn nhiễu ghi "không phân biệt được";
hiệu ứng có cộng dồn khi kết hợp không.

## 5. Suy luận (Bước 3, trên val)
{inf_table}

![tradeoff](figures/tradeoff.png)

Độ trễ: warmup 10 lần, `torch.cuda.synchronize()` trước và sau, 100 lần đo, báo p50/p95/p99, chỉ đo forward
(không tính tiền xử lý), GPU {gpu}. Bảng đầy đủ (batch 1 và 32, FP32/AMP/FP16, có/không gộp BN) ở sheet Latency.

**Phương pháp cho chung kết:** {why_inf} Temperature scaling (T khớp trên val của từng seed) luôn áp dụng thêm.

TODO: TTA tăng bao nhiêu và tốn bao nhiêu lần độ trễ; ECE trước/sau; độ phân giải kiểm tra (FixRes);
phương pháp nào hợp ngoại tuyến, phương pháp nào hợp thời gian thực.

## 6. Cấu hình tốt nhất và kết quả test (Bước 4)
Test chạy đúng một lần cho mỗi seed, sau khi đã chốt mọi lựa chọn trên val. F01rt = cùng mô hình F01, 1 view
(cấu hình thời gian thực, p95 batch 1 = {rt_p95:.1f} ms theo {rt_src}). F01uncal = F01 chưa temperature scaling.

{fin_table}

Theo lớp (test, mean qua seed):

{pc_table}

![confusion](figures/confusion_F01.png)

![errors](figures/errors_chinee_snake.png)

Tự chấm phần I (`eval.py grade`, đề xuất):

```
{grade}
```

TODO: lớp còn nhầm nhiều nhất và giả thuyết (xem ảnh sai Chinee Apple ↔ Snake Weed); so với bài báo
(95,7% / 95,1%, điều kiện huấn luyện khác: 100 epoch, augmentation mạnh).

## 7. Kết luận và khuyến nghị
- TODO: cấu hình tốt nhất, tốt hơn mốc bao nhiêu, có vượt std không.
- TODO: yếu tố đóng góp nhiều nhất (backbone, huấn luyện hay suy luận).
- TODO: triển khai trên robot với ngân sách 30–100 ms/khung: chọn gì và vì sao.

## 8. Hạn chế
- Quét sàng Bước 1–2 chỉ 1 seed; chung kết 3 seed; chỉ fold 0.
- Chia ngẫu nhiên, không theo địa điểm, nên điểm test có thể lạc quan khi gặp địa điểm/mùa mới.
- Giảm bớt do ngân sách GPU (một session Kaggle 12 giờ): {epoch_note}5 backbone, ablation trên 1 backbone. T00 seed 0 dùng lại lần
  chạy Bước 1, F01 seed 0 dùng lại lần chạy tốt nhất Bước 2 (cùng cấu hình, cùng seed; ghi ở cột reused_from).
- TODO: thí nghiệm thất bại hoặc bất thường (nếu có).

## 9. Phụ lục
- Cấu hình đầy đủ từng lần chạy: `runs/<exp_id>/seed<k>/config.json`; bảng đầy đủ trong `results.xlsx`.
- Notebook: `code/lab_day2.ipynb` (link Colab: TODO).
"""


def write_report_draft(path: str | Path, *, env: dict, split_stats: dict | None, backbone: str, bb_df, why_bb,
                       pipeline_checks: dict | None = None, bb_corr: dict | None = None,
                       tr_df, why_combo, recipe_id: str, why_recipe, inf_df, method: str, why_inf, fin_df, pc_df,
                       rt_p95: float, rt_src: str, epochs: int, grade_text: str = "", overwrite: bool = False) -> Path:
    """Bản nháp report.md theo dàn ý GUIDE mục 6.3: điền sẵn BẢNG SỐ LIỆU thật và lý do chọn cấu hình;
    phần nhận xét để dạng TODO. Không ghi đè file đã có (trừ khi overwrite=True)."""
    path = Path(path)
    if path.exists() and not overwrite:
        print(f"{path} đã có -> không ghi đè (overwrite=True để ghi lại)")
        return path
    fmean = fin_df[fin_df.seed.astype(str).str.startswith("mean")].set_index("exp_id")
    get = lambda g, c: fmean.loc[g, c] if g in fmean.index and c in fmean.columns else "?"  # noqa: E731
    split_md = ""
    if split_stats:
        split_md = (f"- Số ảnh: train {split_stats['n']['train']}, val {split_stats['n']['val']}, "
                    f"test {split_stats['n']['test']} ("
                    + ", ".join(f"{k} {v:.2%}" for k, v in split_stats["frac"].items())
                    + f"). Giao từng cặp: {split_stats['overlap']}; hợp ba tập: {split_stats['union']}; "
                    f"file thiếu: {split_stats['missing']}; nhãn lệch labels.csv gốc: {split_stats.get('label_mismatch')}.\n"
                    f"- Tỉ lệ lớp nhiều nhất / ít nhất: {split_stats.get('imbalance_ratio', float('nan')):.2f}. "
                    "Số ảnh mỗi lớp trong từng tập:\n\n"
                    + md_table(pd.DataFrame(split_stats["per_class"]).rename_axis("lớp").reset_index(), digits=0)
                    + "\n\n")
    pipeline_md = ""
    if pipeline_checks:
        pipeline_md = ("- Kiểm tra pipeline (Bước 0, ResNet-50): "
                       + "; ".join(f"{k} = {v}" for k, v in pipeline_checks.items())
                       + f" (kỳ vọng loss ban đầu ≈ ln 9 = {np.log(9):.3f}).\n")
    txt = REPORT_TEMPLATE.format(
        backbone=backbone, recipe_id=recipe_id, epochs=epochs,
        epoch_note=(f"{epochs} epoch cho mọi thí nghiệm (GUIDE gợi ý 10–15), " if epochs < 10 else
                    f"{epochs} epoch (bài báo ~100 epoch nên số tuyệt đối có thể thấp hơn bài báo), "), method=method, n_seed=get("F01", "seed"),
        f_mf1=get("F01", "test_macro_f1"), f_top1=get("F01", "test_top1"), f_ece=get("F01", "test_ece"),
        b_mf1=get("T00", "test_macro_f1"), b_top1=get("T00", "test_top1"), split_md=split_md, pipeline_md=pipeline_md,
        env=", ".join(f"{k} {v}" for k, v in env.items()), gpu=env.get("gpu"), noise=NOISE_F1,
        bb_table=md_table(bb_df, ["exp_id", "backbone", "weights_tag", "params_M", "gmacs", "best_epoch",
                                  "val_macro_f1", "val_top1", "train_time_per_epoch_s", "latency_b1_p50_ms",
                                  "latency_b1_p95_ms"]),
        why_bb=why_bb,
        bb_analysis=md_table(bb_df, ["exp_id", "backbone", "best_epoch", "epochs_to_99pct", "val_loss_min_epoch",
                                     "val_loss_rise", "overfit", "train_val_gap_last"]),
        bb_corr=", ".join(f"{k} = {v:.2f}" for k, v in (bb_corr or {}).items()) or "(chưa tính)",
        tr_table=md_table(tr_df, ["exp_id", "axis", "change_vs_T00", "val_macro_f1", "val_top1", "delta_vs_T00",
                                  "f1_chinee", "f1_snake", "reused_from"]),
        why_combo=why_combo, why_recipe=why_recipe,
        inf_table=md_table(inf_df, ["exp_id", "method", "K", "img_size", "val_macro_f1", "val_top1", "val_ece",
                                    "p50_ms", "p95_ms", "p99_ms", "rel_cost_vs_I00", "note"]),
        why_inf=why_inf, rt_p95=rt_p95, rt_src=rt_src, fin_table=md_table(fin_df),
        pc_table=md_table(pc_df) if len(pc_df) else "(chưa có)", grade=grade_text.strip())
    path.write_text(txt, encoding="utf-8")
    print("Đã ghi bản nháp", path)
    return path

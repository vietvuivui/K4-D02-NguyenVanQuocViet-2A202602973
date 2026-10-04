"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc, cùng định nghĩa lúc chấm.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# eval.py nằm ở gốc repo: <repo>/submissions/<sv>/code/train.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
if (_REPO_ROOT / "eval.py").exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dataset as ds_lib  # noqa: E402
import losses as loss_lib  # noqa: E402
import model as model_lib  # noqa: E402
from eval import compute_metrics, save_predictions  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png (mặc định = backbone)
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | flip_rot
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    cache_images: bool = False        # nạp ảnh vào RAM (dataset.DeepWeedsDataset)
    cache_dir: str | None = None      # cache memmap trên đĩa (khuyên dùng trên Colab, vd /content/cache)
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None   # None: không trọng số; 0: 1/n_c; >0: class-balanced
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    momentum: float = 0.9             # chỉ dùng cho sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True        # bố cục bộ nhớ NHWC: conv nhanh hơn với AMP trên GPU Tensor Core
    num_workers: int = 2
    deterministic: bool = False       # True: cudnn.deterministic (chậm hơn, tái lập tốt hơn)
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"        # ảnh biểu đồ training (nộp cùng bài)
    save_checkpoint: bool = True      # ghi best.pt vào run_dir (tắt nếu thiếu dung lượng)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    """curves/<exp_id>_<desc>.png cho seed 0; các seed khác thêm hậu tố _seed<k>."""
    desc = (cfg.desc or cfg.backbone).replace(" ", "_")
    suffix = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}{suffix}.png"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Cố định random, numpy, torch (CPU và CUDA).

    Mặc định cudnn.benchmark = True cho nhanh: kết quả giữa hai lần chạy cùng seed có thể lệch
    rất nhỏ do thuật toán conv không tất định. deterministic=True tắt điều này (chậm hơn).
    Seed của worker DataLoader do dataset.make_loader xử lý.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD + momentum, trục E) với 3 nhóm tham số (model.param_groups)."""
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr_backbone)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr_backbone, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"optimizer={cfg.optimizer!r} không hợp lệ (adamw | sgd)")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC (iteration).

    Hệ số nhân chung cho mọi nhóm, nên tỉ lệ LR head/backbone (10x) giữ nguyên suốt quá trình.
    warmup_epochs = 0 thì bỏ warmup.
    """
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ một bản sao riêng `self.module` để đánh giá. Áp dụng EMA cho mọi tensor số thực trong
    state_dict, gồm cả buffer running_mean/var của BatchNorm (giống timm ModelEmaV2), nên thống kê
    BN khớp với trọng số đã làm mượt; buffer số nguyên (num_batches_tracked) được chép thẳng.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        ema_sd = self.module.state_dict()
        for k, v in model.state_dict().items():
            e = ema_sd[k]
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                e.copy_(v)


def _autocast(cfg: Config, device):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=cfg.amp and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc" (NaN nếu mix), "lr_backbone", "lr_head", "lrs"}."""
    model_lib.set_train_mode(model)
    total_loss, correct, seen = 0.0, 0, 0
    lrs = []
    mem = torch.channels_last if cfg.channels_last else torch.contiguous_format
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=mem)
        y = y.to(device, non_blocking=True)
        with _autocast(cfg, device):
            if cfg.mix:
                x, targets = loss_lib.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
                logits = model(x)
                loss = loss_lib.mixed_loss(criterion, logits, targets)
            else:
                logits = model(x)
                loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        if ema is not None:
            ema.update(model)

        bs = y.size(0)
        total_loss += loss.item() * bs
        seen += bs
        if not cfg.mix:
            correct += (logits.argmax(1) == y).sum().item()

    lr_of = {g.get("name"): g["lr"] for g in optimizer.param_groups}
    return {"train_loss": total_loss / max(1, seen),
            "train_acc": correct / seen if (seen and not cfg.mix) else float("nan"),
            "lr_backbone": lrs[-1] if lrs else float("nan"),
            "lr_head": lr_of.get("head", float("nan")),
            "lrs": lrs}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool = True, channels_last: bool = False):
    """Chạy model ở chế độ eval. Trả về (filenames, y_true[N], logits[N, 9], loss) theo đúng thứ tự loader."""
    model.eval()
    names, ys, outs = [], [], []
    total_loss, seen = 0.0, 0
    mem = torch.channels_last if channels_last else torch.contiguous_format
    for x, y, f in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=mem)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            total_loss += criterion(logits, y).item() * y.size(0)
        seen += y.size(0)
        names.extend(f)
        ys.append(y.cpu())
        outs.append(logits.cpu())
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy(), total_loss / max(1, seen)


def softmax_np(logits) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics_from_logits(y_true, logits) -> dict:
    probs = softmax_np(logits)
    return compute_metrics(np.asarray(y_true), probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs: list[float] | None = None) -> None:
    """Vẽ đường cong training -> curves/<exp_id>_<desc>.png.

    Ô 1: loss train/val theo epoch. Ô 2: macro-F1 val, top-1 val (và acc train nếu không mix),
    đánh dấu epoch được chọn. Ô 3: LR backbone theo bước (nếu có).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ncol = 3 if lrs else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4))
    ax[0].plot(h["epoch"], h["train_loss"], "o-", label="train")
    ax[0].plot(h["epoch"], h["val_loss"], "o-", label="val")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(h["epoch"], h["val_macro_f1"], "o-", label="val macro-F1")
    ax[1].plot(h["epoch"], h["val_top1"], "s--", label="val top-1")
    if h["train_acc"].notna().any():
        ax[1].plot(h["epoch"], h["train_acc"], "^:", label="train acc")
    best = h.loc[h["val_macro_f1"].idxmax()]
    ax[1].axvline(best["epoch"], color="gray", ls=":", lw=1)
    ax[1].annotate(f"best {best['val_macro_f1']:.4f}\n@ epoch {int(best['epoch'])}",
                   (best["epoch"], best["val_macro_f1"]), textcoords="offset points", xytext=(-60, -30))
    ax[1].set(xlabel="epoch", ylabel="metric", title="Val metric")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    if lrs:
        ax[2].plot(np.arange(len(lrs)), lrs)
        ax[2].set(xlabel="bước", ylabel="LR backbone", title="Lịch LR")
        ax[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Ghi vào run_dir(cfg): config.json, history.csv, best.pt (nếu save_checkpoint), val_logits.npz
    (và test_logits.npz nếu save_test_predictions), summary.json.
    Ghi predictions/<exp_id>_seed<k>_val.csv; _test.csv CHỈ khi save_test_predictions (Bước 4).
    Checkpoint chọn theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn). Test không tham gia quyết định nào.
    """
    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")

    # 2. dữ liệu
    train_df, val_df, test_df = ds_lib.load_split(cfg.labels_dir, cfg.fold)
    ds_lib.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False,
                       labels_df=ds_lib.load_labels(cfg.labels_dir))

    # 4a. model trước loader để lấy đúng mean/std của trọng số
    model = model_lib.build_model(cfg.backbone, pretrained=True, num_classes=ds_lib.NUM_CLASSES,
                                  drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    pcfg = getattr(model, "pretrained_cfg", {}) or {}
    mean, std = pcfg.get("mean", ds_lib.IMAGENET_MEAN), pcfg.get("std", ds_lib.IMAGENET_STD)
    n_params = model_lib.count_params(model)
    try:
        gmacs = model_lib.count_gmacs(model, cfg.img_size)
    except Exception as e:  # đếm FLOPs không được thì không chặn việc train
        print("Không đếm được GMAC:", e)
        gmacs = float("nan")

    # 3. loader
    tf_train = ds_lib.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    tf_eval = ds_lib.build_transforms(False, cfg.img_size, mean=mean, std=std)
    kw = dict(images_dir=cfg.images_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              seed=cfg.seed, cache=cfg.cache_dir or cfg.cache_images)
    train_loader = ds_lib.make_loader(train_df, transform=tf_train, train=True, sampler=cfg.sampler, **kw)
    val_loader = ds_lib.make_loader(val_df, transform=tf_eval, train=False, **kw)

    # 4b. loss, optimizer, scheduler, scaler, EMA
    weight = None
    if cfg.class_weight_beta is not None or cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"], minlength=ds_lib.NUM_CLASSES)
        weight = loss_lib.class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    smoothing = cfg.label_smoothing if cfg.label_smoothing else 0.1
    criterion = loss_lib.build_criterion(cfg.loss, smoothing=smoothing, gamma=cfg.focal_gamma, weight=weight)
    criterion = criterion.to(device)
    eval_criterion = torch.nn.CrossEntropyLoss()  # val loss luôn là CE thường để so được giữa các cấu hình
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    # 5. vòng epoch
    history, all_lrs = [], []
    best_f1, best_epoch, best_state = -1.0, -1, None
    epoch_times = []
    print(f"[{cfg.exp_id} seed{cfg.seed}] {cfg.backbone} ({model.weights_tag}) | {n_params:.1f}M params | "
          f"{gmacs:.2f} GMAC | device {device}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        all_lrs += tr.pop("lrs")

        eval_model = ema.module if ema is not None else model
        _, y_val, logits_val, val_loss = evaluate(eval_model, val_loader, eval_criterion, device, cfg.amp, cfg.channels_last)
        m = metrics_from_logits(y_val, logits_val)
        row = {"epoch": epoch, **tr, "val_loss": val_loss, "val_top1": m["top1"],
               "val_macro_f1": m["macro_f1"], "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"],
               "epoch_time_s": epoch_times[-1]}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        improved = m["macro_f1"] > best_f1  # hòa thì giữ epoch sớm hơn
        if improved:
            best_f1, best_epoch = m["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            if cfg.save_checkpoint:
                torch.save(best_state, out / "best.pt")
                if ema is not None:  # trọng số THƯỜNG cùng epoch: so EMA vs không EMA trong cùng lần chạy (I06)
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "best_raw.pt")
        print(f"  epoch {epoch:2d} | train_loss {tr['train_loss']:.4f} | val_loss {val_loss:.4f} | "
              f"val top-1 {m['top1']:.4f} | val macro-F1 {m['macro_f1']:.4f}{' *' if improved else ''} | "
              f"{epoch_times[-1]:.0f}s")

    # 6. nạp checkpoint tốt nhất, lưu dự đoán val
    model.load_state_dict(best_state)
    names_val, y_val, logits_val, _ = evaluate(model, val_loader, eval_criterion, device, cfg.amp, cfg.channels_last)
    np.savez(out / "val_logits.npz", filenames=np.array(names_val), y_true=y_val, logits=logits_val)
    save_predictions(pred_path(cfg, "val"), names_val, y_val, softmax_np(logits_val))
    m_val = metrics_from_logits(y_val, logits_val)

    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weights_tag": model.weights_tag,
        "params_M": n_params, "gmacs": gmacs, "best_epoch": best_epoch,
        "val_macro_f1": m_val["macro_f1"], "val_top1": m_val["top1"],
        "val_balanced_acc": m_val["balanced_acc"], "val_ece": m_val["ece"],
        "train_time_per_epoch_s": float(np.mean(epoch_times)), "run_dir": str(out),
        "checkpoint": str(out / "best.pt") if cfg.save_checkpoint else None,
        "checkpoint_raw": str(out / "best_raw.pt") if (cfg.save_checkpoint and ema is not None) else None,
    }

    # 7. test: chỉ ở Bước 4, đúng MỘT lần với checkpoint đã chọn trên val
    if cfg.save_test_predictions:
        test_loader = ds_lib.make_loader(test_df, transform=tf_eval, train=False, **kw)
        names_t, y_t, logits_t, _ = evaluate(model, test_loader, eval_criterion, device, cfg.amp, cfg.channels_last)
        np.savez(out / "test_logits.npz", filenames=np.array(names_t), y_true=y_t, logits=logits_t)
        save_predictions(pred_path(cfg, "test"), names_t, y_t, softmax_np(logits_t))
        m_t = metrics_from_logits(y_t, logits_t)
        summary.update({"test_macro_f1": m_t["macro_f1"], "test_top1": m_t["top1"], "test_ece": m_t["ece"]})

    # 8. biểu đồ + tóm tắt
    np.save(out / "lr_steps.npy", np.asarray(all_lrs, dtype=np.float32))
    title = f"{cfg.exp_id} · {cfg.backbone} · seed {cfg.seed} · best epoch {best_epoch} (val macro-F1 {best_f1:.4f})"
    plot_curves(history, curve_path(cfg), title, all_lrs)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{cfg.exp_id} seed{cfg.seed}] xong: best epoch {best_epoch}, val macro-F1 {m_val['macro_f1']:.4f}, "
          f"top-1 {m_val['top1']:.4f}")
    return summary


def _cast(value: str, type_str: str):
    t = str(type_str).replace(" ", "")
    if value.lower() in ("none", "null"):
        if "None" not in t:
            raise ValueError(f"giá trị None không hợp lệ cho kiểu {type_str}")
        return None
    base = [p for p in t.split("|") if p != "None"][0]
    if base == "bool":
        if value.lower() in ("1", "true", "yes"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"không đọc được bool từ {value!r}")
    if base == "int":
        return int(value)
    if base == "float":
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        key, value = pair.split("=", 1)
        if key not in types:
            raise KeyError(f"'{key}' không có trong Config. Các khoá hợp lệ: {sorted(types)}")
        out[key] = _cast(value, types[key])
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè trường của Config")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()

"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
EXPECTED_FRAC = {"train": 0.6, "val": 0.2, "test": 0.2}
AUG_CHOICES = ("basic", "color", "trivial", "randaug", "flip_rot")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame, KHÔNG sửa/lọc/chia lại.
    """
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        assert not missing, f"{split}_subset{fold}.csv thiếu cột {missing}"
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và mỗi lớp (cảnh báo nếu lệch 60/20/20 quá 1 điểm phần trăm)
    2. giao từng cặp tập theo Filename phải rỗng
    3. hợp ba tập = 17.509 ảnh
    4. mọi Filename tồn tại trong images_dir
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    total = sum(len(d) for d in splits.values())

    n = {k: len(d) for k, d in splits.items()}
    frac = {k: v / total for k, v in n.items()}
    per_class = pd.DataFrame({k: d["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, d in splits.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(1)

    names = {k: set(d["Filename"]) for k, d in splits.items()}
    overlap = {"train∩val": len(names["train"] & names["val"]),
               "train∩test": len(names["train"] & names["test"]),
               "val∩test": len(names["val"] & names["test"])}
    union = len(names["train"] | names["val"] | names["test"])
    dup_within = {k: int(d["Filename"].duplicated().sum()) for k, d in splits.items()}

    images_dir = Path(images_dir)
    on_disk = {p.name for p in images_dir.iterdir()} if images_dir.is_dir() else set()
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)

    warnings = [f"{k}: {frac[k]:.2%} lệch kỳ vọng {EXPECTED_FRAC[k]:.0%} quá 1 điểm %"
                for k in splits if abs(frac[k] - EXPECTED_FRAC[k]) > 0.01]

    if verbose:
        print("Số ảnh mỗi tập:", n, "| tỉ lệ:", {k: f"{v:.2%}" for k, v in frac.items()})
        print("Số ảnh mỗi lớp:\n", per_class.to_string())
        print("Giao từng cặp:", overlap, "| hợp ba tập:", union, "| trùng trong tập:", dup_within)
        print(f"File thiếu trong {images_dir}: {len(missing)}", missing[:5])
        for w in warnings:
            print("CẢNH BÁO:", w, "-> báo giảng viên trước khi chạy tiếp")

    assert all(v == 0 for v in overlap.values()), f"Giao giữa các tập khác rỗng: {overlap}"
    assert all(v == 0 for v in dup_within.values()), f"Có tên file trùng trong một tập: {dup_within}"
    assert union == TOTAL_IMAGES, f"Hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    assert images_dir.is_dir(), f"Không thấy thư mục ảnh {images_dir}"
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:5]}"

    return {"n": n, "frac": frac, "per_class": per_class.to_dict(), "overlap": overlap,
            "union": union, "missing": len(missing), "warnings": warnings}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, eval_resize: int | None = None):
    """Tạo transform.

    Train, theo `aug` (trục B của GUIDE.md mục 3):
      - "basic"    : RandomResizedCrop(img_size) + lật ngang
      - "color"    : basic + ColorJitter (độ sáng/tương phản/bão hoà/hue nhẹ)
      - "trivial"  : basic + TrivialAugmentWide
      - "randaug"  : basic + RandAugment(num_ops=2, magnitude=9)
      - "flip_rot" : basic + lật dọc + xoay 90° ngẫu nhiên (ảnh chụp từ trên xuống, không có
                     hướng "trên/dưới" cố định, nên lật dọc/xoay là hợp lệ)
    Val/test: Resize(eval_resize) + CenterCrop(img_size). Mặc định eval_resize = img_size / 0.875
    (img_size = 224 -> 256 = kích thước gốc, tức chỉ center-crop 224). Không augmentation ngẫu nhiên.
    """
    normalize = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        resize = eval_resize or int(round(img_size / 0.875))
        return T.Compose([T.Resize(resize), T.CenterCrop(img_size), *normalize])

    if aug not in AUG_CHOICES:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn trong {AUG_CHOICES}")
    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "flip_rot":
        ops += [T.RandomVerticalFlip(), T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)])]
    return T.Compose([*ops, *normalize])


def build_memmap_cache(filenames: list[str], images_dir: str | Path, cache_dir: str | Path) -> Path:
    """Giải mã mọi ảnh MỘT lần thành mảng uint8 (N, H, W, 3) lưu ở `cache_dir/<băm danh sách file>.npy`.

    Các lần chạy sau (và các worker của DataLoader) đọc bằng np.load(mmap_mode="r"): không giải mã JPEG
    lặp lại, không nhân bản RAM giữa các worker. Chỉ cache dữ liệu đầu vào, không đổi nội dung ảnh.
    Nên đặt cache_dir trên đĩa cục bộ của Colab (vd /content/cache), không đặt trên Drive.
    """
    import hashlib

    key = hashlib.md5("\n".join(filenames).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{key}.npy"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(Path(images_dir) / filenames[0]) as im0:
        w, h = im0.size
    tmp = path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(filenames), h, w, 3))
    for i, f in enumerate(filenames):
        with Image.open(Path(images_dir) / f) as im:
            im = im.convert("RGB")
            assert im.size == (w, h), f"{f}: kích thước {im.size} khác {(w, h)}, không cache được"
            arr[i] = np.asarray(im)
    arr.flush()
    del arr
    tmp.replace(path)
    return path


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    cache:
      - False      : đọc và giải mã JPEG mỗi lần
      - True       : nạp ảnh đã giải mã vào RAM (list PIL)
      - "<thư mục>": cache memmap trên đĩa (build_memmap_cache), khuyên dùng trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool | str = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).to_numpy()
        self._cache = None
        self._mmap_path = None
        self._mmap = None
        if isinstance(cache, (str, Path)) and cache:
            self._mmap_path = build_memmap_cache(self.filenames, self.images_dir, cache)
        elif cache:
            self._cache = [self._load(f) for f in self.filenames]

    def _load(self, filename: str) -> Image.Image:
        with Image.open(self.images_dir / filename) as im:
            return im.convert("RGB")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        if self._mmap_path is not None:
            if self._mmap is None:  # mở lười trong từng worker
                self._mmap = np.load(self._mmap_path, mmap_mode="r")
            img = Image.fromarray(np.array(self._mmap[i]))
        elif self._cache is not None:
            img = self._cache[i]
        else:
            img = self._load(self.filenames[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), self.filenames[i]


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache: bool | str = False) -> DataLoader:
    """Tạo DataLoader.

    - train=True: shuffle, hoặc sampler="balanced" (WeightedRandomSampler, trọng số 1/n_lớp,
      lấy đủ len(df) mẫu mỗi epoch, có hoàn lại); drop_last=True để BatchNorm ổn định.
    - train=False: không shuffle, giữ đúng thứ tự df (để ghép logit với Filename).
    - generator + worker_init_fn theo seed để tái lập thứ tự batch và augmentation.
    """
    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)

    smp = None
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        w = 1.0 / counts[ds.labels]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
    elif sampler not in (None, "balanced"):
        raise ValueError(f"sampler={sampler!r} không hợp lệ (None | 'balanced')")

    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=train and smp is None, sampler=smp,
        drop_last=train, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker, generator=g,
    )

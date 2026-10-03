"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (đọc ảnh, resize, normalize): chỉ đo forward của model trên tensor đã
    nằm sẵn trên thiết bị.
"""
from __future__ import annotations

import copy
import platform
import time

import numpy as np
import torch

DTYPES = ("fp32", "amp", "fp16")


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây). `sync`: hàm đồng bộ (torch.cuda.synchronize) hoặc None trên CPU."""
    assert iters >= 50, "cần >= 50 lần đo"
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "std": float(t.std(ddof=1)),
            "n": iters, "warmup": warmup}


def _device_name(device: str) -> str:
    if torch.device(device).type == "cuda":
        return torch.cuda.get_device_name(torch.device(device))
    return platform.processor() or platform.machine() or "cpu"


def _prepare(model, dtype: str, device: str):
    """Trả về (model đã chuẩn bị, dtype của input, có autocast không). fp16 làm trên bản sao."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype={dtype!r} không hợp lệ, chọn trong {DTYPES}")
    if dtype == "fp16":
        assert torch.device(device).type == "cuda", "fp16 chỉ đo trên GPU"
        return copy.deepcopy(model).half().to(device).eval(), torch.float16, False
    return model.to(device).eval(), torch.float32, dtype == "amp"


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, k_views: int = 1, label: str = "") -> dict:
    """Đo độ trễ forward với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    k_views > 1: mỗi lần đo chạy K forward liên tiếp (TTA tuần tự, mỗi view một lượt).
    Trả về dict ghi thẳng vào sheet `Latency` của results.xlsx.
    """
    m, in_dtype, use_amp = _prepare(model, dtype, device)
    dev = torch.device(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev, dtype=in_dtype)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    @torch.inference_mode()
    def fn():
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            for _ in range(k_views):
                m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {"label": label, "gpu": _device_name(device), "dtype": dtype, "batch": batch_size,
            "img_size": img_size, "k_views": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "mean": r["mean"], "n_iters": r["n"], "warmup": r["warmup"],
            "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__,
            "preprocessing_included": False}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view (K forward tuần tự), kèm so sánh với K * p50 của một lượt (slide trang 63)."""
    single = latency_report(model, k_views=1, **kw)
    multi = latency_report(model, k_views=k_views, **kw)
    multi["single_p50"] = single["p50"]
    multi["ratio_vs_k_single"] = multi["p50"] / (k_views * single["p50"])
    return multi

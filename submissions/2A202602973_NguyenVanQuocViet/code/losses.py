"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOSS_CHOICES = ("ce", "ls", "focal", "ce_weighted")


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`.

      - "ce"          : CrossEntropyLoss
      - "ls"          : LabelSmoothingCE(smoothing=kw["smoothing"], mặc định 0.1)
      - "focal"       : FocalLoss(gamma=kw["gamma"], alpha=kw["alpha"] hoặc kw["weight"])
      - "ce_weighted" : CrossEntropyLoss(weight=kw["weight"]), weight từ class_weights(...)
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        alpha = kw.get("alpha", kw.get("weight"))
        return FocalLoss(kw.get("gamma", 2.0), alpha)
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight=class_weights(counts_train, beta)")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"loss={kind!r} không hợp lệ, chọn trong {LOSS_CHOICES}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt (không dùng label_smoothing của PyTorch):
        loss = (1 - eps) * NLL(y) + eps * mean_k(-log p_k)
    eps = 0 cho đúng CE; kết quả trùng nn.CrossEntropyLoss(label_smoothing=eps).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=-1)
        return ((1 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp (độ dài K). Lấy trung bình theo batch.
    gamma = 0, alpha = None cho đúng cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c ∝ 1 / n_c
    - beta > 0: class-balanced, w_c ∝ (1 - beta) / (1 - beta ** n_c)  (Cui et al. arXiv:1901.05555)
    Cả hai đều chuẩn hoá để tổng trọng số = số lớp (tức trung bình = 1).
    """
    n = np.asarray(counts, dtype=np.float64)
    assert (n > 0).all(), "mỗi lớp phải có ít nhất 1 ảnh"
    w = 1.0 / n if not beta else (1.0 - beta) / (1.0 - np.power(beta, n))
    w = w / w.sum() * len(n)
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn. lam ~ Beta(alpha, alpha).

    - mixup : x_mix = lam * x + (1 - lam) * x[perm]
    - cutmix: dán hộp từ x[perm] vào x (cạnh hộp = sqrt(1 - lam) * cạnh ảnh, tâm ngẫu nhiên),
              rồi tính lại lam = 1 - diện tích THỰC của hộp sau khi cắt theo biên / diện tích ảnh.
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], (y_a, y_b, lam)
    if mode != "cutmix":
        raise ValueError(f"mode={mode!r} không hợp lệ (mixup | cutmix)")

    H, W = x.shape[-2:]
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(H * cut), int(W * cut)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
    x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, (y_a, y_b, float(lam))


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

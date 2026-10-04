"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn)   -> (filenames, y_true, [logits[N, 9]] * K)
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _to_numpy_logits(chunks) -> np.ndarray:
    return torch.cat(chunks).float().cpu().numpy()


@torch.inference_mode()
def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Chạy model trên loader, gom logit theo đúng thứ tự file.

    `view`: hàm biến đổi batch (N, C, H, W) trước khi đưa vào model (vd view_hflip), hoặc None.
    """
    model.eval()
    names, ys, outs = [], [], []
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        if view is not None:
            x = view(x)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                            enabled=amp and torch.device(device).type == "cuda"):
            outs.append(model(x).float())
        names.extend(f)
        ys.append(torch.as_tensor(y))
    return names, torch.cat(ys).numpy(), _to_numpy_logits(outs)


@torch.inference_mode()
def predict_views(model, loader, device, views_fn, amp: bool = False):
    """Như predict_logits nhưng `views_fn(x) -> list[batch]`; trả về list logit, mỗi view một mảng.

    Dùng cho TTA nhiều view (multicrop/multiscale) mà chỉ đọc ảnh một lần.
    """
    model.eval()
    names, ys, per_view = [], [], None
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        views = views_fn(x)
        if per_view is None:
            per_view = [[] for _ in views]
        for k, v in enumerate(views):
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=amp and torch.device(device).type == "cuda"):
                per_view[k].append(model(v).float())
        names.extend(f)
        ys.append(torch.as_tensor(y))
    return names, torch.cat(ys).numpy(), [_to_numpy_logits(c) for c in per_view]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_hflip(x):
    """TTA K = 2: [ảnh gốc, ảnh lật ngang]."""
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch x; flip=True thêm bản lật (K = 10).

    x nên là ảnh CHƯA crop (vd 256x256, loader dùng build_transforms(False, img_size=256)).
    """
    H, W = x.shape[-2:]
    assert crop <= H and crop <= W, f"crop {crop} lớn hơn ảnh {H}x{W}"
    top, left = (H - crop) // 2, (W - crop) // 2
    crops = [x[..., :crop, :crop], x[..., :crop, W - crop:], x[..., H - crop:, :crop],
             x[..., H - crop:, W - crop:], x[..., top:top + crop, left:left + crop]]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias). Trả về list các batch.

    CNN có global pooling nhận được mọi kích thước. ViT/DeiT cần nội suy position embedding,
    Swin cần kích thước chia hết cho cửa sổ: với các mạng đó chỉ dùng kích thước lúc train,
    hoặc tạo model với img_size mới (timm.create_model(..., img_size=s)) - ghi rõ trong báo cáo.
    """
    out = []
    for s in sizes:
        out.append(x if x.shape[-1] == s and x.shape[-2] == s
                   else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True))
    return out


def _softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view của TTA thành xác suất (N, 9).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    stack = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])  # (K, N, C)
    if space == "prob":
        probs = _softmax(stack).mean(0)
    elif space == "logit":
        probs = _softmax(stack.mean(0))
    else:
        raise ValueError(f"space={space!r} không hợp lệ (prob | logit)")
    return probs / probs.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    shapes = {np.shape(p) for p in list_of_probs}
    assert len(shapes) == 1, f"các mô hình phải cùng dạng đầu ra, nhận {shapes}"
    probs = np.mean([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)
    return probs / probs.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels, max_iter: int = 200) -> float:
    """Tìm T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tối ưu log T bằng LBFGS (float64), khởi tạo T = 1. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T) dạng numpy."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(device=conv.weight.device, dtype=conv.weight.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight if bn.weight is not None else torch.ones_like(std)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(std)
    scale = gamma / std
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(beta + (b - bn.running_mean) * scale)
    return fused


def _bn_replacement(bn: nn.Module) -> nn.Module:
    """timm BatchNormAct2d = BN + drop + act trong một module: giữ lại drop/act khi bỏ BN."""
    extra = [m for m in (getattr(bn, "drop", None), getattr(bn, "act", None)) if m is not None]
    return nn.Sequential(*extra) if extra else nn.Identity()


@torch.no_grad()
def fuse_conv_bn(model, check_input=None, inplace: bool = False):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Ghép các cặp (Conv2d, BatchNorm2d) đứng LIỀN NHAU trong cùng module cha (thứ tự đăng ký), đúng
    với ResNet/ResNeXt/EfficientNet/MobileNet của timm. Mặc định làm trên bản sao.
    `check_input`: tensor mẫu; nếu có, in sai số tuyệt đối lớn nhất giữa đầu ra trước và sau gộp.
    ViT/Swin/ConvNeXt dùng LayerNorm, không có cặp nào để gộp (hàm in "0 cặp").
    """
    model.eval()
    fused_model = model if inplace else copy.deepcopy(model)
    n_fused = 0
    for parent in fused_model.modules():
        names = list(parent._modules.keys())
        for a, b in zip(names, names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if (type(conv) is nn.Conv2d and isinstance(bn, nn.BatchNorm2d)
                    and bn.track_running_stats and conv.out_channels == bn.num_features):
                parent._modules[a] = _fuse_pair(conv, bn)
                parent._modules[b] = _bn_replacement(bn)
                n_fused += 1
    fused_model.eval()
    fused_model.n_fused = n_fused
    print(f"fuse_conv_bn: đã gộp {n_fused} cặp Conv2d+BatchNorm2d"
          + ("" if n_fused else " (kiến trúc không có BatchNorm, vd dùng LayerNorm: không áp dụng)"))
    if check_input is not None:
        # tắt TF32 khi so: TF32 (GPU Ampere+) tự làm lệch ~1e-3, che mất sai số thật của phép gộp
        tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        try:
            diff = (model(check_input).float() - fused_model(check_input).float()).abs().max().item()
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
        print(f"fuse_conv_bn: sai số lớn nhất trước/sau gộp = {diff:.2e}"
              + ("" if diff <= 1e-5 else "  <-- CẢNH BÁO: lớn hơn 1e-5, kiểm tra lại phép gộp"))
        fused_model.fuse_max_abs_diff = diff
    return fused_model

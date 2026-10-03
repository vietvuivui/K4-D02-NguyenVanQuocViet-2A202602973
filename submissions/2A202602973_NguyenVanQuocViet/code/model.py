"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (giữ BN ở eval nếu đóng băng)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
from torch import nn

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}
INIT_CHOICES = ("scratch", "frozen", "finetune")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp qua timm (head mới khởi tạo ngẫu nhiên).

    `init` (trục A): "scratch" (không tiền huấn luyện) | "frozen" (chỉ train head) | "finetune".
    Gắn thêm `model.weights_tag` (vd "resnet50.a1_in1k" hoặc "scratch") để ghi vào results.xlsx.
    """
    import timm  # import trong hàm: các hàm còn lại (và test) dùng được khi chưa cài timm

    if init not in INIT_CHOICES:
        raise ValueError(f"init={init!r} không hợp lệ, chọn trong {INIT_CHOICES}")
    use_pretrained = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=use_pretrained, num_classes=num_classes, drop_rate=drop_rate)

    cfg = getattr(model, "pretrained_cfg", {}) or {}
    tag = cfg.get("tag")
    model.weights_tag = (f"{cfg.get('architecture', name)}.{tag}" if tag else name) if use_pretrained else "scratch"
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của backbone đóng băng phải ở chế độ eval, nếu không running_mean/var vẫn bị cập nhật
    theo dữ liệu mới trong khi trọng số conv giữ nguyên -> lệch phân phối. Train loop gọi
    set_train_mode(model) thay cho model.train() để giữ điều này.
    """
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(); nếu backbone đóng băng thì đưa mọi module không thuộc head về eval (BN, dropout)."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head_modules = set(model.get_classifier().modules())
        for m in model.modules():
            if m is not model and m not in head_modules:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone, ndim > 1           : lr_backbone, weight_decay
    - backbone norm/bias (ndim <= 1): lr_backbone, weight_decay = 0
    - head mới                     : lr_head, weight_decay
    Bỏ qua tham số requires_grad == False và nhóm rỗng.
    """
    head = _head_param_ids(model)
    decay, no_decay, head_params = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head:
            head_params.append(p)
        elif p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
        {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (có sẵn trong PyTorch >= 2.1), đếm FLOPs của
    conv, linear, matmul và attention; GMAC = FLOPs / 2. Có thể lệch vài % so với fvcore/ptflops.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9

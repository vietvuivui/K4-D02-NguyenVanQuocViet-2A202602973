"""test_code.py - kiểm tra tự viết cho code/ (chạy trên CPU, không cần GPU, timm hay dữ liệu thật).

Chạy trong thư mục code/:
    python -m unittest -v test_code
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402
import inference as I  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402


class TinyNet(nn.Module):
    """Mạng nhỏ có Conv+BN, Dropout và head `fc`, giao diện giống model timm (get_classifier)."""

    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.act = nn.ReLU()
        self.down = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(16, num_classes)

    def get_classifier(self):
        return self.fc

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.down(self.act(self.bn1(self.conv1(x))))).flatten(1)))


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(128, 9) * 3
        self.y = torch.randint(0, 9, (128,))

    def test_focal_gamma0_equals_ce(self):
        ce = F.cross_entropy(self.z, self.y)
        self.assertLess(abs(L.FocalLoss(gamma=0.0)(self.z, self.y) - ce).item(), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(L.FocalLoss(gamma=2.0)(self.z, self.y).item(), F.cross_entropy(self.z, self.y).item())

    def test_label_smoothing(self):
        ce = F.cross_entropy(self.z, self.y)
        self.assertLess(abs(L.LabelSmoothingCE(0.0)(self.z, self.y) - ce).item(), 1e-6)
        ref = nn.CrossEntropyLoss(label_smoothing=0.1)(self.z, self.y)
        self.assertLess(abs(L.LabelSmoothingCE(0.1)(self.z, self.y) - ref).item(), 1e-6)

    def test_class_weights(self):
        w = L.class_weights([100, 10, 1000])
        self.assertAlmostEqual(w.sum().item(), 3.0, places=5)
        self.assertTrue(w[1] > w[0] > w[2])
        wb = L.class_weights([100, 10, 1000], beta=0.999)
        self.assertAlmostEqual(wb.sum().item(), 3.0, places=5)

    def test_cutmix_mixes_image_and_label(self):
        """Ảnh i tô giá trị i; sau CutMix, tỉ lệ pixel lấy từ ảnh khác phải đúng bằng 1 - lam, và y_b = y[perm]."""
        np.random.seed(0)
        torch.manual_seed(0)
        n, H, W = 8, 32, 40
        x = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 3, H, W).contiguous()
        y = torch.arange(n)
        for _ in range(20):
            xm, (ya, yb, lam) = L.mix_batch(x, y, alpha=1.0, mode="cutmix")
            self.assertTrue(torch.equal(ya, y))
            self.assertFalse(torch.equal(xm, x) and lam < 1.0)
            for i in range(n):
                if yb[i] == i:  # ghép với chính nó: không đổi
                    continue
                frac_other = (xm[i, 0] == yb[i].float()).float().mean().item()
                self.assertAlmostEqual(frac_other, 1.0 - lam, places=6)  # lam theo diện tích THỰC
                self.assertTrue(((xm[i, 0] == i) | (xm[i, 0] == yb[i].float())).all())
            self.assertTrue(torch.equal(x, torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 3, H, W)))

    def test_mixup_and_mixed_loss(self):
        np.random.seed(1)
        x = torch.randn(6, 3, 8, 8)
        y = torch.arange(6)
        xm, (ya, yb, lam) = L.mix_batch(x, y, alpha=0.4, mode="mixup")
        perm = yb  # vì y = arange
        self.assertTrue(torch.allclose(xm, lam * x + (1 - lam) * x[perm]))
        ce = nn.CrossEntropyLoss()
        logits = torch.randn(6, 9)
        expected = lam * ce(logits, ya) + (1 - lam) * ce(logits, yb)
        self.assertAlmostEqual(L.mixed_loss(ce, logits, (ya, yb, lam)).item(), expected.item(), places=6)


class TestModel(unittest.TestCase):
    def test_param_groups(self):
        m = TinyNet()
        groups = {g["name"]: g for g in M.param_groups(m, 1e-4, 1e-3, 0.05)}
        self.assertEqual(groups["head"]["lr"], 1e-3)
        self.assertEqual(groups["backbone"]["weight_decay"], 0.05)
        self.assertEqual(groups["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertTrue(all(p.ndim > 1 for p in groups["backbone"]["params"]))
        self.assertTrue(all(p.ndim <= 1 for p in groups["backbone_no_decay"]["params"]))
        n = sum(len(g["params"]) for g in groups.values())
        self.assertEqual(n, len(list(m.parameters())))  # không sót, không trùng

    def test_freeze_keeps_bn_eval(self):
        m = TinyNet()
        M.freeze_backbone(m)
        trainable = [p for p in m.parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in trainable}, {id(p) for p in m.fc.parameters()})
        self.assertEqual([g["name"] for g in M.param_groups(m, 1e-4, 1e-3, 0.05)], ["head"])
        M.set_train_mode(m)
        self.assertTrue(m.fc.training)
        self.assertFalse(m.bn1.training)
        before = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 16, 16))
        self.assertTrue(torch.equal(before, m.bn1.running_mean))  # BN không bị cập nhật

    def test_finetune_train_mode(self):
        m = TinyNet()
        m.frozen_backbone = False
        M.set_train_mode(m)
        self.assertTrue(all(mod.training for mod in m.modules()))

    def test_count(self):
        m = TinyNet()
        self.assertAlmostEqual(M.count_params(m), sum(p.numel() for p in m.parameters()) / 1e6)
        self.assertGreater(M.count_gmacs(m, 32), 0)


class TestTrain(unittest.TestCase):
    def test_scheduler_warmup_cosine(self):
        m = TinyNet()
        cfg = T.Config(epochs=4, warmup_epochs=1.0)
        opt = T.build_optimizer(m, cfg)
        sch = T.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs, ratios = [], []
        head = [g for g in opt.param_groups if g["name"] == "head"][0]
        for _ in range(40):
            lrs.append(opt.param_groups[0]["lr"])
            if lrs[-1] > 0:
                ratios.append(head["lr"] / lrs[-1])
            opt.step()
            sch.step()
        self.assertAlmostEqual(max(lrs), cfg.lr_backbone)
        self.assertEqual(int(np.argmax(lrs)), 9)                 # đỉnh ở cuối warmup
        self.assertLess(lrs[-1], 0.01 * cfg.lr_backbone)        # cosine về ~0
        np.testing.assert_allclose(ratios, 10.0)                # tỉ lệ head/backbone giữ nguyên

    def test_ema(self):
        m = TinyNet()
        ema = T.EMA(m, decay=0.5)
        w0 = m.fc.weight.detach().clone()
        with torch.no_grad():
            m.fc.weight.add_(1.0)
        ema.update(m)
        self.assertTrue(torch.allclose(ema.module.fc.weight, w0 + 0.5))

    def test_parse_overrides(self):
        o = T.parse_overrides(["seed=1", "ema_decay=none", "amp=false", "lr_head=3e-3", "sampler=balanced"])
        self.assertEqual(o, {"seed": 1, "ema_decay": None, "amp": False, "lr_head": 3e-3, "sampler": "balanced"})
        with self.assertRaises(KeyError):
            T.parse_overrides(["khong_co=1"])

    def test_initial_loss_near_ln9(self):
        torch.manual_seed(0)
        m = TinyNet().eval()
        nn.init.zeros_(m.fc.weight)
        nn.init.zeros_(m.fc.bias)
        loss = F.cross_entropy(m(torch.randn(16, 3, 16, 16)), torch.randint(0, 9, (16,)))
        self.assertAlmostEqual(loss.item(), math.log(9), places=5)


class TestDataAndEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.img = root / "images"
        cls.img.mkdir()
        rng = np.random.default_rng(0)
        names, labels = [], []
        for i in range(40):
            f = f"{i:03d}.jpg"
            Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(cls.img / f)
            names.append(f)
            labels.append(i % 9)
        cls.df = pd.DataFrame({"Filename": names, "Label": labels})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_item_contract(self):
        ds = D.DeepWeedsDataset(self.df, self.img, D.build_transforms(False, 16))
        x, y, f = ds[3]
        self.assertEqual(tuple(x.shape), (3, 16, 16))
        self.assertIsInstance(y, int)
        self.assertEqual(f, "003.jpg")

    def test_eval_loader_keeps_order(self):
        """evaluate giữ thứ tự df; model.eval() được gọi (Dropout tắt -> hai lần chạy giống nhau)."""
        loader = D.make_loader(self.df, self.img, D.build_transforms(False, 16), batch_size=7,
                               train=False, num_workers=0)
        m = TinyNet()
        m.train()
        names, y, logits, _ = T.evaluate(m, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertFalse(m.training)
        self.assertEqual(names, self.df["Filename"].tolist())
        np.testing.assert_array_equal(y, self.df["Label"].to_numpy())
        _, _, logits2, _ = T.evaluate(m, loader, None, torch.device("cpu"))
        np.testing.assert_allclose(logits, logits2)

    def test_train_loader_seeded(self):
        def first_batch(seed):
            ld = D.make_loader(self.df, self.img, D.build_transforms(True, 16), batch_size=8,
                               train=True, num_workers=0, seed=seed)
            torch.manual_seed(123)  # augmentation dùng RNG toàn cục ở num_workers=0
            return next(iter(ld))
        a, b = first_batch(0), first_batch(0)
        self.assertEqual(a[2], b[2])
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertNotEqual(a[2], first_batch(1)[2])

    def test_balanced_sampler(self):
        df = pd.DataFrame({"Filename": [f"{i:03d}.jpg" for i in range(40)], "Label": [0] * 36 + [1] * 4})
        ld = D.make_loader(df, self.img, D.build_transforms(False, 16), batch_size=40, train=True,
                           sampler="balanced", num_workers=0)
        counts = np.zeros(2)
        for _ in range(25):
            for _, y, _ in ld:
                counts += np.bincount(y.numpy(), minlength=2)
        self.assertAlmostEqual(counts[1] / counts.sum(), 0.5, delta=0.05)

    def test_check_split_detects_relabel(self):
        """S1: nhãn trong file split khác labels.csv gốc -> phát hiện và báo chi tiết, KHÔNG dừng
        (dữ liệu gốc fold 0 có sẵn 1 ảnh lệch như vậy; S1 cấm sửa CSV)."""
        ref = self.df.copy()
        bad = self.df.copy()
        bad.loc[0, "Label"] = (bad.loc[0, "Label"] + 1) % 9
        old, D.TOTAL_IMAGES = D.TOTAL_IMAGES, len(self.df)  # dữ liệu giả 40 ảnh thay vì 17.509
        try:
            stats = D.check_split(bad.iloc[:20], bad.iloc[20:30], bad.iloc[30:], self.img, verbose=False, labels_df=ref)
            self.assertEqual(stats["label_mismatch"], 1)
            self.assertEqual(stats["label_mismatch_rows"],
                             [{"Filename": "000.jpg", "split": "train", "label_split_csv": int(bad.loc[0, "Label"]),
                               "label_labels_csv": int(ref.loc[0, "Label"])}])
            stats = D.check_split(ref.iloc[:20], ref.iloc[20:30], ref.iloc[30:], self.img, verbose=False, labels_df=ref)
            self.assertEqual(stats["label_mismatch"], 0)
        finally:
            D.TOTAL_IMAGES = old

    def test_check_split_detects_leak(self):
        tr, va, te = self.df.iloc[:20], self.df.iloc[19:30], self.df.iloc[30:]
        with self.assertRaises(AssertionError):
            D.check_split(tr, va, te, self.img, verbose=False)


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        torch.manual_seed(0)
        m = TinyNet()
        m.train()
        for _ in range(3):
            m(torch.randn(8, 3, 16, 16))  # có running stats khác mặc định
        x = torch.randn(4, 3, 16, 16)
        fused = I.fuse_conv_bn(m, check_input=x)
        self.assertLess(fused.fuse_max_abs_diff, 1e-5)
        self.assertFalse(any(isinstance(mod, nn.BatchNorm2d) for mod in fused.modules()))

    def test_temperature(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(3000, 9)) * 4
        p = I.apply_temperature(z, 2.0)
        y = np.array([rng.choice(9, p=r) for r in p])
        self.assertAlmostEqual(I.fit_temperature(z, y), 2.0, delta=0.15)
        np.testing.assert_array_equal(I.apply_temperature(z, 3.0).argmax(1), z.argmax(1))

    def test_aggregate(self):
        a, b = np.random.randn(5, 9), np.random.randn(5, 9)
        for space in ("prob", "logit"):
            np.testing.assert_allclose(I.aggregate_views([a, b], space).sum(1), 1.0)
        self.assertEqual(len(I.views_multicrop(torch.randn(2, 3, 32, 32), 24, flip=True)), 10)


if __name__ == "__main__":
    unittest.main()

# Lab Day 2 — DeepWeeds · Nguyễn Văn Quốc Việt · 2A202602973

## Notebook chạy lại
- Kaggle: <!-- TODO: dán link notebook Kaggle -->
- File: `code/lab_day2.ipynb` (chạy được trên Kaggle hoặc Colab)

## Môi trường
| Thành phần | Phiên bản |
|---|---|
| Python | <!-- TODO: lấy từ ô cài đặt / X.env_info() --> |
| PyTorch | <!-- TODO --> |
| torchvision | <!-- TODO --> |
| timm | <!-- TODO --> |
| GPU | <!-- TODO --> |

## Dữ liệu
DeepWeeds, fold 0 chia sẵn của tác giả (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`), không sửa.
Notebook tự tải ảnh từ Zenodo (kiểm tra MD5) và CSV từ GitHub của tác giả.

## Cách chạy lại (Kaggle)
1. Tạo notebook mới, import `code/lab_day2.ipynb`. Settings: **Accelerator = GPU**, **Internet = On**.
2. Chọn **Save Version → Save & Run All (Commit)**. Toàn bộ chạy trong một session (ước tính 6–9 giờ GPU):
   ô cài đặt (clone repo) → tải dữ liệu → Bước 0 (kiểm tra split, EDA, kiểm tra pipeline, test) →
   Bước 1 (5 backbone) → Bước 2 (8 ablation + kết hợp) → Bước 3 (suy luận, độ trễ) →
   Bước 4 (chung kết 3 seed, test một lần mỗi seed, `eval.py score/grade`) → Bước 5 (xlsx, hình, bản nháp báo cáo).
3. Tải Output (`/kaggle/working/K4-D02-.../submissions/2A202602973_NguyenVanQuocViet/`) về repo rồi commit.

Chạy một thí nghiệm lẻ từ dòng lệnh: `python code/train.py --set exp_id=B01 backbone=resnet50 seed=0`.
Test tự viết (CPU): `cd code && python -m unittest -v test_code`.

## Cấu hình
- Công thức nền T00: ImageNet pretrained, tinh chỉnh toàn bộ, AdamW (LR backbone 1e-4, head 1e-3, WD 0,05 trừ
  norm/bias), warmup 1 epoch + cosine, CE, batch 64, **12 epoch**, AMP, chọn checkpoint theo macro-F1 val.
- Quy tắc chọn backbone/công thức/suy luận chỉ dùng val, ghi ở đầu `code/experiments.py`.

## Seed đã dùng
- Bước 1–3: seed 0. Chung kết F01 và mốc T00: seed 0, 1, 2.

## Checkpoint
Không commit checkpoint. <!-- TODO: link Kaggle Output nếu cần chia sẻ -->

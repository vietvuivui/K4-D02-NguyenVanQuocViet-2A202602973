# Lab Day 2 — DeepWeeds · Nguyễn Văn Quốc Việt · 2A202602973

## Notebook chạy lại
- Kaggle: https://www.kaggle.com/code/vietnguyen2715/lab02 (phiên bản đã chạy xong; GPU Tesla T4, ~4,1 giờ)
- File: `code/lab_day2.ipynb` (chạy được trên Kaggle hoặc Colab)
- Log đầy đủ của lần chạy: `kaggle_run_log.txt`

## Kết quả chung kết (test, 3 seed, tính bằng `eval.py`)
| Cấu hình | macro-F1 | top-1 | ECE | recall Chinee apple / Snake weed |
|---|---|---|---|---|
| F01: ConvNeXt-T + T10 + I04_256 + temperature scaling | 0.9632 ± 0.0006 | 0.9718 ± 0.0000 | 0.0057 | 0.928 / 0.931 |
| Mốc T00 + I00 | 0.9522 ± 0.0026 | 0.9625 ± 0.0014 | | |

Chi tiết: `results.xlsx`, `report.md`, file dự đoán trong `predictions/`.

## Môi trường
| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| torchvision | 0.26.0+cu128 |
| timm | 1.0.29 |
| numpy / pandas | 2.1.3 / 2.3.3 |
| GPU | Tesla T4 (Kaggle) |

## Dữ liệu
DeepWeeds, fold 0 chia sẵn của tác giả (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`), không sửa.
Notebook tự tải ảnh từ Zenodo (kiểm tra MD5) và CSV từ GitHub của tác giả.

## Cách chạy lại (Kaggle)
1. Tạo notebook mới, import `code/lab_day2.ipynb`. Settings: **Accelerator = GPU**, **Internet = On**.
2. Chọn **Save Version → Save & Run All (Commit)**. Toàn bộ chạy trong một session (~4 giờ trên T4):
   ô cài đặt (clone repo) → tải dữ liệu → Bước 0 (kiểm tra split, EDA, kiểm tra pipeline, test) →
   Bước 1 (5 backbone) → Bước 2 (đo nhiễu T00 × 3 seed, 9 ablation + kết hợp T10) → Bước 3 (suy luận, độ trễ) →
   Bước 4 (chung kết 3 seed, test một lần mỗi seed, `eval.py score/grade`) → Bước 5 (xlsx, hình, bản nháp báo cáo).
3. Tải Output (`/kaggle/working/K4-D02-.../submissions/2A202602973_NguyenVanQuocViet/`) về repo rồi commit.

Chạy một thí nghiệm lẻ từ dòng lệnh: `python code/train.py --set exp_id=B01 backbone=resnet50 seed=0`.
Test tự viết (CPU): `cd code && python -m unittest -v test_code`.

## Cấu hình
- Công thức nền T00: ImageNet-1k pretrained, tinh chỉnh toàn bộ, AdamW (LR backbone 1e-4, head 1e-3, WD 0,05 trừ
  norm/bias), warmup 1 epoch + cosine, CE, batch 64, **12 epoch**, AMP, chọn checkpoint theo macro-F1 val.
- Quy tắc chọn backbone/công thức/suy luận chỉ dùng val, ghi ở đầu `code/experiments.py`.

## Seed đã dùng
- Bước 1–3: seed 0 (Bước 2 đo nhiễu bằng T00 seed 0, 1, 2). Chung kết F01 và mốc T00: seed 0, 1, 2.

## Checkpoint
Không commit checkpoint (nằm trong Output của notebook Kaggle ở trên, thư mục `deepweeds_runs/runs/`).

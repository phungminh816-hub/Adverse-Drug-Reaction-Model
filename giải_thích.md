# Giải thích cấu trúc dự án ADR – HUST 2026-1

> Tài liệu này giải thích **mọi thư mục và file** trong skeleton, mục đích của từng phần, cách chúng kết nối với nhau và trình tự học/làm việc đề xuất.

---

## Tổng quan kiến trúc

```
SMILES → [features] → [model] → scores [27] → [evaluate] → metrics
                                     ↓
                               [predictor] → PredictionResult → [streamlit / CLI]
```

Mọi bước đều có **artifact** được lưu vào `artifacts/<run_id>/` để tái sử dụng và so sánh.

---

## Cây thư mục và giải thích

### 📁 Gốc dự án

| File | Mục đích |
|------|----------|
| `README.md` | Hướng dẫn cài đặt, lệnh chạy, trạng thái dự án |
| `pyproject.toml` | Khai báo package `adr` – cho phép `pip install -e .` và `import adr` từ mọi nơi |
| `requirements.txt` | Danh sách thư viện + ghi chú cách cài (rdkit, torch CPU) |
| `.gitignore` | Bỏ qua `.venv`, data thật, checkpoint binary, kết quả auto-gen |

---

### 📁 `configs/` – Cấu hình YAML

| File | Mục đích |
|------|----------|
| `base.yaml` | **Config gốc**: seed, đường dẫn, tham số data/split/training chung |
| `frequency.yaml` | Override cho baseline tần suất |
| `logreg.yaml` | Override cho LogReg (solver, C, max_iter, fingerprint) |
| `mlp.yaml` | Override cho MLP (hidden_dim, dropout) |
| `cnn.yaml` | Override cho CNN (embedding_dim, channels, kernel_size) |

**Quy tắc merge:** `base.yaml` load trước → model config deep-merge đè lên.  
Deep-merge không làm mất key lồng nhau của base.

```python
from adr.config import load_config, resolve_paths
cfg = resolve_paths(load_config("configs/logreg.yaml"))
# cfg['paths']['data_raw'] → Path tuyệt đối
```

---

### 📁 `data/` – Dữ liệu (KHÔNG commit)

| Thư mục | Nội dung sau pipeline |
|---------|----------------------|
| `data/raw/` | `sider.csv.gz` + `sider.csv` + `download_meta.json` (URL, thời gian, SHA-256) |
| `data/processed/` | `processed.csv`, `label_names.json`, `label_mask.npy`, `drug_id_mapping.json` |
| `data/splits/` | `split_manifest.csv` (drug_id → train/val/test), `split_meta.json` |

**Quan trọng:**
- `label_names.json`: 27 tên nhãn **theo thứ tự cột nguồn** – không sắp alphabet, không tự bịa
- `label_mask.npy`: boolean `[N, 27]` – True = nhãn được quan sát; NaN trong CSV gốc → mask=False
- `drug_id`: SHA-256 của canonical SMILES (ổn định, không dùng Python `hash()`)

---

### 📁 `src/adr/` – Package chính

#### `adr/__init__.py`
Khai báo `__version__`. Import các sub-module sẽ lazy (chỉ khi dùng).

#### `adr/config.py`
```python
load_config(model_config, base)  # load + deep-merge
resolve_paths(cfg)               # str → Path tuyệt đối
save_config(cfg, run_dir)        # lưu resolved config vào artifact
validate_base_config(cfg)        # kiểm tra trường bắt buộc
```
Config không bao giờ hardcode đường dẫn ổ đĩa.

#### `adr/schemas.py`
Hai dataclass trung tâm:
- **`ArtifactMetadata`**: ghi kèm mọi run – model_type, seed, label_names, checksum, library versions, threshold
- **`PredictionResult`**: kết quả inference – scores[27], predicted_labels, warnings

Hằng số: `NUM_LABELS=27`, `PAD_TOKEN_ID=0`, `UNK_TOKEN_ID=1`

---

#### 📁 `adr/data/`

| File | Chức năng |
|------|-----------|
| `download.py` | Tải `sider.csv.gz` từ S3; ghi metadata tải; không ghi đè nếu đã có |
| `validate.py` | Kiểm tra schema (cột smiles, 27 nhãn, giá trị 0/1/NaN, SMILES parse); phát hiện conflict duplicate |
| `prepare.py` | Canonicalize SMILES (RDKit) → drug_id → gộp duplicate → lưu processed.csv + mask |
| `split.py` | Random split 80/10/10 theo drug_id (group-level); scaffold split = NotImplementedError |

**Luồng dữ liệu:**
```
download → raw/sider.csv
             ↓ validate (kiểm tra lỗi)
             ↓ prepare  (canonicalize, drug_id, mask)
           processed.csv + label_mask.npy
             ↓ split
           split_manifest.csv
```

---

#### 📁 `adr/features/`

| File | Chức năng |
|------|-----------|
| `fingerprints.py` | `MorganFeaturizer`: SMILES → `[N, 2048]` float32; `save/load` config JSON |
| `tokenizer.py` | `CharTokenizer`: `fit(train)` → `transform` → `(input_ids, attention_mask, stats)`; `save/load` JSON |

**Tokenizer contract:**
- `fit` chỉ được gọi trên train
- PAD_ID=0, UNK_ID=1
- `max_length` = max độ dài trong train (lưu vào artifact)
- `stats`: n_unk, n_truncated, unk_rate cho mỗi lần transform

---

#### 📁 `adr/models/`

| File | Model | Đặc điểm |
|------|-------|-----------|
| `frequency.py` | `FrequencyBaseline` | Mỗi nhãn = positive rate train; không phụ thuộc SMILES đầu vào |
| `logreg.py` | `PerLabelLogReg` | 1 LogisticRegression / nhãn; nhãn 1 lớp → `ConstantPredictor` |
| `mlp.py` | `MLP` | Linear(2048→256)→ReLU→Dropout→Linear(256→27); không sigmoid trong `forward` |
| `cnn.py` | `SmilesCharCNN` | Embedding→Conv1d→Masked mean pool→Dropout→Linear(→27); không sigmoid |

**Tất cả model đều có `save(run_dir)` và `load(run_dir)`.**  
MLP và CNN cần PyTorch; nếu chưa cài, import sẽ raise ImportError rõ ràng.

**Masked mean pooling (CNN):** Chỉ tính trung bình trên token thật (mask=True). Output **bất biến** khi thêm PAD cuối chuỗi – có unit test kiểm chứng.

---

#### 📁 `adr/train/`

| File | Chức năng |
|------|-----------|
| `losses.py` | `masked_bce_loss`: BCE → nhân mask → chia n_observed; batch toàn mask=0 → RuntimeError |
| `baseline.py` | Train Frequency/LogReg; tạo run_dir với metadata; CLI hoạt động đầy đủ |
| `neural.py` | **TODO (Tuần 5-7)**: stub với `--help` hoạt động và exit code ≠ 0 |

**Loss chi tiết:**
```python
loss = BCEWithLogitsLoss(reduction='none')  # [B, 27]
loss = loss * mask                          # zero ở vị trí missing
loss = loss.sum() / mask.sum()             # chuẩn hóa đúng
```

---

#### 📁 `adr/evaluate/`

| File | Chức năng |
|------|-----------|
| `metrics.py` | Per-label: AP, ROC-AUC, Precision, Recall, F1; Macro AP; null handling rõ ràng |
| `run.py` | Load artifact → load split → predict → evaluate → lưu JSON; guard test lock |

**Null handling:**
- ROC-AUC = `None` nếu split chỉ có 1 lớp (pos=0 hoặc neg=0)
- AP = `None` nếu không có positive trong split
- Không thay `None` bằng 0 rồi trung bình

**Test lock:** `--split test` chỉ chạy khi metadata có `config_locked: true`.

---

#### 📁 `adr/predict/`

| File | Chức năng |
|------|-----------|
| `predictor.py` | **Nơi DUY NHẤT** thực hiện inference: `Predictor(run_dir).predict(smiles)` |
| `run.py` | CLI wrapper; print top-k, 27 nhãn, cảnh báo y tế |

**`PredictionResult`** chứa:
- `scores`: list[float] 27 phần tử theo đúng `label_names`
- `predicted_labels`: nhãn có score ≥ threshold
- `warnings`: UNK, truncation, SMILES lỗi
- `top_k(k)`: trả k nhãn có score cao nhất

Thiếu artifact → `FileNotFoundError` (không sinh điểm ngẫu nhiên).

---

#### 📁 `adr/utils/`

| File | Chức năng |
|------|-----------|
| `seed.py` | `set_seed(seed)`: cố định random/numpy/torch |
| `paths.py` | `get_run_id`, `get_run_dir`, `sha256_file`, `sha256_str`, `get_library_versions` |
| `artifacts.py` | `check_artifact_dir`: kiểm tra run_dir + file bắt buộc → load metadata |

---

### 📁 `scripts/`

#### `smoke.py`
Test end-to-end **offline**, CPU, seed cố định, **dữ liệu synthetic** (ghi rõ không dùng cho nghiên cứu).

Kiểm tra:
1. Config load & validate
2. Schema contracts
3. Tokenizer fit/transform/save/load (encoding bất biến)
4. Frequency baseline fit/predict/save/load
5. LogReg per-label (fingerprint giả, không cần RDKit)
6. Metrics: AP, null handling, padding irrelevant
7. MLP forward shape (skip nếu thiếu torch)
8. CNN forward + pad invariance (skip nếu thiếu torch)
9. Masked BCE loss (skip nếu thiếu torch)
10. Random split không leakage, reproducible
11. Scaffold split raises NotImplementedError

---

### 📁 `tests/` – pytest

| File | Kiểm tra |
|------|----------|
| `test_data_contracts.py` | Validate: schema sai, SMILES lỗi, nhãn lạ, conflict; Split: no overlap, reproducible |
| `test_models.py` | Frequency/LogReg: shape, range, single-class, save/load; CNN: pad invariance |
| `test_metrics.py` | Null AP/ROC, padding irrelevant, masked loss, save JSON |
| `test_artifacts.py` | Metadata save/load, check_artifact_dir, type mismatch, missing file |

---

### 📁 `app/`

#### `streamlit_app.py`
- **Không có artifact:** Hiển thị hướng dẫn tạo model step-by-step
- **Có artifact:** Dropdown chọn model, input SMILES, top-5, xem 27 nhóm, cảnh báo y tế
- Dùng `Predictor` thật (không giả)
- SMILES lỗi → thông báo rõ ràng (không crash)

---

### 📁 `notebooks/`

#### `01_eda.ipynb`
EDA Tuần 2: gọi `adr.data.validate`, vẽ phân phối positive rate, missing rate, độ dài SMILES, số nhãn/thuốc.  
**Không chứa kết quả giả.** Các cell có guard `if raw_csv.exists()`.

---

### 📁 `artifacts/` và `results/`

- `artifacts/<run_id>/`: mỗi run train tạo một thư mục riêng
  - `metadata.json`: ArtifactMetadata (model_type, seed, label_names, checksum...)
  - `config.yaml`: resolved config đã merge
  - `model.json` / `model.pkl` / `model.pt`: model artifact
  - `fingerprint_config.json` hoặc `tokenizer.json`: feature artifact
  - `eval_val.json`: kết quả evaluate
- `results/`: kết quả tổng hợp (bảng so sánh, plots)

**Không commit** artifact vào git (trong `.gitignore`).

---

### 📁 `reports/`

| File | Nội dung |
|------|----------|
| `roadmap.md` | Lộ trình 12 tuần chi tiết: mục tiêu, checklist, ưu tiên |
| `report_outline.md` | Khung báo cáo nghiên cứu (điền dần từ Tuần 9-11) |

---

## Luồng dữ liệu end-to-end

```
1. python -m adr.data.download     → data/raw/sider.csv
2. python -m adr.data.prepare      → data/processed/{processed.csv, label_names.json, label_mask.npy}
3. python -m adr.data.split        → data/splits/split_manifest.csv
4. python -m adr.train.baseline    → artifacts/logreg_YYYYMMDD_HHMMSS_42/{metadata.json, model.pkl, ...}
5. python -m adr.evaluate.run      → artifacts/<run_id>/eval_val.json
6. python -m adr.predict.run       → stdout (scores, top-5, warnings)
7. streamlit run app/streamlit_app.py → web UI
```

---

## Trình tự học đề xuất

1. **Đọc** `adr/schemas.py` – hiểu contract trung tâm
2. **Đọc** `adr/config.py` – hiểu config system
3. **Chạy** `python scripts/smoke.py` – xem pipeline chạy thật
4. **Đọc** `adr/data/validate.py` – hiểu hợp đồng dữ liệu
5. **Đọc** `adr/models/frequency.py` → `logreg.py` → `mlp.py` → `cnn.py`
6. **Đọc** `adr/evaluate/metrics.py` – hiểu null handling
7. **Đọc** `adr/predict/predictor.py` – hiểu luồng inference
8. **Tải data thật** và chạy pipeline từ bước 1 đến 7 (sau khi cài RDKit)
9. **Implement** `adr/train/neural.py` (Tuần 5-7)

---

## Các điểm dễ nhầm

| Dễ nhầm | Đúng |
|---------|------|
| Dùng softmax/argmax cho đầu ra | Sigmoid độc lập từng nhãn (multi-label) |
| `mask=0` nghĩa là nhãn âm | `mask=0` = chưa quan sát; 0.0 chỉ là đệm trong tensor |
| Hardcode 1.427 như điều kiện | Số hàng thực tế đo bằng `len(df)` sau xử lý |
| Python `hash()` cho drug_id | SHA-256 canonical SMILES (ổn định giữa phiên) |
| Thay null AP/ROC bằng 0 | Giữ `None`, báo lý do, không tính vào macro |
| Evaluate test bất kỳ lúc nào | Chỉ sau khi khóa config (`config_locked: true`) |
| Scaffold split fallback về random | `NotImplementedError` – không âm thầm |
| Fit tokenizer trên toàn dataset | Chỉ fit trên train; transform val/test với tokenizer đã fit |

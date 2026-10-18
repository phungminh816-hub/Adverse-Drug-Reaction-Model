# ADR Prediction Project – HUST 2026-1

**Ứng dụng học máy và học sâu dự đoán các nhóm phản ứng có hại của thuốc từ biểu diễn phân tử.**

- **Đầu vào:** SMILES (chuỗi biểu diễn cấu trúc phân tử)
- **Đầu ra:** Điểm cho 27 nhóm ADR MedDRA (phân loại đa nhãn)
- **Benchmark:** SIDER MoleculeNet (~1.427 thuốc, 27 nhãn)
- **Thời gian:** 12 tuần, 12–15 giờ/tuần

> ⚠️ **Giới hạn diễn giải:** Điểm sigmoid chưa hiệu chỉnh, không phải xác suất bệnh nhân gặp ADR. Nhãn 0 không chứng minh thuốc an toàn. Không dùng cho quyết định lâm sàng.

---

## Môi trường thực tế

Đã xác minh: Python 3.14.7 · scikit-learn 1.9.1 · numpy 2.5.3 · pandas 3.0.6  
Cần cài thêm: rdkit, torch, pyyaml, streamlit, pytest

---

## Cài đặt

### Windows PowerShell

```powershell
# Tạo và kích hoạt virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Nếu PowerShell execution policy bị chặn, dùng:
# .\.venv\Scripts\python.exe -m pip install -r requirements.txt
# thay vì kích hoạt venv

# Cài dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# PyTorch CPU (nếu không có GPU):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# RDKit (khuyến nghị qua conda):
# conda install -c conda-forge rdkit
# Hoặc pip (unofficial):
# python -m pip install rdkit

# Cài package ở chế độ editable
python -m pip install -e .
```

### Google Colab

```python
!pip install rdkit pyyaml streamlit scikit-learn
!pip install torch  # Colab đã có GPU
!git clone <your-repo-url>
%cd ADR_Product
!pip install -e .
```

---

## Chạy nhanh

### 1. Smoke test offline (không cần data thật)

```powershell
python scripts/smoke.py
```

Kết quả mong đợi: tất cả test PASS (một số skip nếu rdkit/torch chưa cài).

### 2. Unit tests

```powershell
python -m pytest -q
```

### 3. Pipeline đầy đủ

```powershell
# Tải data
python -m adr.data.download --config configs/base.yaml

# Chuẩn bị (canonicalize, drug_id)
python -m adr.data.prepare --config configs/base.yaml

# Chia split
python -m adr.data.split --config configs/base.yaml --strategy random

# Train baseline Logistic Regression
python -m adr.train.baseline --config configs/logreg.yaml

# Evaluate trên validation (lấy <run_id> từ output của lệnh train)
python -m adr.evaluate.run --run-dir artifacts/<run_id> --split val

# Dự đoán cho một SMILES (CCO = input kiểm tra kỹ thuật, không gán tên thuốc)
python -m adr.predict.run --run-dir artifacts/<run_id> --smiles "CCO"

# Demo Streamlit
python -m streamlit run app/streamlit_app.py
```

> **Ghi chú `<run_id>`:** Sau khi chạy `python -m adr.train.baseline`, output sẽ in đường dẫn artifact, ví dụ:  
> `[baseline] LogReg lưu tại: artifacts/logreg_20261015_203000_42`  
> Dùng tên thư mục đó làm `<run_id>`.

### 4. Xem help cho bất kỳ lệnh nào

```powershell
python -m adr.data.download --help
python -m adr.data.prepare --help
python -m adr.data.split --help
python -m adr.train.baseline --help
python -m adr.train.neural --help   # Thông báo TODO (Tuần 5-7)
python -m adr.evaluate.run --help
python -m adr.predict.run --help
```

---

## Cấu trúc dự án

```text
ADR_Product/
├── README.md               # File này
├── pyproject.toml          # Packaging (cài editable)
├── requirements.txt        # Dependencies + ghi chú cài đặt
├── .gitignore
│
├── configs/                # YAML config cho từng model
│   ├── base.yaml           # Config gốc (seed, paths, data, split)
│   ├── frequency.yaml      # Baseline tần suất
│   ├── logreg.yaml         # Logistic Regression per-label
│   ├── mlp.yaml            # MLP neural
│   └── cnn.yaml            # CNN 1-nhánh
│
├── data/                   # KHÔNG commit data thật
│   ├── raw/                # sider.csv.gz + sider.csv (sau download)
│   ├── processed/          # processed.csv, label_names.json, label_mask.npy
│   └── splits/             # split_manifest.csv, split_meta.json
│
├── src/adr/                # Package chính
│   ├── config.py           # Load/merge/validate YAML
│   ├── schemas.py          # Dataclass contracts (ArtifactMetadata, PredictionResult)
│   ├── data/               # download, validate, prepare, split
│   ├── features/           # fingerprints (Morgan), tokenizer (char-level)
│   ├── models/             # frequency, logreg, mlp, cnn
│   ├── train/              # losses, baseline trainer, neural (TODO)
│   ├── evaluate/           # metrics, evaluation runner
│   ├── predict/            # predictor thống nhất, CLI
│   └── utils/              # seed, paths, artifact helpers
│
├── scripts/
│   └── smoke.py            # End-to-end smoke test offline
│
├── tests/                  # pytest test suite
│   ├── test_data_contracts.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_artifacts.py
│
├── app/
│   └── streamlit_app.py    # Demo web app
│
├── notebooks/
│   └── 01_eda.ipynb        # EDA (gọi module adr, không giữ logic pipeline)
│
├── artifacts/              # Run artifacts (KHÔNG commit)
├── results/                # Kết quả evaluate (KHÔNG commit)
└── reports/
    ├── roadmap.md          # Lộ trình 12 tuần
    └── report_outline.md   # Khung báo cáo nghiên cứu
```

---

## Config

Merge rule: `base.yaml` được load trước, model config (`logreg.yaml`, v.v.) deep-merge đè lên.  
Key lồng nhau của base không bị mất nếu model config chỉ override một sub-key.

```powershell
# Kiểm tra config hợp lệ
python -m adr.config --config configs/logreg.yaml
```

---

## Trạng thái hiện tại (Tuần 1)

| Thành phần | Trạng thái |
|---|---|
| Packaging & config | ✅ Hoạt động |
| Download, validate, prepare | ✅ Skeleton (cần RDKit + data thật) |
| Split (random) | ✅ Hoạt động |
| Split (scaffold) | 🚧 TODO Tuần 10 |
| Morgan fingerprint | ✅ Skeleton (cần RDKit) |
| Char tokenizer | ✅ Hoạt động |
| Frequency baseline | ✅ Hoạt động |
| LogReg per-label | ✅ Hoạt động |
| MLP forward | ✅ Hoạt động (cần torch) |
| CNN forward + pad invariance | ✅ Hoạt động (cần torch) |
| Masked BCE loss | ✅ Hoạt động (cần torch) |
| Metrics (AP, ROC-AUC, null) | ✅ Hoạt động |
| Predictor thống nhất | ✅ Baseline; Neural TODO |
| Streamlit demo | ✅ Hoạt động |
| Training neural (MLP/CNN) | 🚧 TODO Tuần 5-7 |
| Smoke test | ✅ (một số skip nếu thiếu lib) |
| pytest suite | ✅ (một số skip nếu thiếu torch) |

---

## Ghi chú thiết kế

- **Drug ID:** SHA-256 của canonical SMILES (không dùng Python `hash()` vì không ổn định)
- **Mask:** NaN trong dataset = nhãn chưa quan sát được; `mask=0` tại vị trí này, giá trị 0.0 là đệm
- **Leakage:** Cùng drug_id không xuất hiện ở hai split; vocabulary fit chỉ từ train
- **Test lock:** `python -m adr.evaluate.run --split test` chỉ chạy khi metadata có `config_locked: true`
- **Scaffold split:** Raise `NotImplementedError` khi chọn; không fallback âm thầm về random

---

## Lộ trình

Xem [reports/roadmap.md](reports/roadmap.md).

---

## Tài liệu tham khảo

- SIDER benchmark: https://sideeffects.embl.de/
- MoleculeNet SIDER CSV: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz
- DeepChem loader: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/sider_datasets.py
- MoleculeNet docs: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#sider-datasets

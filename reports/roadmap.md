# Roadmap 12 tuần – ADR Project 1 HUST 2026-1

> Dựa trên "Kế hoạch Project 1 ADR trong 12 tuần". Thời lượng ước tính: 12–15 giờ/tuần.  
> Ưu tiên: dữ liệu kiểm tra được → baseline → một model chạy được → test độc lập → demo → README.  
> RF, scaffold split và 3-seed có thể giảm ưu tiên khi thiếu thời gian.

---

## Tuần 1 – Môi trường & Skeleton ✅ (hiện tại)

**Mục tiêu:** Repository có thể cài, import và chạy smoke test offline.

- [x] Khởi tạo repo với layout `src/adr`
- [x] `pyproject.toml`, `requirements.txt`, `.gitignore`
- [x] Config system (base + model, deep-merge)
- [x] Schemas (`ArtifactMetadata`, `PredictionResult`)
- [x] Data pipeline skeleton: download, validate, prepare, split
- [x] Feature: Morgan fingerprint + char tokenizer
- [x] Model skeleton: Frequency, LogReg, MLP, CNN (forward đúng shape)
- [x] Masked BCE loss
- [x] Metrics: AP, ROC-AUC, null-handling, masked F1
- [x] Predictor thống nhất
- [x] CLI cho tất cả bước (`--help` hoạt động)
- [x] Smoke test offline (synthetic fixture)
- [x] Streamlit app (hướng dẫn khi thiếu artifact)
- [ ] **Cần làm:** Cài RDKit, PyTorch, PyYAML, Streamlit; chạy `python -m pip install -e .`
- [ ] **Cần làm:** Xác minh `python scripts/smoke.py` pass tất cả

---

## Tuần 2 – EDA (Exploratory Data Analysis)

**Mục tiêu:** Hiểu dữ liệu SIDER trước khi xây mô hình.

- [ ] Tải SIDER: `python -m adr.data.download --config configs/base.yaml`
- [ ] Validate: `python -m adr.data.prepare --config configs/base.yaml`
- [ ] Mở `notebooks/01_eda.ipynb`
- [ ] Phân tích:
  - Số hàng thực tế (không hardcode 1.427)
  - Phân phối positive rate mỗi nhãn (bar plot)
  - Tỷ lệ missing per nhãn
  - Phân phối độ dài SMILES (histogram)
  - Morgan fingerprint density (bit frequency)
  - SMILES lỗi nếu có
- [ ] Ghi nhận: tên 27 nhãn theo thứ tự cột nguồn
- [ ] Không hardcode kết quả EDA vào code

---

## Tuần 3 – Clean, Split & Features

**Mục tiêu:** Dữ liệu sạch, split cố định, feature NPZ.

- [ ] Canonicalize SMILES bằng RDKit; ghi báo cáo SMILES lỗi
- [ ] Tạo `drug_id` từ SHA-256 canonical SMILES
- [ ] Phát hiện và xử lý duplicate (gộp hoặc báo conflict)
- [ ] Random split 80/10/10: `python -m adr.data.split --config configs/base.yaml`
- [ ] Lưu manifest; không split lại khi đã có manifest
- [ ] Tạo fingerprint NPZ: `X[N, 2048]`, `y[N, 27]`, `mask[N, 27]`
- [ ] Lưu `label_names.json` theo thứ tự cột nguồn
- [ ] Báo support nhãn thực tế mỗi split

---

## Tuần 4 – Baseline & Metrics

**Mục tiêu:** Mốc kiểm tra; pipeline evaluate hoàn chỉnh.

- [ ] Frequency baseline: fit → predict → evaluate
  ```powershell
  python -m adr.train.baseline --config configs/frequency.yaml
  python -m adr.evaluate.run --run-dir artifacts/<run_id> --split val
  ```
- [ ] Logistic Regression per-label:
  ```powershell
  python -m adr.train.baseline --config configs/logreg.yaml
  python -m adr.evaluate.run --run-dir artifacts/<run_id> --split val
  ```
- [ ] So sánh: Frequency vs LogReg; báo macro AP, per-label AP
- [ ] Kiểm tra: null AP/ROC-AUC có lý do rõ
- [ ] Demo Streamlit với baseline artifact

---

## Tuần 5 – MLP Training

**Mục tiêu:** Neural model đầu tiên chạy được.

- [ ] Implement `adr/train/neural.py` cho MLP
  - DataLoader cho fingerprint NPZ
  - Train loop: masked_bce_loss → Adam → early stopping patience=8
  - Val loop: eval mode + no_grad → macro AP
  - Checkpoint theo val macro AP
  - Log epoch vào file
- [ ] Train:
  ```powershell
  python -m adr.train.neural --config configs/mlp.yaml
  ```
- [ ] Evaluate trên val; so sánh với LogReg
- [ ] Implement MLP inference trong `Predictor`

---

## Tuần 6 – CNN Tokenizer & Forward

**Mục tiêu:** CNN pipeline tokenizer → forward hoàn chỉnh.

- [ ] Fit CharTokenizer từ train; báo vocab_size, max_length
- [ ] Báo tỷ lệ UNK và truncation trên val/test
- [ ] Lưu tokenizer artifact cùng run_dir
- [ ] CNN DataLoader với input_ids + attention_mask
- [ ] Kiểm tra pad-invariance bằng test `test_cnn_pad_invariance`
- [ ] Forward pass đúng shape `[B, 27]`

---

## Tuần 7 – CNN Training

**Mục tiêu:** CNN training hoàn chỉnh, so sánh với MLP.

- [ ] Train CNN:
  ```powershell
  python -m adr.train.neural --config configs/cnn.yaml
  ```
- [ ] So sánh: Frequency / LogReg / MLP / CNN trên val
- [ ] Implement CNN inference trong `Predictor`
- [ ] Demo Streamlit với CNN artifact

---

## Tuần 8 – Ổn định & Class Weighting

**Mục tiêu:** Cải thiện robustness; khám phá class weighting.

- [ ] Phân tích imbalance per-label (positive rate train)
- [ ] Thử `pos_weight` cho BCE: tính trên train có mask
- [ ] Quy tắc cho zero-positive / zero-negative; giới hạn max weight
- [ ] Không kết hợp nhiều balancing mechanism mà không giải thích
- [ ] So sánh: weighted vs unweighted

---

## Tuần 9 – Chọn Cấu hình

**Mục tiêu:** Khóa cấu hình tốt nhất; không dùng test.

- [ ] Grid search nhẹ trên val: dropout, hidden_dim (MLP), channels/kernel (CNN)
- [ ] Chọn threshold trên val (không quét seed theo test)
- [ ] Lưu `config_locked: true` vào metadata sau khi chọn
- [ ] Ghi rõ cấu hình đã chọn và lý do

---

## Tuần 10 – Test Set & Phân tích lỗi

**Mục tiêu:** Báo cáo kết quả cuối; hiểu failure mode.

- [ ] Evaluate test chỉ sau khi khóa cấu hình:
  ```powershell
  python -m adr.evaluate.run --run-dir artifacts/<run_id> --split test
  ```
- [ ] Phân tích nhãn: nhãn nào AP thấp? Tại sao?
- [ ] Phân tích lỗi: phân tử nào bị sai nhãn nhất?
- [ ] PR curves per-label (matplotlib)
- [ ] (Tùy chọn) Scaffold split: train lại với manifest scaffold; không dùng test của random split

---

## Tuần 11 – Demo & Báo cáo

**Mục tiêu:** Demo hoàn chỉnh; báo cáo nghiên cứu.

- [ ] Streamlit app hoàn chỉnh: select model, top-5, 27 nhóm, warning y tế
- [ ] Notebook EDA hoàn chỉnh (gọi module, không giữ logic riêng)
- [ ] `reports/report_outline.md` → báo cáo đầy đủ:
  - Bài toán và dữ liệu
  - Phương pháp
  - Kết quả bảng so sánh (Frequency / LogReg / MLP / CNN)
  - Phân tích lỗi
  - Giới hạn và hướng mở rộng
- [ ] README tiếng Việt đầy đủ với lệnh chạy thực tế

---

## Tuần 12 – Tái lập & Bàn giao

**Mục tiêu:** Dự án tái lập được; bàn giao sạch.

- [ ] Chạy lại toàn bộ pipeline từ đầu trên môi trường sạch
- [ ] `python -m pytest -q` xanh tất cả
- [ ] `python scripts/smoke.py` pass tất cả
- [ ] README đủ để người khác chạy mà không cần hỏi
- [ ] Kiểm tra: không hardcode đường dẫn, không commit data/checkpoint
- [ ] Ghi nhận phiên bản thư viện thực tế đã dùng
- [ ] (Tùy chọn) Random Forest baseline
- [ ] (Tùy chọn) 3-seed experiment cho robustness

---

## Ưu tiên giảm khi thiếu thời gian

1. Random Forest (thay bằng LogReg là đủ baseline)
2. Scaffold split (thêm TODO, không âm thầm fallback)
3. 3-seed experiment (1 seed vẫn hợp lệ)
4. PR curves đầy đủ (per-label AP đã đủ)

---

## Không nằm trong phạm vi dự án

- Bài toán DTI (drug-target interaction), dữ liệu protein
- Bài toán hồi quy (IC50, Ki)
- Dữ liệu bệnh nhân, liều, bệnh nền, tương tác nhiều thuốc
- Mức độ nặng ADR
- GNN, Transformer, LLM
- API server, frontend React, Docker, cloud deployment, MLOps

# ADR Predictor — HUST Project 1

Dự đoán 27 nhóm ADR MedDRA từ cấu trúc phân tử SMILES, dùng benchmark SIDER MoleculeNet.
Pipeline chạy được với Frequency, Logistic Regression, MLP và CNN.
Điểm model chưa hiệu chỉnh thành xác suất nguy cơ trên bệnh nhân; chỉ phục vụ học tập/nghiên cứu.

## Chạy trên máy hiện tại

Môi trường `.venv`, dữ liệu thật và model được tạo trực tiếp trong project.
Môi trường hiện tại dùng Python 3.14.7, PyTorch CPU và kế thừa các thư viện đã cài trên máy.

```powershell
cd D:\HUST_20261\Project1\ADR_Product
.\run.ps1
```

Mở http://127.0.0.1:8501, nhập SMILES và bấm **Submit** hoặc nhấn Enter.
Giao diện tự nạp model, ưu tiên MLP, và hiển thị bảng 5 nhóm ADR có điểm cao nhất.
Dừng server bằng Ctrl+C. Nếu PowerShell chặn script:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Có thể chạy trực tiếp, không cần kích hoạt virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

## Cài mới trên máy khác

Cần Python >=3.10 và phiên bản có wheel tương thích cho RDKit/PyTorch.
Các phiên bản đã kiểm chứng trên máy này nằm trong `requirements-tested.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Hoặc chạy `.\setup.ps1`. Dùng `-UseSystemPackages` nếu chủ động muốn tái sử dụng thư viện trên máy.
PyTorch CPU có thể cài từ index chính thức:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Pipeline một lệnh

Lệnh này tải dữ liệu công khai, chuẩn bị dữ liệu, chia split, train và đánh giá **validation**:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --models logreg
.\.venv\Scripts\python.exe scripts/run_pipeline.py --models frequency logreg mlp cnn --device cpu
```

Giới hạn số epoch khi chỉ muốn kiểm tra kỹ thuật:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --models mlp cnn --epochs 2 --device cpu
```

Mỗi lần train tạo thư mục artifact riêng. Kết quả tổng hợp ghi vào `results/run_summary.json`.
Chạy lại pipeline sẽ tạo lại split; chọn cùng chiến lược và seed để giữ phép so sánh nhất quán.
Muốn thử scaffold, dùng `--strategy scaffold --base DUONG_DAN_CONFIG` với các đường dẫn dữ liệu/split/artifact riêng.
`--base` được áp dụng xuyên suốt prepare, split, training và evaluation; tên CSV, nhãn và manifest lấy từ config.

## Training từng model

Dữ liệu phải được chuẩn bị trước; pipeline một lệnh ở trên đã thực hiện các bước này.

```powershell
.\.venv\Scripts\python.exe -m adr.data.download --config configs/base.yaml
.\.venv\Scripts\python.exe -m adr.data.prepare --config configs/base.yaml
.\.venv\Scripts\python.exe -m adr.data.split --config configs/base.yaml --strategy random

.\.venv\Scripts\python.exe -m adr.train.baseline --config configs/frequency.yaml
.\.venv\Scripts\python.exe -m adr.train.baseline --config configs/logreg.yaml
.\.venv\Scripts\python.exe -m adr.train.neural --config configs/mlp.yaml --device cpu
.\.venv\Scripts\python.exe -m adr.train.neural --config configs/cnn.yaml --device cpu
```

MLP/CNN: tối đa 50 epoch, early stopping sau 8 epoch không tăng validation macro AP.
`model.pt` chứa checkpoint tốt nhất; `training_state.pt` chứa trạng thái cuối để resume.
`history.json` và `training_summary.json` ghi loss, AP, epoch tốt nhất và thiết bị.
CNN fit vocabulary bằng train; chiều dài tối đa bị giới hạn 512 ký tự trong config để kiểm soát RAM.
Inference báo nếu có UNK hoặc chuỗi bị cắt. Không fit lại tokenizer bằng validation/test.

```powershell
# Thay TEN_RUN bằng tên thư mục được in sau khi train.
.\.venv\Scripts\python.exe -m adr.train.neural --config configs/mlp.yaml --resume artifacts/TEN_RUN --epochs 80 --device cpu
```

`--epochs` là tổng số epoch. Resume giữ nguyên seed, model, features và các tham số training chính.
Nếu run đã early stop, lệnh resume không huấn luyện thêm; đổi cấu hình thì tạo run mới.
Mọi model đều từ chối training nếu có nhãn hoàn toàn không có quan sát trong tập train.
Checkpoint chỉ cho phép resume khi dataset, mask và split khớp checksum.

## Đánh giá và dự đoán

```powershell
.\.venv\Scripts\python.exe -m adr.evaluate.run --run-dir artifacts/TEN_RUN --split val
.\.venv\Scripts\python.exe -m adr.predict.run --run-dir artifacts/TEN_RUN --smiles "CC(=O)Nc1ccc(O)cc1"
```

Metric chính là macro average precision (AP), không phải accuracy.
ROC-AUC không xác định nếu chỉ có một lớp; AP không xác định khi không có positive.
Không dùng ô nhãn missing để tính loss hoặc metric. Không coi nhãn 0 là bằng chứng thuốc an toàn.

Tập test không được đánh giá tự động. Chỉ khi đã chọn xong cấu hình bằng validation:

```powershell
.\.venv\Scripts\python.exe -m adr.evaluate.run --run-dir artifacts/TEN_RUN --split test --lock-config
```

Kết quả nằm trong `artifacts/TEN_RUN/eval_val.json` hoặc `eval_test.json`.
Đánh giá kiểm tra checksum dataset/split/mask và thứ tự tên nhãn trước khi tính điểm.
Artifact đã train có thể mang sang máy khác để inference; dùng `--base` với config đường dẫn mới nếu muốn evaluate tại máy khác.

## Training ở đâu

Máy hiện tại đủ để chạy bốn model trên SIDER bằng CPU; GPU không bắt buộc.
Nếu thử nhiều cấu hình hoặc muốn train bằng GPU, dùng Google Colab:

1. Tạo gói mã nguồn: `.\.venv\Scripts\python.exe scripts/export_colab.py`.
2. Mở `notebooks/02_train_colab.ipynb` trên Colab.
3. Chọn runtime có GPU, upload `results/ADR_Product_colab.zip`, rồi chạy các cell.
4. Notebook train MLP/CNN và tải về `adr_trained_artifacts.zip`.
5. Giải nén để các thư mục model nằm ngay trong `artifacts/` trên máy này; mở lại app.

Colab cấp tài nguyên/GPU theo tình trạng khả dụng; không bảo đảm luôn có GPU.
Notebook tự chọn CPU nếu CUDA không sẵn sàng. Không cần gắn plugin GitHub để dùng cách upload ZIP.
Nguồn: [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke.py
```

Test bổ sung kiểm tra pipeline neural, save/load, resume khớp train liên tục, nhãn missing,
SMILES lỗi, scaffold không giao nhau và từ chối evaluate khi dữ liệu đã thay đổi.
Smoke dùng dữ liệu tổng hợp; chỉ dùng để kiểm tra kỹ thuật.

## Các thư mục chính

- `configs/`: tham số dữ liệu, split, features và model.
- `src/adr/data/`: tải, validate, chuẩn bị và chia dữ liệu.
- `src/adr/features/`: Morgan fingerprint, tokenizer ký tự.
- `src/adr/models/`, `src/adr/train/`: model, masked loss, training.
- `src/adr/evaluate/`, `src/adr/predict/`: đánh giá và inference.
- `app/streamlit_app.py`: giao diện demo dùng Predictor chung.
- `data/`: dữ liệu; `artifacts/`: model; `results/`: kết quả chạy.
- [Báo cáo kỹ thuật và kết quả thực nghiệm](reports/bao_cao_ky_thuat_va_ket_qua_ADR.md); hình nằm trong `reports/images/`.

## Nguồn kỹ thuật

- [SIDER CSV — DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz)
- [RDKit fingerprint generator](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html)
- [PyTorch checkpoint và resume](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

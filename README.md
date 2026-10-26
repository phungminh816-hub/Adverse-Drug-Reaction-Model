# ADR Predictor

## Abstract

Dự án xây dựng pipeline dự đoán 27 nhóm phản ứng có hại của thuốc (ADR) theo MedDRA từ cấu trúc phân tử biểu diễn bằng SMILES, sử dụng benchmark SIDER MoleculeNet do DeepChem phân phối. Dữ liệu được kiểm tra nhãn và cấu trúc bằng RDKit, chuẩn hóa SMILES, xử lý bản ghi trùng và lưu mask cho nhãn thiếu trước khi chia thành tập train, validation và test. Pipeline hỗ trợ chia ngẫu nhiên hoặc theo scaffold (khung cấu trúc phân tử).

Bốn mô hình hiện được triển khai gồm Frequency, Logistic Regression, MLP và CNN. Logistic Regression và MLP sử dụng Morgan fingerprint; CNN sử dụng chuỗi token ký tự SMILES. MLP/CNN được huấn luyện bằng masked binary cross-entropy và chọn checkpoint theo validation macro average precision (AP). Giao diện Streamlit nhận SMILES và hiển thị năm nhóm ADR có điểm cao nhất. Điểm dự đoán phục vụ nghiên cứu, chưa được hiệu chỉnh thành xác suất nguy cơ trên bệnh nhân.

Hướng phát triển tiếp theo là tích hợp encoder pretrained IBM MoLFormer để tạo biểu diễn phân tử, huấn luyện head dự đoán 27 nhãn và thực hiện fine-tuning. MoLFormer hiện chưa được tích hợp trong mã nguồn.

## Chạy project

### 1. Cài đặt

Yêu cầu Python >=3.10 với phiên bản tương thích RDKit/PyTorch. Mở PowerShell tại thư mục gốc repository. Lần cài đặt và tải dữ liệu đầu tiên cần kết nối Internet.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Có thể thay các lệnh trên bằng:

```powershell
.\setup.ps1
```

Dependency chính được khai báo trong `pyproject.toml`; `requirements-tested.txt` ghi các phiên bản tham chiếu đã được lưu trong dự án.

### 2. Mở ứng dụng

```powershell
.\run.ps1
```

Khi chưa có model, script tự tải dữ liệu, chuẩn bị dữ liệu, chia tập, train Logistic Regression và đánh giá validation trước khi mở Streamlit.

Truy cập <http://127.0.0.1:8501>, nhập SMILES, ví dụ `CC(=O)Nc1ccc(O)cc1`, rồi bấm **Submit**. Ứng dụng ưu tiên MLP, CNN, Logistic Regression, Frequency; trong cùng loại chọn run mới nhất. Dừng server bằng **Ctrl+C**.

Nếu PowerShell chặn script, chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Khi đã có model trong `artifacts/`, có thể mở ứng dụng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

### 3. Huấn luyện mô hình

Chạy toàn bộ pipeline cho bốn mô hình trên CPU:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --models frequency logreg mlp cnn --device cpu
```

Chỉ huấn luyện MLP hoặc chạy thử hai epoch:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --models mlp --device cpu
.\.venv\Scripts\python.exe scripts/run_pipeline.py --models mlp cnn --epochs 2 --device cpu
```

Pipeline tự thực hiện download, prepare, split, train và evaluate validation. Mặc định chia ngẫu nhiên theo tỷ lệ 80/10/10 với seed 42; MLP/CNN tối đa 50 epoch, early stopping sau 8 epoch không tăng validation macro AP. `--device auto` chọn CUDA khi khả dụng, ngược lại dùng CPU.

Mỗi lần chạy pipeline tạo lại split. Để dùng scaffold với các thư mục dữ liệu và kết quả riêng, sao chép `configs/base.yaml`, điều chỉnh `paths`, rồi truyền file đó qua `--base` cùng `--strategy scaffold`.

### 4. Đánh giá và dự đoán bằng CLI

Thay `TEN_RUN` bằng tên thư mục model được in sau khi train hoặc được ghi trong `results/run_summary.json`.

```powershell
.\.venv\Scripts\python.exe -m adr.evaluate.run --run-dir artifacts/TEN_RUN --split val
.\.venv\Scripts\python.exe -m adr.predict.run --run-dir artifacts/TEN_RUN --smiles "CC(=O)Nc1ccc(O)cc1"
```

Sau khi chốt cấu hình bằng validation, đánh giá test bằng:

```powershell
.\.venv\Scripts\python.exe -m adr.evaluate.run --run-dir artifacts/TEN_RUN --split test --lock-config
```

### 5. Huấn luyện trên Colab

Tạo gói mã nguồn:

```powershell
.\.venv\Scripts\python.exe scripts/export_colab.py
```

Mở `notebooks/02_train_colab.ipynb` trên Colab, upload `results/ADR_Product_colab.zip` và chạy lần lượt các cell. Notebook huấn luyện MLP/CNN với `--device auto`. Tải `adr_trained_artifacts.zip` rồi giải nén tại thư mục gốc repository để các run nằm trong `artifacts/`.

### 6. Kiểm tra kỹ thuật

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke.py
```

Smoke test sử dụng dữ liệu tổng hợp để kiểm tra luồng xử lý.

## Dữ liệu và file đầu ra

- `configs/`: cấu hình dữ liệu, chia tập, đặc trưng và mô hình.
- `data/raw/`: dữ liệu SIDER đã tải và metadata nguồn.
- `data/processed/`: dữ liệu chuẩn hóa, tên nhãn, mask và ánh xạ hàng nguồn.
- `data/splits/`: manifest và metadata chia tập.
- `artifacts/<run>/`: model, cấu hình, metadata và kết quả đánh giá; neural có thêm lịch sử train và trạng thái resume.
- `results/run_summary.json`: kết quả validation tổng hợp của lần chạy pipeline gần nhất.

Nguồn dữ liệu: [SIDER CSV do DeepChem phân phối](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz).

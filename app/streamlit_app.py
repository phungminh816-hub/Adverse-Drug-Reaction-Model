"""
app/streamlit_app.py – Demo Streamlit cho ADR prediction.

Trạng thái:
  - Nếu chưa có artifact: hiển thị hướng dẫn tạo model.
  - Nếu có artifact baseline: dùng Predictor thật, trả top-5 và 27 nhóm.
  - SMILES lỗi được giải thích rõ ràng.
  - Cảnh báo giới hạn diễn giải y tế hiển thị mọi lúc.

Chạy:
  python -m streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Thêm src vào path khi chưa cài editable
ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="ADR Predictor – HUST Project 1",
    page_icon="💊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💊 ADR Predictor")
st.markdown(
    "**Dự đoán nhóm phản ứng có hại của thuốc (ADR) từ SMILES phân tử**  \n"
    "Dự án học tập – HUST 2026-1 · Benchmark: SIDER MoleculeNet (27 nhóm MedDRA)"
)

st.warning(
    "⚠️ **Giới hạn diễn giải:**  \n"
    "- Điểm này là output model học máy, **KHÔNG phải xác suất** bệnh nhân gặp ADR.  \n"
    "- Nhãn 0 / score thấp **không chứng minh** thuốc an toàn.  \n"
    "- Các nhãn là **nhóm rộng MedDRA**; model học liên hệ thuốc–nhãn trong benchmark SIDER.  \n"
    "- **Không dùng** kết quả này cho quyết định lâm sàng."
)

# ---------------------------------------------------------------------------
# Tìm artifact
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = ROOT / "artifacts"


def find_run_dirs():
    """Tìm các run_dir hợp lệ trong artifacts/."""
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        [d for d in ARTIFACTS_DIR.iterdir() if d.is_dir() and (d / "metadata.json").exists()
         and "smoke" not in d.name],
        reverse=True,
    )


run_dirs = find_run_dirs()

# ---------------------------------------------------------------------------
# Sidebar: Chọn model
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình")

if not run_dirs:
    st.sidebar.error("Chưa có artifact model.")
    st.info(
        "### Cách tạo model đầu tiên\n\n"
        "**Bước 1:** Cài môi trường\n"
        "```powershell\n"
        "python -m venv .venv\n"
        ".venv\\Scripts\\Activate.ps1\n"
        "pip install -r requirements.txt\n"
        "pip install -e .\n"
        "```\n\n"
        "**Bước 2:** Tải và chuẩn bị dữ liệu\n"
        "```powershell\n"
        "python -m adr.data.download --config configs/base.yaml\n"
        "python -m adr.data.prepare --config configs/base.yaml\n"
        "python -m adr.data.split --config configs/base.yaml --strategy random\n"
        "```\n\n"
        "**Bước 3:** Train baseline\n"
        "```powershell\n"
        "python -m adr.train.baseline --config configs/logreg.yaml\n"
        "```\n\n"
        "**Bước 4:** Chạy lại app này\n"
        "```powershell\n"
        "python -m streamlit run app/streamlit_app.py\n"
        "```"
    )
    st.stop()

run_dir_names = [d.name for d in run_dirs]
selected_name = st.sidebar.selectbox("Chọn model artifact", run_dir_names)
selected_run_dir = ARTIFACTS_DIR / selected_name

# Load metadata
import json
meta_path = selected_run_dir / "metadata.json"
with meta_path.open("r", encoding="utf-8") as fh:
    meta_dict = json.load(fh)

st.sidebar.markdown(f"**Model:** `{meta_dict.get('model_type', 'unknown')}`")
st.sidebar.markdown(f"**Run ID:** `{meta_dict.get('run_id', '?')[:20]}...`")
st.sidebar.markdown(f"**Threshold:** `{meta_dict.get('threshold', 0.5)}`")
st.sidebar.markdown(f"**Labels:** {len(meta_dict.get('label_names', []))} nhóm")

# ---------------------------------------------------------------------------
# Load Predictor
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor(run_dir_str: str):
    from adr.predict.predictor import Predictor
    return Predictor(run_dir_str)


try:
    predictor = load_predictor(str(selected_run_dir))
    st.sidebar.success("✓ Model loaded")
except Exception as e:
    st.sidebar.error(f"Lỗi load model: {e}")
    st.error(f"Không thể load artifact: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Input SMILES
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Nhập SMILES phân tử")

col1, col2 = st.columns([3, 1])
with col1:
    smiles_input = st.text_input(
        "SMILES",
        placeholder="Nhập SMILES, ví dụ: CC(=O)Nc1ccc(O)cc1",
        help="SMILES (Simplified Molecular Input Line Entry System) – biểu diễn cấu trúc phân tử.",
    )
with col2:
    example = st.selectbox(
        "Ví dụ",
        ["", "CCO", "CC(=O)Nc1ccc(O)cc1", "c1ccccc1", "C(C(=O)O)N"],
        format_func=lambda x: x if x else "– chọn ví dụ –",
    )
    if example:
        smiles_input = example

predict_btn = st.button("🔍 Dự đoán ADR", type="primary", disabled=not smiles_input)

# ---------------------------------------------------------------------------
# Kết quả
# ---------------------------------------------------------------------------
if predict_btn and smiles_input:
    with st.spinner("Đang tính toán..."):
        result = predictor.predict(smiles_input.strip())

    if result.canonical_smiles is None:
        st.error(
            f"❌ SMILES không hợp lệ: `{smiles_input}`\n\n"
            + "\n".join(result.warnings)
        )
    else:
        st.success(f"✓ Canonical SMILES: `{result.canonical_smiles}`")
        if result.warnings:
            for w in result.warnings:
                st.warning(w)

        st.markdown("---")

        # Top-5
        st.subheader("🏆 Top 5 nhóm ADR có điểm cao nhất")
        top5 = result.top_k(5)
        for i, (label, score) in enumerate(top5, 1):
            flag = "🔴" if score >= result.threshold else "⚪"
            st.metric(
                label=f"{flag} {i}. {label}",
                value=f"{score:.4f}",
                delta=f"{'≥' if score >= result.threshold else '<'} threshold {result.threshold}",
                delta_color="off",
            )

        # Tất cả 27 nhóm
        st.markdown("---")
        with st.expander("📋 Xem tất cả 27 nhóm ADR"):
            cols = st.columns(3)
            for i, (label, score) in enumerate(zip(result.label_names, result.scores)):
                flag = "🔴" if score >= result.threshold else "⚪"
                with cols[i % 3]:
                    st.markdown(
                        f"**{flag} {label[:40]}**  \n"
                        f"Score: `{score:.4f}`"
                    )

        # Nhãn dự đoán dương
        st.markdown("---")
        if result.predicted_labels:
            st.subheader(f"✅ Nhãn dự đoán (score ≥ {result.threshold})")
            st.write(", ".join(result.predicted_labels) if result.predicted_labels else "_Không có nhãn nào_")
        else:
            st.info(f"Không có nhãn nào vượt threshold {result.threshold}.")

        st.caption(
            f"Model: {result.model_id} | Threshold: {result.threshold} | "
            f"Drug ID: {result.drug_id}"
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "HUST ADR Project 2026-1 · Dữ liệu: SIDER MoleculeNet · "
    "Kết quả chỉ phục vụ nghiên cứu học thuật."
)

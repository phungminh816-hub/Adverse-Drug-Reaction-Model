"""Nhận SMILES qua form và hiển thị năm nhóm ADR có điểm cao nhất."""

import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adr.utils.artifacts import check_artifact_dir, list_run_dirs


st.set_page_config(page_title="Dự đoán ADR", layout="centered")


def find_model_dir() -> Path | None:
    """Chọn model đã lưu, ưu tiên MLP và run mới nhất trong cùng loại."""
    priority = {"mlp": 3, "cnn": 2, "logreg": 1, "frequency": 0}
    candidates = []
    for run_dir in list_run_dirs(ROOT / "artifacts"):
        if "smoke" in run_dir.name:
            continue
        try:
            meta = check_artifact_dir(run_dir)
        except (OSError, ValueError, TypeError):
            continue
        candidates.append((priority[meta.model_type], meta.created_at_utc, run_dir))
    return max(candidates, key=lambda item: (item[0], item[1], item[2].name))[2] if candidates else None


@st.cache_resource
def load_predictor(run_dir: str):
    """Nạp và cache Predictor theo đường dẫn model."""
    from adr.predict.predictor import Predictor
    return Predictor(run_dir)


st.title("Dự đoán ADR")

with st.form("prediction", border=False):
    smiles_input = st.text_input("SMILES", placeholder="CC(=O)Nc1ccc(O)cc1")
    st.caption("Ví dụ SMILES hợp lệ: `CCO`, `c1ccccc1`, `CC(=O)Nc1ccc(O)cc1`.")
    submitted = st.form_submit_button("Submit", type="primary")

if submitted:
    smiles = smiles_input.strip()
    if not smiles:
        st.error("Nhập SMILES trước khi Submit.")
        st.stop()

    run_dir = find_model_dir()
    if run_dir is None:
        st.error("Chưa có mô hình để dự đoán.")
        st.stop()

    try:
        with st.spinner("Đang dự đoán..."):
            result = load_predictor(str(run_dir)).predict(smiles)
    except Exception:
        logging.exception("Không thể dự đoán ADR")
        st.error("Không thể thực hiện dự đoán. Vui lòng thử lại.")
        st.stop()

    if result.canonical_smiles is None:
        st.error("SMILES không hợp lệ. Kiểm tra lại chuỗi đã nhập.")
        st.stop()

    st.subheader("Kết quả")
    top5 = result.top_k(5)
    st.dataframe(
        {"Nhóm ADR": [label for label, _ in top5], "Điểm": [score for _, score in top5]},
        hide_index=True,
        width="stretch",
        column_config={"Điểm": st.column_config.NumberColumn(format="%.4f")},
    )
    for warning in result.warnings:
        st.caption(warning)
    st.caption("5 nhóm có điểm cao nhất. Điểm dùng cho nghiên cứu, không phải nguy cơ ở bệnh nhân.")

"""Kiểm tra schema, nhãn, SMILES và phân tử trùng trong CSV nguồn."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from adr.schemas import NUM_LABELS, SMILES_COL


try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False


@dataclass
class ValidationReport:
    """Kết quả kiểm tra dữ liệu."""
    total_rows: int = 0
    n_labels: int = 0
    label_names: list[str] = field(default_factory=list)
    smiles_errors: list[dict] = field(default_factory=list)
    label_value_errors: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    duplicates_merged: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_rdkit: bool = False

    @property
    def is_valid(self) -> bool:
        return (
            len(self.smiles_errors) == 0
            and len(self.label_value_errors) == 0
            and len(self.conflicts) == 0
            and self.n_labels == NUM_LABELS
        )


def _get_label_cols(df: pd.DataFrame) -> list[str]:
    """Trả danh sách cột nhãn: mọi cột trừ 'smiles'."""
    return [c for c in df.columns if c != SMILES_COL]


def _canonical_smiles(smi: str) -> str | None:
    """Trả canonical isomeric SMILES hoặc None nếu lỗi."""
    if not _RDKIT_OK:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def validate_raw(
    csv_path: Path,
    save_label_names: Path | None = None,
) -> ValidationReport:
    """
    Đọc raw CSV và chạy mọi kiểm tra.

    Args:
        csv_path: Path đến sider.csv (đã giải nén).
        save_label_names: Nếu truyền, lưu label_names.json vào path này.

    Returns:
        ValidationReport
    """
    report = ValidationReport(has_rdkit=_RDKIT_OK)

    if not _RDKIT_OK:
        report.warnings.append(
            "RDKit chưa được cài. Bỏ qua canonicalization và kiểm tra SMILES. "
            "Cài bằng: conda install -c conda-forge rdkit"
        )

    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"File raw không tồn tại: {csv_path}\n"
            "  Chạy: python -m adr.data.download --config configs/base.yaml"
        )

    df = pd.read_csv(csv_path)
    report.total_rows = len(df)

    if SMILES_COL not in df.columns:
        raise ValueError(f"Thiếu cột '{SMILES_COL}' trong {csv_path}")

    label_cols = _get_label_cols(df)
    report.n_labels = len(label_cols)
    report.label_names = label_cols

    if report.n_labels != NUM_LABELS:
        report.warnings.append(
            f"Số cột nhãn là {report.n_labels}, kỳ vọng {NUM_LABELS}. "
            "Kiểm tra file CSV."
        )

    if save_label_names is not None:
        Path(save_label_names).parent.mkdir(parents=True, exist_ok=True)
        with open(save_label_names, "w", encoding="utf-8") as fh:
            json.dump(label_cols, fh, indent=2, ensure_ascii=False)

    for col in label_cols:
        col_vals = df[col].dropna().unique()
        invalid = [v for v in col_vals if v not in (0, 1, 0.0, 1.0)]
        if invalid:
            report.label_value_errors.append({"col": col, "invalid_values": invalid})

    canonical_map: dict[int, str | None] = {}
    for idx, smi in df[SMILES_COL].items():
        if not isinstance(smi, str) or smi.strip() == "":
            report.smiles_errors.append({
                "row": int(idx), "smiles": smi, "reason": "Rỗng hoặc không phải string"
            })
            canonical_map[int(idx)] = None
            continue
        canon = _canonical_smiles(smi.strip())
        if canon is None:
            report.smiles_errors.append({
                "row": int(idx), "smiles": smi, "reason": "RDKit parse lỗi"
            })
        canonical_map[int(idx)] = canon

    df["_canonical"] = df.index.map(canonical_map)
    valid_mask = df["_canonical"].notna()
    df_valid = df[valid_mask].copy()

    groups = df_valid.groupby("_canonical")
    for canon_smi, grp in groups:
        if len(grp) <= 1:
            continue

        label_arrays = [tuple(grp.iloc[i][label_cols].fillna(0).tolist()) for i in range(len(grp))]
        mask_arrays = [tuple(grp.iloc[i][label_cols].isna().tolist()) for i in range(len(grp))]

        if len(set(zip(label_arrays, mask_arrays))) == 1:

            report.duplicates_merged.append({
                "canonical_smiles": canon_smi,
                "source_rows": grp.index.tolist(),
            })
        else:

            report.conflicts.append({
                "canonical_smiles": canon_smi,
                "source_rows": grp.index.tolist(),
                "labels": [list(la) for la in label_arrays],
            })

    df.drop(columns=["_canonical"], inplace=True)
    return report


def print_report(report: ValidationReport) -> None:
    """In tóm tắt ValidationReport ra stdout."""
    print(f"[validate] Tổng hàng: {report.total_rows}")
    print(f"[validate] Số nhãn: {report.n_labels}  (kỳ vọng {NUM_LABELS})")
    print(f"[validate] RDKit: {'có' if report.has_rdkit else 'không có'}")
    if report.smiles_errors:
        print(f"[validate] SMILES lỗi: {len(report.smiles_errors)}")
        for e in report.smiles_errors[:5]:
            print(f"  row={e['row']}: {str(e['smiles'])[:60]}  ({e['reason']})")
    if report.label_value_errors:
        print(f"[validate] Giá trị nhãn lạ: {len(report.label_value_errors)} cột")
    if report.conflicts:
        print(f"[validate] Conflict phân tử trùng: {len(report.conflicts)}")
        print("  → Cần quy tắc giải quyết trước khi chạy prepare.")
    if report.duplicates_merged:
        print(f"[validate] Phân tử trùng có thể gộp: {len(report.duplicates_merged)}")
    if report.warnings:
        print("[validate] Cảnh báo:")
        for w in report.warnings:
            print(f"  ! {w}")
    if report.is_valid:
        print("[validate] ✓ Dữ liệu hợp lệ.")
    else:
        print("[validate] ✗ Có lỗi. Xem chi tiết ở trên.")

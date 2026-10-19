"""Chuẩn hóa SMILES, gộp phân tử trùng và lưu dữ liệu cùng mask nhãn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from adr.config import load_config, resolve_paths
from adr.data.validate import validate_raw, print_report, _canonical_smiles, _RDKIT_OK
from adr.schemas import NUM_LABELS, SMILES_COL, DRUG_ID_COL, CANONICAL_SMILES_COL
from adr.utils.paths import sha256_str


def prepare(
    raw_csv: Path,
    processed_dir: Path,
    label_names_path: Path | None = None,
    num_labels: int = NUM_LABELS,
    processed_filename: str = "processed.csv",
) -> pd.DataFrame:
    """
    Chạy pipeline prepare và lưu artifact.

    Returns:
        DataFrame processed với cột drug_id, smiles, canonical_smiles, 27 nhãn.
    """
    raw_csv = Path(raw_csv)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not _RDKIT_OK:
        raise ImportError("Cần cài RDKit trước khi chuẩn bị dữ liệu.")

    label_names_save = label_names_path or (processed_dir / "label_names.json")
    report = validate_raw(raw_csv, save_label_names=label_names_save)
    print_report(report)
    if report.label_value_errors:
        raise ValueError("Nhãn phải là 0, 1 hoặc NaN; xem báo cáo validation.")

    if report.conflicts:
        print(
            "\n[prepare] DỪNG: Phát hiện conflict phân tử trùng.\n"
            f"  {len(report.conflicts)} phân tử có canonical SMILES giống nhau nhưng nhãn/mask khác.\n"
            "  Cần quyết định quy tắc giải quyết trước khi tiếp tục.\n"
            "  Xem chi tiết trong ValidationReport.",
            file=sys.stderr,
        )
        sys.exit(2)

    if report.n_labels != num_labels:
        print(
            f"[prepare] LỖI: Số nhãn {report.n_labels} != {num_labels}.",
            file=sys.stderr,
        )
        sys.exit(1)

    label_cols = report.label_names
    df = pd.read_csv(raw_csv)

    rows_valid = []
    skipped = []
    for idx, row in df.iterrows():
        smi = str(row[SMILES_COL]).strip()
        canon = _canonical_smiles(smi)
        if canon is None:
            skipped.append({"source_row": int(idx), "smiles": smi, "reason": "RDKit parse lỗi"})
            continue
        rows_valid.append({
            "source_row": int(idx),
            SMILES_COL: smi,
            CANONICAL_SMILES_COL: canon,
            **{col: row[col] for col in label_cols},
        })

    if skipped:
        print(f"[prepare] Bỏ {len(skipped)} hàng SMILES lỗi:")
        for s in skipped[:5]:
            print(f"  row={s['source_row']}: {s['smiles'][:60]}")
        (processed_dir / "smiles_errors.json").write_text(
            json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not rows_valid:
        raise ValueError("Không còn SMILES hợp lệ sau khi kiểm tra dữ liệu.")
    df_valid = pd.DataFrame(rows_valid)

    df_valid[DRUG_ID_COL] = df_valid[CANONICAL_SMILES_COL].map(sha256_str)

    mapping_records = []
    merged_rows = []
    seen: dict[str, list[int]] = {}
    for _, row in df_valid.iterrows():
        did = row[DRUG_ID_COL]
        if did not in seen:
            seen[did] = [int(row["source_row"])]
            merged_rows.append(row)
        else:
            seen[did].append(int(row["source_row"]))

    df_out = pd.DataFrame(merged_rows).drop(columns=["source_row"])

    for did, src_rows in seen.items():
        mapping_records.append({"drug_id": did, "source_rows": src_rows})

    cols_order = [DRUG_ID_COL, SMILES_COL, CANONICAL_SMILES_COL] + label_cols
    df_out = df_out[cols_order].reset_index(drop=True)

    mask_arr = df_out[label_cols].notna().to_numpy(dtype=np.int8)
    df_out[label_cols] = df_out[label_cols].fillna(0.0)

    out_csv = processed_dir / processed_filename
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    mapping_path = processed_dir / "drug_id_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as fh:
        json.dump(mapping_records, fh, indent=2, ensure_ascii=False)

    np.save(out_csv.parent / "label_mask.npy", mask_arr)

    print(f"[prepare] Đã lưu:")
    print(f"  {out_csv}  ({len(df_out)} thuốc, {num_labels} nhãn)")
    print(f"  {label_names_save}")
    print(f"  {mapping_path}")
    print(f"  {out_csv.parent / 'label_mask.npy'}  shape={mask_arr.shape}")

    return df_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dữ liệu SIDER: canonicalize, tạo drug_id, lưu processed.csv",
        epilog=(
            "Ví dụ:\n"
            "  python -m adr.data.prepare --config configs/base.yaml\n"
        ),
    )
    parser.add_argument("--config", required=True, help="Path đến base config YAML")
    args = parser.parse_args(argv)

    cfg = resolve_paths(load_config(args.config))
    raw_dir: Path = cfg["paths"]["data_raw"]
    processed_dir: Path = cfg["paths"]["data_processed"]
    raw_filename = cfg["data"]["raw_filename"].removesuffix(".gz")
    raw_csv = raw_dir / raw_filename

    prepare(
        raw_csv, processed_dir,
        label_names_path=processed_dir / cfg["data"]["label_names_file"],
        num_labels=cfg["data"]["num_labels"],
        processed_filename=cfg["data"]["processed_filename"],
    )


if __name__ == "__main__":
    main()

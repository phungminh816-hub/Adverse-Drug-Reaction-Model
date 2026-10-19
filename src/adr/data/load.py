"""Đọc split và mask theo cùng thứ tự drug_id."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_split(processed_csv: Path, splits_dir: Path, label_names: list[str],
               split_name: str = "train",
               manifest_filename: str = "split_manifest.csv") -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Đọc SMILES, nhãn và mask theo cùng thứ tự drug_id trong manifest."""
    df = pd.read_csv(processed_csv)
    manifest = pd.read_csv(Path(splits_dir) / manifest_filename)
    mask = np.load(Path(processed_csv).parent / "label_mask.npy", allow_pickle=False)
    if not label_names or len(set(label_names)) != len(label_names):
        raise ValueError("Danh sách tên nhãn rỗng hoặc trùng lặp.")
    if not {"drug_id", "canonical_smiles", *label_names}.issubset(df.columns):
        raise ValueError("processed.csv thiếu cột drug_id, canonical_smiles hoặc nhãn.")
    if not {"drug_id", "split"}.issubset(manifest.columns):
        raise ValueError("Manifest thiếu cột drug_id hoặc split.")
    if df.drug_id.isna().any() or manifest.drug_id.isna().any():
        raise ValueError("drug_id bị thiếu trong dữ liệu hoặc manifest.")
    if df.drug_id.duplicated().any() or manifest.drug_id.duplicated().any():
        raise ValueError("drug_id bị trùng trong dữ liệu hoặc manifest.")
    if set(df.drug_id) != set(manifest.drug_id):
        raise ValueError("Manifest không khớp tập drug_id của processed.csv.")
    if not set(manifest.split).issubset({"train", "val", "test"}):
        raise ValueError("Manifest chứa tên split không hợp lệ.")
    if mask.shape != (len(df), len(label_names)):
        raise ValueError("Kích thước label_mask không khớp dữ liệu.")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("label_mask chỉ được chứa 0 hoặc 1.")
    mask = mask.astype(bool)
    targets = df[label_names].to_numpy(dtype=np.float32)
    if not np.isin(targets[mask], [0, 1]).all():
        raise ValueError("Nhãn quan sát được phải là 0 hoặc 1.")
    if df.canonical_smiles.isna().any() or df.canonical_smiles.str.strip().eq("").any():
        raise ValueError("canonical_smiles bị thiếu hoặc rỗng.")
    ids = manifest.loc[manifest.split == split_name, "drug_id"].tolist()
    if not ids:
        raise ValueError(f"Split '{split_name}' không có dữ liệu.")
    indices = df.set_index("drug_id").index.get_indexer(ids)
    rows = df.iloc[indices]
    return ids, rows.canonical_smiles.tolist(), targets[indices], mask[indices]

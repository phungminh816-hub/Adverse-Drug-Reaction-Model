"""Chia phân tử thành train/validation/test theo random hoặc scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from adr.config import load_config, resolve_paths
from adr.schemas import DRUG_ID_COL, SPLIT_COL


def random_split(
    drug_ids: list[str],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, str]:
    """
    Chia drug_ids ngẫu nhiên thành train/val/test.

    Returns:
        dict drug_id → "train" | "val" | "test"
    """
    _validate_ratios(train_ratio, val_ratio)
    if len(set(drug_ids)) != len(drug_ids):
        raise ValueError("drug_id bị trùng; chạy prepare trước khi chia split.")
    rng = np.random.default_rng(seed)
    ids = np.array(drug_ids)
    idx = rng.permutation(len(ids))

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    assignment: dict[str, str] = {}
    for i in train_idx:
        assignment[ids[i]] = "train"
    for i in val_idx:
        assignment[ids[i]] = "val"
    for i in test_idx:
        assignment[ids[i]] = "test"

    return assignment


def scaffold_split(drug_ids: list[str], canonical_smiles: list[str] | None = None,
                   train_ratio: float = 0.8, val_ratio: float = 0.1,
                   seed: int = 42) -> dict[str, str]:
    """Giữ mỗi Bemis-Murcko scaffold trong đúng một split."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if canonical_smiles is None or len(canonical_smiles) != len(drug_ids):
        raise ValueError("Cần canonical_smiles tương ứng với từng drug_id.")
    if len(set(drug_ids)) != len(drug_ids):
        raise ValueError("drug_id bị trùng; chạy prepare trước khi chia split.")
    _validate_ratios(train_ratio, val_ratio)
    groups = {}
    for did, smi in zip(drug_ids, canonical_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"SMILES không hợp lệ: {smi}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.setdefault(scaffold, []).append(did)
    if len(groups) < 3:
        raise ValueError("Cần ít nhất 3 nhóm scaffold để tạo train/val/test.")
    ordered = [groups[k] for k in sorted(groups)]
    np.random.default_rng(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    names = ["train", "val", "test"]
    targets = np.array([train_ratio, val_ratio, 1 - train_ratio - val_ratio]) * len(drug_ids)
    counts = np.zeros(3)
    assignment = {}
    for group in ordered:

        dest = int(np.argmax(targets - counts))
        counts[dest] += len(group)
        assignment.update({did: names[dest] for did in group})
    if set(assignment.values()) != set(names):
        raise ValueError("Nhóm scaffold quá lệch để tạo đủ 3 split; hãy dùng random hoặc đổi tỷ lệ.")
    return assignment


def _validate_ratios(train_ratio, val_ratio):
    """Kiểm tra tỷ lệ train, validation và phần test còn lại đều dương."""
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Tỷ lệ train/val/test phải dương và tổng bằng 1.")


def split_dataset(
    processed_csv: Path,
    splits_dir: Path,
    strategy: str = "random",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    manifest_filename: str = "split_manifest.csv",
) -> pd.DataFrame:
    """
    Đọc processed.csv và tạo split manifest.

    Returns:
        DataFrame manifest với cột drug_id, split.
    """
    processed_csv = Path(processed_csv)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Không tìm thấy processed.csv: {processed_csv}\n"
            "  Chạy: python -m adr.data.prepare --config configs/base.yaml"
        )

    df = pd.read_csv(processed_csv)
    if DRUG_ID_COL not in df.columns:
        raise ValueError(f"Cột '{DRUG_ID_COL}' không có trong {processed_csv}")

    drug_ids = df[DRUG_ID_COL].tolist()
    if len(set(drug_ids)) != len(drug_ids):
        raise ValueError("drug_id bị trùng trong processed.csv.")

    if strategy == "random":
        assignment = random_split(drug_ids, train_ratio, val_ratio, seed)
    elif strategy == "scaffold":
        assignment = scaffold_split(drug_ids, df["canonical_smiles"].tolist(), train_ratio, val_ratio, seed)
    else:
        raise ValueError(f"Chiến lược không hỗ trợ: '{strategy}'. Chọn: random | scaffold")

    if set(assignment.values()) != {"train", "val", "test"}:
        raise ValueError("Dataset hoặc tỷ lệ quá nhỏ để tạo đủ train/val/test.")
    manifest = pd.DataFrame([
        {DRUG_ID_COL: did, SPLIT_COL: split}
        for did, split in assignment.items()
    ])

    counts = manifest[SPLIT_COL].value_counts()
    print("[split] Phân bổ:")
    for split in ["train", "val", "test"]:
        n = counts.get(split, 0)
        print(f"  {split}: {n} ({n/len(manifest)*100:.1f}%)")

    manifest_path = splits_dir / manifest_filename
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    meta = {
        "strategy": strategy,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 4),
        "n_total": len(manifest),
        "counts": counts.to_dict(),
        "processed_csv": str(processed_csv),
    }
    with (splits_dir / "split_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[split] Manifest: {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chia dataset thành train/val/test",
        epilog=(
            "Ví dụ:\n"
            "  python -m adr.data.split --config configs/base.yaml --strategy random\n"
        ),
    )
    parser.add_argument("--config", required=True, help="Path đến base config YAML")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["random", "scaffold"],
        help="Chiến lược split (mặc định từ config)",
    )
    args = parser.parse_args(argv)

    cfg = resolve_paths(load_config(args.config))
    split_cfg = cfg["split"]
    strategy = args.strategy or split_cfg["strategy"]
    seed = split_cfg["seed"]
    train_ratio = split_cfg["train_ratio"]
    val_ratio = split_cfg["val_ratio"]

    processed_csv: Path = cfg["paths"]["data_processed"] / cfg["data"]["processed_filename"]
    splits_dir: Path = cfg["paths"]["data_splits"]

    split_dataset(
        processed_csv, splits_dir, strategy, train_ratio, val_ratio, seed,
        manifest_filename=cfg["data"]["split_manifest"],
    )


if __name__ == "__main__":
    main()

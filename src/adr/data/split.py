"""
adr.data.split – Chia processed.csv thành train/val/test.

Chiến lược hiện có:
  - "random": Group random split theo canonical molecule (drug_id). Mỗi
    drug_id chỉ xuất hiện trong một split.
  - "scaffold": TODO – NotImplementedError khi chọn.

Quy tắc:
  - Chỉ split sau khi validate đã chạy sạch.
  - Cùng drug_id không được xuất hiện ở hai split.
  - Lặp lại cùng seed cho cùng manifest.
  - Lưu manifest: drug_id, split (train|val|test).

CLI:
  python -m adr.data.split --config configs/base.yaml --strategy random
  python -m adr.data.split --config configs/base.yaml --strategy scaffold  # NotImplementedError
"""

from __future__ import annotations

import argparse
import json
import sys
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


def scaffold_split(drug_ids: list[str], *args, **kwargs) -> dict[str, str]:
    """
    Scaffold split – chưa triển khai.

    TODO:
      Input: drug_ids + canonical_smiles mapping.
      Bước:
        1. Tính Bemis-Murcko scaffold bằng RDKit cho mỗi phân tử.
        2. Nhóm theo scaffold.
        3. Phân chia nhóm scaffold (không phải phân tử) vào train/val/test.
        4. Scaffold rỗng → giữ tất cả trong một nhóm riêng; báo tỷ lệ ảnh hưởng.
        5. Đảm bảo không scaffold nào nằm ở hai split.
      Output: Manifest giống random_split.
      Điều kiện hoàn thành: test không giao scaffold với train/val.
    """
    raise NotImplementedError(
        "Scaffold split chưa được triển khai.\n"
        "  Xem TODO trong adr/data/split.py::scaffold_split.\n"
        "  Dùng --strategy random cho phiên bản hiện tại."
    )


def split_dataset(
    processed_csv: Path,
    splits_dir: Path,
    strategy: str = "random",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
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

    if strategy == "random":
        assignment = random_split(drug_ids, train_ratio, val_ratio, seed)
    elif strategy == "scaffold":
        scaffold_split(drug_ids)  # sẽ raise NotImplementedError
    else:
        raise ValueError(f"Chiến lược không hỗ trợ: '{strategy}'. Chọn: random | scaffold")

    manifest = pd.DataFrame([
        {DRUG_ID_COL: did, SPLIT_COL: split}
        for did, split in assignment.items()
    ])

    # Báo số mẫu mỗi split
    counts = manifest[SPLIT_COL].value_counts()
    print("[split] Phân bổ:")
    for split in ["train", "val", "test"]:
        n = counts.get(split, 0)
        print(f"  {split}: {n} ({n/len(manifest)*100:.1f}%)")

    # Lưu manifest
    manifest_path = splits_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Lưu metadata split
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

    split_dataset(processed_csv, splits_dir, strategy, train_ratio, val_ratio, seed)


if __name__ == "__main__":
    main()

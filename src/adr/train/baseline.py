"""
adr.train.baseline – Train và lưu baseline models (Frequency, LogReg).

CLI:
  python -m adr.train.baseline --config configs/frequency.yaml
  python -m adr.train.baseline --config configs/logreg.yaml

Output: artifacts/<run_id>/ với metadata.json và model artifact.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adr.config import load_config, resolve_paths, save_config
from adr.features.fingerprints import MorganFeaturizer
from adr.models.frequency import FrequencyBaseline
from adr.models.logreg import PerLabelLogReg
from adr.schemas import ArtifactMetadata, NUM_LABELS, DRUG_ID_COL, CANONICAL_SMILES_COL
from adr.utils.paths import get_run_id, get_run_dir, sha256_file, get_library_versions
from adr.utils.seed import set_seed


def _load_split_data(
    processed_csv: Path,
    splits_dir: Path,
    label_names: list[str],
    split_name: str = "train",
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """
    Load dữ liệu cho một split từ processed.csv và split_manifest.csv.

    Returns:
        drug_ids, canonical_smiles, y [N, 27], mask [N, 27]
    """
    df = pd.read_csv(processed_csv)
    manifest = pd.read_csv(splits_dir / "split_manifest.csv")

    split_ids = set(manifest.loc[manifest["split"] == split_name, DRUG_ID_COL].tolist())
    df_split = df[df[DRUG_ID_COL].isin(split_ids)].copy()
    df_split = df_split.set_index(DRUG_ID_COL)
    # Giữ thứ tự từ manifest để ổn định
    ordered_ids = [did for did in manifest.loc[manifest["split"] == split_name, DRUG_ID_COL] if did in df_split.index]
    df_split = df_split.loc[ordered_ids]

    canonical_smiles = df_split[CANONICAL_SMILES_COL].tolist()
    y = df_split[label_names].values.astype(np.float32)

    # Load mask
    # mask được lưu trong label_mask.npy (toàn bộ dataset)
    # Cần map lại theo drug_id
    mask_full = np.load(processed_csv.parent / "label_mask.npy").astype(bool)
    # processed.csv có cùng thứ tự với mask_full sau prepare
    df_full = pd.read_csv(processed_csv)
    df_full_indexed = df_full.set_index(DRUG_ID_COL)
    mask_dict = {
        did: mask_full[i]
        for i, did in enumerate(df_full[DRUG_ID_COL].tolist())
    }
    mask = np.stack([mask_dict[did] for did in ordered_ids], axis=0)

    return ordered_ids, canonical_smiles, y, mask


def train_frequency(
    cfg: dict,
    processed_csv: Path,
    splits_dir: Path,
    label_names: list[str],
    artifacts_root: Path,
) -> Path:
    """Train frequency baseline và lưu artifact."""
    seed = cfg["project"]["seed"]
    set_seed(seed)

    drug_ids, smiles, y_train, mask_train = _load_split_data(
        processed_csv, splits_dir, label_names, "train"
    )
    print(f"[baseline] Train set: {len(drug_ids)} thuốc")

    model = FrequencyBaseline(n_labels=NUM_LABELS)
    model.fit(y_train, mask_train, label_names)

    run_id = get_run_id("frequency", seed)
    run_dir = get_run_dir(artifacts_root, run_id)
    model.save(run_dir)

    meta = ArtifactMetadata(
        model_type="frequency",
        run_id=run_id,
        seed=seed,
        label_names=label_names,
        dataset_checksum=sha256_file(processed_csv),
        split_manifest_checksum=sha256_file(splits_dir / "split_manifest.csv"),
        feature_config={"type": "frequency_baseline"},
        library_versions=get_library_versions(),
        threshold=cfg.get("inference", {}).get("threshold", 0.5),
        eval_label_policy="all_27_labels; null AP/ROC if no positive in split",
        created_at_utc=datetime.datetime.utcnow().isoformat() + "Z",
        config_resolved=cfg,
    )
    meta.save(run_dir / "metadata.json")
    save_config(cfg, run_dir)

    print(f"[baseline] Frequency baseline lưu tại: {run_dir}")
    return run_dir


def train_logreg(
    cfg: dict,
    processed_csv: Path,
    splits_dir: Path,
    label_names: list[str],
    artifacts_root: Path,
) -> Path:
    """Train LogReg per-label và lưu artifact."""
    seed = cfg["project"]["seed"]
    set_seed(seed)

    drug_ids, smiles, y_train, mask_train = _load_split_data(
        processed_csv, splits_dir, label_names, "train"
    )
    print(f"[baseline] Train set: {len(drug_ids)} thuốc")

    # Tạo fingerprint
    fp_cfg = cfg.get("features", {}).get("fingerprint", {})
    featurizer = MorganFeaturizer(
        radius=fp_cfg.get("radius", 2),
        n_bits=fp_cfg.get("n_bits", 2048),
        use_chirality=fp_cfg.get("use_chirality", True),
    )
    print("[baseline] Tạo fingerprint...")
    X_train, errors = featurizer.transform(smiles)
    if errors:
        print(f"  Cảnh báo: {len(errors)} SMILES lỗi trong train")

    print(f"[baseline] X_train shape: {X_train.shape}")

    # Train LogReg
    model_cfg = cfg.get("model", {})
    model = PerLabelLogReg(
        n_labels=NUM_LABELS,
        solver=model_cfg.get("solver", "lbfgs"),
        max_iter=model_cfg.get("max_iter", 1000),
        C=model_cfg.get("C", 1.0),
    )
    print("[baseline] Fit LogReg per-label...")
    model.fit(X_train, y_train, mask_train, label_names)

    run_id = get_run_id("logreg", seed)
    run_dir = get_run_dir(artifacts_root, run_id)
    model.save(run_dir)
    featurizer.save(run_dir / "fingerprint_config.json")

    meta = ArtifactMetadata(
        model_type="logreg",
        run_id=run_id,
        seed=seed,
        label_names=label_names,
        dataset_checksum=sha256_file(processed_csv),
        split_manifest_checksum=sha256_file(splits_dir / "split_manifest.csv"),
        feature_config=featurizer.get_config(),
        library_versions=get_library_versions(),
        threshold=cfg.get("inference", {}).get("threshold", 0.5),
        eval_label_policy="all_27_labels; null AP/ROC if no positive in split",
        created_at_utc=datetime.datetime.utcnow().isoformat() + "Z",
        config_resolved=cfg,
    )
    meta.save(run_dir / "metadata.json")
    save_config(cfg, run_dir)

    print(f"[baseline] LogReg lưu tại: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline model (frequency hoặc logreg)",
        epilog=(
            "Ví dụ:\n"
            "  python -m adr.train.baseline --config configs/logreg.yaml\n"
            "  python -m adr.train.baseline --config configs/frequency.yaml\n"
        ),
    )
    parser.add_argument("--config", required=True, help="Path đến model config YAML")
    parser.add_argument("--base", default=None, help="Path đến base.yaml (mặc định: configs/base.yaml)")
    args = parser.parse_args(argv)

    cfg = resolve_paths(load_config(args.config, base=args.base))

    processed_csv: Path = cfg["paths"]["data_processed"] / cfg["data"]["processed_filename"]
    splits_dir: Path = cfg["paths"]["data_splits"]
    artifacts_root: Path = cfg["paths"]["artifacts"]
    label_names_path: Path = cfg["paths"]["data_processed"] / cfg["data"]["label_names_file"]

    if not processed_csv.exists():
        print(f"[baseline] LỖI: Chưa có {processed_csv}. Chạy prepare trước.", file=sys.stderr)
        sys.exit(1)

    if not label_names_path.exists():
        print(f"[baseline] LỖI: Chưa có {label_names_path}. Chạy prepare trước.", file=sys.stderr)
        sys.exit(1)

    with label_names_path.open("r", encoding="utf-8") as fh:
        label_names: list[str] = json.load(fh)

    model_type = cfg.get("model", {}).get("type", "logreg")

    if model_type == "frequency":
        train_frequency(cfg, processed_csv, splits_dir, label_names, artifacts_root)
    elif model_type == "logreg":
        train_logreg(cfg, processed_csv, splits_dir, label_names, artifacts_root)
    else:
        print(f"[baseline] Model type '{model_type}' chưa được hỗ trợ ở đây.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

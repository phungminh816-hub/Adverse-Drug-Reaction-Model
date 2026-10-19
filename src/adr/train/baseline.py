"""Huấn luyện Frequency/LogReg và lưu model, đặc trưng, metadata."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from adr.config import load_config, resolve_paths, save_config
from adr.data.load import load_split
from adr.features.fingerprints import MorganFeaturizer
from adr.models.frequency import FrequencyBaseline
from adr.models.logreg import PerLabelLogReg
from adr.schemas import ArtifactMetadata
from adr.utils.paths import get_run_id, get_run_dir, sha256_file, get_library_versions
from adr.utils.seed import set_seed


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

    manifest_filename = cfg["data"].get("split_manifest", "split_manifest.csv")
    drug_ids, _, y_train, mask_train = load_split(
        processed_csv, splits_dir, label_names, "train", manifest_filename
    )
    print(f"[baseline] Train set: {len(drug_ids)} thuốc")

    model = FrequencyBaseline(n_labels=len(label_names))
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
        label_mask_checksum=sha256_file(processed_csv.parent / "label_mask.npy"),
        split_manifest_checksum=sha256_file(splits_dir / manifest_filename),
        feature_config={"type": "frequency_baseline"},
        library_versions=get_library_versions(),
        threshold=cfg.get("inference", {}).get("threshold", 0.5),
        eval_label_policy="all_27_labels; null AP/ROC if no positive in split",
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
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

    manifest_filename = cfg["data"].get("split_manifest", "split_manifest.csv")
    drug_ids, smiles, y_train, mask_train = load_split(
        processed_csv, splits_dir, label_names, "train", manifest_filename
    )
    print(f"[baseline] Train set: {len(drug_ids)} thuốc")

    fp_cfg = cfg.get("features", {}).get("fingerprint", {})
    featurizer = MorganFeaturizer(
        radius=fp_cfg.get("radius", 2),
        n_bits=fp_cfg.get("n_bits", 2048),
        use_chirality=fp_cfg.get("use_chirality", True),
    )
    print("[baseline] Tạo fingerprint...")
    X_train, errors = featurizer.transform(smiles)
    if errors:
        raise ValueError(f"SMILES lỗi trong train: {errors[:3]}")

    print(f"[baseline] X_train shape: {X_train.shape}")

    model_cfg = cfg.get("model", {})
    model = PerLabelLogReg(
        n_labels=len(label_names),
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
        label_mask_checksum=sha256_file(processed_csv.parent / "label_mask.npy"),
        split_manifest_checksum=sha256_file(splits_dir / manifest_filename),
        feature_config=featurizer.get_config(),
        library_versions=get_library_versions(),
        threshold=cfg.get("inference", {}).get("threshold", 0.5),
        eval_label_policy="all_27_labels; null AP/ROC if no positive in split",
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
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

"""
adr.evaluate.run – Evaluation runner cho một run artifact.

Mặc định đánh giá trên validation. Test chỉ khi có --split test
VÀ metadata đã locked (có flag "config_locked": true trong metadata).

CLI:
  python -m adr.evaluate.run --run-dir artifacts/<run_id> --split val
  python -m adr.evaluate.run --run-dir artifacts/<run_id> --split test  # yêu cầu config_locked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adr.config import load_config, resolve_paths
from adr.evaluate.metrics import evaluate, print_eval_summary
from adr.schemas import ArtifactMetadata, DRUG_ID_COL, CANONICAL_SMILES_COL
from adr.utils.artifacts import check_artifact_dir
from adr.utils.paths import find_project_root


def _load_split_for_eval(
    processed_csv: Path,
    splits_dir: Path,
    label_names: list[str],
    split_name: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load scores targets và mask cho split_name."""
    df = pd.read_csv(processed_csv)
    manifest = pd.read_csv(splits_dir / "split_manifest.csv")

    split_ids = set(manifest.loc[manifest["split"] == split_name, DRUG_ID_COL].tolist())
    df_split = df[df[DRUG_ID_COL].isin(split_ids)].copy()

    smiles = df_split[CANONICAL_SMILES_COL].tolist()
    y = df_split[label_names].values.astype(np.float32)

    # Load mask
    mask_full = np.load(processed_csv.parent / "label_mask.npy").astype(bool)
    df_full = pd.read_csv(processed_csv)
    mask_dict = {did: mask_full[i] for i, did in enumerate(df_full[DRUG_ID_COL].tolist())}
    mask = np.stack([mask_dict[did] for did in df_split[DRUG_ID_COL]], axis=0)

    return smiles, y, mask


def run_evaluation(
    run_dir: Path,
    split: str = "val",
    base_config: str | None = None,
) -> None:
    """Chạy evaluation cho run_dir trên split chỉ định."""
    run_dir = Path(run_dir)
    meta = check_artifact_dir(run_dir)

    # Kiểm tra test lock
    if split == "test":
        config_locked = meta.config_resolved.get("config_locked", False)
        if not config_locked:
            print(
                "[eval] LỖI: Không thể evaluate test khi config chưa locked.\n"
                "  Thêm 'config_locked: true' vào metadata sau khi khóa cấu hình.\n"
                "  Nguyên tắc: test chỉ dùng sau khi khóa, không quét seed theo kết quả test.",
                file=sys.stderr,
            )
            sys.exit(3)

    # Load config để tìm paths
    root = find_project_root()
    cfg = resolve_paths(load_config(base=base_config or str(root / "configs" / "base.yaml")))

    processed_csv = cfg["paths"]["data_processed"] / cfg["data"]["processed_filename"]
    splits_dir = cfg["paths"]["data_splits"]
    label_names = meta.label_names
    threshold = meta.threshold
    model_type = meta.model_type

    print(f"[eval] Đánh giá {model_type} ({run_dir.name}) trên split={split}")

    # Load model và tạo scores
    smiles, y, mask = _load_split_for_eval(processed_csv, splits_dir, label_names, split)
    n_samples = len(smiles)
    print(f"[eval] {n_samples} mẫu trong split={split}")

    if model_type == "frequency":
        from adr.models.frequency import FrequencyBaseline
        model = FrequencyBaseline.load(run_dir)
        scores = model.predict_scores(n_samples)

    elif model_type == "logreg":
        from adr.models.logreg import PerLabelLogReg
        from adr.features.fingerprints import MorganFeaturizer
        featurizer = MorganFeaturizer.load(run_dir / "fingerprint_config.json")
        model = PerLabelLogReg.load(run_dir)
        X, errors = featurizer.transform(smiles)
        scores = model.predict_scores(X)

    else:
        print(
            f"[eval] TODO: Evaluation cho model_type='{model_type}' chưa triển khai.\n"
            "  Xem adr/evaluate/run.py.",
            file=sys.stderr,
        )
        sys.exit(10)

    result = evaluate(scores, y, mask, label_names, threshold, split, meta.run_id)
    print_eval_summary(result)

    out_path = run_dir / f"eval_{split}.json"
    result.save(out_path)
    print(f"[eval] Kết quả lưu tại: {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá model từ run artifact",
        epilog=(
            "Ví dụ:\n"
            "  python -m adr.evaluate.run --run-dir artifacts/logreg_20261015_123000_42 --split val\n"
            "  Test chỉ khi config đã locked: thêm --split test\n"
        ),
    )
    parser.add_argument("--run-dir", required=True, help="Path đến run artifact directory")
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Split cần evaluate (mặc định: val)"
    )
    parser.add_argument("--base", default=None, help="Path đến base.yaml")
    args = parser.parse_args(argv)

    run_evaluation(Path(args.run_dir), split=args.split, base_config=args.base)


if __name__ == "__main__":
    main()

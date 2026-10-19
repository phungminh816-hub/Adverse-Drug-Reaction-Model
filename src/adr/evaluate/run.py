"""Kiểm tra checksum, nạp model và đánh giá một split dữ liệu."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adr.config import load_config, resolve_paths
from adr.data.load import load_split
from adr.evaluate.metrics import evaluate, print_eval_summary
from adr.utils.artifacts import check_artifact_dir
from adr.utils.paths import sha256_file


def run_evaluation(
    run_dir: Path,
    split: str = "val",
    base_config: str | None = None,
    lock_config: bool = False,
) -> None:
    """Chạy evaluation cho run_dir trên split chỉ định."""
    run_dir = Path(run_dir)
    meta = check_artifact_dir(run_dir)

    if lock_config:
        meta.config_resolved["config_locked"] = True
        meta.save(run_dir / "metadata.json")
    if split == "test":
        config_locked = meta.config_resolved.get("config_locked", False)
        if not config_locked:
            print(
                "[eval] LỖI: Không thể evaluate test khi config chưa locked.\n"
                "  Chạy lại với --lock-config sau khi chốt cấu hình bằng validation.\n"
                "  Nguyên tắc: test chỉ dùng sau khi khóa, không quét seed theo kết quả test.",
                file=sys.stderr,
            )
            sys.exit(3)

    cfg = resolve_paths(load_config(base=base_config) if base_config else (meta.config_resolved or load_config()))

    processed_csv = cfg["paths"]["data_processed"] / cfg["data"]["processed_filename"]
    splits_dir = cfg["paths"]["data_splits"]
    label_names = meta.label_names
    threshold = meta.threshold
    model_type = meta.model_type

    print(f"[eval] Đánh giá {model_type} ({run_dir.name}) trên split={split}")

    if sha256_file(processed_csv) != meta.dataset_checksum:
        raise ValueError("processed.csv đã thay đổi sau khi train.")
    manifest_filename = cfg["data"].get("split_manifest", "split_manifest.csv")
    if sha256_file(splits_dir / manifest_filename) != meta.split_manifest_checksum:
        raise ValueError("Split manifest đã thay đổi sau khi train.")
    if meta.label_mask_checksum and sha256_file(processed_csv.parent / "label_mask.npy") != meta.label_mask_checksum:
        raise ValueError("Label mask đã thay đổi sau khi train.")
    names_path = cfg["paths"]["data_processed"] / cfg["data"]["label_names_file"]
    meta.check_label_names(json.loads(names_path.read_text(encoding="utf-8")))

    _, smiles, y, mask = load_split(processed_csv, splits_dir, label_names, split, manifest_filename)
    n_samples = len(smiles)
    print(f"[eval] {n_samples} mẫu trong split={split}")

    from adr.predict.predictor import Predictor
    scores = Predictor(run_dir).score_smiles(smiles)

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
    parser.add_argument("--lock-config", action="store_true", help="Chốt cấu hình trước khi đánh giá test")
    args = parser.parse_args(argv)

    run_evaluation(Path(args.run_dir), split=args.split, base_config=args.base, lock_config=args.lock_config)


if __name__ == "__main__":
    main()

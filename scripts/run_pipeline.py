"""Tải dữ liệu, chia split, train và evaluate validation bằng một lệnh."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    """Chạy download, prepare, split, train và đánh giá validation theo cấu hình."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["frequency", "logreg", "mlp", "cnn"], default=["logreg"])
    parser.add_argument("--base", default=None, help="Cấu hình nền chứa đường dẫn dữ liệu và kết quả")
    parser.add_argument("--strategy", choices=["random", "scaffold"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    from adr.config import load_config, resolve_paths
    from adr.data.download import download_sider, decompress_gz
    from adr.data.prepare import prepare
    from adr.data.split import split_dataset
    from adr.evaluate.run import run_evaluation
    from adr.train.baseline import train_frequency, train_logreg

    cfg = resolve_paths(load_config(base=args.base))
    dc, paths, sc = cfg["data"], cfg["paths"], cfg["split"]
    strategy = args.strategy or sc["strategy"]
    force = dc.get("force_download", False)
    gz = download_sider(dc["source_url"], paths["data_raw"], dc["raw_filename"], timeout=dc.get("download_timeout_s", 60), force=force)
    raw = decompress_gz(gz, paths["data_raw"] / dc["raw_filename"].removesuffix(".gz"), force=force)
    prepare(raw, paths["data_processed"], label_names_path=paths["data_processed"] / dc["label_names_file"],
            num_labels=dc["num_labels"], processed_filename=dc["processed_filename"])
    processed = paths["data_processed"] / dc["processed_filename"]
    split_dataset(processed, paths["data_splits"], strategy, sc["train_ratio"], sc["val_ratio"], sc["seed"],
                  manifest_filename=dc["split_manifest"])
    labels = json.loads((paths["data_processed"] / dc["label_names_file"]).read_text(encoding="utf-8"))
    runs = {}
    for name in args.models:
        model_cfg = resolve_paths(load_config(ROOT / "configs" / f"{name}.yaml", base=args.base))
        model_cfg["split"]["strategy"] = strategy
        if name in {"frequency", "logreg"}:
            trainer = train_frequency if name == "frequency" else train_logreg
            run_dir = trainer(model_cfg, processed, paths["data_splits"], labels, paths["artifacts"])
        else:
            from adr.train.neural import train_neural
            if args.epochs is not None:
                model_cfg["training"]["max_epochs"] = args.epochs
            run_dir = train_neural(model_cfg, device=args.device)
        run_evaluation(run_dir, split="val")
        metrics = json.loads((run_dir / "eval_val.json").read_text(encoding="utf-8"))
        runs[name] = {"run_dir": str(run_dir), "val_macro_ap": metrics["macro_ap"], "val_macro_roc_auc": metrics["macro_roc_auc"]}
    paths["results"].mkdir(parents=True, exist_ok=True)
    summary = paths["results"] / "run_summary.json"
    summary.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"[pipeline] Hoàn thành: {summary}")


if __name__ == "__main__":
    main()

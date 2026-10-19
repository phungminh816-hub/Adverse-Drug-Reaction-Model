"""Kiểm tra đường dẫn cấu hình, dữ liệu thiếu và pipeline bốn model."""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from adr.config import PROJECT_ROOT, load_config, resolve_paths, save_config
from adr.data.download import decompress_gz
from adr.data.split import scaffold_split
from adr.evaluate.metrics import evaluate
from adr.features.tokenizer import CharTokenizer
from adr.models.logreg import PerLabelLogReg
from adr.predict.predictor import Predictor
from adr.schemas import ArtifactMetadata
from adr.utils.paths import sha256_file


def test_config_paths_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("configs/cnn.yaml", base="configs/base.yaml")
    cfg["paths"]["data_raw"] = Path("data/raw")
    resolved = resolve_paths(cfg)
    assert resolved["paths"]["data_raw"] == PROJECT_ROOT / "data/raw"
    assert cfg["paths"]["data_raw"] == Path("data/raw")
    saved = save_config(resolved, tmp_path / "run")
    assert yaml.safe_load(saved.read_text(encoding="utf-8"))["paths"]["data_raw"] == str(PROJECT_ROOT / "data/raw")


@pytest.mark.parametrize("source", ["- item\n", "", "null\n"])
def test_config_rejects_non_mapping(tmp_path, source):
    path = tmp_path / "bad.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(base=path)


def test_logreg_rejects_fully_missing_label():
    mask = np.ones((4, 2), dtype=bool)
    mask[:, 1] = False
    with pytest.raises(ValueError, match="unobserved"):
        PerLabelLogReg(n_labels=2).fit(np.ones((4, 3)), np.zeros((4, 2)), mask, ["observed", "unobserved"])


def test_tokenizer_empty_batch_and_refit():
    tokenizer = CharTokenizer().fit(["CCO"])
    ids, mask, stats = tokenizer.transform([])
    assert ids.shape == mask.shape == (0, 3)
    assert stats["n_unk"] == stats["n_truncated"] == 0
    tokenizer.fit(["NN"])
    assert "C" not in tokenizer.vocab and "O" not in tokenizer.vocab
    assert tokenizer.transform(["C"])[2]["n_unk"] == 1


@pytest.mark.parametrize("length", [0, -1])
def test_tokenizer_rejects_invalid_length(length):
    tokenizer = CharTokenizer().fit(["CCO"])
    with pytest.raises(ValueError, match="max_length"):
        tokenizer.encode("CCO", length)
    with pytest.raises(ValueError, match="max_length"):
        tokenizer.transform(["CCO"], length)


@pytest.mark.parametrize("smiles", [[], [""]])
def test_tokenizer_rejects_empty_training(smiles):
    with pytest.raises(ValueError, match="SMILES"):
        CharTokenizer().fit(smiles)


@pytest.mark.parametrize("problem", ["shape", "score", "target", "mask", "threshold"])
def test_metrics_rejects_invalid_input(problem):
    scores, targets, mask = np.ones((2, 1)) * 0.5, np.ones((2, 1)), np.ones((2, 1))
    threshold = 0.5
    if problem == "shape":
        scores = scores[:1]
    elif problem == "score":
        scores[0, 0] = np.nan
    elif problem == "target":
        targets[0, 0] = 2
    elif problem == "mask":
        mask[0, 0] = 2
    else:
        threshold = 1.1
    with pytest.raises(ValueError):
        evaluate(scores, targets, mask, ["label"], threshold=threshold)


def test_corrupt_gzip_preserves_existing_csv(tmp_path):
    source, target = tmp_path / "bad.gz", tmp_path / "data.csv"
    source.write_bytes(b"invalid gzip")
    target.write_text("existing data", encoding="utf-8")
    with pytest.raises((OSError, EOFError)):
        decompress_gz(source, target, force=True)
    assert target.read_text(encoding="utf-8") == "existing data"
    assert not target.with_name("data.csv.part").exists()


def test_scaffold_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="drug_id"):
        scaffold_split(["same", "same", "other"], ["c1ccccc1", "c1ccncc1", "C1CCCCC1"])


def test_four_model_pipeline_with_custom_paths(tmp_path, monkeypatch):
    from adr.data.prepare import main as prepare_main
    from adr.data.split import main as split_main

    workspace = tmp_path / "project with spaces"
    workspace.mkdir()
    cfg = load_config()
    cfg["paths"] = {key: str(workspace / key) for key in cfg["paths"]}
    cfg["data"].update(raw_filename="custom.csv.gz", processed_filename="nested/molecules.csv",
                       label_names_file="names/labels.json", split_manifest="manifests/assignments.csv")
    cfg["split"].update(strategy="scaffold", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    cfg["training"].update(num_threads=1, batch_size=4)
    base = workspace / "custom.yaml"
    base.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    raw_dir = Path(cfg["paths"]["data_raw"])
    raw_dir.mkdir()
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "Cc1ccncc1", "C1CCCCC1", "CC1CCCCC1",
              "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "CCO", "CCCC", "CCN", "COC",
              "C1CCCC1", "CC1CCCC1", "c1ccoc1", "Cc1ccoc1", "c1ccsc1", "Cc1ccsc1", "C1CC1", "CC1CC1"]
    labels = [f"label_{i}" for i in range(27)]
    data = pd.DataFrame(np.random.default_rng(12).integers(0, 2, (len(smiles), 27)), columns=labels)
    data.insert(0, "smiles", smiles)
    with gzip.open(raw_dir / "custom.csv.gz", "wt", encoding="utf-8") as handle:
        data.to_csv(handle, index=False)
    decompress_gz(raw_dir / "custom.csv.gz", raw_dir / "custom.csv")
    monkeypatch.chdir(tmp_path)
    prepare_main(["--config", str(base)])
    split_main(["--config", str(base)])
    command = [sys.executable, str(PROJECT_ROOT / "scripts/run_pipeline.py"), "--base", str(base),
               "--models", "frequency", "logreg", "mlp", "cnn", "--epochs", "1", "--device", "cpu"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
                            timeout=180, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((Path(cfg["paths"]["results"]) / "run_summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"frequency", "logreg", "mlp", "cnn"}
    processed = Path(cfg["paths"]["data_processed"]) / cfg["data"]["processed_filename"]
    manifest = Path(cfg["paths"]["data_splits"]) / cfg["data"]["split_manifest"]
    for model_type, item in summary.items():
        run = Path(item["run_dir"])
        meta = ArtifactMetadata.load(run / "metadata.json")
        assert meta.model_type == model_type
        assert meta.config_resolved["split"]["strategy"] == "scaffold"
        assert meta.dataset_checksum == sha256_file(processed)
        assert meta.split_manifest_checksum == sha256_file(manifest)
        assert (run / "eval_val.json").exists() and not (run / "eval_test.json").exists()
        scores = Predictor(run).predict("CCO").scores
        assert len(scores) == 27 and np.isfinite(scores).all()

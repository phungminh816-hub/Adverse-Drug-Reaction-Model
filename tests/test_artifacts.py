"""
tests/test_artifacts.py – Kiểm tra artifact save/load và checker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adr.schemas import ArtifactMetadata
from adr.utils.artifacts import check_artifact_dir


def _make_metadata(run_dir: Path, model_type: str = "frequency") -> ArtifactMetadata:
    meta = ArtifactMetadata(
        model_type=model_type,
        run_id="test_run_001",
        seed=42,
        label_names=[f"l{i}" for i in range(27)],
        dataset_checksum="abc123",
        split_manifest_checksum="def456",
        feature_config={"type": "test"},
        library_versions={"numpy": "2.5.3"},
        threshold=0.5,
        eval_label_policy="all",
        created_at_utc="2026-10-15T00:00:00Z",
    )
    meta.save(run_dir / "metadata.json")
    return meta


class TestArtifactMetadata:
    def test_save_load_round_trip(self, tmp_path):
        meta = _make_metadata(tmp_path)
        meta2 = ArtifactMetadata.load(tmp_path / "metadata.json")
        assert meta.model_type == meta2.model_type
        assert meta.label_names == meta2.label_names
        assert meta.threshold == meta2.threshold

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ArtifactMetadata.load(tmp_path / "nonexistent.json")

    def test_check_label_names_mismatch(self, tmp_path):
        meta = _make_metadata(tmp_path)
        with pytest.raises(ValueError, match="Label names"):
            meta.check_label_names(["wrong"] * 27)

    def test_check_label_names_ok(self, tmp_path):
        meta = _make_metadata(tmp_path)
        meta.check_label_names(meta.label_names)  # không raise


class TestCheckArtifactDir:
    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            check_artifact_dir(tmp_path / "nonexistent_run")

    def test_missing_metadata_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            check_artifact_dir(run_dir)

    def test_valid_frequency_artifact(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        meta = _make_metadata(run_dir, "frequency")
        # Tạo model.json giả
        (run_dir / "model.json").write_text(json.dumps({"n_labels": 27}))
        meta2 = check_artifact_dir(run_dir)
        assert meta2.model_type == "frequency"

    def test_model_type_mismatch_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _make_metadata(run_dir, "frequency")
        (run_dir / "model.json").write_text("{}")
        with pytest.raises(ValueError, match="model_type"):
            check_artifact_dir(run_dir, model_type="logreg")

    def test_missing_required_file_raises(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _make_metadata(run_dir, "logreg")
        # Không tạo model.pkl
        with pytest.raises(FileNotFoundError, match="model.pkl"):
            check_artifact_dir(run_dir)

"""
tests/test_data_contracts.py – Kiểm tra hợp đồng dữ liệu.

Không cần dataset thật. Dùng fixture nhỏ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adr.data.validate import validate_raw
from adr.data.split import random_split, scaffold_split
from adr.schemas import NUM_LABELS


N = 10
VALID_SMILES = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "CCCC",
    "OCC(O)CO", "c1ccncc1", "CC(=O)c1ccccc1", "OC(=O)c1ccccc1", "C(C(=O)O)N",
]
LABEL_NAMES = [f"label_{i}" for i in range(NUM_LABELS)]


def _make_csv(tmp_path: Path, smiles=None, labels=None) -> Path:
    """Tạo CSV fixture trong tmp_path."""
    smiles = smiles or VALID_SMILES
    rng = np.random.default_rng(42)
    data = {"smiles": smiles}
    for name in LABEL_NAMES:
        col = rng.integers(0, 2, size=len(smiles)).astype(float)

        missing = rng.random(len(smiles)) < 0.1
        col[missing] = np.nan
        data[name] = col
    if labels is not None:
        for i, (name, vals) in enumerate(zip(LABEL_NAMES, labels)):
            data[name] = vals
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestValidate:
    def test_valid_csv(self, tmp_path):
        csv = _make_csv(tmp_path)
        report = validate_raw(csv)
        assert report.total_rows == N
        assert report.n_labels == NUM_LABELS

        assert len(report.label_value_errors) == 0

    def test_missing_smiles_col(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"not_smiles": ["CCO"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="smiles"):
            validate_raw(csv)

    def test_invalid_label_value(self, tmp_path):
        """Nhãn ngoài {0, 1, NaN} phải bị phát hiện."""
        bad_labels = [[2.0] * N] + [[float(col_idx % 2)] * N for col_idx in range(NUM_LABELS - 1)]
        csv = _make_csv(tmp_path, labels=bad_labels)
        report = validate_raw(csv)
        assert len(report.label_value_errors) > 0

    def test_wrong_label_count(self, tmp_path):
        """Số nhãn sai phải tạo warning."""
        smiles = VALID_SMILES[:3]
        df = pd.DataFrame({"smiles": smiles, "label_1": [0, 1, 0], "label_2": [1, 0, 1]})
        csv = tmp_path / "few_labels.csv"
        df.to_csv(csv, index=False)
        report = validate_raw(csv)
        assert report.n_labels == 2
        assert any("27" in w for w in report.warnings)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_raw(tmp_path / "nonexistent.csv")


class TestSplit:
    def test_random_split_no_overlap(self):
        ids = [f"drug_{i}" for i in range(100)]
        assignment = random_split(ids, train_ratio=0.8, val_ratio=0.1, seed=42)
        train = {k for k, v in assignment.items() if v == "train"}
        val = {k for k, v in assignment.items() if v == "val"}
        test = {k for k, v in assignment.items() if v == "test"}
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)
        assert len(assignment) == 100

    def test_random_split_reproducible(self):
        ids = [f"drug_{i}" for i in range(50)]
        a1 = random_split(ids, seed=42)
        a2 = random_split(ids, seed=42)
        assert a1 == a2

    def test_random_split_different_seeds(self):
        ids = [f"drug_{i}" for i in range(50)]
        a1 = random_split(ids, seed=42)
        a2 = random_split(ids, seed=99)
        assert a1 != a2

    def test_scaffold_split_requires_smiles(self):
        with pytest.raises(ValueError, match="canonical_smiles"):
            scaffold_split(["drug_0"])

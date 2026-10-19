"""Regression tests cho dữ liệu, checkpoint và inference hoàn chỉnh."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from adr.config import load_config
from adr.data.load import load_split
from adr.data.prepare import prepare
from adr.data.split import scaffold_split, split_dataset
from adr.features.fingerprints import MorganFeaturizer
from adr.predict.predictor import Predictor
from adr.schemas import ArtifactMetadata
from adr.train.neural import train_neural


@pytest.fixture
def dataset(tmp_path):
    smiles = ["C" * i for i in range(1, 21)]
    labels = [f"label_{i}" for i in range(27)]
    rng = np.random.default_rng(12)
    values = rng.integers(0, 2, (20, 27)).astype(float)
    values[0, 0] = np.nan
    df = pd.DataFrame(values, columns=labels)
    df.insert(0, "smiles", smiles)
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, index=False)
    processed, splits = tmp_path / "processed", tmp_path / "splits"
    prepare(raw, processed)
    split_dataset(processed / "processed.csv", splits, train_ratio=0.6, val_ratio=0.2)
    return processed, splits, labels


def test_fingerprint_valid_and_invalid():
    fp = MorganFeaturizer(n_bits=64)
    values, errors = fp.transform(["CCO", "OCC", "", "broken"])
    np.testing.assert_array_equal(values[0], values[1])
    assert values[0].sum() > 0
    assert len(errors) == 2 and not values[2:].any()
    assert fp.transform([])[0].shape == (0, 64)


def test_duplicate_nan_labels_merge(tmp_path):
    raw = tmp_path / "duplicates.csv"
    df = pd.DataFrame({"smiles": ["CCO", "OCC"], **{f"l{i}": [np.nan, np.nan] if i == 0 else [1, 1] for i in range(27)}})
    df.to_csv(raw, index=False)
    result = prepare(raw, tmp_path / "out")
    assert len(result) == 1
    mask = np.load(tmp_path / "out" / "label_mask.npy")
    assert not mask[0, 0] and mask[0, 1:].all()


def test_prepare_rejects_invalid_labels(tmp_path):
    raw = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO"], **{f"l{i}": [2 if i == 0 else 0] for i in range(27)}}).to_csv(raw, index=False)
    with pytest.raises(ValueError, match="Nhãn"):
        prepare(raw, tmp_path / "out")


def test_scaffold_groups_do_not_leak():
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccncc1", "Cc1ccncc1", "C1CCCCC1", "CC1CCCCC1", "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1"]
    ids = [str(i) for i in range(len(smiles))]
    assignment = scaffold_split(ids, smiles, train_ratio=0.5, val_ratio=0.25)
    seen = {}
    for did, smi in zip(ids, smiles):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi))
        seen.setdefault(scaffold, set()).add(assignment[did])
    assert all(len(splits) == 1 for splits in seen.values())
    assert set(assignment.values()) == {"train", "val", "test"}


def test_loader_rejects_incomplete_manifest(dataset):
    processed, splits, labels = dataset
    path = splits / "split_manifest.csv"
    df = pd.read_csv(path)
    df.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="Manifest"):
        load_split(processed / "processed.csv", splits, labels)


@pytest.mark.parametrize("problem", ["mask", "label"])
def test_loader_rejects_invalid_observed_data(dataset, problem):
    processed, splits, labels = dataset
    if problem == "mask":
        mask = np.load(processed / "label_mask.npy")
        mask[1, 1] = 2
        np.save(processed / "label_mask.npy", mask)
    else:
        data = pd.read_csv(processed / "processed.csv")
        data.loc[1, labels[1]] = np.nan
        data.to_csv(processed / "processed.csv", index=False)
    with pytest.raises(ValueError):
        load_split(processed / "processed.csv", splits, labels)


@pytest.mark.parametrize("model_type", ["mlp", "cnn"])
def test_train_resume_predict_evaluate(dataset, tmp_path, model_type):
    from adr.evaluate.run import run_evaluation
    processed, splits, labels = dataset
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{model_type}.yaml")
    cfg["paths"].update(data_processed=str(processed), data_splits=str(splits), artifacts=str(tmp_path / "artifacts"))
    cfg["training"].update(max_epochs=2, batch_size=4, num_threads=1)
    cfg["model"].update(hidden_dim=8, embedding_dim=4, channels=8, kernel_size=3)
    cfg["features"]["fingerprint"]["n_bits"] = 64
    run = train_neural(cfg, device="cpu")
    meta = ArtifactMetadata.load(run / "metadata.json")
    assert meta.label_names == labels
    predictor = Predictor(run)
    result = predictor.predict("CCO")
    assert len(result.scores) == 27 and np.isfinite(result.scores).all()
    assert predictor.predict("").canonical_smiles is None
    before = np.asarray(result.scores)
    np.testing.assert_allclose(before, Predictor(run).predict("CCO").scores)
    run_evaluation(run, split="val")
    assert json.loads((run / "eval_val.json").read_text())["macro_ap"] is not None
    with pytest.raises(SystemExit) as exc:
        run_evaluation(run, split="test")
    assert exc.value.code == 3 and not (run / "eval_test.json").exists()
    cfg["training"]["max_epochs"] = 3
    train_neural(cfg, resume=run, device="cpu")
    state_resumed = torch.load(run / "training_state.pt", weights_only=True)
    assert state_resumed["epoch"] == 3
    fresh = train_neural(copy.deepcopy(cfg), device="cpu")
    state_fresh = torch.load(fresh / "training_state.pt", weights_only=True)
    for key in state_fresh["model"]:
        torch.testing.assert_close(state_fresh["model"][key], state_resumed["model"][key], rtol=0, atol=0)
    mask_path = processed / "label_mask.npy"
    mask = np.load(mask_path)
    mask[0, 1] = 1 - mask[0, 1]
    np.save(mask_path, mask)
    with pytest.raises(ValueError, match="mask"):
        run_evaluation(run)
    with pytest.raises(ValueError, match="mask"):
        train_neural(cfg, resume=run, device="cpu")


def test_loss_ignores_nan_at_missing_positions():
    from adr.train.losses import masked_bce_loss
    logits = torch.zeros(1, 2, requires_grad=True)
    loss = masked_bce_loss(logits, torch.tensor([[1.0, float("nan")]]), torch.tensor([[True, False]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0, 1] == 0

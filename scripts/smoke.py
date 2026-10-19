"""Kiểm tra config, model, tokenizer, loss và metric bằng dữ liệu tổng hợp."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SMOKE_DIR: Path

SEED = 42
N_LABELS = 27
N_MOLECULES = 20


FIXTURE_SMILES = [
    "CCO",          "CC(=O)O",      "c1ccccc1",     "CC(C)O",       "CCCC",
    "CC(=O)Nc1ccc(O)cc1",  "OC[C@H](O)[C@@H](O)[C@H](O)CO",  "CN1CCC[C@H]1c2cccnc2",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  "C1CCCCC1",
    "c1ccc2ccccc2c1",  "CCN(CC)CC",  "CCCCCCCC",  "OCC(O)CO",  "CC(C)(C)c1ccc(O)cc1",
    "C(C(=O)O)N",  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  "OC(=O)c1ccccc1",
    "c1ccncc1",    "CC(=O)c1ccccc1",
]

assert len(FIXTURE_SMILES) == N_MOLECULES

rng = np.random.default_rng(SEED)


def _make_fixture() -> tuple[pd.DataFrame, list[str]]:
    """Tạo DataFrame fixture với cột smiles và 27 cột nhãn."""

    labels = rng.integers(0, 2, size=(N_MOLECULES, N_LABELS)).astype(float)

    missing_mask = rng.random((N_MOLECULES, N_LABELS)) < 0.15
    labels[missing_mask] = np.nan

    label_names = [f"smoke_label_{i+1:02d}" for i in range(N_LABELS)]
    df = pd.DataFrame(labels, columns=label_names)
    df.insert(0, "smiles", FIXTURE_SMILES)
    return df, label_names


PASSED = []
FAILED = []


def _run(name: str, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        PASSED.append(name)
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()
        FAILED.append((name, str(e)))


def test_config():
    from adr.config import load_config, validate_base_config
    cfg = load_config(ROOT / "configs" / "logreg.yaml")
    errors = validate_base_config(cfg)
    assert not errors, f"Config lỗi: {errors}"


def test_schemas():
    from adr.schemas import PredictionResult, NUM_LABELS
    assert NUM_LABELS == 27

    try:
        PredictionResult(
            drug_id="test", input_smiles="CCO", canonical_smiles="CCO",
            model_id="x", label_names=["a"] * 5, scores=[0.5] * 3,
            threshold=0.5, predicted_labels=[],
        )
        assert False, "Phải raise ValueError"
    except ValueError:
        pass


def test_tokenizer():
    from adr.features.tokenizer import CharTokenizer
    tok = CharTokenizer()
    tok.fit(FIXTURE_SMILES[:15])
    assert tok.fitted

    ids, masks, stats = tok.transform(FIXTURE_SMILES[15:], max_length=tok.max_length)
    assert ids.shape[1] == tok.max_length
    assert masks.shape == ids.shape

    tok_path = SMOKE_DIR / "tokenizer_smoke.json"
    tok.save(tok_path)
    tok2 = CharTokenizer.load(tok_path)
    ids2, masks2, _ = tok2.transform(FIXTURE_SMILES[15:], max_length=tok.max_length)
    assert np.array_equal(ids, ids2), "Tokenizer save/load encoding thay đổi"

    ids_unk, _, stats_unk = tok2.transform(["$$$"], max_length=tok.max_length)
    assert stats_unk["n_unk"] > 0, "Ký tự lạ phải ra UNK"


def test_frequency_baseline():
    df, label_names = _make_fixture()
    y = df[label_names].values.astype(np.float32)

    mask = ~np.isnan(df[label_names].values)

    y = np.nan_to_num(y, nan=0.0)

    from adr.models.frequency import FrequencyBaseline
    model = FrequencyBaseline(n_labels=N_LABELS)
    model.fit(y[:15], mask[:15], label_names)

    scores = model.predict_scores(5)
    assert scores.shape == (5, N_LABELS)
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))

    model.save(SMOKE_DIR / "freq")
    model2 = FrequencyBaseline.load(SMOKE_DIR / "freq")
    scores2 = model2.predict_scores(5)
    np.testing.assert_allclose(scores, scores2, atol=1e-6, err_msg="Frequency scores thay đổi sau save/load")


def test_logreg_no_rdkit():
    """Kiểm tra LogReg với fingerprint giả (không cần RDKit)."""
    df, label_names = _make_fixture()
    y = df[label_names].values.astype(np.float32)
    mask = ~np.isnan(df[label_names].values)
    y = np.nan_to_num(y, nan=0.0)

    X = rng.integers(0, 2, size=(N_MOLECULES, 2048)).astype(np.float32)

    from adr.models.logreg import PerLabelLogReg
    model = PerLabelLogReg(n_labels=N_LABELS, solver="lbfgs", max_iter=100)
    model.fit(X[:15], y[:15], mask[:15], label_names)

    scores = model.predict_scores(X[15:])
    assert scores.shape == (len(X[15:]), N_LABELS), f"Shape sai: {scores.shape}"
    assert np.all(np.isfinite(scores)), "Scores không hữu hạn"
    assert np.all((scores >= 0) & (scores <= 1)), "Scores ngoài [0,1]"

    model.save(SMOKE_DIR / "logreg")
    model2 = PerLabelLogReg.load(SMOKE_DIR / "logreg")
    scores2 = model2.predict_scores(X[15:])
    np.testing.assert_allclose(scores, scores2, atol=1e-6, err_msg="LogReg scores thay đổi sau save/load")


def test_metrics():
    from adr.evaluate.metrics import evaluate

    N = 10
    scores = rng.random((N, N_LABELS)).astype(np.float32)
    targets = rng.integers(0, 2, size=(N, N_LABELS)).astype(np.float32)
    mask = rng.random((N, N_LABELS)) > 0.2

    label_names = [f"label_{i}" for i in range(N_LABELS)]
    result = evaluate(scores, targets, mask, label_names, threshold=0.5, split="val", model_id="smoke")

    assert result.n_valid_ap >= 0
    if result.macro_ap is not None:
        assert 0 <= result.macro_ap <= 1

    targets_alt = targets.copy()
    targets_alt[~mask] = 999.0
    result_alt = evaluate(scores, targets_alt, mask, label_names, threshold=0.5, split="val", model_id="smoke")
    if result.macro_ap is not None and result_alt.macro_ap is not None:
        assert abs(result.macro_ap - result_alt.macro_ap) < 1e-9, \
            "Metrics bị ảnh hưởng bởi giá trị padding (mask=False)"


def test_mlp_forward():
    try:
        import torch
    except ImportError:
        print("    [skip] torch chưa cài")
        return

    from adr.models.mlp import MLP
    model = MLP(input_dim=2048, hidden_dim=64, n_labels=N_LABELS, dropout=0.0)
    model.eval()
    x = torch.randn(4, 2048)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, N_LABELS), f"MLP shape sai: {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "MLP logits không hữu hạn"


def test_cnn_forward_and_pad_invariance():
    try:
        import torch
    except ImportError:
        print("    [skip] torch chưa cài")
        return

    from adr.features.tokenizer import CharTokenizer
    from adr.models.cnn import SmilesCharCNN

    tok = CharTokenizer()
    tok.fit(FIXTURE_SMILES)

    smi = "CCO"
    L = tok.max_length

    ids1, mask1, _ = tok.transform([smi], max_length=L)

    ids2, mask2, _ = tok.transform([smi], max_length=L + 5)

    model = SmilesCharCNN(
        vocab_size=tok.vocab_size,
        embedding_dim=32, channels=64, kernel_size=3,
        n_labels=N_LABELS, dropout=0.0,
    )
    model.eval()

    t_ids1 = torch.tensor(ids1, dtype=torch.long)
    t_mask1 = torch.tensor(mask1, dtype=torch.bool)
    t_ids2 = torch.tensor(ids2, dtype=torch.long)
    t_mask2 = torch.tensor(mask2, dtype=torch.bool)

    with torch.no_grad():
        out1 = model(t_ids1, t_mask1)
        out2 = model(t_ids2, t_mask2)

    assert out1.shape == (1, N_LABELS), f"CNN output shape sai: {out1.shape}"
    assert torch.allclose(out1, out2, atol=1e-5), \
        f"CNN output thay đổi khi padding: max_diff={torch.max(torch.abs(out1-out2))}"

    sigmoid_scores = torch.sigmoid(out1).numpy()
    assert np.all(sigmoid_scores >= 0) and np.all(sigmoid_scores <= 1)


def test_masked_loss():
    try:
        import torch
    except ImportError:
        print("    [skip] torch chưa cài")
        return

    from adr.train.losses import masked_bce_loss

    B, L = 4, N_LABELS
    logits = torch.randn(B, L)
    targets = torch.randint(0, 2, (B, L)).float()
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, :5] = False

    loss1 = masked_bce_loss(logits, targets, mask)
    assert torch.isfinite(loss1)

    targets_alt = targets.clone()
    targets_alt[~mask] = 999.0
    loss2 = masked_bce_loss(logits, targets_alt, mask)
    assert torch.allclose(loss1, loss2, atol=1e-6), "Masked loss bị ảnh hưởng bởi giá trị padding"

    mask_empty = torch.zeros(B, L, dtype=torch.bool)
    try:
        masked_bce_loss(logits, targets, mask_empty, empty_batch_policy="error")
        assert False, "Phải raise RuntimeError"
    except RuntimeError:
        pass


def test_split_no_leakage():
    from adr.data.split import random_split
    drug_ids = [f"drug_{i}" for i in range(50)]
    assignment = random_split(drug_ids, train_ratio=0.8, val_ratio=0.1, seed=SEED)

    train_set = {k for k, v in assignment.items() if v == "train"}
    val_set = {k for k, v in assignment.items() if v == "val"}
    test_set = {k for k, v in assignment.items() if v == "test"}

    assert len(train_set & val_set) == 0, "Train/Val rò rỉ"
    assert len(train_set & test_set) == 0, "Train/Test rò rỉ"
    assert len(val_set & test_set) == 0, "Val/Test rò rỉ"
    assert len(assignment) == 50

    assignment2 = random_split(drug_ids, train_ratio=0.8, val_ratio=0.1, seed=SEED)
    assert assignment == assignment2, "Split không ổn định với cùng seed"


def test_scaffold_split_raises():
    from adr.data.split import scaffold_split
    try:
        scaffold_split(["a", "b"])
        assert False, "scaffold_split cần canonical_smiles"
    except ValueError:
        pass


def main() -> int:
    """Chạy các kiểm tra offline trong thư mục tạm và trả exit code tổng hợp."""
    global SMOKE_DIR
    PASSED.clear()
    FAILED.clear()
    print("\n" + "="*60)
    print("  ADR Project – Smoke Test (Synthetic Fixture)")
    print("  CẢNH BÁO: Dữ liệu fixture là synthetic, không dùng cho nghiên cứu")
    print("="*60)

    tests = [
        ("Config load & validate", test_config),
        ("Schemas & contracts", test_schemas),
        ("Tokenizer fit/transform/save/load", test_tokenizer),
        ("Frequency baseline", test_frequency_baseline),
        ("LogReg per-label (fingerprint giả)", test_logreg_no_rdkit),
        ("Metrics & null handling", test_metrics),
        ("MLP forward shape", test_mlp_forward),
        ("CNN forward & pad invariance", test_cnn_forward_and_pad_invariance),
        ("Masked BCE loss", test_masked_loss),
        ("Split no leakage", test_split_no_leakage),
        ("Scaffold split requires SMILES", test_scaffold_split_raises),
    ]

    with tempfile.TemporaryDirectory(prefix="adr-smoke-") as temporary_dir:
        SMOKE_DIR = Path(temporary_dir)
        for name, fn in tests:
            _run(name, fn)

    print(f"\n{'='*60}")
    print(f"  Passed: {len(PASSED)}/{len(tests)}")
    if FAILED:
        print(f"  Failed: {len(FAILED)}")
        for name, err in FAILED:
            print(f"    - {name}: {err}")
    print("="*60)

    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())

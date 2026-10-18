"""
tests/test_models.py – Kiểm tra forward pass và hợp đồng model.
"""

from __future__ import annotations

import numpy as np
import pytest

from adr.schemas import NUM_LABELS


BATCH = 4
N_LABELS = NUM_LABELS
FINGERPRINT_DIM = 2048
rng = np.random.default_rng(42)


class TestFrequencyBaseline:
    def _make_data(self, N=20):
        y = rng.integers(0, 2, (N, N_LABELS)).astype(np.float32)
        mask = rng.random((N, N_LABELS)) > 0.1
        y[~mask] = 0.0
        return y, mask

    def test_fit_predict_shape(self):
        from adr.models.frequency import FrequencyBaseline
        y, mask = self._make_data()
        labels = [f"l{i}" for i in range(N_LABELS)]
        model = FrequencyBaseline()
        model.fit(y, mask, labels)
        scores = model.predict_scores(BATCH)
        assert scores.shape == (BATCH, N_LABELS)

    def test_scores_in_range(self):
        from adr.models.frequency import FrequencyBaseline
        y, mask = self._make_data()
        model = FrequencyBaseline()
        model.fit(y, mask)
        scores = model.predict_scores(5)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_no_observation_raises(self):
        from adr.models.frequency import FrequencyBaseline
        y = rng.integers(0, 2, (5, N_LABELS)).astype(np.float32)
        mask = np.ones((5, N_LABELS), dtype=bool)
        mask[:, 0] = False  # nhãn 0 không có observation
        model = FrequencyBaseline()
        with pytest.raises(ValueError, match="không có sample"):
            model.fit(y, mask)

    def test_save_load_consistency(self, tmp_path):
        from adr.models.frequency import FrequencyBaseline
        y, mask = self._make_data()
        model = FrequencyBaseline()
        model.fit(y, mask)
        model.save(tmp_path)
        model2 = FrequencyBaseline.load(tmp_path)
        s1 = model.predict_scores(3)
        s2 = model2.predict_scores(3)
        np.testing.assert_allclose(s1, s2, atol=1e-6)


class TestLogReg:
    def _make_data(self, N=30):
        X = rng.integers(0, 2, (N, FINGERPRINT_DIM)).astype(np.float32)
        y = rng.integers(0, 2, (N, N_LABELS)).astype(np.float32)
        mask = rng.random((N, N_LABELS)) > 0.1
        y[~mask] = 0.0
        return X, y, mask

    def test_fit_predict_shape(self):
        from adr.models.logreg import PerLabelLogReg
        X, y, mask = self._make_data()
        model = PerLabelLogReg()
        model.fit(X[:20], y[:20], mask[:20])
        scores = model.predict_scores(X[20:])
        assert scores.shape == (len(X[20:]), N_LABELS)

    def test_scores_in_range(self):
        from adr.models.logreg import PerLabelLogReg
        X, y, mask = self._make_data()
        model = PerLabelLogReg()
        model.fit(X[:20], y[:20], mask[:20])
        scores = model.predict_scores(X[20:])
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_single_class_label_still_returns_27_cols(self):
        """Nhãn chỉ có 1 lớp vẫn trả đúng 27 cột."""
        from adr.models.logreg import PerLabelLogReg
        X = rng.random((10, FINGERPRINT_DIM)).astype(np.float32)
        y = rng.integers(0, 2, (10, N_LABELS)).astype(np.float32)
        mask = np.ones((10, N_LABELS), dtype=bool)
        # Buộc nhãn 0 chỉ có class 0
        y[:, 0] = 0.0
        model = PerLabelLogReg()
        model.fit(X[:8], y[:8], mask[:8])
        scores = model.predict_scores(X[8:])
        assert scores.shape[1] == N_LABELS

    def test_save_load_consistency(self, tmp_path):
        from adr.models.logreg import PerLabelLogReg
        X, y, mask = self._make_data()
        model = PerLabelLogReg()
        model.fit(X[:20], y[:20], mask[:20])
        model.save(tmp_path)
        model2 = PerLabelLogReg.load(tmp_path)
        s1 = model.predict_scores(X[20:])
        s2 = model2.predict_scores(X[20:])
        np.testing.assert_allclose(s1, s2, atol=1e-6)


class TestTokenizer:
    SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccncc1"]

    def test_fit_transform(self):
        from adr.features.tokenizer import CharTokenizer
        tok = CharTokenizer()
        tok.fit(self.SMILES)
        ids, masks, stats = tok.transform(self.SMILES)
        assert ids.shape == (len(self.SMILES), tok.max_length)
        assert masks.shape == ids.shape

    def test_unk_for_unseen_chars(self):
        from adr.features.tokenizer import CharTokenizer, UNK_ID
        tok = CharTokenizer()
        tok.fit(["CCO"])
        ids, _, stats = tok.transform(["$$$"])
        assert stats["n_unk"] > 0

    def test_save_load_same_encoding(self, tmp_path):
        from adr.features.tokenizer import CharTokenizer
        tok = CharTokenizer()
        tok.fit(self.SMILES)
        path = tmp_path / "tok.json"
        tok.save(path)
        tok2 = CharTokenizer.load(path)
        ids1, masks1, _ = tok.transform(self.SMILES)
        ids2, masks2, _ = tok2.transform(self.SMILES)
        assert np.array_equal(ids1, ids2)
        assert np.array_equal(masks1, masks2)

    def test_not_fitted_raises(self):
        from adr.features.tokenizer import CharTokenizer
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="chưa được fit"):
            tok.transform(["CCO"])


try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


@pytest.mark.skipif(not _TORCH, reason="torch chưa cài")
class TestMLP:
    def test_forward_shape(self):
        import torch
        from adr.models.mlp import MLP
        model = MLP(input_dim=FINGERPRINT_DIM, hidden_dim=64, n_labels=N_LABELS, dropout=0.0)
        model.eval()
        x = torch.randn(BATCH, FINGERPRINT_DIM)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, N_LABELS)

    def test_save_load(self, tmp_path):
        import torch
        from adr.models.mlp import MLP
        model = MLP(input_dim=32, hidden_dim=16, n_labels=N_LABELS, dropout=0.0)
        model.eval()
        x = torch.randn(2, 32)
        with torch.no_grad():
            out1 = model(x)
        model.save(tmp_path)
        model2 = MLP.load(tmp_path)
        model2.eval()
        with torch.no_grad():
            out2 = model2(x)
        assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.skipif(not _TORCH, reason="torch chưa cài")
class TestCNN:
    def test_forward_shape(self):
        import torch
        from adr.features.tokenizer import CharTokenizer
        from adr.models.cnn import SmilesCharCNN

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC"]
        tok = CharTokenizer()
        tok.fit(smiles)
        ids, masks, _ = tok.transform(smiles)

        model = SmilesCharCNN(
            vocab_size=tok.vocab_size, embedding_dim=16, channels=32,
            kernel_size=3, n_labels=N_LABELS, dropout=0.0,
        )
        model.eval()
        t_ids = torch.tensor(ids, dtype=torch.long)
        t_masks = torch.tensor(masks, dtype=torch.bool)
        with torch.no_grad():
            out = model(t_ids, t_masks)
        assert out.shape == (BATCH, N_LABELS)

    def test_pad_invariance(self):
        """Output không đổi khi thêm PAD cuối chuỗi."""
        import torch
        from adr.features.tokenizer import CharTokenizer
        from adr.models.cnn import SmilesCharCNN

        tok = CharTokenizer()
        tok.fit(["CCO", "c1ccccc1"])
        L = tok.max_length

        # Encode "CCO" với max_length=L (có padding)
        ids1, mask1, _ = tok.transform(["CCO"], max_length=L)
        # Encode "CCO" với max_length=L+5 (thêm 5 PAD)
        ids2, mask2, _ = tok.transform(["CCO"], max_length=L + 5)

        model = SmilesCharCNN(
            vocab_size=tok.vocab_size, embedding_dim=16, channels=32,
            kernel_size=3, n_labels=N_LABELS, dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            out1 = model(torch.tensor(ids1, dtype=torch.long), torch.tensor(mask1, dtype=torch.bool))
            out2 = model(torch.tensor(ids2, dtype=torch.long), torch.tensor(mask2, dtype=torch.bool))

        assert torch.allclose(out1, out2, atol=1e-4), \
            f"CNN thay đổi khi thêm PAD: max_diff={torch.max(torch.abs(out1-out2)):.6f}"

"""
tests/test_metrics.py – Kiểm tra metrics và artifact.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from adr.schemas import NUM_LABELS
from adr.evaluate.metrics import evaluate


N_LABELS = NUM_LABELS
rng = np.random.default_rng(42)


class TestMetrics:
    def _make_eval_data(self, N=50, seed=42):
        r = np.random.default_rng(seed)
        scores = r.random((N, N_LABELS)).astype(np.float32)
        targets = r.integers(0, 2, (N, N_LABELS)).astype(np.float32)
        mask = r.random((N, N_LABELS)) > 0.15
        return scores, targets, mask

    def test_eval_output_structure(self):
        scores, targets, mask = self._make_eval_data()
        labels = [f"l{i}" for i in range(N_LABELS)]
        result = evaluate(scores, targets, mask, labels)
        assert len(result.per_label) == N_LABELS
        assert result.n_valid_ap >= 0
        assert result.n_excluded >= 0

    def test_macro_ap_in_range(self):
        scores, targets, mask = self._make_eval_data()
        labels = [f"l{i}" for i in range(N_LABELS)]
        result = evaluate(scores, targets, mask, labels)
        if result.macro_ap is not None:
            assert 0 <= result.macro_ap <= 1

    def test_null_for_no_positive(self):
        """Nhãn không có positive → AP = None."""
        N = 20
        scores = rng.random((N, N_LABELS)).astype(np.float32)
        targets = np.zeros((N, N_LABELS), dtype=np.float32)
        mask = np.ones((N, N_LABELS), dtype=bool)
        labels = [f"l{i}" for i in range(N_LABELS)]
        result = evaluate(scores, targets, mask, labels)
        for lm in result.per_label:
            assert lm.ap is None, f"AP phải là None khi không có positive: {lm.label_name}"
        assert result.macro_ap is None

    def test_null_for_single_class_roc(self):
        """Nhãn chỉ một lớp → ROC-AUC = None."""
        N = 10
        scores = rng.random((N, N_LABELS)).astype(np.float32)
        targets = np.ones((N, N_LABELS), dtype=np.float32)
        mask = np.ones((N, N_LABELS), dtype=bool)
        labels = [f"l{i}" for i in range(N_LABELS)]
        result = evaluate(scores, targets, mask, labels)
        for lm in result.per_label:
            assert lm.roc_auc is None

    def test_padding_irrelevant(self):
        """Giá trị tại mask=False không ảnh hưởng metrics."""
        scores, targets, mask = self._make_eval_data()
        labels = [f"l{i}" for i in range(N_LABELS)]
        result1 = evaluate(scores, targets, mask, labels)

        targets_alt = targets.copy()
        targets_alt[~mask] = 999.0
        result2 = evaluate(scores, targets_alt, mask, labels)

        if result1.macro_ap is not None and result2.macro_ap is not None:
            assert abs(result1.macro_ap - result2.macro_ap) < 1e-9

    def test_no_observed_label(self):
        """Nhãn không có observation → valid=False."""
        N = 10
        scores = rng.random((N, N_LABELS)).astype(np.float32)
        targets = rng.integers(0, 2, (N, N_LABELS)).astype(np.float32)
        mask = np.ones((N, N_LABELS), dtype=bool)
        mask[:, 0] = False
        labels = [f"l{i}" for i in range(N_LABELS)]
        result = evaluate(scores, targets, mask, labels)
        assert not result.per_label[0].valid
        assert result.per_label[0].ap is None

    def test_save_json(self, tmp_path):
        scores, targets, mask = self._make_eval_data()
        labels = [f"l{i}" for i in range(N_LABELS)]
        result = evaluate(scores, targets, mask, labels)
        out = tmp_path / "eval.json"
        result.save(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "macro_ap" in data
        assert len(data["per_label"]) == N_LABELS


try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


@pytest.mark.skipif(not _TORCH, reason="torch chưa cài")
class TestLoss:
    def test_masked_loss_basic(self):
        import torch
        from adr.train.losses import masked_bce_loss
        logits = torch.zeros(4, N_LABELS)
        targets = torch.zeros(4, N_LABELS)
        mask = torch.ones(4, N_LABELS, dtype=torch.bool)
        loss = masked_bce_loss(logits, targets, mask)
        assert torch.isfinite(loss)

    def test_masked_loss_padding_irrelevant(self):
        import torch
        from adr.train.losses import masked_bce_loss
        logits = torch.randn(4, N_LABELS)
        targets = torch.randint(0, 2, (4, N_LABELS)).float()
        mask = torch.ones(4, N_LABELS, dtype=torch.bool)
        mask[0, :5] = False

        loss1 = masked_bce_loss(logits, targets, mask)
        t2 = targets.clone()
        t2[~mask] = 999.0
        loss2 = masked_bce_loss(logits, t2, mask)
        assert torch.allclose(loss1, loss2, atol=1e-5)

    def test_empty_mask_raises(self):
        import torch
        from adr.train.losses import masked_bce_loss
        logits = torch.randn(2, N_LABELS)
        targets = torch.randint(0, 2, (2, N_LABELS)).float()
        mask = torch.zeros(2, N_LABELS, dtype=torch.bool)
        with pytest.raises(RuntimeError, match="toàn mask=0"):
            masked_bce_loss(logits, targets, mask, empty_batch_policy="error")

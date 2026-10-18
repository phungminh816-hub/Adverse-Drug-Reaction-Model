"""
adr.models.frequency – Baseline tần suất.

Mỗi nhãn trả tỷ lệ dương trong train tính trên nhãn được quan sát.
Kết quả: [N, 27] – mọi hàng đều như nhau (không phụ thuộc đầu vào).

Nếu một nhãn không có nhãn train quan sát được:
  → Raise ValueError rõ và báo nhãn đó; không tự bịa prior.

Nhãn chỉ có dương (không âm) hoặc chỉ có âm vẫn trả tỷ lệ đúng.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adr.schemas import NUM_LABELS


class FrequencyBaseline:
    """
    Baseline: predict positive rate per label từ train set.

    fit(y_train, mask_train):
      - y_train: float32 [N, 27]
      - mask_train: bool [N, 27] – True nếu nhãn quan sát được
    predict_scores(n_samples):
      - Trả [n_samples, 27] với mỗi cột = positive rate của nhãn đó.
    """

    def __init__(self, n_labels: int = NUM_LABELS) -> None:
        self.n_labels = n_labels
        self.pos_rates: np.ndarray | None = None  # [n_labels]
        self.label_names: list[str] | None = None
        self._fitted = False

    def fit(
        self,
        y_train: np.ndarray,
        mask_train: np.ndarray,
        label_names: list[str] | None = None,
    ) -> "FrequencyBaseline":
        """
        Tính tỷ lệ dương cho mỗi nhãn trên mask=True.

        Raises:
            ValueError: nếu một nhãn không có sample quan sát được.
        """
        y = np.asarray(y_train, dtype=np.float32)
        m = np.asarray(mask_train, dtype=bool)

        if y.shape[1] != self.n_labels or m.shape[1] != self.n_labels:
            raise ValueError(
                f"Số nhãn không khớp: y={y.shape[1]}, mask={m.shape[1]}, n_labels={self.n_labels}"
            )

        pos_rates = np.zeros(self.n_labels, dtype=np.float32)
        no_obs_labels = []

        for j in range(self.n_labels):
            observed = m[:, j]
            n_obs = observed.sum()
            if n_obs == 0:
                no_obs_labels.append(j if label_names is None else label_names[j])
                pos_rates[j] = np.nan
            else:
                pos_rates[j] = y[observed, j].mean()

        if no_obs_labels:
            raise ValueError(
                f"Nhãn không có sample quan sát trong train: {no_obs_labels}\n"
                "  Không thể tính tỷ lệ dương. Kiểm tra dữ liệu hoặc split."
            )

        self.pos_rates = pos_rates
        self.label_names = label_names
        self._fitted = True
        return self

    def predict_scores(self, n_samples: int) -> np.ndarray:
        """
        Trả ma trận scores [n_samples, n_labels].
        Mọi hàng đều giống nhau (không phụ thuộc đầu vào SMILES).
        """
        if not self._fitted or self.pos_rates is None:
            raise RuntimeError("FrequencyBaseline chưa được fit.")
        return np.tile(self.pos_rates, (n_samples, 1)).astype(np.float32)

    def save(self, run_dir: Path) -> None:
        """Lưu model vào run_dir/model.json."""
        if not self._fitted:
            raise RuntimeError("Chưa fit, không thể lưu.")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "n_labels": self.n_labels,
            "pos_rates": self.pos_rates.tolist(),
            "label_names": self.label_names,
        }
        with (run_dir / "model.json").open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, run_dir: Path) -> "FrequencyBaseline":
        """Load từ run_dir/model.json."""
        run_dir = Path(run_dir)
        model_path = run_dir / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"model.json không tìm thấy: {model_path}")
        with model_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        obj = cls(n_labels=data["n_labels"])
        obj.pos_rates = np.array(data["pos_rates"], dtype=np.float32)
        obj.label_names = data["label_names"]
        obj._fitted = True
        return obj

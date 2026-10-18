"""
adr.models.logreg – Logistic Regression per-label trên Morgan fingerprint.

Mỗi nhãn có một LogisticRegression riêng.
Fit từng nhãn chỉ dùng hàng có mask=True cho nhãn đó.
Nhãn chỉ có một lớp train dùng ConstantPredictor (trả 0 hoặc 1).

Đảm bảo: predict_scores luôn trả [N, 27] dù một số nhãn chỉ có 1 lớp.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from adr.schemas import NUM_LABELS


class ConstantPredictor:
    """Fallback khi nhãn chỉ có 1 lớp trong train."""

    def __init__(self, constant: float) -> None:
        self.constant = constant

    def predict_proba_pos(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.constant, dtype=np.float32)


class PerLabelLogReg:
    """
    Logistic Regression per-label.

    fit(X_train, y_train, mask_train):
      - X_train: [N, 2048] fingerprints
      - y_train: [N, 27] labels (float32)
      - mask_train: [N, 27] bool mask

    predict_scores(X):
      - X: [M, 2048]
      - Trả: [M, 27] float32 scores (probability positive)
    """

    def __init__(
        self,
        n_labels: int = NUM_LABELS,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        C: float = 1.0,
    ) -> None:
        self.n_labels = n_labels
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.estimators: list = []  # list[LogisticRegression | ConstantPredictor]
        self.label_names: list[str] | None = None
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        mask_train: np.ndarray,
        label_names: list[str] | None = None,
    ) -> "PerLabelLogReg":
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.float32)
        m = np.asarray(mask_train, dtype=bool)

        self.estimators = []
        self.label_names = label_names

        for j in range(self.n_labels):
            obs_mask = m[:, j]
            n_obs = obs_mask.sum()

            if n_obs == 0:
                print(f"  [logreg] Nhãn {j}: không có sample quan sát → ConstantPredictor(0)")
                self.estimators.append(ConstantPredictor(0.0))
                continue

            X_j = X[obs_mask]
            y_j = y[obs_mask, j].astype(int)
            classes = np.unique(y_j)

            if len(classes) == 1:
                c = float(classes[0])
                name = label_names[j] if label_names else str(j)
                print(f"  [logreg] Nhãn '{name}': chỉ có 1 lớp ({int(c)}) → ConstantPredictor({c})")
                self.estimators.append(ConstantPredictor(c))
                continue

            clf = LogisticRegression(
                solver=self.solver,
                max_iter=self.max_iter,
                C=self.C,
            )
            clf.fit(X_j, y_j)
            self.estimators.append(clf)

        self._fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Trả [M, n_labels] float32 scores.
        Dùng predict_proba(...)[:, 1] với LogisticRegression hợp lệ;
        Dùng constant cho ConstantPredictor.
        """
        if not self._fitted:
            raise RuntimeError("PerLabelLogReg chưa được fit.")
        X = np.asarray(X, dtype=np.float32)
        M = len(X)
        scores = np.zeros((M, self.n_labels), dtype=np.float32)

        for j, est in enumerate(self.estimators):
            if isinstance(est, ConstantPredictor):
                scores[:, j] = est.predict_proba_pos(M)
            else:
                # LogisticRegression: lấy prob của class 1
                proba = est.predict_proba(X)
                if proba.shape[1] == 2:
                    scores[:, j] = proba[:, 1]
                else:
                    # Chỉ một lớp được học (edge case)
                    scores[:, j] = float(est.classes_[0])

        return scores

    def save(self, run_dir: Path) -> None:
        """Lưu tất cả estimator vào run_dir/model.pkl."""
        if not self._fitted:
            raise RuntimeError("Chưa fit.")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "n_labels": self.n_labels,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "C": self.C,
            "label_names": self.label_names,
            "estimators": self.estimators,
        }
        with (run_dir / "model.pkl").open("wb") as fh:
            pickle.dump(data, fh)

    @classmethod
    def load(cls, run_dir: Path) -> "PerLabelLogReg":
        """Load từ run_dir/model.pkl."""
        run_dir = Path(run_dir)
        pkl_path = run_dir / "model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"model.pkl không tìm thấy: {pkl_path}")
        with pkl_path.open("rb") as fh:
            data = pickle.load(fh)
        obj = cls(
            n_labels=data["n_labels"],
            solver=data["solver"],
            max_iter=data["max_iter"],
            C=data["C"],
        )
        obj.label_names = data["label_names"]
        obj.estimators = data["estimators"]
        obj._fitted = True
        return obj

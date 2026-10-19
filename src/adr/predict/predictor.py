"""Nạp model, chuẩn hóa SMILES và dự đoán điểm ADR cho app/CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adr.schemas import PredictionResult, ArtifactMetadata
from adr.utils.artifacts import check_artifact_dir
from adr.utils.paths import sha256_str

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False


def _canonicalize(smiles: str) -> tuple[str | None, list[str]]:
    """Trả (canonical_smiles, warnings)."""
    warnings = []
    if not _RDKIT_OK:
        warnings.append(
            "RDKit chưa cài: dùng SMILES gốc mà không canonicalize. "
            "Cài rdkit để có kết quả chính xác hơn."
        )
        return smiles, warnings
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None, [f"SMILES không hợp lệ (RDKit parse lỗi): '{smiles[:80]}'"]
    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canon, warnings


class Predictor:
    """
    Predictor thống nhất cho mọi model type.

    Sử dụng:
      pred = Predictor(run_dir="artifacts/logreg_20261015_123000_42")
      result = pred.predict("CCO")
      print(result.top_k(5))
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.meta: ArtifactMetadata = check_artifact_dir(self.run_dir)
        self._model = None
        self._featurizer = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model theo model_type từ metadata."""
        mt = self.meta.model_type

        if mt == "frequency":
            from adr.models.frequency import FrequencyBaseline
            self._model = FrequencyBaseline.load(self.run_dir)

        elif mt == "logreg":
            from adr.models.logreg import PerLabelLogReg
            from adr.features.fingerprints import MorganFeaturizer
            self._featurizer = MorganFeaturizer.load(self.run_dir / "fingerprint_config.json")
            self._model = PerLabelLogReg.load(self.run_dir)

        elif mt == "mlp":
            from adr.models.mlp import MLP
            from adr.features.fingerprints import MorganFeaturizer
            self._model = MLP.load(self.run_dir).eval()
            self._featurizer = MorganFeaturizer.load(self.run_dir / "fingerprint_config.json")

        elif mt == "cnn":
            from adr.models.cnn import SmilesCharCNN
            from adr.features.tokenizer import CharTokenizer
            self._model = SmilesCharCNN.load(self.run_dir).eval()
            self._tokenizer = CharTokenizer.load(self.run_dir / "tokenizer.json")

        else:
            raise ValueError(f"model_type không nhận dạng được: '{mt}'")

        model_labels = getattr(self._model, "label_names", None)
        if model_labels is not None:
            self.meta.check_label_names(model_labels)
        if getattr(self._model, "n_labels", len(self.meta.label_names)) != len(self.meta.label_names):
            raise ValueError("Số nhãn model không khớp metadata.")
        if self._featurizer and self._featurizer.check_config(self.meta.feature_config):
            raise ValueError("Fingerprint config không khớp metadata.")
        if mt == "mlp" and self._model.input_dim != self._featurizer.n_bits:
            raise ValueError("Kích thước fingerprint không khớp MLP.")
        if mt == "cnn":
            if self._tokenizer.check_config(self.meta.feature_config) or self._model.vocab_size != self._tokenizer.vocab_size:
                raise ValueError("Tokenizer không khớp checkpoint hoặc metadata.")

    def score_smiles(self, smiles: list[str], batch_size: int = 32) -> np.ndarray:
        """Tạo ma trận điểm cho SMILES đã canonicalize."""
        if batch_size <= 0:
            raise ValueError("batch_size phải dương.")
        if not smiles:
            return np.empty((0, len(self.meta.label_names)), dtype=np.float32)
        mt = self.meta.model_type
        if mt == "frequency":
            scores = self._model.predict_scores(len(smiles))
        else:
            batches = []
            for start in range(0, len(smiles), batch_size):
                chunk = smiles[start:start + batch_size]
                if mt in {"logreg", "mlp"}:
                    values, errors = self._featurizer.transform(chunk)
                    if errors:
                        raise ValueError(f"SMILES không hợp lệ: {errors[:3]}")
                if mt == "logreg":
                    batches.append(self._model.predict_scores(values))
                else:
                    import torch
                    with torch.inference_mode():
                        if mt == "mlp":
                            logits = self._model(torch.from_numpy(values))
                        else:
                            ids, attention, _ = self._tokenizer.transform(chunk)
                            logits = self._model(torch.from_numpy(ids), torch.from_numpy(attention))
                        batches.append(logits.sigmoid().numpy())
            scores = np.concatenate(batches)
        if scores.shape != (len(smiles), len(self.meta.label_names)) or not np.isfinite(scores).all():
            raise ValueError("Model trả điểm sai kích thước hoặc không hữu hạn.")
        return np.clip(scores, 0, 1)

    def predict(
        self,
        smiles: str,
        drug_id: str | None = None,
    ) -> PredictionResult:
        """
        Dự đoán ADR scores cho một SMILES.

        Args:
            smiles: SMILES input (chưa nhất thiết phải canonical).
            drug_id: ID tùy chọn (nếu biết từ dataset).

        Returns:
            PredictionResult với scores [27], predicted_labels, warnings.

        SMILES không hợp lệ trả canonical_smiles=None và thông báo trong warnings.
        """
        warnings: list[str] = []
        mt = self.meta.model_type
        label_names = self.meta.label_names
        threshold = self.meta.threshold
        n_labels = len(label_names)

        canon, canon_warns = _canonicalize(smiles)
        warnings.extend(canon_warns)

        if canon is None:

            return PredictionResult(
                drug_id=drug_id,
                input_smiles=smiles,
                canonical_smiles=None,
                model_id=self.meta.run_id,
                label_names=label_names,
                scores=[float("nan")] * n_labels,
                threshold=threshold,
                predicted_labels=[],
                warnings=warnings,
            )

        scores_arr = self.score_smiles([canon])[0]
        if mt == "cnn":
            _, _, stats = self._tokenizer.transform([canon])
            if stats["n_unk"]:
                warnings.append(f"Có {stats['n_unk']} ký tự chưa thấy trong train (UNK).")
            if stats["n_truncated"]:
                warnings.append(f"SMILES bị cắt ở {self._tokenizer.max_length} ký tự.")

        scores_arr = np.clip(scores_arr, 0.0, 1.0)
        scores = scores_arr.tolist()

        predicted_labels = [
            label_names[j]
            for j, s in enumerate(scores)
            if s >= threshold
        ]

        return PredictionResult(
            drug_id=drug_id or sha256_str(canon),
            input_smiles=smiles,
            canonical_smiles=canon,
            model_id=self.meta.run_id,
            label_names=label_names,
            scores=scores,
            threshold=threshold,
            predicted_labels=predicted_labels,
            warnings=warnings,
        )

    def predict_batch(
        self,
        smiles_list: list[str],
    ) -> list[PredictionResult]:
        """Dự đoán cho danh sách SMILES."""
        return [self.predict(smi) for smi in smiles_list]

"""
adr.predict.predictor – Predictor thống nhất cho mọi model type.

Đây là nơi DUY NHẤT thực hiện inference nghiệp vụ.
Streamlit app, CLI và bất kỳ script nào khác đều gọi qua đây.

Luồng:
  1. check_artifact_dir → load metadata → kiểm tra label_names, kích thước.
  2. Canonicalize SMILES (RDKit nếu có).
  3. Featurize theo model_type (fingerprint / tokenizer).
  4. Load model → predict_scores → sigmoid nếu cần.
  5. Trả PredictionResult với warnings.

Thiếu artifact → raise FileNotFoundError (không sinh điểm ngẫu nhiên).
Model neural (mlp/cnn) đang là TODO: load contract có thể hoàn chỉnh sau khi train.
"""

from __future__ import annotations

import sys
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
    if mol is None:
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
            # TODO (Tuần 5): Load MLP sau khi training loop hoàn thành
            # Input: run_dir/model.pt + run_dir/fingerprint_config.json
            # Output: self._model = MLP đã load, self._featurizer = MorganFeaturizer
            # Điều kiện: model.pt phải tồn tại (check_artifact_dir đã kiểm tra)
            raise NotImplementedError(
                "MLP inference chưa triển khai (TODO Tuần 5).\n"
                "  Sau khi train MLP, implement _load_model cho mt='mlp'."
            )

        elif mt == "cnn":
            # TODO (Tuần 6-7): Load CNN + tokenizer
            # Input: run_dir/model.pt + run_dir/tokenizer.json
            # Output: self._model = SmilesCharCNN, self._tokenizer = CharTokenizer
            # Điều kiện: eval mode, no_grad khi predict
            raise NotImplementedError(
                "CNN inference chưa triển khai (TODO Tuần 6-7).\n"
                "  Sau khi train CNN, implement _load_model cho mt='cnn'."
            )

        else:
            raise ValueError(f"model_type không nhận dạng được: '{mt}'")

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

        Raises:
            ValueError: nếu SMILES không hợp lệ (RDKit).
            FileNotFoundError: nếu artifact thiếu.
        """
        warnings: list[str] = []
        mt = self.meta.model_type
        label_names = self.meta.label_names
        threshold = self.meta.threshold
        n_labels = len(label_names)

        # Canonicalize
        canon, canon_warns = _canonicalize(smiles)
        warnings.extend(canon_warns)

        if canon is None:
            # SMILES không hợp lệ
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

        # Featurize và predict
        if mt == "frequency":
            scores_arr = self._model.predict_scores(1)[0]  # [27]

        elif mt == "logreg":
            X, fp_errors = self._featurizer.transform([canon])
            if fp_errors:
                warnings.extend(fp_errors)
            scores_arr = self._model.predict_scores(X)[0]  # [27]

        else:
            raise RuntimeError(f"predict chưa hỗ trợ model_type='{mt}'")

        # Đảm bảo scores hữu hạn [0, 1]
        scores_arr = np.clip(scores_arr, 0.0, 1.0)
        scores = scores_arr.tolist()

        predicted_labels = [
            label_names[j]
            for j, s in enumerate(scores)
            if s >= threshold
        ]

        return PredictionResult(
            drug_id=drug_id or sha256_str(canon)[:8],
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

"""
adr.schemas – Dataclass contracts cho artifact metadata và prediction output.

Tất cả artifact phải có ArtifactMetadata.
Inference trả PredictionResult.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Artifact metadata
# ---------------------------------------------------------------------------


@dataclass
class ArtifactMetadata:
    """
    Metadata được lưu cùng mọi run artifact.
    Dùng để kiểm tra tính nhất quán khi load lại.
    """
    model_type: str                      # "frequency" | "logreg" | "mlp" | "cnn"
    run_id: str                          # timestamp-based unique ID
    seed: int
    label_names: list[str]               # 27 tên theo thứ tự cột nguồn
    dataset_checksum: str                # SHA-256 của processed.csv
    split_manifest_checksum: str         # SHA-256 của split_manifest.csv
    feature_config: dict[str, Any]       # fingerprint hoặc tokenizer params
    library_versions: dict[str, str]     # {"numpy": "2.5.3", ...}
    threshold: float                     # inference threshold
    eval_label_policy: str               # mô tả nhãn nào được dùng trong eval
    created_at_utc: str                  # ISO 8601
    config_resolved: dict[str, Any] = field(default_factory=dict)
    # Tokenizer artifact path (chỉ dùng cho CNN)
    tokenizer_path: str | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ArtifactMetadata":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata không tìm thấy: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    def check_label_names(self, expected: list[str]) -> None:
        """Raise ValueError nếu label_names không khớp expected."""
        if self.label_names != expected:
            raise ValueError(
                f"Label names trong artifact ({len(self.label_names)}) "
                f"không khớp expected ({len(expected)}).\n"
                f"  Artifact: {self.label_names[:3]}...\n"
                f"  Expected: {expected[:3]}..."
            )


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """
    Kết quả inference cho một phân tử.

    scores: sigmoid scores [0, 1] cho 27 nhãn theo thứ tự label_names.
    predicted_labels: nhãn có score >= threshold.
    warnings: cảnh báo UNK/truncation hoặc vấn đề SMILES.
    """
    drug_id: str | None
    input_smiles: str
    canonical_smiles: str | None
    model_id: str
    label_names: list[str]
    scores: list[float]                  # độ dài == len(label_names) == 27
    threshold: float
    predicted_labels: list[str]          # nhãn có score >= threshold
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.label_names):
            raise ValueError(
                f"scores có {len(self.scores)} phần tử, "
                f"nhưng label_names có {len(self.label_names)}."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        """Trả k nhãn có score cao nhất (label, score)."""
        pairs = sorted(
            zip(self.label_names, self.scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return list(pairs[:k])


# ---------------------------------------------------------------------------
# Dataset schema constants
# ---------------------------------------------------------------------------

NUM_LABELS: int = 27
SMILES_COL: str = "smiles"
DRUG_ID_COL: str = "drug_id"
CANONICAL_SMILES_COL: str = "canonical_smiles"
SPLIT_COL: str = "split"
VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})
VALID_LABEL_VALUES: frozenset = frozenset({0, 1, 0.0, 1.0})

# PAD/UNK token IDs cho tokenizer
PAD_TOKEN_ID: int = 0
UNK_TOKEN_ID: int = 1

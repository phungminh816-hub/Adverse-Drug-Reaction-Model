"""
adr.utils.artifacts – Helpers để kiểm tra và load artifact.

Mọi inference phải đi qua check_artifact_dir để bắt lỗi sớm.
"""

from __future__ import annotations

import sys
from pathlib import Path

from adr.schemas import ArtifactMetadata


REQUIRED_FILES = {
    "frequency": ["metadata.json", "model.json"],
    "logreg": ["metadata.json", "model.pkl"],
    "mlp": ["metadata.json", "model.pt"],
    "cnn": ["metadata.json", "model.pt", "tokenizer.json"],
}


def check_artifact_dir(run_dir: Path, model_type: str | None = None) -> ArtifactMetadata:
    """
    Kiểm tra run_dir tồn tại và chứa đủ file bắt buộc.
    Load và trả ArtifactMetadata.

    Raises:
        FileNotFoundError: nếu run_dir hoặc file quan trọng thiếu.
        ValueError: nếu model_type không khớp metadata.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory không tồn tại: {run_dir}\n"
            "  Chạy train trước để tạo artifact."
        )

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Thiếu metadata.json trong: {run_dir}\n"
            "  Artifact có thể bị hỏng hoặc chưa hoàn chỉnh."
        )

    meta = ArtifactMetadata.load(meta_path)

    # Kiểm tra model_type khớp
    if model_type is not None and meta.model_type != model_type:
        raise ValueError(
            f"model_type yêu cầu '{model_type}' nhưng artifact có '{meta.model_type}'."
        )

    # Kiểm tra file bắt buộc theo model_type
    required = REQUIRED_FILES.get(meta.model_type, ["metadata.json"])
    missing = [f for f in required if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Artifact thiếu file: {missing}\n"
            f"  Run dir: {run_dir}"
        )

    return meta


def list_run_dirs(artifacts_root: Path) -> list[Path]:
    """Liệt kê tất cả run_dir hợp lệ (có metadata.json) trong artifacts_root."""
    root = Path(artifacts_root)
    if not root.exists():
        return []
    return sorted(
        [d for d in root.iterdir() if d.is_dir() and (d / "metadata.json").exists()],
        reverse=True,  # mới nhất trước
    )

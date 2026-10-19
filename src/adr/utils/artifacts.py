"""Kiểm tra file bắt buộc và đọc metadata của model đã lưu."""

from __future__ import annotations

from pathlib import Path

from adr.schemas import ArtifactMetadata


REQUIRED_FILES = {
    "frequency": ["metadata.json", "model.json"],
    "logreg": ["metadata.json", "model.pkl", "fingerprint_config.json"],
    "mlp": ["metadata.json", "model.pt", "fingerprint_config.json"],
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

    if model_type is not None and meta.model_type != model_type:
        raise ValueError(
            f"model_type yêu cầu '{model_type}' nhưng artifact có '{meta.model_type}'."
        )

    if meta.model_type not in REQUIRED_FILES:
        raise ValueError(f"model_type không hỗ trợ: {meta.model_type}")
    required = REQUIRED_FILES[meta.model_type]
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
        reverse=True,
    )

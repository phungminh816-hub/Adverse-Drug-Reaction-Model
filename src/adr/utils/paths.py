"""
adr.utils.paths – Quản lý đường dẫn artifact và run directory.

Không hardcode ổ đĩa. Mọi path tính từ project root hoặc được
truyền tường minh qua config.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Any


def get_run_id(model_type: str, seed: int) -> str:
    """
    Tạo run ID dạng: {model_type}_{YYYYMMDD_HHMMSS}_{seed}
    Ví dụ: logreg_20261015_203000_42
    """
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{ts}_{seed}"


def get_run_dir(artifacts_root: Path, run_id: str) -> Path:
    """Trả path đến run directory; tạo nếu chưa có."""
    run_dir = Path(artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sha256_file(path: Path, buf_size: int = 65536) -> str:
    """Tính SHA-256 hex của file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(text: str) -> str:
    """Tính SHA-256 hex của chuỗi UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_project_root() -> Path:
    """Tìm project root chứa pyproject.toml từ cwd."""
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def get_library_versions() -> dict[str, str]:
    """Thu thập phiên bản các thư viện quan trọng."""
    versions: dict[str, str] = {}
    libs = ["numpy", "pandas", "sklearn", "torch", "rdkit", "yaml", "streamlit"]
    import importlib

    for lib in libs:
        try:
            mod = importlib.import_module(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not_installed"

    import sys
    versions["python"] = sys.version.split()[0]
    return versions

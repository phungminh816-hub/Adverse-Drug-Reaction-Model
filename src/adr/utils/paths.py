"""Xác định đường dẫn dự án, tạo run ID và tính checksum."""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path


def get_run_id(model_type: str, seed: int) -> str:
    """Tạo run ID từ model type, thời gian UTC đến microsecond và seed."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
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
    for p in list(Path(__file__).resolve().parents) + [Path.cwd()] + list(Path.cwd().parents):
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

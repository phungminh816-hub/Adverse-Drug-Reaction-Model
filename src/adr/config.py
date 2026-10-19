"""Đọc, hợp nhất, kiểm tra và lưu cấu hình YAML."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

from adr.utils.paths import find_project_root

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def _require_yaml() -> None:
    if not _YAML_OK:
        print(
            "[adr.config] ERROR: pyyaml chưa được cài.\n"
            "  Chạy: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override vào base; trả dict mới, không sửa đầu vào."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


PROJECT_ROOT: Path = find_project_root()
DEFAULT_BASE: Path = PROJECT_ROOT / "configs" / "base.yaml"


def _config_path(path: str | Path) -> Path:
    """Tìm file cấu hình từ đường dẫn tuyệt đối, cwd hoặc gốc dự án."""
    path = Path(path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    return path


def _load_yaml(path: Path) -> dict:
    """Đọc một YAML mapping và báo lỗi nếu nội dung khác kiểu dictionary."""
    _require_yaml()
    if not path.exists():
        raise FileNotFoundError(f"Config không tìm thấy: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config phải là YAML mapping: {path}")
    return data


def load_config(
    model_config: str | Path | None = None,
    base: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load và merge config.

    Args:
        model_config: Path đến model-specific YAML (logreg.yaml, mlp.yaml …).
                      Nếu None, chỉ load base.
        base: Path đến base.yaml. Mặc định: configs/base.yaml trong project root.

    Returns:
        dict config đã merge.
    """
    base_path = _config_path(base) if base else DEFAULT_BASE
    cfg = _load_yaml(base_path)

    if model_config is not None:
        model_path = _config_path(model_config)
        model_cfg = _load_yaml(model_path)
        cfg = _deep_merge(cfg, model_cfg)

    return cfg


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Chuyển các path string trong cfg['paths'] thành Path tuyệt đối
    tính từ project root. Trả copy; không sửa đầu vào.
    """
    cfg = copy.deepcopy(cfg)
    paths_section = cfg.get("paths", {})
    for key, val in paths_section.items():
        if isinstance(val, (str, Path)):
            p = Path(val).expanduser()
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            paths_section[key] = p
    cfg["paths"] = paths_section
    return cfg


def _pathify(obj: Any) -> Any:
    """Chuyển Path lồng trong dictionary/list thành chuỗi để ghi YAML."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _pathify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_pathify(v) for v in obj]
    return obj


def save_config(cfg: dict[str, Any], run_dir: Path) -> Path:
    """Lưu resolved config vào run_dir/config.yaml."""
    _require_yaml()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.yaml"

    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_pathify(cfg), fh, allow_unicode=True, default_flow_style=False)
    return out


def validate_base_config(cfg: dict[str, Any]) -> list[str]:
    """
    Kiểm tra các trường bắt buộc của base config.
    Trả list lỗi (rỗng = hợp lệ).
    """
    errors: list[str] = []
    required = [
        ("project", "seed"),
        ("paths", "data_raw"),
        ("paths", "artifacts"),
        ("data", "source_url"),
        ("data", "num_labels"),
        ("split", "strategy"),
    ]
    for section, key in required:
        if section not in cfg or key not in cfg.get(section, {}):
            errors.append(f"Thiếu cfg['{section}']['{key}']")
    return errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kiểm tra load config ADR")
    parser.add_argument("--config", default=str(DEFAULT_BASE), help="Path đến YAML")
    parser.add_argument("--base", default=None, help="Path đến base YAML (nếu config là model config)")
    args = parser.parse_args()

    cfg = load_config(args.config, base=args.base)
    errors = validate_base_config(cfg)
    if errors:
        print("Lỗi config:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    cfg = resolve_paths(cfg)
    _require_yaml()
    print(yaml.safe_dump(_pathify(cfg), allow_unicode=True, default_flow_style=False))
    print("Config hợp lệ.")

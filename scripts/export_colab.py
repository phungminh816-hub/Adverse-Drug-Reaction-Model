"""Gói mã nguồn cho Colab; bỏ môi trường, dữ liệu và model cục bộ."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Đóng gói mã nguồn, notebook, tài liệu và script chạy vào ZIP dùng trên Colab."""
    output = ROOT / "results" / "ADR_Product_colab.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for folder in ("src", "configs", "scripts", "app", "tests", "notebooks", "reports"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and ".ipynb_checkpoints" not in path.parts and not path.suffix == ".pyc":
                    archive.write(path, Path("ADR_Product") / path.relative_to(ROOT))
        for name in ("pyproject.toml", "README.md", "requirements.txt", "requirements-tested.txt", "setup.ps1", "run.ps1"):
            archive.write(ROOT / name, Path("ADR_Product") / name)
    print(f"Colab ZIP: {output}")


if __name__ == "__main__":
    main()

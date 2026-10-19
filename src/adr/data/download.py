"""Tải, giải nén SIDER và lưu checksum của file nguồn."""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import shutil
import sys
import urllib.request
import urllib.error
from pathlib import Path

from adr.config import load_config, resolve_paths
from adr.utils.paths import sha256_file


def download_sider(
    url: str,
    dest_dir: Path,
    filename: str,
    timeout: int = 60,
    force: bool = False,
) -> Path:
    """
    Tải file từ url về dest_dir/filename.
    Bỏ qua nếu file đã tồn tại và force=False.

    Returns:
        Path đến file đã tải.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    if dest.exists() and not force:
        print(f"[download] File đã tồn tại: {dest}")
        print("  Dùng --force để tải lại.")
        return dest

    print(f"[download] Đang tải: {url}")
    print(f"  → {dest}")
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            with part.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
        part.replace(dest)
    except urllib.error.URLError as exc:
        part.unlink(missing_ok=True)
        print(f"[download] LỖI kết nối: {exc}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError:
        part.unlink(missing_ok=True)
        print(f"[download] LỖI timeout sau {timeout}s", file=sys.stderr)
        sys.exit(1)

    checksum = sha256_file(dest)
    meta = {
        "url": url,
        "filename": filename,
        "downloaded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256": checksum,
        "size_bytes": dest.stat().st_size,
    }
    meta_path = dest_dir / "download_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[download] Hoàn thành. SHA-256: {checksum}")
    print(f"[download] Metadata: {meta_path}")
    return dest


def decompress_gz(gz_path: Path, out_path: Path, force: bool = False) -> Path:
    """Giải nén .gz thành out_path."""
    gz_path, out_path = Path(gz_path), Path(out_path)
    if out_path.exists() and not force:
        print(f"[download] Đã giải nén: {out_path}")
        return out_path
    print(f"[download] Giải nén: {gz_path} → {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")
    try:
        with gzip.open(gz_path, "rb") as f_in, part.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        part.replace(out_path)
    finally:
        part.unlink(missing_ok=True)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tải SIDER dataset về data/raw/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python -m adr.data.download --config configs/base.yaml\n"
            "  python -m adr.data.download --config configs/base.yaml --force\n"
        ),
    )
    parser.add_argument("--config", required=True, help="Path đến base config YAML")
    parser.add_argument("--force", action="store_true", help="Tải lại dù file đã tồn tại")
    args = parser.parse_args(argv)

    cfg = resolve_paths(load_config(args.config))
    data_cfg = cfg["data"]
    raw_dir: Path = cfg["paths"]["data_raw"]

    url = data_cfg["source_url"]
    gz_filename = data_cfg["raw_filename"]
    csv_filename = gz_filename.removesuffix(".gz")

    force = args.force or data_cfg.get("force_download", False)

    gz_path = download_sider(url, raw_dir, gz_filename, timeout=data_cfg.get("download_timeout_s", 60), force=force)
    decompress_gz(gz_path, raw_dir / csv_filename, force=force)

    print(f"[download] Raw CSV sẵn sàng: {raw_dir / csv_filename}")


if __name__ == "__main__":
    main()

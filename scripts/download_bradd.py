"""Resumable, md5-verified download and unzip of BraDD-S1TS from Zenodo (PRD Phase 0).

Usage: python scripts/download_bradd.py --config configs/data/downloads.yaml [--dest-dir DIR] [--delete-zip]
Proxy and CA bundle come from the environment (HTTPS_PROXY, REQUESTS_CA_BUNDLE), which requests honours.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path

import requests
import yaml

MB = 1024 * 1024


def free_gb(path: Path) -> float:
    """Free disk space in GB on the filesystem holding `path`."""
    return shutil.disk_usage(path).free / 1024 ** 3


def md5_of_file(path: Path, chunk_bytes: int) -> "hashlib._Hash":
    """Streaming md5 hasher already fed with the whole file."""
    hasher = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            hasher.update(chunk)
    return hasher


def download_resumable(session: requests.Session, url: str, part_path: Path, chunk_bytes: int,
                       timeout_s: float) -> str:
    """Download `url` into `part_path`, resuming with an HTTP Range request; return the md5 hex digest."""
    offset = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, headers=headers, stream=True, timeout=timeout_s) as response:
        if offset and response.status_code == 416:
            return md5_of_file(part_path, chunk_bytes).hexdigest()
        response.raise_for_status()
        resumed = offset > 0 and response.status_code == 206
        if offset and not resumed:
            print(f"server ignored Range (HTTP {response.status_code}); restarting from byte 0")
        hasher = md5_of_file(part_path, chunk_bytes) if resumed else hashlib.md5()
        with open(part_path, "ab" if resumed else "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                fh.write(chunk)
                hasher.update(chunk)
    return hasher.hexdigest()


def safe_unzip(zip_path: Path, dest: Path) -> int:
    """Extract every member under `dest`, refusing paths that escape it; return the member count."""
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in archive: {name}")
        archive.extractall(root)
        return len(archive.namelist())


def fetch_verify_unzip(cfg: Mapping, session: requests.Session) -> Path:
    """Download (or reuse) the zip, verify its md5, unzip it, and delete it when space is short."""
    dest = Path(cfg["dest_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / cfg["zip_name"]
    part_path = zip_path.with_name(zip_path.name + ".part")
    chunk_bytes = int(cfg["chunk_mb"] * MB)
    space = free_gb(dest)
    delete_zip = bool(cfg["delete_zip_after_unzip"]) or space < float(cfg["min_free_gb"])
    print(f"free space {space:.1f} GB (threshold {cfg['min_free_gb']} GB); delete zip after unzip: {delete_zip}")
    if zip_path.exists():
        digest = md5_of_file(zip_path, chunk_bytes).hexdigest()
    else:
        digest = download_resumable(session, cfg["url"], part_path, chunk_bytes, float(cfg["timeout_s"]))
    if digest != cfg["md5"]:
        bad = zip_path if zip_path.exists() else part_path
        raise ValueError(f"md5 mismatch: got {digest}, expected {cfg['md5']}; remove {bad} and retry")
    if not zip_path.exists():
        part_path.rename(zip_path)
    count = safe_unzip(zip_path, dest)
    print(f"md5 ok, unzipped {count} members into {dest}")
    if delete_zip:
        zip_path.unlink()
        print(f"deleted {zip_path}")
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: read the `bradd` section of the downloads config and run the download."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dest-dir", default=None)
    parser.add_argument("--delete-zip", action="store_true", help="delete the zip after unzip regardless of space")
    args = parser.parse_args(argv)
    cfg = dict(yaml.safe_load(Path(args.config).read_text())["bradd"])
    if args.dest_dir:
        cfg["dest_dir"] = args.dest_dir
    cfg["delete_zip_after_unzip"] = cfg["delete_zip_after_unzip"] or args.delete_zip
    try:
        fetch_verify_unzip(cfg, requests.Session())
    except (requests.RequestException, ValueError) as err:
        print(f"download failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

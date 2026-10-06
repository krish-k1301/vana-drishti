"""Download DETER-B Amazonia alerts and PRODES yearly deforestation from TerraBrasilis (PRD Phase 3).

Usage: python scripts/download_deter_prodes.py --config configs/data/downloads.yaml [--layers deter_amz ...]
       [--mode wfs|files] [--verify PATH]
WFS pages are cached as JSON files (resumable), then merged into one GeoJSON per layer.
Endpoints are unverified until terrabrasilis.dpi.inpe.br is reachable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import requests
import yaml

PRD_DETER_CLASSES = ("DESMATAMENTO_CR", "DESMATAMENTO_VEG", "CICATRIZ_DE_QUEIMADA", "DEGRADACAO",
                     "CS_DESORDENADO", "CS_GEOMETRICO", "MINERACAO")


def wfs_params(layer: Mapping, version: str, output_format: str, start: int, count: int) -> dict:
    """GeoServer WFS GetFeature query parameters for one page."""
    params = {"service": "WFS", "version": version, "request": "GetFeature", "typeNames": layer["type_name"],
              "outputFormat": output_format, "startIndex": start, "count": count, "srsName": "EPSG:4326"}
    if layer.get("sort_by"):
        params["sortBy"] = layer["sort_by"]
    if layer.get("cql_filter"):
        params["CQL_FILTER"] = layer["cql_filter"]
    return params


def fetch_wfs_pages(session: requests.Session, wfs_url: str, layer: Mapping, version: str,
                    output_format: str, page_size: int, timeout_s: float, pages_dir: Path) -> int:
    """Fetch every page (startIndex/count) into `pages_dir`, reusing cached pages; return the feature count."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    start, total = 0, 0
    while True:
        page_path = pages_dir / f"page_{start:09d}.json"
        if page_path.exists():
            page = json.loads(page_path.read_text())
        else:
            params = wfs_params(layer, version, output_format, start, page_size)
            response = session.get(wfs_url, params=params, timeout=timeout_s)
            response.raise_for_status()
            page = response.json()
            page_path.write_text(json.dumps(page))
        n_features = len(page.get("features", []))
        total += n_features
        print(f"{layer['type_name']}: startIndex={start} -> {n_features} features")
        if n_features < page_size:
            return total
        start += page_size


def merge_pages(pages_dir: Path, out_path: Path) -> int:
    """Merge cached WFS pages into one GeoJSON FeatureCollection; return the feature count."""
    features = []
    for page_path in sorted(pages_dir.glob("page_*.json")):
        features.extend(json.loads(page_path.read_text()).get("features", []))
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return len(features)


def download_file(session: requests.Session, url: str, out_path: Path, timeout_s: float) -> Path:
    """Stream one file-delivery URL to disk."""
    with session.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
    return out_path


def verify_deter_classes(path: str | Path, attribute: str = "classname",
                         expected: Sequence[str] = PRD_DETER_CLASSES) -> dict:
    """Print the unique values of the DETER class attribute and compare them with the PRD's list."""
    features = json.loads(Path(path).read_text()).get("features", [])
    values = sorted({str(f.get("properties", {}).get(attribute)) for f in features})
    report = {"found": values, "missing": sorted(set(expected) - set(values)),
              "unexpected": sorted(set(values) - set(expected))}
    print(f"{attribute} values ({len(features)} features): {values}")
    print(f"expected but missing: {report['missing']}")
    print(f"present but not in the PRD list: {report['unexpected']}")
    return report


def run_layers(cfg: Mapping, names: Sequence[str], mode: str, session: requests.Session) -> list[Path]:
    """Download the selected layers by WFS or by the file-delivery URLs; return the written paths."""
    dest = Path(cfg["dest_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        if mode == "files":
            url = cfg["file_downloads"][name]
            written.append(download_file(session, url, dest / f"{name}_download", float(cfg["timeout_s"])))
            continue
        layer = cfg["layers"][name]
        pages_dir = dest / f"{name}_pages"
        fetch_wfs_pages(session, cfg["wfs_url"], layer, cfg["wfs_version"], cfg["output_format"],
                        int(cfg["page_size"]), float(cfg["timeout_s"]), pages_dir)
        out_path = dest / layer["out_name"]
        print(f"merged {merge_pages(pages_dir, out_path)} features into {out_path}")
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: read the `terrabrasilis` section of the downloads config and download."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--layers", nargs="*", default=None, help="layer keys; default: all configured")
    parser.add_argument("--mode", choices=("wfs", "files"), default="wfs")
    parser.add_argument("--dest-dir", default=None)
    parser.add_argument("--verify", default=None, help="only check DETER classes in this GeoJSON")
    args = parser.parse_args(argv)
    cfg = dict(yaml.safe_load(Path(args.config).read_text())["terrabrasilis"])
    if args.dest_dir:
        cfg["dest_dir"] = args.dest_dir
    if args.verify:
        verify_deter_classes(args.verify, cfg["deter_class_attribute"], cfg["expected_deter_classes"])
        return 0
    names = args.layers or list(cfg["layers"])
    try:
        run_layers(cfg, names, args.mode, requests.Session())
    except requests.RequestException as err:
        print(f"download failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

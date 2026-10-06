"""Read-only Phase 0 inspection of an unzipped BraDD-S1TS archive: writes a Markdown report and CSVs.

Usage: python scripts/inspect_bradd.py --config configs/data/bradd.yaml [--root DIR] [--out-dir DIR]
       [--n-samples N] [--label TEXT]
`--root` is the unzipped archive top (meta.csv may sit in a sub-folder). Nothing under it is modified.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import inspection as ins  # noqa: E402
from src.data.samples import load_sample, read_meta  # noqa: E402


def collect(top: Path, cfg: Mapping, split_column: str) -> dict[str, pd.DataFrame | Path | bool]:
    """Run every inspection step and return the tables keyed by CSV name."""
    root = ins.find_dataset_root(top)
    meta = read_meta(root)
    files = ins.list_archive(top)
    samples, schema = ins.inspect_samples(root, meta, int(cfg["n_samples"]), int(cfg["seed"]))
    first_keys = list(load_sample(root / "Samples" / meta["file"].iloc[0]).keys())
    return {
        "dataset_root": root,
        "close_stats_present": any(Path(p).name == "close_stats.pt" for p in files["path"]),
        "files": files,
        "files_by_extension": ins.count_by(files, "extension"),
        "files_by_directory": ins.count_by(files, "directory"),
        "non_sample_files": ins.non_sample_files(files),
        "geolocation_hits": ins.geolocation_hits(list(meta.columns), first_keys, files,
                                                 cfg["geo_pattern"], cfg["geo_extensions"]),
        "meta_columns": pd.DataFrame({"column": meta.columns, "dtype": meta.dtypes.astype(str).values}),
        "samples": samples,
        "sample_keys": schema,
        "split_counts": ins.split_counts(meta, split_column, cfg["paper_split_counts"],
                                         cfg["notebook_split_counts"]),
        "sampling_type_counts": meta["sampling_type"].value_counts().rename_axis("sampling_type").reset_index(),
    }


def _describe(samples: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Summary statistics of selected per-sample columns."""
    present = [c for c in columns if c in samples.columns]
    return samples[present].describe().T.reset_index().rename(columns={"index": "quantity"}).round(3)


def render_report(tables: Mapping, label: str, top: Path) -> str:
    """Markdown report text from the collected tables."""
    s = tables["samples"]
    dates = ["T", "n_label_dates", "first_label_minus_alert", "last_label_minus_alert",
             "first_image_minus_alert", "last_image_minus_alert", "image_span_days", "median_revisit_days"]
    values = [c for c in s.columns if c.startswith(("VV_", "VH_"))] + ["nonfinite"]
    transitions = s[["n_0to0", "n_0to1", "n_1to0", "n_1to1"]].sum()
    share = (100 * transitions / transitions.sum()).round(3)
    hits = tables["geolocation_hits"]
    parts = [
        f"# BraDD-S1TS inspection report ({label})", "",
        f"Archive top: `{top}`; dataset root (meta.csv): `{tables['dataset_root']}`.",
        f"Samples opened: {len(s)} of {int(tables['split_counts']['meta_csv'].sum())}.", "",
        "## Files", f"Total files: {len(tables['files'])}.", "",
        ins.markdown_table(tables["files_by_extension"]), "", "By directory:", "",
        ins.markdown_table(tables["files_by_directory"], 30), "", "Every file outside Samples/:", "",
        ins.markdown_table(tables["non_sample_files"]), "",
        f"`close_stats.pt` ships in the archive: **{tables['close_stats_present']}**.", "",
        "## Geolocation search",
        "No candidate found." if hits.empty else ins.markdown_table(hits), "",
        "## meta.csv columns", ins.markdown_table(tables["meta_columns"]), "",
        "## Sample keys, types, shapes", ins.markdown_table(tables["sample_keys"]), "",
        "## T and date spacing (days, relative to meta `date`)", ins.markdown_table(_describe(s, dates)), "",
        "## dB values per channel", ins.markdown_table(_describe(s, values)), "",
        "## Label transitions (first -> last label mask, opened samples)",
        ins.markdown_table(pd.DataFrame({"transition": share.index, "pixels": transitions.values,
                                         "percent": share.values})), "",
        "## Split counts", ins.markdown_table(tables["split_counts"]), "",
        "## Sampling types", ins.markdown_table(tables["sampling_type_counts"]), "",
    ]
    return "\n".join(parts)


def write_outputs(tables: Mapping, report: str, out_dir: Path) -> None:
    """Write report.md plus one CSV per table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(out_dir / f"{name}.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=None, help="unzipped archive top; default: config `root`")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--label", default="real data", help="text shown in the report title, e.g. SMOKE")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    inspect_cfg = dict(cfg["inspect"])
    if args.n_samples is not None:
        inspect_cfg["n_samples"] = args.n_samples
    top = Path(args.root or cfg["root"])
    tables = collect(top, inspect_cfg, cfg["split_column"])
    out_dir = Path(args.out_dir or inspect_cfg["out_dir"])
    write_outputs(tables, render_report(tables, args.label, top), out_dir)
    print(f"wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

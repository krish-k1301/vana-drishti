"""QA report writing (markdown, CSV, JSON summary) and the pilot-passed gate for full exports."""
from __future__ import annotations

import json
import os

import pandas as pd

SUMMARY_FILE = "qa_summary.json"


def write_qa_outputs(qa_dir: str, table: pd.DataFrame, passed: bool, reasons: list[str], conversion: dict,
                     spot_check: pd.DataFrame, config_hash: str) -> dict[str, str]:
    """Write qa_checks.csv, qa_spot_check.csv, qa_summary.json and qa_report.md; return their paths."""
    os.makedirs(qa_dir, exist_ok=True)
    paths = {name: os.path.join(qa_dir, name) for name in
             ("qa_checks.csv", "qa_spot_check.csv", SUMMARY_FILE, "qa_report.md")}
    table.to_csv(paths["qa_checks.csv"], index=False)
    spot_check.to_csv(paths["qa_spot_check.csv"], index=False)
    summary = {"passed": passed, "reasons": reasons, "config_hash": config_hash, "n_samples": int(len(table)),
               "n_failed": int((~table["ok"]).sum()) if len(table) else 0,
               "converted": conversion["converted"], "rejected": conversion["rejected"],
               "missing": conversion["missing"]}
    with open(paths[SUMMARY_FILE], "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    with open(paths["qa_report.md"], "w", encoding="utf-8") as handle:
        handle.write(render_markdown(table, summary, spot_check))
    return paths


def render_markdown(table: pd.DataFrame, summary: dict, spot_check: pd.DataFrame) -> str:
    """Markdown QA report: verdict, per-check failure counts, rejections and the NICFI spot-check list."""
    lines = ["# Pilot QA report", "", f"Verdict: {'PASSED' if summary['passed'] else 'FAILED'}",
             f"Config hash: `{summary['config_hash']}`", f"Samples checked: {summary['n_samples']}", ""]
    lines += [f"- {reason}" for reason in summary["reasons"]]
    lines += ["", "## Failures per check", "", "| check | failed |", "|---|---|"]
    for col in [c for c in table.columns if c not in ("patch_id", "sampling_type", "ok")]:
        lines.append(f"| {col} | {int((~table[col]).sum())} |")
    reasons = pd.Series(list(summary["rejected"].values()), dtype=str).value_counts()
    lines += ["", "## Conversion rejections", ""] + [f"- {k}: {v}" for k, v in reasons.items()]
    lines += [f"- missing raw files: {len(summary['missing'])}", "", "## NICFI spot-check sample", ""]
    lines += markdown_table(spot_check)
    return "\n".join(lines) + "\n"


def markdown_table(frame: pd.DataFrame) -> list[str]:
    """Render a DataFrame as markdown table lines (no optional dependencies)."""
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return lines


def pilot_passed(qa_dir: str, config_hash: str) -> tuple[bool, str]:
    """Return (ok, reason): a pilot QA summary exists, passed, and matches the current config hash."""
    path = os.path.join(qa_dir, SUMMARY_FILE)
    if not os.path.exists(path):
        return False, f"no pilot QA summary at {path}; run with --pilot first"
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("passed"):
        return False, f"pilot QA did not pass: {summary.get('reasons')}"
    if summary.get("config_hash") != config_hash:
        return False, "pilot QA was run with different S1/label/window settings; re-run the pilot"
    return True, "pilot QA passed"

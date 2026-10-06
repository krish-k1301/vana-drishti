"""CLI: plan, export, convert and QA a dated set. Usage: gee/run_export.py --config configs/gee/<x>.yaml [--pilot]."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gee.auth import EarthEngineUnavailable, initialize  # noqa: E402
from gee.config import config_hash, load_config, mode_paths, unresolved_placeholders  # noqa: E402
from gee.convert import convert_all  # noqa: E402
from gee.pipeline import export_all  # noqa: E402
from gee.plan import plan_patches  # noqa: E402
from gee.qa import qa_passed, random_qa_sample, run_qa  # noqa: E402
from gee.qa_report import pilot_passed, write_qa_outputs  # noqa: E402
from gee.spec import load_specs, save_specs  # noqa: E402
from gee.tasklog import TaskLog  # noqa: E402

EXIT_BLOCKED = 2
EXIT_REFUSED = 3
EXIT_QA_FAILED = 4


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/gee/<region>.yaml")
    parser.add_argument("--pilot", action="store_true", help="small pilot bbox; required to pass QA before full")
    return parser.parse_args(argv)


def planned_specs(cfg: dict, mode: str, plan_path: str) -> list:
    """Load the saved plan for resume, or plan the patches once and save them."""
    if os.path.exists(plan_path):
        return load_specs(plan_path)
    specs = plan_patches(cfg, cfg["modes"][mode])
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    save_specs(specs, plan_path)
    return specs


def run(cfg: dict, mode: str) -> int:
    """Plan, export (resume-safe), convert to BraDD format and, for pilots, write the QA report."""
    paths = mode_paths(cfg, mode)
    specs = planned_specs(cfg, mode, paths["plan"])
    backend = cfg["export"]["backend"][mode]
    counts = export_all(cfg, specs, paths["raw_dir"], TaskLog(paths["task_log"]), backend)
    print(f"export ({backend}): {counts}")
    conversion = convert_all(cfg, specs, paths["raw_dir"], paths["out_root"])
    print(f"converted {conversion['converted']}, rejected {len(conversion['rejected'])}, "
          f"missing {len(conversion['missing'])}")
    if mode != "pilot":
        return 0
    table = run_qa(paths["out_root"], cfg["qa"])
    passed, reasons = qa_passed(table, cfg["qa"])
    spot = random_qa_sample(paths["out_root"], cfg["qa"]["spot_check_size"], cfg["split"]["seed"])
    outputs = write_qa_outputs(paths["qa_dir"], table, passed, reasons, conversion, spot, config_hash(cfg))
    print(f"pilot QA {'PASSED' if passed else 'FAILED'}: {outputs['qa_report.md']}")
    return 0 if passed else EXIT_QA_FAILED


def main(argv: list[str] | None = None) -> int:
    """Entry point: fail fast without credentials; refuse a full export until the pilot QA passed."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    mode = "pilot" if args.pilot else "full"
    try:
        initialize()
    except EarthEngineUnavailable as error:
        print(error, file=sys.stderr)
        return EXIT_BLOCKED
    placeholders = unresolved_placeholders(cfg, mode)
    if placeholders:
        print(f"REFUSED: unresolved VERIFY placeholders in {args.config}: {placeholders}", file=sys.stderr)
        return EXIT_REFUSED
    if mode == "full":
        ok, reason = pilot_passed(mode_paths(cfg, "pilot")["qa_dir"], config_hash(cfg))
        if not ok:
            print(f"REFUSED: full export needs a passed pilot QA ({reason})", file=sys.stderr)
            return EXIT_REFUSED
    return run(cfg, mode)


if __name__ == "__main__":
    sys.exit(main())

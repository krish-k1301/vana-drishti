"""CLI tests: BLOCKED without credentials, placeholder refusal, and the full-export pilot-QA gate."""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from gee import auth, run_export
from gee.config import config_hash, load_config, mode_paths

REPO = Path(__file__).resolve().parents[1]


def clean_env(home: Path) -> dict:
    """Environment without any Earth Engine credentials."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("EE_")}
    env["HOME"] = str(home)
    return env


def test_cli_pilot_without_credentials_prints_blocked(tmp_path):
    result = subprocess.run([sys.executable, "gee/run_export.py", "--config", "configs/gee/amazon_dated.yaml",
                             "--pilot"], cwd=REPO, env=clean_env(tmp_path), capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == run_export.EXIT_BLOCKED
    assert result.stderr.startswith("BLOCKED: no Earth Engine credentials")


def test_initialize_raises_typed_error_and_detects_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "user_credentials_path", lambda: str(tmp_path / "none"))
    assert auth.credential_source({}) is None
    try:
        auth.initialize({})
    except auth.EarthEngineUnavailable as error:
        assert str(error) == auth.BLOCKED_MESSAGE
    else:
        raise AssertionError("expected EarthEngineUnavailable")
    key = tmp_path / "key.json"
    key.write_text("{}")
    env = {auth.SERVICE_ACCOUNT_ENV: "sa@x.iam.gserviceaccount.com", auth.KEY_FILE_ENV: str(key)}
    assert auth.credential_source(env) == "service_account"
    assert auth.credential_source({auth.SERVICE_ACCOUNT_ENV: "sa", auth.KEY_FILE_ENV: str(tmp_path / "x")}) is None


def test_placeholders_refuse_the_amazon_pilot(monkeypatch, capsys):
    monkeypatch.setattr(run_export, "initialize", lambda: "user")
    code = run_export.main(["--config", "configs/gee/amazon_dated.yaml", "--pilot"])
    assert code == run_export.EXIT_REFUSED
    assert "labels.fogo.asset" in capsys.readouterr().err


def resolved_config(tmp_path: Path) -> Path:
    """Congo config with the batch bucket filled in and outputs under tmp_path."""
    cfg = load_config("configs/gee/congo_pilot.yaml")
    cfg["export"]["batch"]["bucket"] = "test-bucket"
    for mode in ("pilot", "full"):
        cfg["modes"][mode]["out_root"] = str(tmp_path / mode)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_full_export_refused_until_pilot_qa_passed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_export, "initialize", lambda: "user")
    monkeypatch.setattr(run_export, "run", lambda cfg, mode: 0)
    path = resolved_config(tmp_path)
    assert run_export.main(["--config", str(path)]) == run_export.EXIT_REFUSED
    assert "REFUSED: full export needs a passed pilot QA" in capsys.readouterr().err
    cfg = load_config(str(path))
    qa_dir = Path(mode_paths(cfg, "pilot")["qa_dir"])
    qa_dir.mkdir(parents=True)
    (qa_dir / "qa_summary.json").write_text(json.dumps({"passed": True, "config_hash": config_hash(cfg)}))
    assert run_export.main(["--config", str(path)]) == 0
    cfg["window"]["days_after"] = 90
    path.write_text(yaml.safe_dump(cfg))
    assert run_export.main(["--config", str(path)]) == run_export.EXIT_REFUSED

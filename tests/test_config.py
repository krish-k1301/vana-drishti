"""src.config: base inheritance, _replace, overrides, require; the shipped experiment configs."""
import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

from src.config import load_config, require
from src.losses.build_loss import build_loss
from src.models.build_model import build_model

MODEL_CONFIGS = ["convlstm", "convgru", "unet3d", "utae_seq2seq"]


def write(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path`, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content))
    return path


def test_base_inheritance_nested_and_relative(tmp_path: Path) -> None:
    """Top-level and nested `base:` resolve relative to the including file and deep-merge."""
    write(tmp_path / "data" / "set.yaml", {"root": "/x", "batch": {"size": 4, "workers": 2}})
    write(tmp_path / "parent.yaml", {"seed": 1, "data": {"base": "data/set.yaml", "batch": {"size": 8}},
                                     "model": {"params": {"a": 1, "b": 2}}})
    child = write(tmp_path / "sub" / "child.yaml", {"base": "../parent.yaml", "seed": 2,
                                                   "model": {"params": {"_replace": True, "c": 3}}})
    cfg = load_config(child)
    assert cfg == {"seed": 2, "data": {"root": "/x", "batch": {"size": 8, "workers": 2}},
                   "model": {"params": {"c": 3}}}


def test_replace_marker_removed_without_base(tmp_path: Path) -> None:
    """A `_replace` marker with nothing to replace never leaks into the loaded config."""
    cfg = load_config(write(tmp_path / "a.yaml", {"model": {"params": {"_replace": True, "x": 1}}}))
    assert cfg == {"model": {"params": {"x": 1}}}


def test_overrides(tmp_path: Path) -> None:
    """Dotted overrides are YAML-parsed and may only touch existing keys."""
    path = write(tmp_path / "a.yaml", {"trainer": {"max_epochs": 200, "widths": [1]}, "name": "x"})
    cfg = load_config(path, ["trainer.max_epochs=2", "trainer.widths=[3, 4]", "name=y"])
    assert cfg == {"trainer": {"max_epochs": 2, "widths": [3, 4]}, "name": "y"}
    with pytest.raises(KeyError, match="max_epoch"):
        load_config(path, ["trainer.max_epoch=2"])
    with pytest.raises(KeyError, match="no 'optim'"):
        load_config(path, ["optim.lr=0.1"])
    with pytest.raises(ValueError, match="key=value"):
        load_config(path, ["trainer.max_epochs"])


def test_require_and_cycles(tmp_path: Path) -> None:
    """require returns nested values or names the missing part; inheritance cycles are refused."""
    assert require({"a": {"b": {"c": 0}}}, "a.b.c") == 0
    with pytest.raises(KeyError, match="no 'x' under 'a.b'"):
        require({"a": {"b": {"c": 0}}}, "a.b.x")
    write(tmp_path / "one.yaml", {"base": "two.yaml"})
    write(tmp_path / "two.yaml", {"base": "one.yaml"})
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "one.yaml")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


@pytest.fixture(name="configs_dir")
def fixture_configs_dir(tmp_path: Path) -> Path:
    """Copy of configs/ with a stub data/bradd.yaml when the data-agent's file is absent."""
    target = tmp_path / "configs"
    shutil.copytree(REPO / "configs", target)
    if not (target / "data" / "bradd.yaml").exists():
        write(target / "data" / "bradd.yaml", {"batch_size": 4})
    return target


def test_baseline_configs(configs_dir: Path) -> None:
    """Baselines carry the PRD 4.2 recipe; the focal one differs only in loss and name."""
    ce = load_config(configs_dir / "baseline_utae.yaml")
    focal = load_config(configs_dir / "baseline_utae_focal.yaml")
    assert ce["seed"] == 42 and require(ce, "model.params.n_head") == 8
    assert require(ce, "optim.weight_decay") == 1.0e-6 and require(ce, "optim.lr") == 1.0e-3
    assert require(ce, "data.batch_size") == 4 and ce["loss"] == {"name": "cross_entropy"}
    assert focal["loss"] == {"name": "focal", "alpha": 0.75, "gamma": 1.0}
    stripped = {k: v for k, v in focal.items() if k not in ("loss", "experiment_name")}
    assert stripped == {k: v for k, v in ce.items() if k not in ("loss", "experiment_name")}
    for cfg in (ce, focal):
        assert type(build_model(cfg["model"])).__name__ == "SegmentNetwork"
        build_loss(cfg["loss"])


@pytest.mark.parametrize("name", MODEL_CONFIGS)
def test_model_configs_build(configs_dir: Path, name: str) -> None:
    """Each benchmark config swaps only the model, keeps focal 0.75/1.0, and builds."""
    cfg = load_config(configs_dir / "models" / f"{name}.yaml")
    focal = load_config(configs_dir / "baseline_utae_focal.yaml")
    assert cfg["experiment_name"] == name and cfg["model"]["name"] == name
    assert cfg["loss"] == {"name": "focal", "alpha": 0.75, "gamma": 1.0}
    assert {k: v for k, v in cfg.items() if k not in ("model", "experiment_name")} == \
        {k: v for k, v in focal.items() if k not in ("model", "experiment_name")}
    assert "_replace" not in cfg["model"]["params"]
    build_model(cfg["model"])

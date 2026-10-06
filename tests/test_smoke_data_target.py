"""SMOKE target-biome copy (src.smoke_data.make_target_biome): Phase 6 GEE export layout on synthetic data."""
from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.data.loaders import build_dataset
from src.data.samples import EPOCH, load_sample, read_meta
from src.data.stats import load_or_compute_stats
from src.smoke_data import BRAZIL_ONLY_KEYS, make_target_biome, write_smoke_dataset

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "smoke" / "synthetic_target_biome.yaml"


@pytest.fixture(scope="module", name="pair")
def fixture_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A tiny SMOKE dated set and its target-biome copy."""
    tmp = tmp_path_factory.mktemp("target")
    source = write_smoke_dataset(tmp / "dated", {"train": 3, "validation": 2, "test": 3}, 0, dated=True)
    positive = load_config(CONFIG)["target_biome"]["positive_class_label"]
    return source, make_target_biome(source, tmp / "target", positive)


def test_keys_stripped_shapes_and_hansen_labels(pair: tuple[Path, Path]) -> None:
    """Brazil-only keys gone, tensors keep their shapes, label is the Hansen stack, RADD kept."""
    source, target = pair
    for name in read_meta(source)["file"]:
        before, after = load_sample(source / "Samples" / name), load_sample(target / "Samples" / name)
        assert not set(BRAZIL_ONLY_KEYS) & set(after)
        assert set(after) == set(before) - set(BRAZIL_ONLY_KEYS)
        assert torch.equal(after["label"], before["label_hansen"]) and torch.equal(after["label_hansen"],
                                                                                    before["label_hansen"])
        for key in ("image", "label", "radd_date", "event_mask", "ref_day"):
            assert after[key].shape == before[key].shape
        assert torch.equal(after["image"], before["image"]) and torch.equal(after["radd_date"], before["radd_date"])
        assert after["deter_class"] == ("hansen_loss" if before["deter_class"] else "")


def test_ref_day_rebuilds_hansen_labels(pair: tuple[Path, Path]) -> None:
    """ref_day is the first Hansen-positive label date: cumulative labels rebuilt from it equal `label`."""
    _, target = pair
    for path in (target / "Samples").glob("*.pt"):
        sample = load_sample(path)
        ref = sample["ref_day"].long()
        days = torch.tensor([(d - EPOCH).days for d in sample["label_dates"]])
        rebuilt = (ref[None] >= 0) & (ref[None] <= days[:, None, None])
        assert torch.equal(rebuilt.long(), sample["label"])


def test_meta_and_marker(pair: tuple[Path, Path]) -> None:
    """meta.csv keeps every row; positives carry hansen_loss, no DETER date; SMOKE marker; no stats copied."""
    source, target = pair
    before, after = read_meta(source), read_meta(target)
    assert list(after["file"]) == list(before["file"])
    assert list(after["deter_class"]) == ["hansen_loss" if c else "" for c in before["deter_class"]]
    assert (after["event_date"] == "").all()
    assert (target / "SMOKE.txt").read_text().startswith("SMOKE")
    assert not list(target.glob("*stats*"))


def test_loads_as_dated_dataset(pair: tuple[Path, Path], tmp_path: Path) -> None:
    """The copy loads through the dated loader with no DETER event day and the Hansen labels as targets."""
    _, target = pair
    data = load_config(REPO / "configs" / "data" / "dated_amazon.yaml",
                       [f"root={target}", f"stats_path={tmp_path / 'stats.pt'}", "num_workers=0",
                        "prefix_truncation=null"])
    load_or_compute_stats(target, data["split_column"], data["stats_path"])
    dataset = build_dataset({**data, "seed": 0}, "test")
    item = dataset[0]
    stored = load_sample(target / "Samples" / dataset.meta["file"][0])
    assert int(item["EventDay"]) == -1 and bool((item["BurnDay"] == -1).all())
    assert torch.equal(item["Targets"].long(), stored["label_hansen"])


def test_refuses_non_smoke_source_and_non_empty_target(pair: tuple[Path, Path], tmp_path: Path) -> None:
    """Only SMOKE sources, never into a non-empty directory."""
    source, target = pair
    with pytest.raises(ValueError, match="SMOKE"):
        make_target_biome(tmp_path, tmp_path / "out", "hansen_loss")
    with pytest.raises(FileExistsError):
        make_target_biome(source, target, "hansen_loss")

"""DatedDataset extra keys and PrefixTruncation semantics on the synthetic dated fixture."""
from __future__ import annotations

import datetime as dt

import pytest
import torch

from src.data import DatedDataset, PrefixTruncation, build_dataloaders, collate_batch
from src.data.prefix import reference_days
from src.data.samples import EPOCH
from tests.fixtures import data_config


def _positive_index(ds: DatedDataset) -> int:
    return int(ds.meta.index[ds.meta["deter_class"] != ""][0])


def test_extra_keys_on_image_day_origin(dated_root) -> None:
    """EventDay/RaddDay/EventMask are on the ImageDays origin."""
    ds = DatedDataset(dated_root, "train", normalization="none")
    i = _positive_index(ds)
    raw, item = ds.load_raw(i), ds[i]
    origin = min(raw["image_dates"] + raw["label_dates"])
    assert item["EventDay"].item() == (raw["event_date"] - origin).days + 1
    inside = raw["radd_date"] >= 0
    expected = raw["radd_date"][inside].long() - (origin - EPOCH).days + 1
    assert torch.equal(item["RaddDay"][inside], expected) and (item["RaddDay"][~inside] == -1).all()
    assert torch.equal(item["EventMask"], raw["event_mask"]) and item["BurnDay"].dtype == torch.long
    assert item["Targets"].shape[0] == len(raw["label_dates"]) > 2


def test_negative_has_no_event(dated_root) -> None:
    """Negatives carry EventDay -1, no burn and an empty event mask."""
    ds = DatedDataset(dated_root, "train", normalization="none")
    neg = int(ds.meta.index[ds.meta["deter_class"] == ""][0])
    item = ds[neg]
    assert item["EventDay"].item() == -1 and (item["BurnDay"] == -1).all() and item["EventMask"].sum() == 0


@pytest.mark.parametrize("anchor", ["deter", "burn"])
def test_prefix_at_semantics(dated_root, anchor) -> None:
    """No image after t_c, monotone labels, last label = pre OR event day <= t_c."""
    ds = DatedDataset(dated_root, "train", normalization="none")
    item = ds[_positive_index(ds)]
    trunc = PrefixTruncation(anchor, min_dates=3, label_interval_days=30, seed=0)
    event_days = trunc.pixel_event_days(item)
    for cutoff in item["ImageDays"][2:].tolist():
        out = trunc.prefix_at(item, cutoff)
        assert out["ImageDays"].max() == cutoff and (out["ImageDays"] <= cutoff).all()
        assert out["Images"].shape[0] == int((item["ImageDays"] <= cutoff).sum())
        assert out["TargetDays"][-1] == cutoff and len(out["TargetDays"]) >= 2
        assert (out["Targets"][1:] >= out["Targets"][:-1]).all(), "labels monotone in time"
        expected = item["Targets"][0].bool() | ((event_days >= 0) & (event_days <= cutoff))
        assert torch.equal(out["Targets"][-1].bool(), expected)
    if anchor == "burn":
        assert ((event_days >= 0) <= item["EventMask"].bool()).all()


def test_label_days_spacing() -> None:
    """Label days step back from t_c by the interval and stay >= 1."""
    trunc = PrefixTruncation("deter", 1, 30, seed=0)
    assert trunc.label_days(95).tolist() == [5, 35, 65, 95]
    assert trunc.label_days(20).tolist() == [1, 20]


def test_random_cutoff_respects_min_dates(dated_root) -> None:
    """Random cutoffs keep at least min_dates images and vary."""
    ds = DatedDataset(dated_root, "train", normalization="none",
                      prefix=PrefixTruncation("deter", min_dates=5, label_interval_days=30, seed=1))
    lengths = {ds[0]["ImageDays"].shape[0] for _ in range(20)}
    assert min(lengths) >= 5 and len(lengths) > 1


def test_dated_loader_collates_variable_labels(dated_root, tmp_path) -> None:
    """Dated loader pads variable label counts; last label day = last image day."""
    cfg = data_config("dated_amazon", root=str(dated_root), stats_path=str(tmp_path / "d.pt"), batch_size=4,
                      num_workers=0, seed=0)
    batch = next(iter(build_dataloaders(cfg)["train"]))
    assert batch["EventDay"].shape == (4,) and batch["BurnDay"].shape == (4, 48, 48)
    assert batch["Targets"].shape[1] == batch["TargetDays"].shape[1]
    for b in range(4):
        valid = batch["TargetDays"][b] > 0
        last_image_day = batch["ImageDays"][b][~batch["PadMask"][b]].max()
        assert batch["TargetDays"][b][valid].max() == last_image_day


def test_collate_pads_label_axis() -> None:
    """Targets/TargetDays are end-padded with 0 along the label axis."""
    items = [{"ImageDays": torch.arange(1, n + 1), "TargetDays": torch.arange(1, t + 1),
              "Targets": torch.ones(t, 2, 2, dtype=torch.long)} for n, t in ((3, 2), (5, 4))]
    batch = collate_batch(items)
    assert batch["TargetDays"].tolist() == [[1, 2, 0, 0], [1, 2, 3, 4]]
    assert batch["Targets"][0, 2:].sum() == 0 and batch["PadMask"].tolist()[0] == [False] * 3 + [True] * 2


def test_ref_day_reproduces_stored_labels(dated_root) -> None:
    """Deter-anchored labels rebuilt at a stored label day equal the stored mask, neighbour polygons included."""
    ds = DatedDataset(dated_root, "train", normalization="none")
    trunc = PrefixTruncation("deter", min_dates=1, label_interval_days=30, seed=0)
    for i in range(len(ds)):
        item = ds[i]
        outside = (item["RefDay"] >= 0) & ~item["EventMask"].bool()
        assert ds.meta.loc[i, "deter_class"] == "" or outside.any(), "positives carry a second polygon"
        for k in range(1, len(item["TargetDays"])):
            cut = trunc.prefix_at(item, int(item["TargetDays"][k]))
            assert torch.equal(cut["Targets"][-1], item["Targets"][k])


def test_burn_day_is_a_month_end(dated_root) -> None:
    """Burn months are encoded as their last day, so BurnDay never precedes the end of the burn month."""
    ds = DatedDataset(dated_root, "train", normalization="none")
    for i in range(len(ds)):
        raw = ds.load_raw(i)
        for value in raw["burn_month"][raw["burn_month"] >= 0].unique().tolist():
            day = EPOCH + dt.timedelta(days=int(value))
            assert (day + dt.timedelta(days=1)).day == 1


def test_derived_ref_day_without_key(dated_root) -> None:
    """Without `ref_day` the dataset derives RefDay from EventDay/EventMask and the stored label onsets."""
    ds = DatedDataset(dated_root, "train", normalization="none")
    item = ds[0]
    stripped = {k: v for k, v in item.items() if k != "RefDay"}
    derived = reference_days(stripped)
    stored_positive = item["Targets"][-1].bool() & ~item["Targets"][0].bool()
    assert torch.equal(derived >= 0, stored_positive | (item["EventMask"].bool() & (item["EventDay"] >= 0)))
    assert ((derived >= item["RefDay"]) | (derived < 0)).all(), "derived days are never earlier than RefDay"

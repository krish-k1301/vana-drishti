"""Tests for Earth Engine label builders (mocked ee), patch-image band schema and patch planning."""
import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import Point, box

from gee import labels, pipeline, plan
from gee.config import load_config
from gee.dates import Window
from gee.grid import lonlat_bounds, make_grid
from gee.sampling import block_index, split_of_block

WINDOW = Window(dt.date(2021, 6, 1), dt.date(2022, 9, 29))


def test_label_image_concatenates_requested_sources(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(labels, "ee", fake)
    cfg = load_config("configs/gee/amazon_dated.yaml")["labels"]
    labels.label_image(cfg, ["radd", "modis"], "region", WINDOW)
    assert len(fake.Image.cat.call_args.args[0]) == 2
    fake.ImageCollection.assert_any_call("projects/radar-wur/raddalert/v1")
    fake.ImageCollection.assert_any_call("MODIS/061/MCD64A1")
    with pytest.raises(KeyError):
        labels.label_image(cfg, ["nicfi"], "region", WINDOW)


def test_fogo_selects_one_band_per_window_year(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(labels, "ee", fake)
    labels.fogo_image({"asset": "A", "band_template": "burned_{year}"}, WINDOW.years)
    fake.Image.return_value.select.assert_called_once_with(["burned_2021", "burned_2022"], ["fogo_2021", "fogo_2022"])
    names = pipeline.label_band_names({"pull": ["radd", "fogo", "modis"]}, WINDOW.years)
    assert names == ["radd_alert", "radd_date", "fogo_2021", "fogo_2022", "modis_burn"]


def test_class_codes_and_positive_class(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(labels, "ee", fake)
    cfg = load_config("configs/gee/congo_pilot.yaml")
    codes = labels.class_codes(cfg["negatives"])
    assert codes == {"forest": 1, "oldDeforest": 2, "herbaceous": 3, "agriculture": 4, "shrubs": 5}
    labels.class_image(cfg["negatives"], cfg["labels"], WINDOW, True)
    where_codes = [c.args[1] for c in fake.Image.constant.return_value.where.call_args_list]
    assert where_codes[0] == 1
    chain = fake.Image.constant.return_value
    for _ in range(2 + len(cfg["negatives"]["nonforest_classes"])):
        chain = chain.where.return_value
    assert chain.where.call_args.args[1] == labels.POSITIVE_CODE
    collections = [c.args[0] for c in fake.ImageCollection.call_args_list]
    assert cfg["labels"]["radd"]["collection"] not in collections


def fake_points(cfg: dict, bbox: list[float]) -> dict:
    """FeatureCollection-like dict: a grid of points of every class; every point carries Hansen lossyear 22."""
    features = []
    codes = list(labels.class_codes(cfg["negatives"]).values()) + [labels.POSITIVE_CODE]
    for i in range(240):
        lon = bbox[0] + 0.05 + (i % 20) * 0.045
        lat = bbox[1] + 0.05 + (i // 20) * 0.075
        props = {"cls": codes[i % len(codes)], "hansen_lossyear": 22}
        features.append({"geometry": Point(lon, lat).__geo_interface__, "properties": props})
    return {"features": features}


def fake_states(bbox: list[float]) -> dict:
    """Two state polygons splitting the bbox at mid-longitude (west 'Para', east 'Mato Grosso')."""
    mid = (bbox[0] + bbox[2]) / 2
    return {"features": [
        {"geometry": box(bbox[0] - 1, bbox[1] - 1, mid, bbox[3] + 1).__geo_interface__,
         "properties": {"ADM1_NAME": "Para"}},
        {"geometry": box(mid, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1).__geo_interface__,
         "properties": {"ADM1_NAME": "Mato Grosso"}}]}


def assert_no_overlap(specs: list) -> None:
    """No two planned patch footprints intersect (in any split or class)."""
    grids = [s.grid({"patch_size": 48, "pixel_size_m": 10.0}) for s in specs]
    for i, a in enumerate(grids):
        for b in grids[i + 1:]:
            if a.epsg == b.epsg:
                ax0, ay0, ax1, ay1 = a.bounds_xy
                bx0, by0, bx1, by1 = b.bounds_xy
                assert ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0


def test_spacing_below_patch_diagonal_is_rejected(tmp_path):
    cfg = yaml.safe_load(Path("configs/gee/amazon_dated.yaml").read_text(encoding="utf-8"))
    cfg["split"]["min_center_distance_m"] = 480
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="diagonal"):
        load_config(str(tmp_path / "c.yaml"))


def test_plan_hansen_split_by_blocks(monkeypatch):
    cfg = load_config("configs/gee/congo_pilot.yaml")
    mode = cfg["modes"]["pilot"]
    monkeypatch.setattr(labels, "class_image", MagicMock())
    monkeypatch.setattr(labels, "sample_class_points", lambda *a, **k: fake_points(cfg, mode["bbox"]))
    specs = plan.plan_patches(cfg, mode)
    positives = [s for s in specs if s.is_positive]
    assert 0 < len(positives) <= mode["max_positives"]
    assert all(s.event_date == dt.date(2022, 12, 31) for s in specs)
    assert {s.state for s in specs} == {cfg["region_label"]}
    assert_no_overlap(specs)
    for s in specs:
        bounds = lonlat_bounds(make_grid(s.lon, s.lat, 48, 10.0))
        block = block_index(bounds[0], bounds[1], cfg["split"]["block_size_deg"])
        assert s.split == split_of_block(block, cfg["split"]["seed"], cfg["split"]["fractions"])
    assert len({s.patch_id for s in specs}) == len(specs)
    assert {s.sampling_type for s in specs} - {"positive"} <= set(cfg["negatives"]["type_weights"])


def test_plan_deter_positives_and_event_screening(tmp_path, monkeypatch):
    cfg = load_config("configs/gee/amazon_dated.yaml")
    mode = {**cfg["modes"]["pilot"], "max_positives": 5}
    polys = [box(-55.9 + 0.1 * i, -7.4, -55.895 + 0.1 * i, -7.395) for i in range(6)]
    deter = gpd.GeoDataFrame({"view_date": ["2021-07-0%d" % (i + 1) for i in range(6)],
                              "classname": ["DESMATAMENTO_CR"] * 5 + ["DEGRADACAO"]},
                             geometry=polys, crs=4326)
    deter.to_file(tmp_path / "deter.geojson")
    cfg["events"]["deter"]["path"] = str(tmp_path / "deter.geojson")
    points = fake_points(cfg, mode["bbox"])
    centre = polys[0].centroid
    points["features"].append({"geometry": Point(centre.x, centre.y).__geo_interface__,
                               "properties": {"cls": 1, "hansen_lossyear": 0}})
    monkeypatch.setattr(labels, "class_image", MagicMock())
    monkeypatch.setattr(labels, "sample_class_points", lambda *a, **k: points)
    monkeypatch.setattr(labels, "state_features", lambda *a, **k: fake_states(mode["bbox"]))
    specs = plan.plan_patches(cfg, mode)
    positives = [s for s in specs if s.is_positive]
    assert len(positives) == 5 and all(s.deter_class == "DESMATAMENTO_CR" for s in positives)
    assert all(s.event_wkt and s.polygon_area_ha > 0 for s in positives)
    negatives = [s for s in specs if not s.is_positive]
    assert negatives and all(abs(s.lon - centre.x) > 1e-6 for s in negatives)
    assert {s.event_date for s in negatives} <= {s.event_date for s in positives}
    vocabulary = {"Para", "Mato Grosso", cfg["region_label"]}
    assert {s.state for s in positives} <= vocabulary and {s.state for s in negatives} <= vocabulary
    assert {s.state for s in positives} == {"Para", "Mato Grosso"} == {s.state for s in negatives}
    assert_no_overlap(specs)
    on_event = plan.point_specs(cfg, [{"lon": centre.x, "lat": centre.y}], "forest", [dt.date(2021, 7, 1)], 0, "")
    far = plan.point_specs(cfg, [{"lon": -55.05, "lat": -6.6}], "forest", [dt.date(2021, 7, 1)], 1, "")
    assert plan.drop_near_events(on_event + far, plan.load_deter(cfg), cfg) == far

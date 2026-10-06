"""Tests for the pure grid and sampling logic: UTM grids, rasterisation, block split and leakage, negatives."""
import datetime as dt
from collections import Counter

import numpy as np
from shapely.geometry import Point, box

from gee.grid import area_ha, lonlat_bounds, lonlat_to_rowcol, make_grid, rasterize, utm_epsg
from gee.sampling import (allocate_negatives, assign_block, block_index, parse_sample_points, sample_negative_dates,
                          split_of_block, thin_by_distance)

FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def test_grid_is_snapped_48x48_utm():
    grid = make_grid(-55.4, -7.05, 48, 10.0)
    assert grid.epsg == utm_epsg(-55.4, -7.05) == 32721
    assert grid.x0 % 10 == 0 and grid.y0 % 10 == 0
    xmin, ymin, xmax, ymax = grid.bounds_xy
    assert xmax - xmin == 480 and ymax - ymin == 480
    assert grid.compute_pixels_grid()["affineTransform"]["scaleY"] == -10.0


def test_rasterize_and_rowcol_agree():
    grid = make_grid(25.3, 0.9, 48, 10.0)
    lon0, lat0, lon1, lat1 = lonlat_bounds(grid)
    half = box(lon0 - 1, lat0 - 1, (lon0 + lon1) / 2, lat1 + 1)
    mask = rasterize([half], grid)
    assert mask.shape == (48, 48) and 0.4 < mask.mean() < 0.6
    assert mask[:, 0].all() and not mask[:, -1].any()
    rows, cols = lonlat_to_rowcol(np.array([(lon0 + lon1) / 2]), np.array([(lat0 + lat1) / 2]), grid)
    assert 22 <= rows[0] <= 25 and 22 <= cols[0] <= 25
    assert not rasterize([], grid).any()


def test_area_ha_of_one_km_square_near_equator():
    square = box(25.0, 0.0, 25.0 + 1000 / 111_320, 1000 / 110_574)
    assert abs(area_ha(square) - 100.0) < 2.0


def test_block_split_is_deterministic_and_roughly_proportional():
    blocks = [(ix, iy) for ix in range(-200, -150) for iy in range(-40, 0)]
    first = [split_of_block(b, 42, FRACTIONS) for b in blocks]
    assert first == [split_of_block(b, 42, FRACTIONS) for b in blocks]
    counts = Counter(first)
    assert 0.6 < counts["train"] / len(blocks) < 0.8
    assert counts["validation"] > 0 and counts["test"] > 0
    assert first != [split_of_block(b, 7, FRACTIONS) for b in blocks]


def test_patches_crossing_block_edge_are_dropped_and_splits_never_share_pixels():
    rng = np.random.default_rng(0)
    kept = []
    for lon, lat in zip(rng.uniform(-56, -55, 400), rng.uniform(-7.5, -6.5, 400)):
        bounds = lonlat_bounds(make_grid(lon, lat, 48, 10.0))
        block = assign_block(bounds, 0.5)
        if block is None:
            assert block_index(bounds[0], bounds[1], 0.5) != block_index(bounds[2], bounds[3], 0.5)
            continue
        kept.append((box(*bounds), split_of_block(block, 42, FRACTIONS)))
    assert 0 < len(kept) < 400
    for i, (geom_a, split_a) in enumerate(kept):
        for geom_b, split_b in kept[i + 1:]:
            if split_a != split_b:
                assert not geom_a.intersects(geom_b)


def test_thin_by_distance_keeps_spaced_points_in_order():
    lons = [0.0, 0.001, 0.01, 0.0105]
    lats = [0.0, 0.0, 0.0, 0.0]
    assert thin_by_distance(lons, lats, 240.0) == [0, 2]
    assert thin_by_distance(lons, lats, 0.0) == [0, 1, 2, 3]


def test_negative_dates_follow_positive_distribution():
    positives = [dt.date(2021, 7, 1)] * 90 + [dt.date(2022, 2, 1)] * 10
    dates = sample_negative_dates(positives, 2000, seed=3)
    share = sum(d == dt.date(2021, 7, 1) for d in dates) / len(dates)
    assert 0.85 < share < 0.95
    assert dates == sample_negative_dates(positives, 2000, seed=3)


def test_allocate_negatives_matches_total_and_weights():
    counts = allocate_negatives(100, 1.72, {"oldDeforest": 7557, "forest": 5472, "herbaceous": 1780,
                                            "agriculture": 152, "shrubs": 21})
    assert sum(counts.values()) == 172
    assert counts["oldDeforest"] > counts["forest"] > counts["herbaceous"] >= counts["agriculture"]


def test_parse_sample_points():
    props = {"cls": 99, "radd_date": 21050}
    info = {"features": [{"geometry": Point(25.1, 0.7).__geo_interface__, "properties": props}]}
    assert parse_sample_points(info, "cls", "radd_date") == [{"lon": 25.1, "lat": 0.7, "code": 99, "extra": 21050}]

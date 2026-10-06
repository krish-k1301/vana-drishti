"""Earth Engine label images: RADD, Hansen GFC, MapBiomas Fogo, MODIS MCD64A1 and the sampling class image."""
from __future__ import annotations

import datetime as dt

import ee

from gee.dates import Window, encode_yyddd
from gee.grid import PatchGrid

RADD_ALERT = "radd_alert"
RADD_DATE = "radd_date"
HANSEN_LOSS = "hansen_lossyear"
MODIS_BURN = "modis_burn"
CLASS_BAND = "cls"
POSITIVE_CODE = 99


def fogo_band_name(year: int) -> str:
    """Return the pulled band name holding MapBiomas Fogo burn months of one year."""
    return f"fogo_{year}"


def patch_region(grid: PatchGrid) -> object:
    """Return the patch footprint as a planar ee.Geometry in the grid CRS."""
    return ee.Geometry.Rectangle(list(grid.bounds_xy), grid.crs, False)


def radd_image(cfg: dict, region: object) -> object:
    """Latest RADD alert mosaic with bands radd_alert, radd_date (YYDDD), 0 where no alert."""
    col = (ee.ImageCollection(cfg["collection"])
           .filter(ee.Filter.eq(cfg["layer_property"], cfg["layer_value"]))
           .filter(ee.Filter.eq(cfg["geoidx_property"], cfg["geoidx_value"]))
           .filterBounds(region)
           .sort("system:time_end"))
    bands = col.mosaic().select([cfg["alert_band"], cfg["date_band"]], [RADD_ALERT, RADD_DATE])
    return bands.unmask(0).toInt32()


def hansen_image(cfg: dict) -> object:
    """Hansen GFC lossyear as band hansen_lossyear (0 = no loss)."""
    return ee.Image(cfg["asset"]).select([cfg["lossyear_band"]], [HANSEN_LOSS]).unmask(0).toInt32()


def fogo_image(cfg: dict, years: list[int]) -> object:
    """MapBiomas Fogo monthly burned area, one band per year holding the burn month (0 = unburned)."""
    image = ee.Image(cfg["asset"])
    sources = [cfg["band_template"].format(year=y) for y in years]
    return image.select(sources, [fogo_band_name(y) for y in years]).unmask(0).toInt32()


def modis_image(cfg: dict, window: Window) -> object:
    """MODIS MCD64A1 burn flag over the window (1 if any BurnDate > 0), used only at patch level."""
    band = cfg["band"]
    zero = ee.ImageCollection([ee.Image.constant(0).rename(band).toInt16()])
    col = (ee.ImageCollection(cfg["collection"])
           .filterDate(window.start.isoformat(), (window.end + dt.timedelta(days=1)).isoformat())
           .select(band).merge(zero))
    return col.max().gt(0).rename(MODIS_BURN).toInt32()


def label_image(labels_cfg: dict, pull: list[str], region: object, window: Window) -> object:
    """Concatenate the configured label sources (radd, hansen, fogo, modis) into one image."""
    builders = {
        "radd": lambda: radd_image(labels_cfg["radd"], region),
        "hansen": lambda: hansen_image(labels_cfg["hansen"]),
        "fogo": lambda: fogo_image(labels_cfg["fogo"], window.years),
        "modis": lambda: modis_image(labels_cfg["modis"], window),
    }
    unknown = set(pull) - set(builders)
    if unknown:
        raise KeyError(f"unknown label sources: {sorted(unknown)}")
    return ee.Image.cat([builders[name]() for name in pull])


def class_codes(neg_cfg: dict) -> dict[str, int]:
    """Return sampling_type -> class code: forest 1, oldDeforest 2, then non-forest types in config order."""
    codes = {"forest": 1, "oldDeforest": 2}
    for i, name in enumerate(neg_cfg["nonforest_classes"]):
        codes[name] = 3 + i
    return codes


def class_image(neg_cfg: dict, labels_cfg: dict, period: Window, positives: bool, bbox: list[float]) -> object:
    """Return a sampling image: band cls (codes from class_codes, POSITIVE_CODE for Hansen+RADD events) + radd_date."""
    hansen = ee.Image(labels_cfg["hansen"]["asset"])
    base = labels_cfg["hansen"]["base_year"]
    loss = hansen.select(labels_cfg["hansen"]["lossyear_band"])
    cover = hansen.select(labels_cfg["hansen"]["treecover_band"])
    wc_cfg = neg_cfg["worldcover"]
    land = ee.ImageCollection(wc_cfg["collection"]).first().select(wc_cfg["band"])
    codes = class_codes(neg_cfg)
    cls = ee.Image.constant(0)
    forest = cover.gte(neg_cfg["forest_treecover_min"]).And(loss.eq(0)).And(land.eq(wc_cfg["tree_class"]))
    cls = cls.where(forest, codes["forest"])
    cls = cls.where(loss.gt(0).And(loss.lt(period.start.year - base)), codes["oldDeforest"])
    for name, wc_class in neg_cfg["nonforest_classes"].items():
        open_land = land.eq(wc_class).And(cover.lte(neg_cfg["nonforest_treecover_max"])).And(loss.eq(0))
        cls = cls.where(open_land, codes[name])
    radd = radd_image(labels_cfg["radd"], ee.Geometry.Rectangle(bbox))
    if positives:
        radd_base = labels_cfg["radd"]["base_year"]
        in_years = loss.gte(period.start.year - base).And(loss.lte(period.end.year - base))
        dated = radd.select(RADD_DATE).gte(encode_yyddd(period.start, radd_base)).And(
            radd.select(RADD_DATE).lte(encode_yyddd(period.end, radd_base)))
        cls = cls.where(in_years.And(dated), POSITIVE_CODE)
    return cls.rename(CLASS_BAND).selfMask().toInt32().addBands(radd.select(RADD_DATE))


def sample_class_points(image: object, bbox: list[float], counts: dict[int, int], scale_m: float, seed: int,
                        tile_scale: int) -> dict:
    """Stratified-sample points of the class image inside a lon/lat bbox; returns the FeatureCollection getInfo()."""
    codes = sorted(counts)
    sample = image.stratifiedSample(
        numPoints=0, classBand=CLASS_BAND, region=ee.Geometry.Rectangle(bbox), scale=scale_m, seed=seed,
        classValues=codes, classPoints=[counts[c] for c in codes], geometries=True, tileScale=tile_scale)
    return sample.getInfo()

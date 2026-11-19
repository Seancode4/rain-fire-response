"""Stage 3a': hours of rain during W per window, for PARDIG and IMERG.

Reads  data/interim/windows.parquet
Writes data/interim/rain_duration.parquet  window_id, pardig_dur_h, imerg_dur_h

At each pixel, count the time slots in W with a rate >= 0.1 mm/h, multiply by the slot length,
then average over the footprint (area-weighted at 1 km). Slots are assigned to W by their nominal
time, as in rain.py: IMERG [t, t + 30 min), PARDIG [t - 7.5, t + 7.5 min).
"""

import sys
from pathlib import Path

import ee
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
import ee_common as E  # noqa: E402
from rain import PRODUCTS, SCALE_M  # noqa: E402

MIN_RATE_MM_H = 0.1


def duration(coll, band, mult, off0, off1, a, b):
    """Hours with rate >= MIN_RATE_MM_H among slots whose midpoint falls in [a, b)."""
    ms = 60_000
    mid = (off0 + off1) / 2  # slot midpoint offset from time_start, minutes
    slot_h = (off1 - off0) / 60
    ic = ee.ImageCollection(coll).select(band).filterDate(
        ee.Date(ee.Number(a).subtract(mid * ms)), ee.Date(ee.Number(b).subtract(mid * ms)))
    return ic.map(lambda img: img.multiply(mult).gte(MIN_RATE_MM_H)).sum().multiply(slot_h)


def compute(fc):
    def per_window(f):
        bands = [duration(coll, band, mult, off0, off1, f.get("w_start"), f.get("w_end")).rename(f"{name}_dur_h")
                 for name, (coll, band, mult, off0, off1) in PRODUCTS.items()]
        vals = ee.Image.cat(bands).reduceRegion(ee.Reducer.mean(), f.geometry(), SCALE_M, maxPixels=1e8)
        return ee.Feature(None, vals).set("window_id", f.get("window_id"))
    return fc.map(per_window)


def main(limit=None):
    E.init()
    w = gpd.read_parquet(E.ROOT / "data/interim/windows.parquet")
    if limit:
        w = w.iloc[:limit]
    df = E.run_batches("rain_duration" if not limit else f"rain_duration_test{limit}", w,
                       ["window_id", "w_start", "w_end"], compute)
    if not limit:
        df.sort_values("window_id").to_parquet(E.ROOT / "data/interim/rain_duration.parquet", index=False)
    print(df.drop(columns="window_id").describe().round(2).T.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

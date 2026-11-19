"""Robustness (added after results): PARDIG coarsened to IMERG's 30-minute slots.

Reads  data/interim/windows.parquet
Writes data/interim/rain_pardig30.parquet  window_id, pardig30_{pre,w,post}

Each 30-minute slot [t, t + 30 min), on IMERG's grid, gets the trapezoid integral of PARDIG's instantaneous
rates at t, t + 15 and t + 30 min, and is then accumulated with the same overlap weighting as IMERG.
Per PARDIG image at time s:
  s at :15 or :45  weight 0.5  x overlap hours of [s - 15, s + 15)
  s at :00 or :30  weight 0.25 x (overlap of [s, s + 30) + overlap of [s - 30, s))
"""

import sys
from functools import reduce
from pathlib import Path

import ee
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
import ee_common as E  # noqa: E402
from rain import INTERVALS, PRODUCTS, SCALE_M  # noqa: E402

COLL, BAND, MULT = PRODUCTS["pardig"][:3]
MS_MIN, MS_H = 60_000, 3_600_000


def accumulate30(a, b):
    a, b = ee.Number(a), ee.Number(b)
    ov = lambda s, e: b.min(e).subtract(a.max(s)).max(0).divide(MS_H)  # noqa: E731
    ic = ee.ImageCollection(COLL).select(BAND).filterDate(
        ee.Date(a.subtract(30 * MS_MIN)), ee.Date(b.add(30 * MS_MIN)))

    def weighted(img):
        s = ee.Number(img.get("system:time_start"))
        mid = s.divide(MS_MIN).mod(30).eq(15)  # 1 at :15/:45, 0 at :00/:30
        w_mid = ov(s.subtract(15 * MS_MIN), s.add(15 * MS_MIN)).multiply(0.5)
        w_edge = ov(s, s.add(30 * MS_MIN)).add(ov(s.subtract(30 * MS_MIN), s)).multiply(0.25)
        w = w_mid.multiply(mid).add(w_edge.multiply(ee.Number(1).subtract(mid)))
        return img.multiply(w.multiply(MULT))

    return ic.map(weighted).sum()


def compute_for(iv):
    a, b = INTERVALS[iv]

    def compute(fc):
        def per_window(f):
            img = accumulate30(f.get(a), f.get(b)).rename(f"pardig30_{iv}")
            vals = img.reduceRegion(ee.Reducer.mean(), f.geometry(), SCALE_M, maxPixels=1e8)
            return ee.Feature(None, vals).set("window_id", f.get("window_id"))
        return fc.map(per_window)
    return compute


def main(limit=None):
    E.init()
    w = gpd.read_parquet(E.ROOT / "data/interim/windows.parquet")
    if limit:
        w = w.iloc[:limit]
    parts = []
    for iv in ["pre", "w", "post"]:
        a, b = INTERVALS[iv]
        name = f"rain_pardig30_{iv}" if not limit else f"rain_pardig30_{iv}_test{limit}"
        parts.append(E.run_batches(name, w, ["window_id", a, b], compute_for(iv)))
    df = reduce(lambda x, y: x.merge(y, on="window_id", validate="1:1"), parts)
    if not limit:
        df.sort_values("window_id").to_parquet(E.ROOT / "data/interim/rain_pardig30.parquet", index=False)
    print(df.drop(columns="window_id").describe().round(2).T.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

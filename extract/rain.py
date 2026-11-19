"""Stage 3a: footprint-mean rain totals (mm) from PARDIG-NRT and IMERG-Final per window.

Reads  data/interim/windows.parquet
Writes data/interim/rain.parquet  window_id + {pardig,imerg}_{ante7,pre,w,post,lead}

Intervals (from stage 2), which tile the timeline: R_ante7 = [ante_start, pre_start), R_pre = [pre_start, w_start),
W = [w_start, w_end), R_post = [w_end, post_end). Each interval is its own cached Earth Engine job.
lead = [post_end, post_end + 24 h) is the placebo interval: rain after the post-period can't affect the outcome.
Each image is weighted by the hours its time slot overlaps the interval:
  IMERG  half-hourly mean rate (mm/h) over [t, t + 30 min)
  PARDIG instantaneous rate (m/h, x1000 -> mm/h) taken to represent [t - 7.5 min, t + 7.5 min)
"""

import sys
from functools import reduce

import ee
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import ee_common as E  # noqa: E402

SCALE_M = 1000  # reduce at 1 km so partial 0.1-degree pixels are area-weighted
PRODUCTS = {
    # name: (collection, band, multiplier to mm/h, slot start offset min, slot end offset min)
    "pardig": ("projects/gcp-public-data-weathernext/assets/PARDIG_1_NRT", "precipitation", 1000, -7.5, 7.5),
    "imerg": ("NASA/GPM_L3/IMERG_V07", "precipitation", 1, 0, 30),
}
INTERVALS = {"ante7": ("ante_start", "pre_start"), "pre": ("pre_start", "w_start"),
             "w": ("w_start", "w_end"), "post": ("w_end", "post_end"), "lead": ("post_end", "lead_end")}


def accumulate(coll, band, mult, off0, off1, a, b):
    """mm accumulated over [a, b) (epoch ms, server-side numbers)."""
    a, b = ee.Number(a), ee.Number(b)
    ms = 60_000
    ic = (ee.ImageCollection(coll).select(band)
          .filterDate(ee.Date(a.subtract(off1 * ms)), ee.Date(b.subtract(off0 * ms))))

    def weighted(img):
        t = ee.Number(img.get("system:time_start"))
        hours = b.min(t.add(off1 * ms)).subtract(a.max(t.add(off0 * ms))).max(0).divide(3_600_000)
        return img.multiply(hours.multiply(mult))

    return ic.map(weighted).sum().rename("mm")


def compute_for(iv):
    a, b = INTERVALS[iv]

    def compute(fc):
        def per_window(f):
            bands = [accumulate(coll, band, mult, off0, off1, f.get(a), f.get(b)).rename(f"{name}_{iv}")
                     for name, (coll, band, mult, off0, off1) in PRODUCTS.items()]
            vals = ee.Image.cat(bands).reduceRegion(ee.Reducer.mean(), f.geometry(), SCALE_M, maxPixels=1e8)
            return ee.Feature(None, vals).set("window_id", f.get("window_id"))
        return fc.map(per_window)
    return compute


def main(limit=None):
    E.init()
    w = gpd.read_parquet(E.ROOT / "data/interim/windows.parquet")
    w["lead_end"] = w.post_end + pd.Timedelta(hours=24)
    if limit:
        w = w.iloc[:limit]
    parts = []
    for iv, (a, b) in INTERVALS.items():
        name = f"rain_{iv}" if not limit else f"rain_{iv}_test{limit}"
        parts.append(E.run_batches(name, w, ["window_id", a, b], compute_for(iv)))
    df = reduce(lambda x, y: x.merge(y, on="window_id", validate="1:1"), parts)
    assert len(df) == len(w)
    if not limit:
        df.sort_values("window_id").to_parquet(E.ROOT / "data/interim/rain.parquet", index=False)
    print(df.drop(columns="window_id").describe().round(2).T.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

"""Stage 3c: ERA5 cloud cover per observation and weather controls per window.

Reads  data/interim/windows.parquet, data/interim/observations.parquet
Writes data/interim/era5.parquet  window_id +
         cc_{k}_{sat}  footprint-mean total cloud cover (0-1) at the hour nearest each daytime overpass,
                       k in m3, m2, m1, p0, p1 (dates d-3 .. d+1), sat in N, N20
         t2m_pre, rh_pre, vpd_pre   means over the pre-period (pre_start .. w_start), deg C, %, kPa
         wind_post                  mean 10 m wind speed over the post-period (w_end .. post_end), m/s
         t2m_post, rh_post, vpd_post means over the post-period; only for the bounding model, since rain
                                    causes them (added after results)
         t2m_w_min                  minimum hourly 2 m temperature during W, deg C (snow flag)
"""

import sys
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ee_common as E  # noqa: E402

ERA5 = "ECMWF/ERA5/HOURLY"
SCALE_M = 1000
OFFSETS = {"m3": -3, "m2": -2, "m1": -1, "p0": 0, "p1": 1}
SATS = ["N", "N20"]
HOUR_MS = 3_600_000


def cloud_time_columns(w: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """For each window, the overpass time (rounded to the hour) of each satellite on d-3 .. d+1."""
    obs = obs.assign(t=(obs.t_first + (obs.t_last - obs.t_first) / 2).dt.round("h")).set_index(["fire_id", "satellite", "date"]).t
    w = w.copy()
    for k, off in OFFSETS.items():
        for sat in SATS:
            key = pd.MultiIndex.from_arrays([w.fire_id, [sat] * len(w), w.d + pd.Timedelta(days=off)])
            w[f"t_{k}_{sat}"] = obs.reindex(key).array
    return w


def weather(img):
    t = img.select("temperature_2m").subtract(273.15)
    td = img.select("dewpoint_temperature_2m").subtract(273.15)
    es = lambda x: x.multiply(17.67).divide(x.add(243.5)).exp().multiply(0.6112)  # kPa  # noqa: E731
    rh = es(td).divide(es(t)).multiply(100)
    vpd = es(t).subtract(es(td))
    wind = img.select("u_component_of_wind_10m").hypot(img.select("v_component_of_wind_10m"))
    return ee.Image.cat([t.rename("t2m"), rh.rename("rh"), vpd.rename("vpd"), wind.rename("wind")])


def compute(fc):
    era = ee.ImageCollection(ERA5)

    def per_window(f):
        def hours(a, b):
            return era.filterDate(ee.Date(f.get(a)), ee.Date(f.get(b))).map(weather)

        bands = []
        for k in OFFSETS:
            for sat in SATS:
                t = ee.Date(f.get(f"t_{k}_{sat}"))
                img = era.filterDate(t, t.advance(1, "hour")).first().select("total_cloud_cover")
                bands.append(ee.Image(img).rename(f"cc_{k}_{sat}"))
        pre = hours("pre_start", "w_start").mean()
        post = hours("w_end", "post_end").mean()
        bands += [pre.select("t2m").rename("t2m_pre"), pre.select("rh").rename("rh_pre"),
                  pre.select("vpd").rename("vpd_pre"),
                  post.select("wind").rename("wind_post"), post.select("t2m").rename("t2m_post"),
                  post.select("rh").rename("rh_post"), post.select("vpd").rename("vpd_post"),
                  hours("w_start", "w_end").select("t2m").min().rename("t2m_w_min")]
        vals = ee.Image.cat(bands).reduceRegion(ee.Reducer.mean(), f.geometry(), SCALE_M, maxPixels=1e8)
        return ee.Feature(None, vals).set("window_id", f.get("window_id"))

    return fc.map(per_window)


def main(limit=None):
    E.init()
    w = gpd.read_parquet(E.ROOT / "data/interim/windows.parquet")
    obs = pd.read_parquet(E.ROOT / "data/interim/observations.parquet")
    if limit:
        w = w.iloc[:limit]
    w = cloud_time_columns(w, obs)
    tcols = [c for c in w.columns if c.startswith("t_")]
    assert w[tcols].notna().all().all(), "every window needs an overpass time on d-3 .. d+1"
    props = ["window_id", "pre_start", "w_start", "w_end", "post_end"] + tcols
    df = E.run_batches("era5" if not limit else f"era5_test{limit}", w, props, compute)
    if not limit:
        df.sort_values("window_id").to_parquet(E.ROOT / "data/interim/era5.parquet", index=False)
    print(df.drop(columns="window_id").describe().round(2).T.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)

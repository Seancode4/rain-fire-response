"""Stage 3b: NLCD fuel fractions per window footprint, and cropland share per fire event.

Reads  data/interim/windows.parquet, data/interim/detections.parquet, data/interim/events.parquet
Writes data/interim/landcover.parquet  window_id, frac_{forest,shrub,grass,crop,developed}, fuel, nlcd_year
       data/interim/cropland.parquet   fire_id, crop_share (from up to 200 sampled detections)

Uses the latest NLCD year before the fire year, so a fire's own burn scar doesn't change its fuel class.
"""

import sys
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ee_common as E  # noqa: E402

NLCD = {2016: "USGS/NLCD_RELEASES/2019_REL/NLCD/2016", 2019: "USGS/NLCD_RELEASES/2019_REL/NLCD/2019",
        2021: "USGS/NLCD_RELEASES/2021_REL/NLCD/2021"}
CLASSES = {"forest": [41, 42, 43], "shrub": [51, 52], "grass": [71], "crop": [82], "developed": [21, 22, 23, 24]}
FUELS = ["forest", "shrub", "grass"]
SAMPLE_PER_FIRE = 200


def nlcd_year(fire_year: int) -> int:
    return max(y for y in NLCD if y < fire_year)


def class_bands(year: int) -> ee.Image:
    lc = ee.Image(NLCD[year]).select("landcover")
    return ee.Image.cat([lc.remap(codes, [1] * len(codes), 0).rename(name) for name, codes in CLASSES.items()])


def footprint_fractions(w: gpd.GeoDataFrame) -> pd.DataFrame:
    out = []
    for year, g in w.groupby("nlcd_year"):
        img = class_bands(year)
        compute = lambda fc, img=img: img.reduceRegions(fc, ee.Reducer.mean(), 30, tileScale=4).map(  # noqa: E731
            lambda f: ee.Feature(None, f.toDictionary()))
        out.append(E.run_batches(f"landcover_{year}", g, ["window_id"], compute, batch_size=50).assign(nlcd_year=year))
    df = pd.concat(out, ignore_index=True).rename(columns={c: f"frac_{c}" for c in CLASSES})
    df["fuel"] = df[[f"frac_{c}" for c in FUELS]].idxmax(axis=1).str.removeprefix("frac_")
    return df


def crop_share(det: pd.DataFrame) -> pd.DataFrame:
    s = det.sample(frac=1, random_state=0).groupby("fire_id").head(SAMPLE_PER_FIRE)
    s["nlcd_year"] = s.date.dt.year.map(nlcd_year)
    s = gpd.GeoDataFrame(s[["fire_id", "nlcd_year"]].reset_index(drop=True),
                         geometry=gpd.points_from_xy(s.longitude, s.latitude), crs=4326)
    out = []
    for year, g in s.groupby("nlcd_year"):
        img = class_bands(year).select("crop")
        compute = lambda fc, img=img: img.reduceRegions(fc, ee.Reducer.first(), 30).map(  # noqa: E731
            lambda f: ee.Feature(None, f.toDictionary()))
        out.append(E.run_batches(f"cropland_{year}", g, ["fire_id"], compute, batch_size=2000))
    pts = pd.concat(out, ignore_index=True)
    return pts.groupby("fire_id").agg(crop_share=("first", "mean"), n_sampled=("first", "size")).reset_index()


def main():
    E.init()
    w = gpd.read_parquet(E.ROOT / "data/interim/windows.parquet")
    w["nlcd_year"] = w.d.dt.year.map(nlcd_year)
    lc = footprint_fractions(w)
    lc.sort_values("window_id").to_parquet(E.ROOT / "data/interim/landcover.parquet", index=False)
    print("fuel counts (windows):", lc.fuel.value_counts().to_dict())
    print(lc.filter(like="frac_").describe().round(2).to_string())

    events = pd.read_parquet(E.ROOT / "data/interim/events.parquet")
    det = pd.read_parquet(E.ROOT / "data/interim/detections.parquet", columns=["fire_id", "date", "latitude", "longitude"])
    det = det[det.fire_id.isin(events.fire_id[events.pass_filters])]
    cs = crop_share(det)
    cs.to_parquet(E.ROOT / "data/interim/cropland.parquet", index=False)
    print(f"fires with crop_share >= 0.5: {(cs.crop_share >= 0.5).sum()} of {len(cs)}")


if __name__ == "__main__":
    main()

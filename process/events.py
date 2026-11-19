"""Stage 1: filter FIRMS VIIRS detections and cluster them into fire events.

Reads  data/raw/firms/*.csv, data/raw/states/cb_2023_us_state_20m.zip
Writes data/interim/detections.parquet  one row per kept detection, with fire_id
       data/interim/events.parquet      one row per fire event, with pass_filters

The cropland filter needs NLCD and is applied in stage 3.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
FIRMS_DIR = ROOT / "data/raw/firms"
STATES_ZIP = ROOT / "data/raw/states/cb_2023_us_state_20m.zip"
OUT_DIR = ROOT / "data/interim"

WEST = ["WA", "OR", "CA", "ID", "NV", "UT", "AZ", "MT", "WY", "CO", "NM"]
WEST_BBOX = (-125.0, 31.0, -102.0, 49.5)  # lon_min, lat_min, lon_max, lat_max (pre-filter for speed)
SEASON = ("05-15", "12-10")  # detections loaded for each year (windows are sampled June-Nov later)

STATIC_BUFFER_M = 1000  # drop detections within this distance of any type = 2 detection
PERSIST_CELL_M = 1000  # grid cell for the persistence filter
PERSIST_MIN_MONTHS, PERSIST_MIN_YEARS = 6, 3
CLUSTER_RADIUS_M = 2000
LOOKBACK_DAYS = 5
MIN_DAYS, MIN_DETECTIONS = 3, 20

COLS = ["latitude", "longitude", "scan", "track", "acq_date", "acq_time",
        "satellite", "confidence", "frp", "daynight", "type"]


def load() -> pd.DataFrame:
    """All detections in the western bounding box, all months (full years are needed for persistence)."""
    lon0, lat0, lon1, lat1 = WEST_BBOX
    frames = []
    for f in sorted(FIRMS_DIR.glob("*.csv")):
        d = pd.read_csv(f, usecols=COLS, dtype={"acq_time": str})
        frames.append(d[d.longitude.between(lon0, lon1) & d.latitude.between(lat0, lat1)])
    d = pd.concat(frames, ignore_index=True)
    d["time_utc"] = pd.to_datetime(d.acq_date + " " + d.acq_time.str.zfill(4), format="%Y-%m-%d %H%M", utc=True)
    d["date"] = d.time_utc.dt.tz_localize(None).dt.normalize()
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(d.longitude.values, d.latitude.values)
    d["x"], d["y"] = x, y  # metres, CONUS Albers equal-area
    return d.drop(columns=["acq_date", "acq_time"])


def static_source_mask(d: pd.DataFrame) -> pd.Series:
    """True for detections near a type = 2 detection, or in a cell that is persistently active."""
    xy = d[["x", "y"]].values
    t2 = cKDTree(xy[d.type.values == 2])
    dist, _ = t2.query(xy, k=1, distance_upper_bound=STATIC_BUFFER_M)
    near_static = np.isfinite(dist)

    cell = (d.x // PERSIST_CELL_M).astype(np.int64).astype(str) + "_" + (d.y // PERSIST_CELL_M).astype(np.int64).astype(str)
    stats = pd.DataFrame({"cell": cell, "ym": d.date.dt.to_period("M"), "yr": d.date.dt.year})
    g = stats.groupby("cell").agg(n_months=("ym", "nunique"), n_years=("yr", "nunique"))
    persistent = g.index[(g.n_months >= PERSIST_MIN_MONTHS) & (g.n_years >= PERSIST_MIN_YEARS)]
    in_persistent = cell.isin(persistent).values

    print(f"  static buffer drops {near_static.sum():,}; persistence drops {(in_persistent & ~near_static).sum():,} more "
          f"({len(persistent):,} cells)")
    return pd.Series(near_static | in_persistent, index=d.index)


def in_west_states(d: pd.DataFrame) -> np.ndarray:
    states = gpd.read_file(STATES_ZIP)
    west = states[states.STUSPS.isin(WEST)].to_crs(5070).union_all()
    pts = gpd.GeoSeries(gpd.points_from_xy(d.x, d.y), crs=5070)
    return pts.within(west).values


def filter_detections(d: pd.DataFrame) -> pd.DataFrame:
    n0 = len(d)
    d = d[~static_source_mask(d)]
    md = d.date.dt.strftime("%m-%d")
    d = d[(md >= SEASON[0]) & (md <= SEASON[1])]
    d = d[(d.type == 0) & d.confidence.isin(["n", "h"])]
    d = d[in_west_states(d)].reset_index(drop=True)
    print(f"  kept {len(d):,} of {n0:,} detections in the bounding box")
    return d


def cluster(d: pd.DataFrame) -> np.ndarray:
    """fire_id per detection.

    1. Within each UTC date, single-linkage clusters at 2 km (every detection belongs to one).
    2. A day's cluster joins any fire with a detection within 2 km in the previous 5 days.
       A cluster that touches two fires merges them.
    """
    caller_index = d.index
    d = d.sort_values("time_utc")
    xy = d[["x", "y"]].values
    day = d.date.values.astype("datetime64[D]").astype(np.int64)

    cluster_id = np.empty(len(d), dtype=np.int64)
    n_clusters = 0
    days = np.unique(day)
    starts = np.searchsorted(day, days)
    ends = np.append(starts[1:], len(day))
    for s, e in zip(starts, ends):  # step 1
        pairs = cKDTree(xy[s:e]).query_pairs(CLUSTER_RADIUS_M, output_type="ndarray")
        n = e - s
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        k, labels = connected_components(g, directed=False)
        cluster_id[s:e] = labels + n_clusters
        n_clusters += k

    links = []
    for dd, s, e in zip(days, starts, ends):  # step 2 (rows are sorted by day)
        p = np.searchsorted(day, dd - LOOKBACK_DAYS)
        if p == s:
            continue
        dist, j = cKDTree(xy[p:s]).query(xy[s:e], k=1, distance_upper_bound=CLUSTER_RADIUS_M)
        hit = np.isfinite(dist)
        links.append(np.column_stack([cluster_id[s:e][hit], cluster_id[p:s][j[hit]]]))
    links = np.unique(np.vstack(links), axis=0)
    g = coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(n_clusters, n_clusters))
    _, fire_of_cluster = connected_components(g, directed=False)

    fire_id = pd.Series(fire_of_cluster[cluster_id], index=d.index)
    return fire_id.reindex(caller_index).values  # back in the caller's row order


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    ev = d.groupby("fire_id").agg(
        start=("date", "min"), end=("date", "max"), n_det=("frp", "size"),
        n_day_det=("daynight", lambda s: (s == "D").sum()),
        lat=("latitude", "mean"), lon=("longitude", "mean"),
        x_min=("x", "min"), x_max=("x", "max"), y_min=("y", "min"), y_max=("y", "max"),
        total_frp=("frp", "sum"),
    )
    ev["n_days"] = (ev.end - ev.start).dt.days + 1
    ev["bbox_km"] = np.hypot(ev.x_max - ev.x_min, ev.y_max - ev.y_min) / 1000  # extent diagonal, for chaining checks
    ev["pass_filters"] = (ev.n_days >= MIN_DAYS) & (ev.n_det >= MIN_DETECTIONS)
    return ev.drop(columns=["x_min", "x_max", "y_min", "y_max"]).reset_index()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading"); d = load()
    print("filtering"); d = filter_detections(d)
    print("clustering"); d["fire_id"] = cluster(d)
    ev = summarize(d)
    print(f"  {len(ev):,} fires; {ev.pass_filters.sum():,} pass duration and detection filters "
          f"({d.fire_id.isin(ev.fire_id[ev.pass_filters]).mean():.0%} of detections)")
    d.drop(columns=["type"]).to_parquet(OUT_DIR / "detections.parquet", index=False)
    ev.to_parquet(OUT_DIR / "events.parquet", index=False)


if __name__ == "__main__":
    main()

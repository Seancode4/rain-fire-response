"""Stage 2: daytime observations and overpass-aligned windows.

Reads  data/interim/detections.parquet, data/interim/events.parquet
Writes data/interim/observations.parquet  one row per fire x satellite x UTC date (daytime overpass)
       data/interim/windows.parquet       one row per fire x day d (GeoParquet, footprint in EPSG:4326)

Timing for window d (all UTC):
  pre      overpasses on d-2 and d-1
  W        (latest pass on d-1, earliest pass on d]
  post     overpasses on d and d+1
  R_post   (earliest pass on d, latest pass on d+1]   rain the post-period sees that W doesn't count
  R_ante7  [earliest pass on d-2 minus 7 days, earliest pass on d-2)

Cloud screening and the remaining validity rules need ERA5 and happen in stage 4.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"

SATS = ["N", "N20"]
ORBIT_GAP = pd.Timedelta(minutes=30)  # passes further apart than this are separate orbits
PAD_DAYS = 3  # observation rows kept before a fire's first and after its last detection
FOOTPRINT_LOOKBACK, FOOTPRINT_LOOKBACK_MAX = 3, 5
FOOTPRINT_BUFFER_M = 5000
MIN_PRE_DAY_DET = 5
SEASON_MONTHS = range(6, 12)
LAST_D = {2025: pd.Timestamp("2025-09-29")}  # IMERG-Final ends 2025-09-30; R_post needs d+1


def build_observations(det: pd.DataFrame, fires: pd.DataFrame) -> pd.DataFrame:
    """Daytime observations for every fire x satellite x date in [start - PAD, end + PAD]."""
    day = det[det.daynight == "D"].sort_values(["fire_id", "satellite", "time_utc"]).copy()
    day["pix_area"] = day.scan * day.track
    new_orbit = (
        (day.fire_id != day.fire_id.shift())
        | (day.satellite != day.satellite.shift())
        | (day.time_utc - day.time_utc.shift() > ORBIT_GAP)
    )
    day["orbit"] = new_orbit.cumsum()
    orbits = day.groupby(["fire_id", "satellite", "date", "orbit"]).agg(
        frp=("frp", "sum"), n_det=("frp", "size"), pix_sum=("pix_area", "sum"),
        t_first=("time_utc", "min"), t_last=("time_utc", "max"))
    obs = orbits.groupby(["fire_id", "satellite", "date"]).agg(
        frp=("frp", "mean"), n_det=("n_det", "sum"), n_orbits=("frp", "size"), pix_sum=("pix_sum", "sum"),
        t_first=("t_first", "min"), t_last=("t_last", "max")).reset_index()
    obs["pix_area"] = obs.pix_sum / obs.n_det
    obs = obs.drop(columns="pix_sum")

    # full grid, so satellite-days with no detections exist (FRP = 0 if the sky turns out to be clear)
    pad = pd.Timedelta(days=PAD_DAYS)
    grid = pd.concat(
        pd.DataFrame({"fire_id": f, "date": pd.date_range(s - pad, e + pad, freq="D")})
        for f, s, e in fires[["fire_id", "start", "end"]].itertuples(index=False))
    grid = grid.merge(pd.DataFrame({"satellite": SATS}), how="cross")
    obs = grid.merge(obs, on=["fire_id", "satellite", "date"], how="left")
    obs["frp"] = obs.frp.fillna(0.0)
    obs["n_det"] = obs.n_det.fillna(0).astype(int)
    obs["n_orbits"] = obs.n_orbits.fillna(0).astype(int)

    # missing overpass time: that fire's median time of day for that satellite (else the overall median)
    tod = day.time_utc - day.time_utc.dt.floor("D")
    med = tod.groupby([day.fire_id, day.satellite]).median().rename("tod").reset_index()
    sat_med = tod.groupby(day.satellite).median()
    obs = obs.merge(med, on=["fire_id", "satellite"], how="left")
    obs["tod"] = obs.tod.fillna(obs.satellite.map(sat_med))
    obs["time_imputed"] = obs.t_first.isna()
    fill = obs.date.dt.tz_localize("UTC") + obs.tod
    obs["t_first"] = obs.t_first.fillna(fill)
    obs["t_last"] = obs.t_last.fillna(fill)
    return obs.drop(columns="tod")


def candidate_days(fires: pd.DataFrame) -> pd.DataFrame:
    """Every day d whose pre-period lies inside the fire's detected dates, in season."""
    w = pd.concat(
        pd.DataFrame({"fire_id": f, "d": pd.date_range(s + pd.Timedelta(days=2), e + pd.Timedelta(days=2), freq="D"),
                      "fire_start": s})
        for f, s, e in fires[["fire_id", "start", "end"]].itertuples(index=False))
    w = w[w.d.dt.month.isin(SEASON_MONTHS)]
    for year, last in LAST_D.items():
        w = w[(w.d.dt.year != year) | (w.d <= last)]
    return w.reset_index(drop=True)


def add_timing(w: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Period boundaries and pre-period daytime detection count from the observation table."""
    by_day = obs.groupby(["fire_id", "date"]).agg(
        earliest=("t_first", "min"), latest=("t_last", "max"), n_det=("n_det", "sum"))

    def at(offset, col):
        key = pd.MultiIndex.from_arrays([w.fire_id, w.d + pd.Timedelta(days=offset)])
        return by_day[col].reindex(key).array  # .array keeps the UTC timezone

    w = w.copy()
    w["pre_start"] = at(-2, "earliest")
    w["w_start"] = at(-1, "latest")
    w["w_end"] = at(0, "earliest")
    w["post_end"] = at(1, "latest")
    w["ante_start"] = w.pre_start - pd.Timedelta(days=7)
    w["n_pre_day_det"] = at(-2, "n_det") + at(-1, "n_det")
    w["fire_age"] = (w.d - w.fire_start).dt.days
    return w.drop(columns="fire_start")


def add_footprints(w: pd.DataFrame, det: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convex hull of the fire's detections (day and night) on d-3..d-1, else d-5..d-1, buffered 5 km."""
    geoms, lookback = [], []
    det_by_fire = {f: g for f, g in det.groupby("fire_id")}
    for f, d in zip(w.fire_id, w.d):
        g = det_by_fire[f]
        for lb in (FOOTPRINT_LOOKBACK, FOOTPRINT_LOOKBACK_MAX):
            pts = g[(g.date >= d - pd.Timedelta(days=lb)) & (g.date < d)]
            if len(pts):
                break
        if len(pts):
            geoms.append(MultiPoint(pts[["x", "y"]].values).convex_hull.buffer(FOOTPRINT_BUFFER_M, quad_segs=4))
            lookback.append(lb)
        else:
            geoms.append(None)
            lookback.append(np.nan)
    gw = gpd.GeoDataFrame(w.assign(footprint_lookback=lookback), geometry=geoms, crs=5070)
    gw = gw[gw.geometry.notna()].copy()
    gw["footprint_km2"] = gw.area / 1e6
    return gw.to_crs(4326)


def main():
    det = pd.read_parquet(INTERIM / "detections.parquet")
    events = pd.read_parquet(INTERIM / "events.parquet")
    fires = events[events.pass_filters]
    det = det[det.fire_id.isin(fires.fire_id)]

    obs = build_observations(det, fires)
    w = candidate_days(fires)
    n_cand = len(w)
    w = add_timing(w, obs)
    w = w[w.n_pre_day_det >= MIN_PRE_DAY_DET]
    w = add_footprints(w, det)
    w.insert(0, "window_id", np.arange(len(w)))

    print(f"observations: {len(obs):,} rows ({(obs.n_det > 0).mean():.0%} with detections, "
          f"{obs.time_imputed.mean():.0%} with imputed overpass time)")
    print(f"windows: {n_cand:,} candidate fire-days in season -> {len(w):,} with >= {MIN_PRE_DAY_DET} pre-period "
          f"daytime detections, from {w.fire_id.nunique():,} fires")
    print(f"  W length (h) quantiles: {((w.w_end - w.w_start).dt.total_seconds() / 3600).quantile([.01, .5, .99]).round(1).to_dict()}")
    print(f"  footprint km2 quantiles: {w.footprint_km2.quantile([.1, .5, .9, .99]).round(0).to_dict()}")
    print(f"  footprint lookback extended to 5 days: {(w.footprint_lookback == 5).sum():,}")

    obs.to_parquet(INTERIM / "observations.parquet", index=False)
    w.to_parquet(INTERIM / "windows.parquet", index=False)


if __name__ == "__main__":
    main()

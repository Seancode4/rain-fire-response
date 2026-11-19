"""Stage 4: cloud screening, validity rules, outcome, controls -> analysis table.

Reads  data/interim/{windows,observations,rain,rain_duration,landcover,cropland,era5}.parquet
Writes data/final/analysis.parquet          one row per window x cloud threshold (0.2, 0.3, 0.4);
                                            `valid` marks rows passing screening; outcome only where valid
       data/final/cloud_validation.parquet  dry-window d-1 observations whose neighbours (d-2, d) were
                                            clear and detected, for the cloud validation plot

Blind: prints counts and input distributions only, never the outcome.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM, FINAL = ROOT / "data/interim", ROOT / "data/final"

THRESHOLDS = [0.2, 0.3, 0.4]
SATS = ["N", "N20"]
DAYS = {"m3": -3, "m2": -2, "m1": -1, "p0": 0, "p1": 1}
PRE, POST, TREND = ["m2", "m1"], ["p0", "p1"], ["m3", "m2", "m1"]
MIN_VALID = 2
MAX_CROP_SHARE = 0.5
NEAR_EXT = 0.1
SNOW_T_C = 1.0
DRY_MM = 0.1  # "< 0.1 mm" bin and dry windows for the validation plot
VALIDATION_CLEAR = 0.2  # neighbours must be this clear in the validation plot
RAIN_BINS = [-np.inf, 0.1, 2, 5, np.inf]
RAIN_LABELS = ["<0.1", "0.1-2", "2-5", "5+"]


def load() -> pd.DataFrame:
    w = gpd.read_parquet(INTERIM / "windows.parquet").drop(columns="geometry")
    df = w
    for name in ["rain", "rain_duration", "rain_pardig30", "era5"]:
        df = df.merge(pd.read_parquet(INTERIM / f"{name}.parquet"), on="window_id", validate="1:1")
    lc = pd.read_parquet(INTERIM / "landcover.parquet")
    df = df.merge(lc, on="window_id", validate="1:1")
    df = df.merge(pd.read_parquet(INTERIM / "cropland.parquet")[["fire_id", "crop_share"]], on="fire_id", how="left")
    n0 = len(df)
    df = df[df.crop_share < MAX_CROP_SHARE].copy()
    print(f"windows: {n0:,}; after cropland filter: {len(df):,} from {df.fire_id.nunique():,} fires")
    return df


def add_observations(df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """frp_{day}_{sat}, ndet_{day}_{sat}, pix_{day}_{sat} for d-3 .. d+1."""
    o = obs.set_index(["fire_id", "satellite", "date"])
    cols = {}
    for k, off in DAYS.items():
        for s in SATS:
            key = pd.MultiIndex.from_arrays([df.fire_id, [s] * len(df), df.d + pd.Timedelta(days=off)])
            r = o.reindex(key)
            cols[f"frp_{k}_{s}"] = r.frp.values
            cols[f"ndet_{k}_{s}"] = r.n_det.values
            cols[f"pix_{k}_{s}"] = r.pix_area.values
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    assert out[[c for c in cols if c.startswith("frp_")]].notna().all().all()
    return out


def offset_constant(df: pd.DataFrame) -> float:
    """c: 5th percentile of nonzero per-observation FRP over the pre- and post-period observations."""
    frp = df[[f"frp_{k}_{s}" for k in PRE + POST for s in SATS]].to_numpy().ravel()
    return float(np.percentile(frp[frp > 0], 5))


def screen(df: pd.DataFrame, thr: float, c: float) -> pd.DataFrame:
    """Validity, period FRP, outcome, and cloud-dependent controls at one cloud threshold."""
    out = df[["window_id"]].copy()
    out["cc_thr"] = thr

    def period(days):
        frp = df[[f"frp_{k}_{s}" for k in days for s in SATS]].to_numpy()
        cc = df[[f"cc_{k}_{s}" for k in days for s in SATS]].to_numpy()
        ok = cc < thr
        n = ok.sum(axis=1)
        mean = np.where(n > 0, np.where(ok, frp, 0).sum(axis=1) / np.maximum(n, 1), np.nan)
        cc_valid = np.where(n > 0, np.where(ok, cc, 0).sum(axis=1) / np.maximum(n, 1), np.nan)
        return mean, n, cc_valid

    # cc_*_valid: mean cloud cover of the valid looks (cloud-control check, added after results)
    out["frp_pre"], out["n_valid_pre"], out["cc_pre_valid"] = period(PRE)
    out["frp_post"], out["n_valid_post"], out["cc_post_valid"] = period(POST)
    out["valid"] = (out.n_valid_pre >= MIN_VALID) & (out.n_valid_post >= MIN_VALID)
    out.loc[~out.valid, ["frp_pre", "frp_post"]] = np.nan
    out["y"] = np.log((out.frp_post + c) / (out.frp_pre + c))
    out["near_ext"] = (out.frp_post < NEAR_EXT * out.frp_pre).where(out.valid)
    out["log_frp_pre"] = np.log(out.frp_pre + c)

    # pre-rain trend: slope of log daily FRP over d-3..d-1, valid observations only
    daily = []
    for k in TREND:
        frp = df[[f"frp_{k}_{s}" for s in SATS]].to_numpy()
        ok = df[[f"cc_{k}_{s}" for s in SATS]].to_numpy() < thr
        n = ok.sum(axis=1)
        daily.append(np.where(n > 0, np.log(np.where(ok, frp, 0).sum(axis=1) / np.maximum(n, 1) + c), np.nan))
    daily = np.column_stack(daily)
    t = np.array([-3.0, -2.0, -1.0])
    have = ~np.isnan(daily)
    n_days = have.sum(axis=1)
    tm = np.where(have, t, 0).sum(axis=1) / np.maximum(n_days, 1)
    ym = np.where(have, daily, 0).sum(axis=1) / np.maximum(n_days, 1)
    num = np.where(have, (t - tm[:, None]) * (daily - ym[:, None]), 0).sum(axis=1)
    den = np.where(have, (t - tm[:, None]) ** 2, 0).sum(axis=1)
    out["trend_missing"] = n_days < 2
    out["trend"] = np.where(out.trend_missing, 0.0, num / np.where(den > 0, den, 1))
    return out


def static_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columns that don't depend on the cloud threshold."""
    s = df[["window_id", "fire_id", "d", "fire_age", "n_pre_day_det", "footprint_km2", "nlcd_year",
            "t2m_pre", "rh_pre", "vpd_pre", "wind_post", "t2m_w_min",
            "t2m_post", "rh_post", "vpd_post"]  # post-period air: bounding model only
           + [f"{p}_{iv}" for p in ["pardig", "imerg"] for iv in ["ante7", "pre", "w", "post", "lead"]]
           + [f"pardig30_{iv}" for iv in ["pre", "w", "post"]]
           + ["pardig_dur_h", "imerg_dur_h"]
           + [c for c in df.columns if c.startswith("frac_")]].copy()
    s["year"], s["month"] = s.d.dt.year, s.d.dt.month
    s["fuel"] = np.where(df.fuel == "forest", "forest", "non-forest")
    s["log_pre_det"] = np.log(df.n_pre_day_det)
    # pre-period mean pixel area over observations with detections (detection-weighted)
    pix = df[[f"pix_{k}_{x}" for k in PRE for x in SATS]].to_numpy()
    nd = df[[f"ndet_{k}_{x}" for k in PRE for x in SATS]].to_numpy()
    s["pix_pre"] = np.nansum(pix * nd, axis=1) / nd.sum(axis=1)
    s["snow_flag"] = (df.t2m_w_min <= SNOW_T_C) & ((df.pardig_w >= DRY_MM) | (df.imerg_w >= DRY_MM))
    for p in ["pardig", "imerg"]:
        s[f"{p}_bin"] = pd.cut(df[f"{p}_w"], RAIN_BINS, labels=RAIN_LABELS, right=False)
    s["cc_post_mean"] = df[[f"cc_{k}_{x}" for k in POST for x in SATS]].mean(axis=1)
    return s


def cloud_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Dry windows: the d-1 observation, where the same satellite saw the fire clear on d-2 and d."""
    dry = (df.pardig_pre < DRY_MM) & (df.pardig_w < DRY_MM)
    rows = []
    for s in SATS:
        nb_ok = ((df[f"cc_m2_{s}"] < VALIDATION_CLEAR) & (df[f"ndet_m2_{s}"] > 0)
                 & (df[f"cc_p0_{s}"] < VALIDATION_CLEAR) & (df[f"ndet_p0_{s}"] > 0))
        sel = df[dry & nb_ok]
        rows.append(pd.DataFrame({"window_id": sel.window_id, "fire_id": sel.fire_id, "satellite": s,
                                  "cc": sel[f"cc_m1_{s}"], "no_detection": sel[f"ndet_m1_{s}"] == 0}))
    return pd.concat(rows, ignore_index=True)


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    df = add_observations(load(), pd.read_parquet(INTERIM / "observations.parquet"))
    c = offset_constant(df)
    print(f"c (5th pct of nonzero observation FRP) = {c:.2f} MW")

    static = static_columns(df)
    table = pd.concat([static.merge(screen(df, thr, c), on="window_id") for thr in THRESHOLDS], ignore_index=True)
    table.attrs["c"] = c
    table.to_parquet(FINAL / "analysis.parquet", index=False)

    val = cloud_validation(df)
    val.to_parquet(FINAL / "cloud_validation.parquet", index=False)

    # blind summaries: counts and inputs only
    v = table[table.valid]
    print("\nvalid windows by threshold:", v.groupby("cc_thr").size().to_dict())
    print("valid windows by PARDIG bin x fuel (30%):")
    print(pd.crosstab(v[v.cc_thr == 0.3].pardig_bin, v[v.cc_thr == 0.3].fuel, margins=True).to_string())
    v3 = v[v.cc_thr == 0.3]
    print(f"\n30%: trend missing {v3.trend_missing.mean():.1%}; snow flag {v3.snow_flag.sum()} windows; "
          f"fires {v3.fire_id.nunique():,}")
    ctrl = ["log_frp_pre", "log_pre_det", "trend", "fire_age", "pix_pre", "t2m_pre", "rh_pre", "vpd_pre",
            "wind_post", "pardig_pre", "pardig_post", "pardig_ante7"]
    print("\ncontrols (30%, valid):")
    print(v3[ctrl].describe().T[["count", "mean", "50%", "min", "max"]].round(2).to_string())
    print(f"\nmissing values in controls: {v3[ctrl].isna().sum()[lambda x: x > 0].to_dict() or 'none'}")
    print(f"cloud validation rows: {len(val):,} ({val.fire_id.nunique():,} fires)")


if __name__ == "__main__":
    main()

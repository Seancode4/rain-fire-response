"""Stage 1 sanity checks: largest events (for chained fires) and known-fire spot checks.

Reads  data/interim/detections.parquet, data/interim/events.parquet
Writes data/interim/checks/largest_events.png, data/interim/checks/known_fires.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/interim/checks"
WEST = ["WA", "OR", "CA", "ID", "NV", "UT", "AZ", "MT", "WY", "CO", "NM"]
PIXEL_ACRES = 375 * 375 / 4047  # one VIIRS I-band pixel at nadir

# name, point of origin (lat, lon), reported start, reported containment, reported acres
KNOWN = [
    ("Dixie 2021", 39.87, -121.39, "2021-07-13", "2021-10-25", 963_309),
    ("Creek 2020", 37.19, -119.26, "2020-09-04", "2020-12-24", 379_895),
]


def burned_acres(det: pd.DataFrame) -> float:
    """Rough area: distinct 375 m cells with a detection."""
    cells = set(zip((det.x // 375).astype(int), (det.y // 375).astype(int)))
    return len(cells) * PIXEL_ACRES


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = pd.read_parquet(ROOT / "data/interim/detections.parquet")
    ev = pd.read_parquet(ROOT / "data/interim/events.parquet")
    ev = ev[ev.pass_filters]
    states = gpd.read_file(ROOT / "data/raw/states/cb_2023_us_state_20m.zip")
    west = states[states.STUSPS.isin(WEST)]

    print(f"{len(ev):,} events. Duration (days) quantiles:",
          ev.n_days.quantile([.5, .9, .99, 1]).round().astype(int).to_dict())

    top = ev.nlargest(20, "n_det").copy()
    top["acres"] = [burned_acres(det[det.fire_id == f]) for f in top.fire_id]
    cols = ["fire_id", "start", "end", "n_days", "n_det", "bbox_km", "acres", "lat", "lon"]
    print("\nLargest 20 events by detections:")
    print(top[cols].round({"bbox_km": 1, "acres": 0, "lat": 2, "lon": 2}).to_string(index=False))

    print("\nWidest 10 events by extent (chaining candidates):")
    print(ev.nlargest(10, "bbox_km")[cols[:-3] + ["lat", "lon"]].round({"bbox_km": 1, "lat": 2, "lon": 2}).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 10))
    west.boundary.plot(ax=ax, color="0.6", lw=0.5)
    sub = det[det.fire_id.isin(top.fire_id)]
    for k, (f, g) in enumerate(sub.groupby("fire_id")):
        ax.scatter(g.longitude, g.latitude, s=0.3, color=plt.cm.tab20(k % 20))
        ax.annotate(str(f), (g.longitude.mean(), g.latitude.mean()), fontsize=7)
    ax.set_title("Largest 20 events by detection count")
    ax.set_aspect(1.25)
    fig.savefig(OUT / "largest_events.png", dpi=130, bbox_inches="tight")

    print("\nKnown fires:")
    fig, axes = plt.subplots(1, len(KNOWN), figsize=(6 * len(KNOWN), 6))
    for ax, (name, lat, lon, start, end, acres) in zip(axes, KNOWN):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        near = det[(det.date.between(start, start + pd.Timedelta(days=4)))
                   & (np.hypot(det.latitude - lat, det.longitude - lon) < 0.1)]
        if near.empty:
            print(f"  {name}: no detections near origin in first 4 days"); continue
        fid = near.fire_id.mode()[0]
        g = det[det.fire_id == fid]
        e = ev[ev.fire_id == fid]
        print(f"  {name}: fire_id {fid}  detected {g.date.min():%Y-%m-%d} → {g.date.max():%Y-%m-%d} "
              f"(reported {start:%Y-%m-%d} → contained {end:%Y-%m-%d}), {len(g):,} detections, "
              f"~{burned_acres(g):,.0f} acres of detected cells (reported {acres:,}), "
              f"passes filters: {not e.empty}")
        sc = ax.scatter(g.longitude, g.latitude, s=1, c=(g.date - g.date.min()).dt.days, cmap="viridis")
        ax.plot(lon, lat, "r*", ms=12)
        ax.set_title(f"{name} (fire_id {fid})")
        ax.set_aspect(1.25)
        fig.colorbar(sc, ax=ax, label="days since first detection")
    fig.savefig(OUT / "known_fires.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()

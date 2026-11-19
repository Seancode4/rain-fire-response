"""Shared Earth Engine helpers: init, and resumable batched computation over windows."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

PROJECT = "PLACEHOLDER"  # your Earth Engine-registered Google Cloud project ID
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/interim/ee"


def init():
    ee.Initialize(project=PROJECT)


def to_fc(gdf: gpd.GeoDataFrame, props: list[str]) -> ee.FeatureCollection:
    """Client-side GeoDataFrame -> ee.FeatureCollection. Timestamps become epoch milliseconds."""
    feats = []
    for row in gdf.itertuples(index=False):
        p = {}
        for c in props:
            v = getattr(row, c)
            p[c] = int(v.timestamp() * 1000) if isinstance(v, pd.Timestamp) else v
        feats.append(ee.Feature(ee.Geometry(json.loads(gpd.GeoSeries([row.geometry]).to_json())["features"][0]["geometry"]), p))
    return ee.FeatureCollection(feats)


def run_batches(name: str, gdf: gpd.GeoDataFrame, props: list[str], compute, batch_size=100, workers=8) -> pd.DataFrame:
    """Apply compute(fc) -> fc to gdf in batches; each batch is cached to data/interim/ee/<name>/.

    compute must return features that carry an `id` property and the output columns.
    """
    out_dir = CACHE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [gdf.iloc[i:i + batch_size] for i in range(0, len(gdf), batch_size)]
    todo = [(k, b) for k, b in enumerate(batches) if not (out_dir / f"{k:05d}.parquet").exists()]
    print(f"{name}: {len(batches)} batches, {len(todo)} to run")

    def one(k, b):
        res = compute(to_fc(b, props)).getInfo()["features"]
        df = pd.DataFrame([f["properties"] for f in res])
        df.to_parquet(out_dir / f"{k:05d}.parquet", index=False)
        return k

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(one, k, b): k for k, b in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:  # keep going; rerun to retry failures
                failed.append(futs[f])
                print(f"  batch {futs[f]} failed: {str(e)[:200]}")
            if n % 20 == 0:
                print(f"  {n}/{len(todo)} done")
    if failed:
        raise RuntimeError(f"{len(failed)} batches failed; rerun to retry them")
    return pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))], ignore_index=True)

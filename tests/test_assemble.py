"""Cloud screening, period FRP and pre-rain trend in process/assemble.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "process"))
import assemble as A  # noqa: E402


def window(frp: dict, cc: dict) -> pd.DataFrame:
    """One window; frp/cc keyed by (day, sat), defaulting to FRP 0 and clear sky."""
    row = {"window_id": 0}
    for k in A.DAYS:
        for s in A.SATS:
            row[f"frp_{k}_{s}"] = frp.get((k, s), 0.0)
            row[f"cc_{k}_{s}"] = cc.get((k, s), 0.0)
    return pd.DataFrame([row])


def test_cloudy_observations_are_excluded_from_period_means():
    df = window(frp={("m2", "N"): 100, ("m1", "N"): 50, ("m1", "N20"): 999, ("p0", "N"): 10},
                cc={("m1", "N20"): 0.9})  # the 999 is cloudy and must not count
    r = A.screen(df, 0.3, c=1.0).iloc[0]
    assert r.n_valid_pre == 3 and r.frp_pre == pytest.approx(50)  # (100 + 0 + 50) / 3
    assert r.n_valid_post == 4 and r.frp_post == pytest.approx(2.5)  # clear zeros count as FRP 0
    assert r.valid and r.y == pytest.approx(np.log(3.5 / 51))


def test_too_few_valid_observations_invalidates_window():
    df = window(frp={}, cc={("p0", "N"): 0.5, ("p0", "N20"): 0.5, ("p1", "N"): 0.5})
    r = A.screen(df, 0.3, c=1.0).iloc[0]
    assert r.n_valid_post == 1 and not r.valid and np.isnan(r.y)
    assert A.screen(df, 0.6, c=1.0).iloc[0].valid  # passes at a looser threshold


def test_trend_uses_valid_days_and_flags_missing():
    c = 1.0
    frp = {("m3", "N"): np.e ** 1 - c, ("m2", "N"): np.e ** 2 - c, ("m1", "N"): np.e ** 3 - c}
    cc = {(k, "N20"): 0.9 for k in ["m3", "m2", "m1"]}  # only N counts
    r = A.screen(window(frp, cc), 0.3, c).iloc[0]
    assert not r.trend_missing and r.trend == pytest.approx(1.0)  # log FRP rises 1 per day

    cc_missing = {**cc, ("m3", "N"): 0.9, ("m2", "N"): 0.9}  # only d-1 is clear
    r = A.screen(window(frp, cc_missing), 0.3, c).iloc[0]
    assert r.trend_missing and r.trend == 0

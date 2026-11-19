"""Timing edge cases for process/windows.py on a tiny synthetic fire."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "process"))
import windows as W  # noqa: E402

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
D = pd.Timestamp


def det_row(t, sat, frp, x=0.0, y=0.0, daynight="D", fire_id=1):
    t = T(t)
    return dict(fire_id=fire_id, satellite=sat, time_utc=t, date=t.tz_localize(None).normalize(),
                frp=frp, scan=0.4, track=0.4, daynight=daynight, x=x, y=y)


@pytest.fixture
def fire():
    rows = [
        det_row("2021-08-01 20:40", "N20", 5), det_row("2021-08-01 21:30", "N", 5),
        det_row("2021-08-02 20:10", "N20", 4), det_row("2021-08-02 20:12", "N20", 6),  # one orbit: sum 10
        det_row("2021-08-02 21:50", "N20", 30),                                         # next orbit: 30
        det_row("2021-08-02 21:20", "N", 5),
        det_row("2021-08-03 20:30", "N20", 5),                                          # N has no detection
        det_row("2021-08-03 09:00", "N", 50, daynight="N"),                              # night: not an observation
    ]
    fires = pd.DataFrame({"fire_id": [1], "start": [D("2021-08-01")], "end": [D("2021-08-03")]})
    return pd.DataFrame(rows), fires


def test_orbits_are_averaged_and_night_ignored(fire):
    det, fires = fire
    obs = W.build_observations(det, fires).set_index(["satellite", "date"])
    o = obs.loc[("N20", D("2021-08-02"))]
    assert o.n_orbits == 2 and o.n_det == 3 and o.frp == pytest.approx(20.0)  # mean of 10 and 30
    assert obs.loc[("N", D("2021-08-03"))].frp == 0


def test_missing_overpass_uses_fire_median_time(fire):
    det, fires = fire
    obs = W.build_observations(det, fires).set_index(["satellite", "date"])
    o = obs.loc[("N", D("2021-08-03"))]
    assert o.time_imputed and o.n_det == 0
    assert o.t_first == o.t_last == T("2021-08-03 21:25")  # median of N times 21:30 and 21:20


def test_window_boundaries_follow_overpasses(fire):
    det, fires = fire
    obs = W.build_observations(det, fires)
    w = W.add_timing(W.candidate_days(fires), obs).set_index("d").loc[D("2021-08-03")]
    assert w.pre_start == T("2021-08-01 20:40")   # earliest pass on d-2
    assert w.w_start == T("2021-08-02 21:50")     # latest pass on d-1 (second N20 orbit)
    assert w.w_end == T("2021-08-03 20:30")       # earliest pass on d (N20, actual)
    assert w.post_end == T("2021-08-04 21:25")    # latest pass on d+1 (both imputed; N later)
    assert w.ante_start == T("2021-07-25 20:40")
    assert w.n_pre_day_det == 6
    assert w.fire_age == 2


def test_candidate_days_season_and_2025_cutoff():
    fires = pd.DataFrame({"fire_id": [1, 2], "start": [D("2021-05-25"), D("2025-09-20")],
                          "end": [D("2021-06-03"), D("2025-10-05")]})
    w = W.candidate_days(fires)
    f1, f2 = w[w.fire_id == 1].d, w[w.fire_id == 2].d
    assert f1.min() == D("2021-06-01") and f1.max() == D("2021-06-05")  # May dropped; runs to end + 2
    assert f2.min() == D("2025-09-22") and f2.max() == D("2025-09-29")  # 2025 stops at Sep 29


def test_footprint_lookback_extends_to_5_days_then_drops():
    det = pd.DataFrame([det_row("2021-08-05 21:00", "N", 5, x=1000, y=1000),
                        det_row("2021-08-05 21:00", "N", 5, x=3000, y=1000)])
    w = pd.DataFrame({"fire_id": [1, 1], "d": [D("2021-08-10"), D("2021-08-12")]})
    gw = W.add_footprints(w, det)
    assert len(gw) == 1 and gw.iloc[0].footprint_lookback == 5  # d-5 found; d=08-12 has nothing in 5 days
    assert gw.iloc[0].footprint_km2 == pytest.approx(2 * 2 * 5 + 3.14159 * 25, rel=0.05)  # 2 km line buffered 5 km

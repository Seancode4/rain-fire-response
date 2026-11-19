"""Stage 5c: checks added after the main results were seen. All at the 30% cutoff.

Reads  data/final/analysis.parquet
Writes analysis/output/robustness.csv          dose-response under each check
       analysis/output/robustness_heldout.csv  PARDIG vs IMERG held-out comparison under each check

1. Placebo: rain in the 24 h after the post-period (can't affect y), entered as bins next to the real W bins.
2. Cloud control: mean cloud cover of the valid pre- and post-period looks, in the dose-response model and
   the held-out PARDIG vs IMERG comparison.
3. Bounding model: post-period temperature and RH added (both caused by the storm), so the rain effect is
   estimated with the storm's air held fixed. A lower bound on the effect of the rain itself.
4. PARDIG at 30 minutes: the held-out comparison (Table 2) with PARDIG coarsened to IMERG's time slots.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import models as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLOUD = " + cc_pre_valid + cc_post_valid"
POST_AIR = " + t2m_post + rh_post"


def dr_rows(check, dr):
    return [dict(check=check, model="dose-response", term=f"{r.fuel} {r.bin} mm", est=r.est, lo=r.lo, hi=r.hi, p=r.p)
            for _, r in dr.iterrows()]


def main():
    t = pd.read_parquet(ROOT / "data/final/analysis.parquet")
    d = M.data(t, M.MAIN)
    d["lead_bin"] = pd.Categorical(
        pd.cut(d.pardig_lead, [-np.inf, 0.1, 2, 5, np.inf], labels=M.BINS, right=False).astype(str), M.BINS)
    rows, ho = [], []

    base_dr, _ = M.dose_response(d)
    rows += dr_rows("main (reference)", base_dr)
    ho.append({"check": "main (reference)", **M.heldout(d)})

    # 1. placebo
    pl_dr, m = M.dose_response(d, extra=" + C(lead_bin, Treatment('<0.1'))")
    rows += dr_rows("placebo: W effects with lead-rain bins added", pl_dr)
    for b in M.BINS[1:]:
        k = f"C(lead_bin, Treatment('<0.1'))[T.{b}]"
        lo, hi = m.conf_int().loc[k]
        rows.append(dict(check="placebo: lead-rain bins (should be ~0)", model="dose-response",
                         term=f"lead {b} mm (n={(d.lead_bin == b).sum()})", est=m.params[k], lo=lo, hi=hi,
                         p=m.pvalues[k]))

    # 2. cloud control
    rows += dr_rows("cloud control", M.dose_response(d, extra=CLOUD)[0])
    ho.append({"check": "cloud control", **M.heldout(d, extra=CLOUD)})

    # 3. bounding model
    rows += dr_rows("bounding: post-storm temperature and RH held fixed", M.dose_response(d, extra=POST_AIR)[0])

    # 4. PARDIG at IMERG's 30-minute resolution
    ho.append({"check": "PARDIG coarsened to 30 min", **M.heldout(d, pardig="pardig30")})

    out = pd.DataFrame(rows)
    out["pct"], out["pct_lo"], out["pct_hi"] = M.pct(out.est), M.pct(out.lo), M.pct(out.hi)
    out.round(4).to_csv(M.OUT / "robustness.csv", index=False)

    pd.DataFrame(ho).round(4).to_csv(M.OUT / "robustness_heldout.csv", index=False)

    for check, g in out.groupby("check", sort=False):
        print(f"\n{check}")
        for _, r in g.iterrows():
            print(f"    {r.term:<26} {r.pct:+6.1f}%  [{r.pct_lo:+6.1f}, {r.pct_hi:+6.1f}]  p={r.p:.2g}")
    print("\nPARDIG vs IMERG, held-out prediction under each check")
    for r in ho:
        M.print_heldout(r, r["check"])


if __name__ == "__main__":
    main()

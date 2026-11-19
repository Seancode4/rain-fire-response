"""Stage 5b: models.

Reads  data/final/analysis.parquet
Writes analysis/output/fig1_dose_response.png
       analysis/output/dose_response.csv       bin x fuel estimates, all thresholds (+ snow-dropped, near-extinguished)
       analysis/output/table2_heldout.csv      PARDIG vs IMERG: held-out prediction of fire response (Table 2)
       analysis/output/joint_model.csv         both products in one regression (supporting detail)

Rain controls enter as log(1 + mm). Effects are on y = log((FRP_post + c) / (FRP_pre + c)) and are shown
as percent change, 100 * (exp(b) - 1), relative to windows with < 0.1 mm of W rain.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/output"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
FUEL_COLOR = {"forest": "#2a78d6", "non-forest": "#eb6834"}
BINS = ["<0.1", "0.1-2", "2-5", "5+"]
THRESHOLDS = [0.2, 0.3, 0.4]
MAIN = 0.3
CV_FOLDS, CV_REPEATS, N_BOOT = 5, 5, 2000

FIRE = "log_frp_pre + log_pre_det + trend + trend_missing + fire_age + pix_pre"
WEATHER = "t2m_pre + rh_pre + vpd_pre + wind_post + C(month)"


def rain_controls(p):
    return f"np.log1p({p}_pre) + np.log1p({p}_post) + np.log1p({p}_ante7)"


def fit(formula, d):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.fire_id})


def data(t, thr, drop_snow=False):
    d = t[(t.cc_thr == thr) & t.valid].copy()
    if drop_snow:
        d = d[~d.snow_flag]
    d["pardig_bin"] = pd.Categorical(d.pardig_bin.astype(str), BINS)
    d["near_ext"] = d.near_ext.astype(float)
    d["trend_missing"] = d.trend_missing.astype(float)
    return d


def dose_response(d, outcome="y", extra=""):
    """Bin x fuel effects relative to < 0.1 mm, from one model with the interaction. extra: added terms."""
    f = (f"{outcome} ~ C(pardig_bin, Treatment('<0.1')) * C(fuel, Treatment('forest')) + "
         f"{FIRE} + {WEATHER} + {rain_controls('pardig')}{extra}")
    m = fit(f, d)
    names = m.params.index
    rows = []
    for fuel in ["forest", "non-forest"]:
        for b in BINS[1:]:
            main = f"C(pardig_bin, Treatment('<0.1'))[T.{b}]"
            terms = [main] + ([f"{main}:C(fuel, Treatment('forest'))[T.non-forest]"] if fuel == "non-forest" else [])
            L = np.array([[1.0 if n in terms else 0.0 for n in names]])
            r = m.t_test(L)
            est, se = float(r.effect[0]), float(r.sd[0][0])
            n = int(((d.pardig_bin == b) & (d.fuel == fuel)).sum())
            rows.append(dict(fuel=fuel, bin=b, est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se,
                             p=float(r.pvalue), n=n, fires=d[(d.pardig_bin == b) & (d.fuel == fuel)].fire_id.nunique()))
    return pd.DataFrame(rows), m


def both_products(d, extra="", pardig="pardig"):
    """pardig: column prefix for PARDIG rain ("pardig", or "pardig30" for the 30-minute version)."""
    pardig_controls = rain_controls(pardig) if pardig == "pardig" else \
        f"np.log1p({pardig}_pre) + np.log1p({pardig}_post) + np.log1p(pardig_ante7)"
    controls = f"{FIRE} + {WEATHER} + C(fuel) + {pardig_controls} + {rain_controls('imerg')}{extra}"
    rows = []
    for label, rain in [("PARDIG alone", [pardig]), ("IMERG alone", ["imerg"]), ("both", [pardig, "imerg"])]:
        m = fit(f"y ~ {' + '.join(f'np.log1p({p}_w)' for p in rain)} + {controls}", d)
        for p in rain:
            k = f"np.log1p({p}_w)"
            lo, hi = m.conf_int().loc[k]
            rows.append(dict(model=label, product=p.upper(), coef=m.params[k], lo=lo, hi=hi, p=m.pvalues[k],
                             r2=m.rsquared, n=int(m.nobs)))
    return pd.DataFrame(rows)


def pardig_rain_terms(pardig="pardig"):
    """PARDIG W rain plus its rain controls; pardig30 has no 7-day series, so it reuses the 15-min one."""
    ante = "pardig_ante7"
    return f"np.log1p({pardig}_w) + np.log1p({pardig}_pre) + np.log1p({pardig}_post) + np.log1p({ante})"


def heldout(d, extra="", pardig="pardig"):
    """Grouped cross-validation: how much each product's rain improves prediction of y on unseen fires.

    Folds hold out whole fires; 5 folds x 5 random fold assignments, squared errors averaged per window.
    The PARDIG-vs-IMERG difference in squared error is bootstrapped over fires.
    """
    import patsy

    base = f"{FIRE} + {WEATHER} + C(fuel){extra}"
    specs = {"no rain": base,
             "PARDIG": f"{base} + {pardig_rain_terms(pardig)}",
             "IMERG": f"{base} + np.log1p(imerg_w) + {rain_controls('imerg')}"}
    d = d.reset_index(drop=True)
    fires = d.fire_id.unique()
    sq = {}
    for name, rhs in specs.items():
        y, X = patsy.dmatrices(f"y ~ {rhs}", d, return_type="dataframe")
        y, X = y.values.ravel(), X.values
        acc = np.zeros(len(d))
        for rep in range(CV_REPEATS):
            perm = np.random.default_rng(rep).permutation(fires)
            fold = d.fire_id.map(dict(zip(perm, np.arange(len(perm)) % CV_FOLDS))).values
            for k in range(CV_FOLDS):
                tr, te = fold != k, fold == k
                beta = OLS(y[tr], X[tr]).fit().params
                acc[te] += (y[te] - X[te] @ beta) ** 2
        sq[name] = acc / CV_REPEATS
    sst = ((d.y - d.y.mean()) ** 2).sum()
    r2 = {k: 1 - v.sum() / sst for k, v in sq.items()}

    def boot(mask):
        g = pd.DataFrame({"fire": d.fire_id[mask], "e": (sq["PARDIG"] - sq["IMERG"])[mask]}).groupby("fire").e.agg(["sum", "size"])
        rng = np.random.default_rng(1)
        draws = []
        for _ in range(N_BOOT):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            draws.append(s["sum"].sum() / s["size"].sum())
        draws = np.array(draws)
        return g["sum"].sum() / g["size"].sum(), *np.percentile(draws, [2.5, 97.5]), (draws < 0).mean()

    rainy = ((d.pardig_w >= 0.1) | (d.imerg_w >= 0.1)).values
    mse_all, lo_all, hi_all, pb_all = boot(np.ones(len(d), bool))
    mse_wet, lo_wet, hi_wet, pb_wet = boot(rainy)
    gain_p, gain_i = r2["PARDIG"] - r2["no rain"], r2["IMERG"] - r2["no rain"]
    return dict(n=len(d), n_rainy=int(rainy.sum()), r2_no_rain=r2["no rain"], r2_pardig=r2["PARDIG"],
                r2_imerg=r2["IMERG"], gain_pardig=gain_p, gain_imerg=gain_i, gain_ratio=gain_p / gain_i,
                dmse=mse_all, dmse_lo=lo_all, dmse_hi=hi_all, pardig_better_share=pb_all,
                dmse_rainy=mse_wet, dmse_rainy_lo=lo_wet, dmse_rainy_hi=hi_wet, pardig_better_share_rainy=pb_wet)


def print_heldout(r, label):
    print(f"    {label}: held-out R2 no rain {r['r2_no_rain']:.3f} | PARDIG {r['r2_pardig']:.3f} | IMERG {r['r2_imerg']:.3f}"
          f"  -> PARDIG gain {r['gain_ratio']:.2f}x IMERG's")
    print(f"      PARDIG - IMERG squared error: all {r['dmse']:+.4f} [{r['dmse_lo']:+.4f}, {r['dmse_hi']:+.4f}], "
          f"PARDIG better in {r['pardig_better_share']:.0%} of fire bootstraps; rainy (n={r['n_rainy']:,}) "
          f"{r['dmse_rainy']:+.4f} [{r['dmse_rainy_lo']:+.4f}, {r['dmse_rainy_hi']:+.4f}], {r['pardig_better_share_rainy']:.0%}")


def pct(x):
    return 100 * (np.exp(x) - 1)


def figure(dr):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    x = np.arange(len(BINS))
    for ax, thr in zip(axes, THRESHOLDS):
        ax.set_facecolor(SURFACE)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=MUTED, lw=1)
        for i, (fuel, col) in enumerate(FUEL_COLOR.items()):
            g = dr[(dr.thr == thr) & (dr.fuel == fuel) & (dr.outcome == "y") & ~dr.drop_snow].set_index("bin").reindex(BINS)
            est = np.r_[0, pct(g.est.values[1:])]
            lo, hi = np.r_[0, pct(g.lo.values[1:])], np.r_[0, pct(g.hi.values[1:])]
            xx = x + (i - 0.5) * 0.18
            ax.vlines(xx[1:], lo[1:], hi[1:], color=col, lw=2)
            ax.plot(xx, est, color=col, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=fuel.capitalize(), zorder=3)
            if thr == MAIN:
                ax.annotate(fuel.capitalize(), (xx[-1], est[-1]), textcoords="offset points", xytext=(8, 0),
                            va="center", fontsize=8.5, color=INK2)
        ax.set_xticks(x, ["< 0.1\n(ref.)", "0.1–2", "2–5", "5+"])
        ax.set_xlim(-0.4, 3.6)
        n = dr[(dr.thr == thr) & (dr.outcome == "y") & ~dr.drop_snow].n_total.iloc[0]
        ax.set_title(f"Cloud cutoff {thr:.0%}{'  (main)' if thr == MAIN else ''}   n = {n:,}",
                     loc="left", fontsize=9.5, color=INK if thr == MAIN else INK2)
        ax.set_xlabel("PARDIG rain between overpasses (mm)", color=INK2, fontsize=9)
    axes[0].set_ylabel("Change in fire radiative power vs. no rain", color=INK2, fontsize=9)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
    fig.suptitle("More rain, bigger drop in fire power", x=0.06, ha="left", y=1.04, fontsize=13, color=INK)
    fig.text(0.06, 0.965, "Western US wildfires 2019–2025. Next-day FRP vs. the day before the rain, adjusted for fire "
             "state, weather, season and other rain. Bars: 95% CI, clustered by fire.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "fig1_dose_response.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(ROOT / "data/final/analysis.parquet")

    dr_all, bp_all, ho_all = [], [], []
    for thr in THRESHOLDS:
        d = data(t, thr)
        dr, _ = dose_response(d)
        dr_all.append(dr.assign(thr=thr, outcome="y", drop_snow=False, n_total=len(d)))
        bp_all.append(both_products(d).assign(thr=thr))
        ho_all.append({"thr": thr, **heldout(d)})
    d = data(t, MAIN, drop_snow=True)
    dr_all.append(dose_response(d)[0].assign(thr=MAIN, outcome="y", drop_snow=True, n_total=len(d)))
    d = data(t, MAIN)
    dr_all.append(dose_response(d, "near_ext")[0].assign(thr=MAIN, outcome="near_ext", drop_snow=False, n_total=len(d)))

    dr = pd.concat(dr_all, ignore_index=True)
    dr["pct"], dr["pct_lo"], dr["pct_hi"] = pct(dr.est), pct(dr.lo), pct(dr.hi)
    bp = pd.concat(bp_all, ignore_index=True)
    dr.round(4).to_csv(OUT / "dose_response.csv", index=False)
    bp.round(4).to_csv(OUT / "joint_model.csv", index=False)
    ho = pd.DataFrame(ho_all)
    ho.round(4).to_csv(OUT / "table2_heldout.csv", index=False)
    figure(dr)

    show = dr[(dr.outcome == "y") & ~dr.drop_snow]
    print("Dose-response: % change in FRP vs < 0.1 mm  [95% CI]  (n windows)")
    for thr in THRESHOLDS:
        print(f"\n  cutoff {thr:.0%}")
        for _, r in show[show.thr == thr].iterrows():
            print(f"    {r.fuel:<10} {r.bin:>5} mm: {r.pct:+6.1f}%  [{r.pct_lo:+6.1f}, {r.pct_hi:+6.1f}]  (n={r.n}, {r.fires} fires)")
    print("\nSnow-flagged windows dropped (30%):")
    for _, r in dr[dr.drop_snow].iterrows():
        print(f"    {r.fuel:<10} {r.bin:>5} mm: {r.pct:+6.1f}%  [{r.pct_lo:+6.1f}, {r.pct_hi:+6.1f}]")
    print("\nNear-extinguished (FRP_post < 10% of FRP_pre), change in probability vs < 0.1 mm (30%):")
    for _, r in dr[dr.outcome == "near_ext"].iterrows():
        print(f"    {r.fuel:<10} {r.bin:>5} mm: {100 * r.est:+5.1f} pts  [{100 * r.lo:+5.1f}, {100 * r.hi:+5.1f}]")
    print("\nTable 2: held-out prediction (grouped CV by fire)")
    for r in ho_all:
        print_heldout(r, f"cutoff {r['thr']:.0%}")
    print("\nJoint model (supporting detail): coefficient on log(1 + W rain mm)  [95% CI]")
    for thr in THRESHOLDS:
        print(f"\n  cutoff {thr:.0%}")
        for _, r in bp[bp.thr == thr].iterrows():
            print(f"    {r.model:<13} {r["product"]:<7} {r.coef:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]  p={r.p:.3g}  R2={r.r2:.3f}")


if __name__ == "__main__":
    main()

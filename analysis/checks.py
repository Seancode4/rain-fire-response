"""Stage 5a: checks that don't use the outcome.

Reads  data/final/cloud_validation.parquet, data/final/analysis.parquet
Writes analysis/output/fig2_cloud_validation.png, analysis/output/table1_dropout.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/output"
INK, INK2, MUTED, GRID, SURFACE, BLUE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb", "#2a78d6"
CC_BINS = np.round(np.arange(0, 1.01, 0.1), 1)
THRESHOLDS = [0.2, 0.3, 0.4]
N_BOOT = 1000


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def cloud_validation():
    v = pd.read_parquet(ROOT / "data/final/cloud_validation.parquet")
    v["bin"] = pd.cut(v.cc, CC_BINS, include_lowest=True, right=False)
    v.loc[v.cc >= 1.0, "bin"] = v.bin.cat.categories[-1]
    share = v.groupby("bin", observed=False).no_detection.agg(["mean", "size"])

    # 95% CI by resampling fires (observations from one fire aren't independent)
    rng = np.random.default_rng(0)
    fires = v.fire_id.unique()
    by_fire = {f: g for f, g in v.groupby("fire_id")}
    boots = []
    for _ in range(N_BOOT):
        s = pd.concat([by_fire[f] for f in rng.choice(fires, len(fires))])
        boots.append(s.groupby("bin", observed=False).no_detection.mean())
    boots = pd.concat(boots, axis=1)
    share["lo"], share["hi"] = boots.quantile(0.025, axis=1), boots.quantile(0.975, axis=1)
    share["mid"] = [b.left + 0.05 for b in share.index]

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style(ax)
    for t in THRESHOLDS:
        ax.axvline(t, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(t + 0.008, 0.98, f"{t:.0%}", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=7.5, color=INK2)
    ok = share["size"] > 0
    ax.fill_between(share.mid[ok], share.lo[ok], share.hi[ok], color=BLUE, alpha=0.15, lw=0)
    ax.plot(share.mid[ok], share["mean"][ok], color=BLUE, lw=2, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for x, n in zip(share.mid[ok], share["size"][ok]):  # counts along the bottom, clear of the line
        ax.text(x, 0.015, f"n={n:,}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, share.hi.max() * 1.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("ERA5 total cloud cover over the footprint at the overpass (dashed: tested cutoffs)", color=INK2, fontsize=9)
    ax.set_ylabel("Share of looks with no detection", color=INK2, fontsize=9)
    ax.set_title("Missed detections rise with cloud cover", loc="left", color=INK, fontsize=12, pad=18)
    fig.text(0.125, 0.905, "Dry days; fire detected on clear looks the day before and after. Band: 95% CI, "
             "resampling fires.", color=INK2, fontsize=8)
    fig.savefig(OUT / "fig2_cloud_validation.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Cloud validation: share of looks with no detection, by cloud cover")
    print(share[["size", "mean", "lo", "hi"]].round(3).to_string())


def dropout():
    t = pd.read_parquet(ROOT / "data/final/analysis.parquet")
    rows = []
    for b, g in t[t.cc_thr == 0.3].groupby("pardig_bin", observed=True):
        r = {"PARDIG W rain": b, "windows": len(g)}
        for thr in THRESHOLDS:
            r[f"kept at {thr:.0%}"] = t[(t.cc_thr == thr) & (t.pardig_bin == b)].valid.mean()
        for label, s in [("kept", g[g.valid]), ("dropped", g[~g.valid])]:
            r[f"rain mm, {label}"] = s.pardig_w.median()
            r[f"duration h, {label}"] = s.pardig_dur_h.median()
            r[f"Oct-Nov share, {label}"] = s.month.isin([10, 11]).mean()
            r[f"forest share, {label}"] = (s.fuel == "forest").mean()
            r[f"post cloud, {label}"] = s.cc_post_mean.mean()
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("PARDIG W rain")
    tab.round(3).to_csv(OUT / "table1_dropout.csv")
    print("\nDropout table (30% unless noted; medians for rain and duration, shares and mean post cloud otherwise)")
    print(tab.T.round(2).to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cloud_validation()
    dropout()


if __name__ == "__main__":
    main()

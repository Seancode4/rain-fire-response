"""Builds and executes analysis/results.ipynb, which reproduces every figure and table from data/final/."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

CELLS = [
    ("md", """# Rain and wildfire power: results

Reproduces every figure and table in the [README](../README.md) from `data/final/` (built by the pipeline; see the README's
*Reproducing* section): cloud screening, then the main results, then robustness checks."""),
    ("code", """import sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, str(Path().resolve()))
import checks, models, robustness

pd.set_option("display.float_format", "{:.3f}".format)
t = pd.read_parquet("../data/final/analysis.parquet")
v = t[(t.cc_thr == 0.3) & t.valid]
print(f"{t.window_id.nunique():,} fire-days after filters; {len(v):,} pass cloud screening at 30% "
      f"({v.fire_id.nunique():,} fires, {v.year.min()}-{v.year.max()})")
pd.crosstab(v.pardig_bin, v.fuel, margins=True)"""),
    ("md", """## 1. Cloud screening

**Figure 2.** On dry days, the share of overpasses that miss a fire that was detected on clear looks the day before
and after, by ERA5 cloud cover. It motivates cloud screening and supports the 30% cutoff."""),
    ("code", """checks.OUT.mkdir(parents=True, exist_ok=True)
checks.cloud_validation()
display(Image(checks.OUT / "fig2_cloud_validation.png", width=700))"""),
    ("md", """**Table 1.** The share of fire-days kept by cloud screening in each rain bin, and how kept windows differ from
dropped ones. This defines what the claim covers."""),
    ("code", """checks.dropout()
pd.read_csv(checks.OUT / "table1_dropout.csv", index_col=0).T"""),
    ("md", """## 2. Results

**Figure 1** (dose-response by fuel at three cloud cutoffs) and **Table 2** (PARDIG vs IMERG: how much each product's
rain improves prediction of fire response on held-out fires, 5-fold cross-validation grouped by fire)."""),
    ("code", """models.main()
display(Image(models.OUT / "fig1_dose_response.png", width=950))"""),
    ("code", """ho = pd.read_csv(models.OUT / "table2_heldout.csv")
ho[["thr", "n", "r2_no_rain", "r2_pardig", "r2_imerg", "gain_pardig", "gain_imerg", "gain_ratio",
    "pardig_better_share", "pardig_better_share_rainy"]]"""),
    ("md", """Supporting detail: both products in one regression. The two rain series are correlated (r = 0.78), so the
individual coefficients are imprecise; the held-out comparison above is the main result."""),
    ("code", """jm = pd.read_csv(models.OUT / "joint_model.csv")
jm.pivot_table(index=["thr", "model"], columns="product", values="coef", sort=False)"""),
    ("md", """## 3. Robustness

Placebo (rain after the measurement), cloud-cover control, the bounding model (post-storm temperature and humidity
held fixed), PARDIG coarsened to 30-minute slots, and the held-out comparison under the cloud control. These checks
were added after the main results were seen."""),
    ("code", "robustness.main()"),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, HERE / "results.ipynb")
    print(f"wrote {HERE / 'results.ipynb'}")


if __name__ == "__main__":
    main()

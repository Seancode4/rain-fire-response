# Rain and Wildfire Power

**How much does rain suppress active wildfires based on PARDIG and IMERG?**

Western US wildfires, June–November 2019–2025. 9,279 fire-days from 1,913 fires, measured with VIIRS active-fire data, with rain from PARDIG-NRT and IMERG-Final V07 (NASA). Figures and tables are reproduced by [`analysis/results.ipynb`](analysis/results.ipynb).

## Findings

1. **Fire power drops with rain amount.** Change in next-day fire radiative power (FRP) compared with a dry day, adjusted for fire state, weather, season and other rain:

   | Rain | Forest | Non-forest |
   |---|---|---|
   | 0.1–2 mm | −42% [−50, −33] | −22% [−44, +9] |
   | 2–5 mm | −66% [−74, −57] | −70% [−82, −50] |
   | 5+ mm | −85% [−89, −80] | −74% [−85, −55] |

   Non-forest has fewer rainy fire-days, so its intervals are wider.

2. **Most of the heavy-rain effect remains with post-storm air held fixed.** Controlling for post-storm temperature and humidity gives −74% [−81, −66] at 5+ mm and −18% [−28, −6] at 0.1–2 mm (forest). This is a lower bound on the effect of the rain itself.

3. **PARDIG predicts fire response better than IMERG.** On fires held out of training, adding PARDIG rain improves prediction 1.4× as much as adding IMERG-Final rain (R² gain 0.038 vs 0.028). PARDIG is ahead in 100% of fire-level bootstrap resamples and at every cloud cutoff.

![Dose-response](analysis/output/fig1_dose_response.png)

## Method

- **Fires:** VIIRS 375 m active-fire detections (Suomi-NPP and NOAA-20) clustered into fire events: 2 km within a day, linked across days with a 5-day lookback. Industrial sources (gas flares, plants) and cropland burns are removed. Spot checks match reported fires (Dixie 2021, Creek 2020).
- **Unit:** one fire × one day. FRP at the two daytime overpasses before the rain window is compared with the two after it. The rain window runs between overpasses (~23 h) rather than UTC days, so afternoon storms aren't split.
- **Outcome:** log((FRP_after + c) / (FRP_before + c)), reported as % change.
- **Clouds:** VIIRS can't see fires through cloud, so overpasses with ERA5 cloud cover ≥ 30% over the fire are dropped, and each period needs at least 2 clear overpasses.
- **Rain:** footprint-mean rain total during the window, with each 15- or 30-minute image weighted by its overlap with the window. Rain before, after and in the prior week enter as controls.
- **Model:** OLS on rain bin × fuel (forest / non-forest, from NLCD) with controls for fire state (FRP, detection count, recent trend, age), weather (temperature, humidity, VPD, wind) and month. Standard errors are clustered by fire.
- **PARDIG vs IMERG:** 5-fold cross-validation with whole fires held out, comparing how much each product's rain improves prediction over a model with no rain.

## Robustness

- Same curve at cloud cutoffs of 20, 30 and 40%.
- Placebo: rain after the measurement has no effect.
- Controlling for cloud cover: curve and PARDIG's edge unchanged (1.41×).
- PARDIG coarsened to 30 min: edge unchanged.

The placebo, cloud-control and held-out comparisons were added after the main results. Details in the [notebook](analysis/results.ipynb).

## Scope

- About 1 in 5 fire-days with 2 mm or more pass cloud screening, so results mostly reflect short summer storms between clear days.
- Firefighting effort is unobserved.

## Data sources

- **PARDIG-NRT**: Google DeepMind and Google Research ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **IMERG-Final V07**: NASA GPM
- **VIIRS active fire**: NASA [FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- **ERA5**: contains modified Copernicus Climate Change Service information (2026)
- **NLCD**: U.S. Geological Survey
- **State boundaries**: U.S. Census Bureau

Gridded data were accessed through Google Earth Engine. The data providers don't endorse this analysis.

---

## Reproducing

```
uv sync
uv run earthengine authenticate   # then set PROJECT in extract/ee_common.py to your Earth Engine Cloud project
```

FIRMS yearly country files (VIIRS S-NPP and NOAA-20, United States, 2019–2025) go in `data/raw/firms/`. Census state boundaries go in `data/raw/states/` ([cb_2023_us_state_20m.zip](https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip)).

| Step | Command                                                              | Output                                                    |
| ---- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| 1    | `uv run python process/events.py`                                    | fire events → `data/interim/`                             |
| 2    | `uv run python process/check_events.py`                              | sanity checks → `data/interim/checks/`                    |
| 3    | `uv run python process/windows.py`                                   | observations and windows → `data/interim/`                |
| 4    | `uv run python extract/rain.py` (~10 min)                            | PARDIG and IMERG rain per interval                        |
| 5    | `uv run python extract/rain_duration.py`, `extract/rain_pardig30.py` | rain duration, 30-min PARDIG                              |
| 6    | `uv run python extract/landcover.py` (~8 min)                        | NLCD fuel, cropland share                                 |
| 7    | `uv run python extract/era5.py` (~2 min)                             | cloud cover per overpass, weather                         |
| 8    | `uv run python process/assemble.py`                                  | `data/final/analysis.parquet`, `cloud_validation.parquet` |
| 9    | `uv run python analysis/checks.py`                                   | Figure 2, Table 1 (no outcome used)                       |
| 10   | `uv run python analysis/models.py`                                   | Figure 1, Table 2, all estimates                          |
| 11   | `uv run python analysis/robustness.py`                               | post-results checks                                       |

Or run steps 9–11 at once with the notebook `analysis/results.ipynb`. Earth Engine steps cache each batch in `data/interim/ee/` and resume if interrupted. Tests: `uv run pytest -q tests`.

Layout: `extract/` pulls data from Earth Engine; `process/` builds the analysis table; `analysis/` fits models and makes figures from `data/final/` only.

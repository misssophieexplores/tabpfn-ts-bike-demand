# TabPFN-TS for bike demand forecasting

Code, tuning results and evaluation results for hourly bike-sharing demand forecasts in Seoul, London and Washington D.C., with observed weather, with simulated weather-forecast errors, and without weather.

## Models

TabPFN, TabPFN_NoWeather, XGBoost, XGBoost_NoWeather, Seasonal_Naive, ARIMA, SARIMAX, Prophet, NeuralProphet, NeuralProphet_NoWeather (`forecasting/run_experiments.py`, `MODEL_KEYS`).

## Contents

- `forecasting/`: models, cross-validation, weather degradation (incl. the measured NWP error model and its calibration files), experiment runner, tuning scripts, tests
- `data/`: bike and weather data, their provenance and licences (`README_BIKES.md`, `README_WEATHER.md`, `provenance/README.md`)
- `results/`: aggregated and fold-level results of the 10 models (`results_master_v7calfix.csv`, `detailed_results_master_v7calfix.csv`), the tuned parameters (`tuning/`) and the calendar-feature check (`tabpfn_ts_calendar_check/`)
- `results_v7calfix/`: tables and figures
- `analysis_visualizations/`: the notebook and scripts that build the tables and figures
- `check_paper_numbers.py`: prints the numbers of the dataset summary
- `ARCHITECTURE.md`: design, configuration and result files in detail

## Data not included

`data/LondonBikeData.csv` and `data/WashingtonBikeData.csv` are not included: they contain visibility values from Visual Crossing, whose licence does not allow redistributing them. Rebuild both with your own Visual Crossing API key: `VISUALCROSSING_KEY=... python data/rebuild_london_washington.py` (details in `data/README_WEATHER.md`).

## Note on the TabPFN-TS calendar features

**Problem.** In `tabpfn-time-series` (we use 1.0.9; unchanged up to 1.4.0), `CalendarFeature.generate`
encodes each calendar index *x* ∈ {0, …, P−1} as sin/cos(2π·x / (P − 1)) instead of sin/cos(2π·x / P).
Because the indices already start at 0, the last value of each cycle lands on the same angle as the first:
23:00 and 00:00, Sunday and Monday, December and January, and 31 December and 1 January get identical
features. The model therefore cannot distinguish, for example, a Sunday from a Monday through the calendar
features. `forecasting/testing/tabpfn_ts_calendar_features.py` prints the collisions
(tables in `results/tabpfn_ts_calendar_check/`).

**Fix.** `forecasting/models/calendar_fix.py` replaces only this feature by the conventional encoding
sin/cos(2π·x / P), passed to the pipeline through its `temporal_features` argument. Model, checkpoint
(TabPFN v2.5), inputs, folds and all other features (running index, automatic seasonal features) are unchanged.

**Why we apply it.** The issue is acknowledged by the maintainers (GitHub issue #145). They did not merge the
correction (PR #146), because on 12 GIFT-Eval datasets it slightly worsened their default configuration
(about +0.6 % MASE). On our task, hourly bike demand with strong weekday/weekend and daily patterns,
the corrected encoding lowers the error in all 48 city × horizon × setting combinations (MASE −16 % with
observed weather, −15 % with forecast weather, −9 % without weather). We report TabPFN-TS with the corrected features;
the results with the library's encoding are in the previous versions of this archive on Zenodo.

**Scope.** Only TabPFN-TS (v2.5) is affected in this study. The baselines (XGBoost, Prophet, SARIMAX, …)
do not use these features.

## Changes in this version

TabPFN-TS is run with corrected cyclical calendar features (see "Note on the TabPFN-TS calendar features");
all TabPFN-TS results, figures and tables are updated accordingly. The other models are unchanged.

## Setup and run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Weights & Biases logging (wandb.init in forecasting/run_experiments.py):
# set WANDB_API_KEY and WANDB_ENTITY, e.g. in a .env file

python forecasting/main.py                         # all three cities
python forecasting/main.py --cities seoul          # one city
```

Tuning: `forecasting/models/tuning/` (each script documents its options at the top). Configuration: `forecasting/config.py` and `forecasting/config_<city>.py`.

Tables and figures (written to `results_v7calfix/`):

```bash
cd analysis_visualizations && jupyter nbconvert --to notebook --execute results_analysis.ipynb && cd ..
python analysis_visualizations/effort_forecast_runtime.py --v7calfix   # forecasting times
python analysis_visualizations/make_tab_effort.py --v7calfix           # effort table (tuning times from logs/tuning/)
python analysis_visualizations/make_fig_datasets.py --v7calfix         # dataset figure
```

The last cell of the notebook (daily rentals per city), `make_fig_datasets.py` and `check_paper_numbers.py` need the rebuilt London and Washington files (see "Data not included"); everything else runs from the repository alone.

`forecasting/build_v7calfix.py` documents how the two result files were assembled (corrected TabPFN-TS runs, MASE rescaled to the daily seasonal naive of each fold); its inputs are not part of this version.

## Disclaimer

Weather data may be inaccurate or incomplete. The data and results in this repository are provided as is, without any guarantee of accuracy, completeness or suitability for any purpose, and must not be used as the sole basis for critical decisions.

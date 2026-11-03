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

**Problem.** `tabpfn-time-series` (1.0.9 used here; unchanged up to 1.4.0) encodes its calendar features as sin/cos(2πx/(P−1)) instead of sin/cos(2πx/P). The last value of each cycle therefore gets the same features as the first: 23:00 and 00:00, Sunday and Monday, December and January. The maintainers have confirmed the issue (GitHub issue #145).

**Fix.** We fix the encoding for all TabPFN-TS results in this repository: `forecasting/models/calendar_fix.py` replaces only the calendar feature with sin/cos(2πx/P). Model (TabPFN v2.5), inputs, folds and all other features are unchanged. `forecasting/testing/tabpfn_ts_calendar_features.py` shows the collisions (tables in `results/tabpfn_ts_calendar_check/`). The other models do not use these features and are not affected.

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

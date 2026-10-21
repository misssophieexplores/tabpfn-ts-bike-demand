# TabPFN-TS for bike demand forecasting

Code, tuning results and evaluation results for hourly bike-sharing demand forecasts in Seoul, London and Washington D.C., with observed weather, with simulated weather-forecast errors, and without weather.

## Models

TabPFN, TabPFN_NoWeather, XGBoost, XGBoost_NoWeather, Seasonal_Naive, ARIMA, SARIMAX, Prophet, NeuralProphet, NeuralProphet_NoWeather (`forecasting/run_experiments.py`, `MODEL_KEYS`).

## Contents

- `forecasting/`: models, cross-validation, weather degradation (incl. the measured NWP error model and its calibration files), experiment runner, tuning scripts, tests
- `data/`: bike and weather data, their provenance and licences (`README_BIKES.md`, `README_WEATHER.md`, `provenance/README.md`)
- `results/`: aggregated and fold-level results of the 10 models (`results_master_v7.csv`, `detailed_results_master_v7.csv`, `comparative_metrics_v7.csv`) and the tuned parameters (`tuning/`)
- `results_v7/`: tables and figures
- `analysis_visualizations/`: notebooks and scripts that build the tables and figures
- `check_paper_numbers.py`: prints the numbers of the dataset summary
- `ARCHITECTURE.md`: design, configuration and result files in detail

## Data not included

`data/LondonBikeData.csv` and `data/WashingtonBikeData.csv` are not included: they contain visibility values from Visual Crossing, whose licence does not allow redistributing them. Rebuild both with your own Visual Crossing API key: `VISUALCROSSING_KEY=... python data/rebuild_london_washington.py` (details in `data/README_WEATHER.md`).

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

## Disclaimer

Weather data may be inaccurate or incomplete. The data and results in this repository are provided as is, without any guarantee of accuracy, completeness or suitability for any purpose, and must not be used as the sole basis for critical decisions.

"""
Forecasting time per fold of the paper models (effort table, appendix).

Input
-----
results/results_master_v7.csv (path relative to the repository root)

    Written by forecasting/run_experiments.py: one row per experiment, i.e.
    city (column `dataset`: seoul, london, washington) x model x horizon
    (6, 24, 48, 168 h) x weather scenario (clean_only = observed weather,
    degraded = weather forecast errors). This script uses only the rows of
    the 10 paper models (MODELS below).

    These rows come from the main run of 7-10 Oct 2026
    (run 20261007_185700_98536, cluster jobs 303 and 304): 10 models x 3 cities
    x 4 horizons in clean_only, plus SARIMAX, XGBoost, NeuralProphet and TabPFN
    in degraded = 168 rows. Models without weather covariates have no degraded
    rows (their forecast would be identical).

    Runtime columns used here (seconds; timed in forecasting/fold_runner.py
    with time.perf_counter):
        fit_time_mean_s      model.fit() per fold, mean over the folds
        predict_time_mean_s  model.predict() per fold, mean over the folds
        runtime_mean_s       fit + predict per fold, mean over the folds
        runtime_std_s        standard deviation of fit + predict over the folds
        fit_time_total_s, predict_time_total_s, runtime_total_s
                             the same, summed over all folds
        n_folds              folds per experiment: 980 (6 h), 245 (24 h),
                             123 (48 h), 35 (168 h)
    Not included in these times: data and feature preparation, weather
    degradation, scoring.

    What one fold is: fit on the 720-h lookback window, forecast the test
    window (horizon h). TabPFN does all its work in predict() (zero-shot;
    fit() only stores the context), so fit + predict is the comparable number
    for all models.

Hardware and threads
--------------------
    All rows: one SLURM job with 16 CPU cores and 32 GB on NOVA DAH 02
    (2 x 56-core x86-64 CPUs, 2 TB RAM). All models ran on CPU; the server's
    GPUs were not used.
    - SARIMAX fits 16 folds at the same time, each in its own process with ONE
      thread (forecasting/fold_runner.py). Its runtime is the time of one fold
      on one core.
    - All other models fit their folds one after another, with all 16 cores
      available to the model.

Output (relative to the repository root)
----------------------------------------
    results_v7/tables/paper/effort_forecast_runtime_long.csv
        one row per city x model x horizon x scenario (the 168 rows, runtime
        columns only)
    results_v7/tables/paper/effort_forecast_runtime_per_fold.csv
        seconds per fold (fit + predict), observed weather, mean of the three
        cities, one row per model, one column per horizon
    and the same table printed to the screen, with checks.

Run
---
    python analysis_visualizations/effort_forecast_runtime.py
    (from any folder; the repository root is found by looking for
    results/results_master_v7.csv in the script's folder and every folder
    above it. In a notebook cell the search starts in the working folder.)
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RESULTS_FILE = Path("results") / "results_master_v7.csv"     # relative to the repo root
OUTPUT_DIR = Path("results_v7") / "tables" / "paper"          # relative to the repo root

# Paper models, in the order of the table (names as in the results file)
MODELS = [
    "Seasonal_Naive",
    "ARIMA",
    "SARIMAX",
    "Prophet",
    "NeuralProphet",
    "NeuralProphet_NoWeather",
    "XGBoost",
    "XGBoost_NoWeather",
    "TabPFN",
    "TabPFN_NoWeather",
]
MODELS_WITH_WEATHER = ["SARIMAX", "XGBoost", "NeuralProphet", "TabPFN"]  # also run in degraded
CITIES = ["seoul", "london", "washington"]
HORIZONS = [6, 24, 48, 168]
FOLDS_PER_HORIZON = {6: 980, 24: 245, 48: 123, 168: 35}

# The main run (jobs 303 and 304) wrote its rows between these two times
MAIN_RUN_START = pd.Timestamp("2026-10-07 18:57:00")
MAIN_RUN_END = pd.Timestamp("2026-10-10 21:21:00")

KEY = ["dataset", "model", "horizon", "weather_scenario"]
RUNTIME_COLUMNS = [
    "n_folds", "fit_time_mean_s", "predict_time_mean_s", "runtime_mean_s",
    "runtime_std_s", "fit_time_total_s", "predict_time_total_s", "runtime_total_s",
]


def find_repo_root() -> Path:
    """Folder that contains results/results_master_v7.csv: the script's folder or
    any folder above it (in a notebook: the working folder or any folder above)."""
    try:
        start = Path(__file__).resolve().parent
    except NameError:                      # notebook cell: no __file__
        start = Path.cwd().resolve()
    for folder in [start, *start.parents]:
        if (folder / RESULTS_FILE).is_file():
            return folder
    raise FileNotFoundError(
        f"{RESULTS_FILE} not found in {start} or any folder above it. "
        f"Put this script in analysis_visualizations/ of the repository."
    )


def load_paper_rows(path: Path) -> pd.DataFrame:
    """The 168 rows of the paper models from the main run, checked."""
    df = pd.read_csv(path, usecols=KEY + ["timestamp", "git_commit"] + RUNTIME_COLUMNS)
    df = df[df["model"].isin(MODELS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Exactly one row per experiment
    duplicated = df[df.duplicated(KEY, keep=False)]
    if len(duplicated):
        raise ValueError(
            "More than one row for these experiments (city, model, horizon, scenario):\n"
            + duplicated.sort_values(KEY)[KEY + ["timestamp"]].to_string(index=False)
        )

    # Every expected experiment is present, and nothing else
    expected = {(c, m, h, "clean_only") for c in CITIES for m in MODELS for h in HORIZONS}
    expected |= {(c, m, h, "degraded") for c in CITIES for m in MODELS_WITH_WEATHER for h in HORIZONS}
    found = set(map(tuple, df[KEY].itertuples(index=False)))
    if found != expected:
        raise ValueError(
            f"Missing experiments: {sorted(expected - found)}\n"
            f"Unexpected experiments: {sorted(found - expected)}"
        )

    # All rows from the main run of 7-10 Oct 2026
    outside = df[(df["timestamp"] < MAIN_RUN_START) | (df["timestamp"] > MAIN_RUN_END)]
    if len(outside):
        raise ValueError(
            f"{len(outside)} rows were not written by the main run "
            f"({MAIN_RUN_START} to {MAIN_RUN_END}):\n"
            + outside[KEY + ["timestamp"]].to_string(index=False)
        )

    # Every experiment has all its folds
    wrong_folds = df[df["n_folds"] != df["horizon"].map(FOLDS_PER_HORIZON)]
    if len(wrong_folds):
        raise ValueError("Experiments with a wrong number of folds:\n"
                         + wrong_folds[KEY + ["n_folds"]].to_string(index=False))
    return df


def per_fold_table(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds per fold (fit + predict), observed weather, mean of the three cities:
    one row per model, one column per horizon, plus the share of fit() in the total."""
    clean = df[df["weather_scenario"] == "clean_only"]
    table = (clean.pivot_table(index="model", columns="horizon",
                               values="runtime_mean_s", aggfunc="mean")
                  .reindex(index=MODELS, columns=HORIZONS))
    table.columns = [f"{h}h_s_per_fold" for h in HORIZONS]
    fit_share = (clean.groupby("model")["fit_time_total_s"].sum()
                 / clean.groupby("model")["runtime_total_s"].sum())
    table["fit_share_pct"] = (100 * fit_share).reindex(MODELS)
    table.index.name = "model"
    return table


def degraded_vs_clean(df: pd.DataFrame) -> pd.Series:
    """Largest difference (%) between degraded and observed weather in seconds per fold,
    over cities and horizons, for the four models run in both scenarios."""
    wide = df.pivot_table(index=["model", "dataset", "horizon"], columns="weather_scenario",
                          values="runtime_mean_s").dropna()
    change = 100 * (wide["degraded"] / wide["clean_only"] - 1)
    return change.abs().groupby("model").max().reindex(MODELS_WITH_WEATHER)


def main() -> None:
    root = find_repo_root()
    df = load_paper_rows(root / RESULTS_FILE)

    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    long_file = out_dir / "effort_forecast_runtime_long.csv"
    table_file = out_dir / "effort_forecast_runtime_per_fold.csv"

    (df[KEY + RUNTIME_COLUMNS + ["timestamp", "git_commit"]]
       .sort_values(["model", "dataset", "horizon", "weather_scenario"])
       .to_csv(long_file, index=False))
    table = per_fold_table(df)
    table.round(3).to_csv(table_file)

    print(f"Input: {RESULTS_FILE} | {len(df)} rows (10 models, main run "
          f"{df['timestamp'].min():%d %b %H:%M} to {df['timestamp'].max():%d %b %Y %H:%M}) | "
          f"code: {', '.join(sorted(df['git_commit'].astype(str).unique()))}")
    print()
    print("Seconds per fold (fit + predict), observed weather, mean of 3 cities")
    print("(SARIMAX: one core per fold; all other models: 16 cores)")
    print(table.to_string(float_format=lambda v: f"{v:8.3f}"))
    print()
    print("Check: largest difference degraded vs observed weather (%), seconds per fold")
    print(degraded_vs_clean(df).to_string(float_format=lambda v: f"{v:6.1f}"))
    print()
    print(f"Written: {long_file.relative_to(root)}")
    print(f"         {table_file.relative_to(root)}")


if __name__ == "__main__":
    main()

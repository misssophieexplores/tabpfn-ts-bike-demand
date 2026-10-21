"""
Appendix table of the computational effort of the paper models (tab:effort).

For each model: the tuning time per city and the forecasting time per fold.
Writes results_v7/tables/paper/table_effort.tex (and table_effort.csv) and
prints the LaTeX.

Also writes results_v7/tables/paper/effort_stability.csv: two numbers that show
how stable the forecasting times are (observed weather only, i.e.
weather_scenario == clean_only; computed from the same 168 checked rows that
effort_forecast_runtime.py writes to effort_forecast_runtime_long.csv):
   X = largest runtime_std_s / runtime_mean_s (spread of the time per fold
       across the folds of one experiment) over TabPFN, TabPFN_NoWeather,
       XGBoost and XGBoost_NoWeather, all cities and horizons, in %.
   Y = largest (max over cities / min over cities - 1) of runtime_mean_s
       (difference between the cities) over the horizons and over TabPFN and
       TabPFN_NoWeather, in %.
   Both rounded to one decimal; the file also names the model, city and horizon
   where each maximum occurs.

Inputs (paths relative to the repository root)
----------------------------------------------
1. results/results_master_v7.csv -> forecasting time per fold
   Read and checked by effort_forecast_runtime.py (same folder; see its header
   for the file and its runtime columns): the 168 rows of the 10 paper models
   from the main run of 7-10 Oct 2026. The time per fold is fit() + predict()
   of one fold: fit on the 720-h lookback window, forecast the test window.

2. forecasting/config_<city>.py -> which tuning result each model used
   The `*_params_file` paths that the experiments read (ARIMA, SARIMAX,
   XGBoost, XGBoost_NoWeather, Prophet, NeuralProphet, NeuralProphet_NoWeather).
   SARIMAX: the config names the re-selected file
   results/tuning/sarimax_best_params_<city>_clean_only_720_converged.json
   (written in seconds by main_job.sh on 7 Oct, nothing refitted); its field
   reselection.source_file names the tuning file it was made from.

3. logs/tuning/tune_<model>_<city>_<jobid>.log -> tuning time per city
   One log per tuning job (written by tune_job.sh, xgboost_tune_job.sh,
   sarimax_tune_job.sh). Line 1: "Job <id> | <model> | <city> | <node> |
   CPUs <n> | <start time>"; near the end "Results saved to: <params file>"
   and "Finished: <end time>". For every model and city the script takes the
   ONE log whose "Results saved to:" is the params file of input 2, so jobs
   that were rerun or never used (e.g. the 3 Oct XGBoost and SARIMAX jobs that
   crashed while saving) are left out without listing job numbers here.
   Tuning time = end - start of that job: the whole search plus the final
   check on the 90 tuning folds (search settings: Section 3.2 of the paper).

Hardware
--------
   All jobs ran on CPU on NOVA DAH 02 (2 x 56-core x86-64 CPUs, 2 TB RAM);
   no job requested a GPU and none of these models used one.
   Tuning jobs: the cores in each log header (4; XGBoost 8).
   Forecasting: one job with 16 cores (submit_main.sh). SARIMAX fitted 16
   folds at a time, each in its own process on one core; all other models
   fitted their folds one after another with all 16 cores.

Run
---
   python analysis_visualizations/effort_forecast_runtime.py   (optional: its own outputs)
   python analysis_visualizations/make_tab_effort.py
   (from any folder; also works pasted into a notebook cell whose working
   folder is inside the repository)
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# effort_forecast_runtime.py lives in the same folder as this script
try:
    _HERE = Path(__file__).resolve().parent
except NameError:                                   # notebook cell: no __file__
    _HERE = next(p / "analysis_visualizations" for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "analysis_visualizations" / "effort_forecast_runtime.py").is_file())
sys.path.insert(0, str(_HERE))
import effort_forecast_runtime as efr  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TUNING_LOG_DIR = Path("logs") / "tuning"                      # relative to the repo root
CONFIG_FILE = Path("forecasting") / "config_{city}.py"        # relative to the repo root
OUTPUT_DIR = efr.OUTPUT_DIR                                    # results_v7/tables/paper
TEX_FILE = "table_effort.tex"
CSV_FILE = "table_effort.csv"
STABILITY_FILE = "effort_stability.csv"

# Models of the two stability numbers (see the header)
STABILITY_X_MODELS = ["TabPFN", "TabPFN_NoWeather", "XGBoost", "XGBoost_NoWeather"]
STABILITY_Y_MODELS = ["TabPFN", "TabPFN_NoWeather"]

# Tuned models -> config attribute of their params file
PARAMS_ATTRIBUTE = {
    "ARIMA": "arima_params_file",
    "SARIMAX": "sarimax_params_file",
    "Prophet": "prophet_params_file",
    "NeuralProphet": "neuralprophet_params_file",
    "NeuralProphet_NoWeather": "neuralprophet_noweather_params_file",
    "XGBoost": "xgb_params_file",
    "XGBoost_NoWeather": "xgb_noweather_params_file",
}
FORECAST_JOB_CORES = 16                                        # submit_main.sh
HARDWARE = "one server with 2~$\\times$~56-core x86-64 CPUs and 2~TB RAM (NOVA DAH~02)"

HEADER_RE = re.compile(r"^Job (\d+) \| (.+?) \| (\w+) \| (\S+) \| CPUs (\d+) \| (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
SAVED_RE = re.compile(r"^Results saved to: (\S+)")
FINISHED_RE = re.compile(r"^Finished: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
CONFIG_RE = re.compile(r'^\s*config\.(\w+_params_file)\s*=\s*"([^"]+)"', re.MULTILINE)


# ---------------------------------------------------------------------------
# Tuning time
# ---------------------------------------------------------------------------
def params_files_used(root: Path, city: str) -> dict:
    """Model -> tuning file (path as written in the tuning log) the experiments used."""
    text = (root / str(CONFIG_FILE).format(city=city)).read_text()
    paths = dict(CONFIG_RE.findall(text))
    used = {}
    for model, attribute in PARAMS_ATTRIBUTE.items():
        if attribute not in paths:
            raise ValueError(f"{CONFIG_FILE.as_posix().format(city=city)}: config.{attribute} not set")
        path = paths[attribute]
        if model == "SARIMAX" and path.endswith("_converged.json"):
            reselected = json.loads((root / path).read_text())
            path = reselected["reselection"]["source_file"]
        used[model] = path
    return used


def read_tuning_logs(root: Path) -> dict:
    """Params file -> facts of the tuning log that wrote it."""
    log_dir = root / TUNING_LOG_DIR
    if not log_dir.is_dir():
        raise FileNotFoundError(f"{TUNING_LOG_DIR} not found in {root}: copy the cluster's "
                                f"logs/tuning/ folder there.")
    logs = {}
    for log in sorted(log_dir.glob("tune_*.log")):
        lines = log.read_text(errors="replace").splitlines()
        header = HEADER_RE.match(lines[0]) if lines else None
        saved = [m.group(1) for m in map(SAVED_RE.match, lines) if m]
        finished = [m.group(1) for m in map(FINISHED_RE.match, lines) if m]
        if not (header and saved and finished):
            continue                       # crashed or unfinished job: wrote no params file
        job, model_key, city, node, cores, start = header.groups()
        logs[saved[-1]] = {
            "log": log.name, "job": int(job), "city": city, "node": node, "cores": int(cores),
            "start": datetime.strptime(start, "%Y-%m-%d %H:%M:%S"),
            "end": datetime.strptime(finished[-1], "%Y-%m-%d %H:%M:%S"),
        }
    return logs


def tuning_times(root: Path) -> pd.DataFrame:
    """One row per tuned model and city: job, cores, node, tuning time (hours)."""
    logs = read_tuning_logs(root)
    rows = []
    for city in efr.CITIES:
        for model, params_file in params_files_used(root, city).items():
            if params_file not in logs:
                raise FileNotFoundError(
                    f"No log in {TUNING_LOG_DIR} says 'Results saved to: {params_file}' "
                    f"({model}, {city}).")
            log = logs[params_file]
            if log["city"] != city:
                raise ValueError(f"{log['log']}: city {log['city']}, expected {city}")
            hours = (log["end"] - log["start"]).total_seconds() / 3600
            rows.append({"model": model, "city": city, "job": log["job"], "log": log["log"],
                         "node": log["node"], "cores": log["cores"], "tuning_h": hours,
                         "params_file": params_file})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def fmt_tuning(lo_h: float, hi_h: float) -> str:
    """Range over the cities, in minutes below one hour, else in hours."""
    if hi_h < 1:
        lo, hi, unit = f"{60 * lo_h:.1f}", f"{60 * hi_h:.1f}", "min"
    else:
        lo, hi, unit = f"{lo_h:.1f}", f"{hi_h:.1f}", "h"
    return f"{lo} {unit}" if lo == hi else f"{lo}--{hi} {unit}"


def fmt_seconds(s: float) -> str:
    if s < 0.005:
        return "$<$0.01"
    if s < 10:
        return f"{s:.2f}"
    if s < 100:
        return f"{s:.1f}"
    return f"{s:.0f}"


def tex_esc(text: str) -> str:
    return text.replace("_", "\\_")


def build_table(tuning: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    per_fold = efr.per_fold_table(forecast).drop(columns="fit_share_pct")
    span = tuning.groupby("model")["tuning_h"].agg(["min", "max"])
    cores = tuning.groupby("model")["cores"].agg(lambda c: "/".join(map(str, sorted(set(c)))))
    table = per_fold.copy()
    table.insert(0, "tuning_h_min", span["min"].reindex(efr.MODELS))
    table.insert(1, "tuning_h_max", span["max"].reindex(efr.MODELS))
    table.insert(2, "tuning_cores", cores.reindex(efr.MODELS))
    return table


def to_latex(table: pd.DataFrame, tuning: pd.DataFrame, max_degraded_pct: float) -> str:
    cores = tuning.groupby("model")["cores"].first().reindex([m for m in efr.MODELS if m in PARAMS_ATTRIBUTE])
    base = int(cores.mode().iloc[0])
    groups = [(int(c), [tex_esc(m) for m in cores[cores == c].index])
              for c in sorted(set(cores)) if c != base]
    other_text = "".join(f" ({' and '.join(ms)}: {c})" for c, ms in groups)
    untuned = [tex_esc(m) for m in efr.MODELS if m not in PARAMS_ATTRIBUTE]
    caption = (
        "Computational effort per model. Tuning: wall-clock time of the tuning run "
        "(search and final check on the 90 tuning folds, Section~3.2), range over the "
        f"three cities, on {base} CPU cores{other_text}; "
        f"{', '.join(untuned[:-1])} and {untuned[-1]} are not tuned. "
        "Forecasting: time to fit the model and forecast one test window, mean over all "
        "folds and the three cities with observed weather (with forecast weather the "
        f"times differ by at most {max_degraded_pct:.0f}\\,\\%), in one run with "
        f"{FORECAST_JOB_CORES} CPU cores; SARIMAX fitted {FORECAST_JOB_CORES} folds at a "
        "time, each on one core, all other models one fold at a time with all cores. "
        f"All runs on CPU, on {HARDWARE}."
    )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:effort}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        " & Tuning time & \\multicolumn{4}{c}{Forecasting time per fold (s)} \\\\",
        "\\cmidrule(lr){3-6}",
        "Model & per city & " + " & ".join(f"{h}~h" for h in efr.HORIZONS) + " \\\\",
        "\\midrule",
    ]
    for model, row in table.iterrows():
        tuned = fmt_tuning(row["tuning_h_min"], row["tuning_h_max"]) if pd.notna(row["tuning_h_min"]) else "--"
        times = [fmt_seconds(row[f"{h}h_s_per_fold"]) for h in efr.HORIZONS]
        lines.append(f"{tex_esc(model)} & {tuned} & " + " & ".join(times) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stability of the forecasting times
# ---------------------------------------------------------------------------
def stability(forecast: pd.DataFrame) -> pd.DataFrame:
    """X and Y of the header, observed weather only, with where each maximum occurs."""
    clean = forecast[forecast["weather_scenario"] == "clean_only"]

    # X: spread across the folds of one experiment (coefficient of variation)
    x_rows = clean[clean["model"].isin(STABILITY_X_MODELS)].copy()
    x_rows["pct"] = 100 * x_rows["runtime_std_s"] / x_rows["runtime_mean_s"]
    x_max = x_rows.loc[x_rows["pct"].idxmax()]

    # Y: difference between the cities, per model and horizon
    y_rows = clean[clean["model"].isin(STABILITY_Y_MODELS)]
    by_city = y_rows.groupby(["model", "horizon"])["runtime_mean_s"]
    y_pct = 100 * (by_city.max() / by_city.min() - 1)
    y_model, y_horizon = y_pct.idxmax()

    return pd.DataFrame([
        {"quantity": "X", "value_pct": round(float(x_max["pct"]), 1),
         "model": x_max["model"], "city": x_max["dataset"], "horizon": int(x_max["horizon"]),
         "definition": "max runtime_std_s / runtime_mean_s, observed weather, over "
                       + ", ".join(STABILITY_X_MODELS) + ", all cities and horizons"},
        {"quantity": "Y", "value_pct": round(float(y_pct.max()), 1),
         "model": y_model, "city": "", "horizon": int(y_horizon),
         "definition": "max over horizons and models of (max over cities / min over cities - 1) "
                       "of runtime_mean_s, observed weather, over " + ", ".join(STABILITY_Y_MODELS)},
    ])


def main() -> None:
    root = efr.find_repo_root()
    forecast = efr.load_paper_rows(root / efr.RESULTS_FILE)
    tuning = tuning_times(root)
    table = build_table(tuning, forecast)
    max_degraded_pct = float(efr.degraded_vs_clean(forecast).max())

    out_dir = root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / CSV_FILE, float_format="%.4f")
    tex = to_latex(table, tuning, max_degraded_pct)
    (out_dir / TEX_FILE).write_text(tex)
    stable = stability(forecast)
    stable.to_csv(out_dir / STABILITY_FILE, index=False)

    print("Tuning jobs used (the ones that wrote the params files in forecasting/config_<city>.py):")
    print(tuning.pivot(index="model", columns="city", values="tuning_h")
                .reindex(index=[m for m in efr.MODELS if m in PARAMS_ATTRIBUTE], columns=efr.CITIES)
                .to_string(float_format=lambda v: f"{60 * v:5.1f} min" if v < 1 else f"{v:5.2f} h"))
    print(tuning.pivot(index="model", columns="city", values="job")
                .reindex(index=[m for m in efr.MODELS if m in PARAMS_ATTRIBUTE], columns=efr.CITIES)
                .to_string())
    print(f"Nodes: {', '.join(sorted(tuning['node'].unique()))}")
    print()
    print(tex)
    print("Stability of the forecasting times (observed weather):")
    for _, row in stable.iterrows():
        where = f"{row['model']}, {row['city'] + ', ' if row['city'] else ''}{row['horizon']} h"
        print(f"  {row['quantity']} = {row['value_pct']:.1f} %  ({where})")
    print()
    print(f"Written: {(OUTPUT_DIR / TEX_FILE).as_posix()}, {(OUTPUT_DIR / CSV_FILE).as_posix()}, "
          f"{(OUTPUT_DIR / STABILITY_FILE).as_posix()}")


if __name__ == "__main__":
    main()

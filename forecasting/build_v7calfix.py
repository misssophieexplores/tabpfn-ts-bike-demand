"""
v7 results with the corrected TabPFN (v2.5) runs: results/results_master_v7calfix.csv and
results/detailed_results_master_v7calfix.csv, for analysis_visualizations/
results_analysis.ipynb. The v7 files are only read.

= the v7 results, with the rows of TabPFN and TabPFN_NoWeather (clean_only, degraded)
  replaced by the runs with corrected calendar features (models/calendar_fix.py),
  TabPFN_CalFix and TabPFN_CalFix_NoWeather, under the v7 names. All other models:
  v7 rows as they are. The published files contain the rows of the 10 paper models only.
  The inputs (v7 results with the library's encoding, CalFix run files) are not part of
  this version of the repository.

The CalFix runs were scored with the v8 MASE (scaled by the weekly naive). v7 scales by
the daily naive. In every fold MASE = MAE / scale, and the scale depends only on the
fold's training window, so the daily MASE of a CalFix fold = its MAE / the daily scale
of the same (city, horizon, fold) in v7 (MAE / MASE of the v7 TabPFN row; identical for
every v7 model up to 1e-14). MASE_mean / MASE_std are recomputed from these folds
(pandas mean / std as run_experiments). Everything else is the CalFix run's own value;
the rows keep version 'v7' (the notebook filters on it), run_name and git_commit of
the CalFix run.

Run from the repository root (seconds):
    python forecasting/build_v7calfix.py [--calfix-archive results_v8calfix.tar.gz]
The archive can also be replaced by the extracted files in results/ (--calfix-dir results).
"""
import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

RENAME = {"TabPFN_CalFix": "TabPFN", "TabPFN_CalFix_NoWeather": "TabPFN_NoWeather"}
SCENARIOS = ["clean_only", "degraded"]          # the v7 scenarios
KEY = ["dataset", "model", "horizon", "weather_scenario"]
FOLD = ["dataset", "horizon", "fold"]


def read_calfix(name, archive, directory):
    if directory:
        return pd.read_csv(Path(directory) / name)
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if Path(m.name).name == name)
        return pd.read_csv(io.BytesIO(tar.extractfile(member).read()))


def main():
    parser = argparse.ArgumentParser(description="v7 results with the corrected TabPFN runs")
    parser.add_argument("--calfix-archive", default="results_v8calfix.tar.gz")
    parser.add_argument("--calfix-dir", default=None, help="folder with the extracted *_v8calfix.csv instead")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    res = Path(args.results_dir)

    det7 = pd.read_csv(res / "detailed_results_master_v7.csv")
    agg7 = pd.read_csv(res / "results_master_v7.csv")
    detc = read_calfix("detailed_results_master_v8calfix.csv", args.calfix_archive, args.calfix_dir)
    aggc = read_calfix("results_master_v8calfix.csv", args.calfix_archive, args.calfix_dir)
    take = lambda d: d[d["model"].isin(RENAME) & d["weather_scenario"].isin(SCENARIOS)].copy()
    detc, aggc = take(detc), take(aggc)

    # daily scale per fold from the v7 TabPFN rows
    ref = det7[det7["model"] == "TabPFN"].drop_duplicates(FOLD)
    scale = ref.set_index(FOLD)["MAE"] / ref.set_index(FOLD)["MASE"]
    s = scale.reindex(pd.MultiIndex.from_frame(detc[FOLD])).to_numpy()
    if np.isnan(s).sum() > detc["MASE"].isna().sum():
        raise RuntimeError("folds of the CalFix runs without a v7 fold: the CV folds differ")
    detc["MASE"] = detc["MAE"].to_numpy() / s
    detc["model"] = detc["model"].map(RENAME)
    detc["version"] = "v7"

    stats = detc.groupby(KEY)["MASE"].agg(["mean", "std"])
    aggc["model"] = aggc["model"].map(RENAME)
    aggc["version"] = "v7"
    idx = pd.MultiIndex.from_frame(aggc[KEY])
    aggc["MASE_mean"] = stats.loc[idx, "mean"].to_numpy()
    aggc["MASE_std"] = stats.loc[idx, "std"].to_numpy()

    replaced = lambda d: d["model"].isin(RENAME.values()) & d["weather_scenario"].isin(SCENARIOS)
    for name, v7, new in [("detailed_results_master_v7calfix.csv", det7, detc),
                          ("results_master_v7calfix.csv", agg7, aggc)]:
        n_old = int(replaced(v7).sum())
        if n_old != len(new):
            raise RuntimeError(f"{name}: {n_old} v7 TabPFN rows but {len(new)} CalFix rows")
        out = pd.concat([v7[~replaced(v7)], new[v7.columns]], ignore_index=True)
        out.to_csv(res / name, index=False)
        print(f"{res / name}: {len(out)} rows ({n_old} TabPFN rows replaced by the corrected runs)")

    before = agg7[replaced(agg7)].set_index(KEY)["MASE_mean"]
    after = aggc.set_index(KEY)["MASE_mean"]
    print("\nMASE (daily naive), geometric mean over city x horizon:")
    print(pd.DataFrame({"v7": before, "CalFix": after}).groupby(level=["model", "weather_scenario"])
          .agg(lambda x: float(np.exp(np.log(x).mean()))).round(4).to_string())


if __name__ == "__main__":
    main()

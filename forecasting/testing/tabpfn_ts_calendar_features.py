"""
Calendar features exactly as tabpfn-time-series builds them (its own default
generators RunningIndexFeature and CalendarFeature, with their default settings),
on synthetic timestamps, written as two tables:

  A. 25 hours: Monday 2018-01-01 00:00 to Tuesday 2018-01-02 00:00 (hour of day)
  B.  8 days:  Sunday 2017-12-31 to Sunday 2018-01-07, at 12:00   (day of week)

Background: CalendarFeature.generate encodes each index as
sin/cos(2*pi*index / (period - 1)) ("Adjust for 0-based indexing"). With
hour_of_day in 0..23 and period 24, hour 23 gets angle 2*pi = hour 0, so 23:00 and
00:00 have identical features; day_of_week (0..6, period 7): Sunday = Monday; etc.
After each table, 'collisions' lists the timestamps that get the same sin/cos values
of a feature although its calendar value differs (e.g. hour 0 and hour 23).

AutoSeasonalFeature (the third default generator) is not included: it depends on
the target series, not on the timestamps.

Run in the environment whose tabpfn-time-series version you want to check:
    python forecasting/testing/tabpfn_ts_calendar_features.py [--out-dir DIR]
Writes tabpfn_ts_calendar_features_<version>_{25h,8d}.csv and prints both tables
with their collisions.
"""
import argparse
from importlib import metadata
from pathlib import Path

import gluonts.time_feature
import numpy as np
import pandas as pd

from tabpfn_time_series.features.basic_features import CalendarFeature, RunningIndexFeature


def features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Features of the default generators, on a frame indexed like a TimeSeriesDataFrame."""
    idx = pd.MultiIndex.from_arrays([["series_1"] * len(timestamps), timestamps], names=["item_id", "timestamp"])
    df = pd.DataFrame({"target": 0.0}, index=idx)
    for gen in (RunningIndexFeature(), CalendarFeature()):
        df = gen.generate(df)
    out = df.drop(columns="target").reset_index(level="item_id", drop=True).reset_index()
    out.insert(1, "weekday", out["timestamp"].dt.day_name())
    return out


def collisions(t: pd.DataFrame) -> list:
    """'<feature>: <ts> (index i) = <ts> (index j)' for timestamps whose sin/cos of a feature are
    equal although the feature's index (gluonts <feature>_index, as CalendarFeature uses) differs."""
    out = []
    ts = pd.DatetimeIndex(t["timestamp"])
    for feat in CalendarFeature().seasonal_features:
        idx = getattr(gluonts.time_feature, f"{feat}_index")(ts).astype(int)
        enc = np.round(t[[f"{feat}_sin", f"{feat}_cos"]].to_numpy(), 9) + 0.0      # -0.0 -> 0.0
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                if idx[i] != idx[j] and (enc[i] == enc[j]).all():
                    out.append(f"{feat}: {ts[i]} (index {idx[i]}) = {ts[j]} (index {idx[j]})")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    version = metadata.version("tabpfn-time-series")
    tables = {
        "25h": features(pd.date_range("2018-01-01 00:00", "2018-01-02 00:00", freq="h")),
        "8d": features(pd.date_range("2017-12-31 12:00", "2018-01-07 12:00", freq="D")),
    }
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 50)
    for name, t in tables.items():
        path = Path(args.out_dir) / f"tabpfn_ts_calendar_features_{version}_{name}.csv"
        t.to_csv(path, index=False, float_format="%.6f")
        print(f"\n===== tabpfn-time-series {version}: {name} ({path})")
        print(t.round(4).to_string(index=False))
        found = collisions(t)
        print("collisions:" + ("".join(f"\n  {c}" for c in found) if found else " none"))


if __name__ == "__main__":
    main()

"""Read-only check of the numbers in Section 3.1 / Table "Dataset summary and
preprocessing". Reads the three bike CSVs, prints the numbers, writes nothing.

Usage:  python check_paper_numbers.py <folder with SeoulBikeData.csv, LondonBikeData.csv, WashingtonBikeData.csv>
        (default folder: data)

Lines start with the claim number (C1, C2, ...).
Rows are loaded like run_experiments.load_and_prepare_data: stable sort,
duplicate timestamps dropped (first kept), Seoul visibility x 0.01.
Evaluation period = last 35 x 168 = 5,880 h (evaluation/cv.py, config.py).
"""
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
EVAL_HOURS = 35 * 168

CITIES = {
    "Seoul": dict(file="SeoulBikeData.csv", ts="Date", y="Rented Bike Count", hol="Holiday",
                  season="Seasons", vis="Visibility", vis_scale=0.01, tz="Asia/Seoul"),
    "London": dict(file="LondonBikeData.csv", ts="timestamp", y="cnt", hol="is_holiday",
                   season="season", vis="visibility_km", vis_scale=1.0, tz="Europe/London"),
    "Washington": dict(file="WashingtonBikeData.csv", ts="timestamp", y="cnt", hol="holiday",
                       season="season", vis="visibility_km", vis_scale=1.0, tz="America/New_York"),
}


def clock_change_days(tz, first, last):
    """[(date, 'spring'/'autumn', skipped or repeated local hour)]"""
    z = ZoneInfo(tz)
    out = []
    d = first
    while d <= last:
        nd = d + timedelta(days=1)
        o1 = datetime(d.year, d.month, d.day, tzinfo=z).utcoffset()
        o2 = datetime(nd.year, nd.month, nd.day, tzinfo=z).utcoffset()
        if o1 != o2:
            for h in range(24):
                t = datetime(d.year, d.month, d.day, h, tzinfo=z)
                if o2 > o1 and t.astimezone(timezone.utc).astimezone(z).hour != h:
                    out.append((d, "spring", h))
                if o2 < o1 and t.replace(fold=0).utcoffset() != t.replace(fold=1).utcoffset():
                    out.append((d, "autumn", h))
        d = nd
    return out


for name, c in CITIES.items():
    ts, y = c["ts"], c["y"]
    print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)

    raw = pd.read_csv(f"{DATA}/{c['file']}")
    raw[ts] = pd.to_datetime(raw[ts])
    raw = raw.sort_values(ts, kind="stable").reset_index(drop=True)
    n_dup = int(raw[ts].duplicated().sum())
    df = raw.drop_duplicates(subset=ts, keep="first").set_index(ts)

    yv = df[y].astype(float)
    imp = (df["Functioning Day"] == "No").to_numpy()

    # ---- C1 ----
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    print(f"C1 first {df.index.min()}, last {df.index.max()}")
    print(f"C1 rows in file {len(raw)}, duplicate timestamps {n_dup}, rows after dropping duplicates {len(df)}, "
          f"missing hours in the range {len(full.difference(df.index))}")

    # ---- C2 / C3 ----
    days = clock_change_days(c["tz"], df.index.min().date(), df.index.max().date())
    if not days:
        print("C2 clock-change days in the data: none")
    for d, kind, h in days:
        day_raw = raw[raw[ts].dt.normalize() == pd.Timestamp(d)]
        n_after = int((df.index.normalize() == pd.Timestamp(d)).sum())
        word = "skipped" if kind == "spring" else "repeated"
        print(f"C2/C3 {d} {kind} ({word} hour {h:02d}:00): rows in file {len(day_raw)}, after dropping duplicates {n_after}")
        for _, r in day_raw[day_raw[ts].dt.hour.between(max(h - 1, 0), h + 1)].iterrows():
            print(f"        {r[ts]}  count {r[y]}  Functioning Day {r['Functioning Day']}")

    # ---- C4 ----
    v = yv.to_numpy()
    d24 = yv.reindex(df.index - pd.Timedelta(hours=24)).to_numpy()
    d168 = yv.reindex(df.index - pd.Timedelta(hours=168)).to_numpy()
    m24 = imp & np.isclose(v, d24)
    m168 = imp & ~m24 & np.isclose(v, d168)
    other = imp & ~m24 & ~m168
    print(f"C4 imputed hours {imp.sum()}: equal to value 24 h earlier {m24.sum()}, "
          f"else equal to value 168 h earlier {m168.sum()}, neither {other.sum()}")
    for t, val in yv[other].items():
        print(f"        neither: {t}  value {val}")

    # ---- C5 ----
    end = df.index.max()
    ev = (df.index > end - pd.Timedelta(hours=EVAL_HOURS))
    print(f"C5 evaluation period {df.index[ev].min()} to {end} ({ev.sum()} h); "
          f"imputed hours total {imp.sum()}, in evaluation period {(imp & ev).sum()}")

    # ---- C6 ----
    h = df[c["hol"]]
    is_h = (h == "Holiday") if name == "Seoul" else (pd.to_numeric(h, errors="coerce") == 1)
    is_h = is_h.to_numpy()
    hol_days = pd.Series(df.index[is_h].normalize()).value_counts().sort_index()
    print(f"C6 holiday hours {is_h.sum()}, holiday dates {len(hol_days)}")
    print("        " + ", ".join(f"{d.date()} ({n} h)" for d, n in hol_days.items()))

    # ---- C7 ----
    s = df[c["season"]]
    print("C7 season changes (first hour of each new value):")
    for t, val in s[s.ne(s.shift())].items():
        print(f"        {t}  season = {val}")

    # ---- C9 / C10 ----
    p = df["precipitation_mm"].astype(float)
    if name == "Seoul":
        r = df["Rainfall"].astype(float)
        winter = df.index.month.isin([11, 12, 1, 2, 3])
        print(f"C9 wet hours: Rainfall > 0 {(r > 0).sum()}, precipitation_mm > 0 {(p > 0).sum()}; "
              f"totals Rainfall {r.sum():.1f}, precipitation_mm {p.sum():.1f}")
        print(f"C9 Nov-Mar wet hours: Rainfall > 0 {((r > 0) & winter).sum()}, precipitation_mm > 0 {((p > 0) & winter).sum()}; "
              f"Apr-Oct hours where precipitation_mm != Rainfall: {((~winter) & ~np.isclose(r, p)).sum()}")
        sd = df["snow_depth_cm"].astype(float)
        print(f"C10 hours where snow_depth_cm != Snowfall: {(~np.isclose(sd, df['Snowfall'].astype(float))).sum()}")
    else:
        calc = (df["rainfall_mm"].astype(float) + df["snowfall_cm"].astype(float) / 0.7).round(1)
        print(f"C9 hours where precipitation_mm != round(rainfall_mm + snowfall_cm / 0.7, 1): {((p - calc).abs() > 0.051).sum()}; "
              f"wet hours (precipitation_mm > 0) {(p > 0).sum()}; hours with snowfall_cm > 0 {(df['snowfall_cm'] > 0).sum()}")
        sd = df["snow_depth_cm"].astype(float)
        print(f"C10 hours with snow_depth_cm > 0: {(sd > 0).sum()}, max {sd.max()}")

    # ---- C11 ----
    vis_raw = df[c["vis"]].astype(float)
    vis = vis_raw * c["vis_scale"]
    print(f"C11 visibility raw max {vis_raw.max()}, in km: min {vis.min():.2f}, max {vis.max():.2f}, "
          f"hours at the max {(np.isclose(vis, vis.max())).sum()}, missing {vis.isna().sum()}")
    print("C11 five largest values (km): " + ", ".join(f"{t} {x}" for t, x in vis.nlargest(5).items()))
    if name != "Seoul":
        for cap in (16.0, 65.0):
            print(f"C11 hours with visibility > {cap} km: {(vis > cap).sum()}")
        off = ~np.isclose(vis * 10, np.round(vis * 10))
        spring = [pd.Timestamp(d) + pd.Timedelta(hours=h) for d, k, h in days if k == "spring"]
        spring_in = [t for t in spring if t in vis.index]
        n_interp = int(off.sum()) + sum(1 for t in spring_in if not off[vis.index.get_loc(t)])
        print(f"C11 identifiable interpolated hours: {n_interp} "
              f"(values not on the 0.1 km grid {off.sum()}, plus skipped spring hours on the grid)")
        for t, x in vis[off].items():
            print(f"        off grid: {t}  {x}")
        for t in spring_in:
            print(f"        skipped spring hour: {t}  {vis[t]}")

    # ---- C13 ----
    print(f"C13 mean rentals per hour: all hours {yv.mean():.1f}, observed hours only {yv[~imp].mean():.1f}")

    # ---- C14 ----
    if name == "Seoul":
        per_day = pd.Series(df.index[imp].normalize()).value_counts().sort_index()
        print(f"C14 non-operational days: full (24 h) {(per_day == 24).sum()}, partial {(per_day < 24).sum()}")
        for d, n in per_day.items():
            print(f"        {d.date()}  {n} h")

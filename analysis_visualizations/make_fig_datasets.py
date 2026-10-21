"""Dataset overview figure for the paper -> results_v7/figures/paper/fig_datasets.pdf

Place this file in analysis_visualizations/ and run it there (python make_fig_datasets.py,
or paste it into a notebook cell in that folder). It finds data/ and results_v7/ itself:
it looks in its own folder first, then in each folder above.

Daily rentals per city BEFORE imputation: the daily sum of the recorded hours.
Hours with Functioning Day == "No" are the imputed hours; they are left out of the sum
(Seoul: the system was closed, the original count is 0; London, Washington: no record).
Periods are counted back from the last hour of each dataset.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


def find_up(start, rel):
    """First path start/rel, start.parent/rel, ... that exists."""
    for folder in (start, *start.parents):
        if (folder / rel).exists():
            return folder / rel
    raise FileNotFoundError(f"'{rel}' not found in {start} or any folder above it")


# Script: the folder of this file. Notebook (no __file__): the notebook's working folder.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
DATA_DIR = find_up(HERE, "data/SeoulBikeData.csv").parent
OUT_FILE = find_up(HERE, "results_v7") / "figures" / "paper" / "fig_datasets.pdf"

# name, file, timestamp column, target column, panel title
CITIES = [
    ("Seoul", "SeoulBikeData.csv", "Date", "Rented Bike Count", "Seoul (Dec 2017 – Nov 2018)"),
    ("London", "LondonBikeData.csv", "timestamp", "cnt", "London (Jan 2015 – Jan 2017)"),
    ("Washington", "WashingtonBikeData.csv", "timestamp", "cnt", "Washington (Jan 2011 – Dec 2012)"),
]

EVAL_HOURS = 5880      # last 5,880 h = 245 days
TUNE_DAYS = 90
LOOKBACK_DAYS = 30
AXIS_DAYS = 731        # same time scale in every panel

FIG_W, FIG_H = 6.5, 5.3  # inches

LINE_COLOR = "#1d2b3a"
MARKER = dict(marker="o", markersize=3.2, markerfacecolor="white", markeredgecolor=LINE_COLOR, markeredgewidth=0.7)
SWATCH_EDGE = "0.55"

# label, span style; order = time order
# Fills from PRGn, the colormap of the paper's heatmaps: not used = PRGn neutral (0.50),
# tuning = PRGn green (0.72), evaluation = PRGn purple (0.42); lookback = grey + hatching.
# Also separate in greyscale (L*: 97 / 79 / 89; lookback 72 + hatching) and for red-green colour blindness.
PERIODS = [
    ("Not used", dict(facecolor="#f6f7f6")),
    ("First lookback window (30 days)", dict(facecolor="#b0b0b0", edgecolor="#5f5f5f", hatch="//////")),
    ("Tuning period (90 days)", dict(facecolor="#96d192")),
    ("Evaluation period (245 days)", dict(facecolor="#eadbeb")),
]
IMPUTED_LABEL = "Days with imputed hours"

RC = {
    "font.family": "DejaVu Sans",        # ships with matplotlib: same font on every machine
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": "line",            # grid above the shading, below the data
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,                  # embed TrueType, no Type 3 fonts
    "ps.fonttype": 42,
    "savefig.bbox": "standard",          # keep the page exactly FIG_W wide
}


def load_city(file, tcol, ycol):
    """Hourly data with duplicate timestamps dropped (the pipeline keeps the first copy)."""
    df = pd.read_csv(DATA_DIR / file)
    ts = pd.to_datetime(df[tcol])
    keep = ~ts.duplicated(keep="first")
    ts, df = ts[keep].reset_index(drop=True), df[keep].reset_index(drop=True)
    if not (ts.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise RuntimeError(f"{file}: rows are not consecutive hours")
    imputed = df["Functioning Day"].astype(str).str.strip().eq("No")
    return pd.DataFrame({"ts": ts, "y": df[ycol].astype(float), "imputed": imputed})


def daily_before_imputation(h):
    """Daily total of the recorded (non-imputed) hours and the number of imputed hours per day."""
    day = h["ts"].dt.normalize()
    d = pd.DataFrame({
        "total": h["y"].where(~h["imputed"], 0.0).groupby(day).sum(),
        "n_imputed": h["imputed"].groupby(day).sum().astype(int),
        "n_hours": h["y"].groupby(day).size(),
    })
    if not (d["n_hours"] == 24).all():
        raise RuntimeError("days without 24 hourly rows")
    return d


def periods(h):
    """Period boundaries (Timestamps), counted back from the last hour."""
    start = h["ts"].min()
    end = h["ts"].max() + pd.Timedelta(hours=1)          # midnight after the last hour
    eval_start = end - pd.Timedelta(hours=EVAL_HOURS)
    tune_start = eval_start - pd.Timedelta(days=TUNE_DAYS)
    look_start = tune_start - pd.Timedelta(days=LOOKBACK_DAYS)
    if look_start < start:
        raise RuntimeError("dataset shorter than lookback + tuning + evaluation")
    return dict(start=start, look=look_start, tune=tune_start, eval=eval_start, end=end)


def month_formatter(first_tick):
    """Month on every tick; year below it on January ticks and on the first tick of the panel."""
    def fmt(x, pos=None):
        d = mdates.num2date(x)
        return f"{d:%b}\n{d:%Y}" if d.month == 1 or abs(x - first_tick) < 1e-6 else f"{d:%b}"
    return FuncFormatter(fmt)


def num(t):
    return mdates.date2num(pd.Timestamp(t).to_pydatetime())


def draw_panel(ax, d, p, title):
    bounds = [p["start"], p["look"], p["tune"], p["eval"], p["end"]]
    for (label, style), a, b in zip(PERIODS, bounds[:-1], bounds[1:]):
        if b > a:  # "Not used" is empty for Seoul
            ax.axvspan(num(a), num(b), linewidth=0, zorder=0.5, **style)
    for t in (p["look"], p["tune"], p["eval"]):
        ax.axvline(num(t), color="0.2", linewidth=0.7, linestyle=(0, (1, 1.5)), zorder=1.6)

    x = mdates.date2num((d.index + pd.Timedelta(hours=12)).to_pydatetime())  # value at mid-day
    ax.plot(x, d["total"].to_numpy(), color=LINE_COLOR, linewidth=0.65, solid_joinstyle="round", zorder=2)
    imp = (d["n_imputed"] > 0).to_numpy()
    ax.plot(x[imp], d["total"].to_numpy()[imp], linestyle="none", zorder=3, clip_on=False, **MARKER)

    x0, x1 = num(p["end"] - pd.Timedelta(days=AXIS_DAYS)), num(p["end"])
    ax.set_xlim(x0, x1)
    major = mdates.MonthLocator(bymonth=(1, 4, 7, 10))
    ax.xaxis.set_major_locator(major)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(month_formatter(min(t for t in major() if t >= x0)))

    ax.set_ylim(0, d["total"].max() * 1.06)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:,.0f}"))
    ax.yaxis.grid(True, color="white", linewidth=0.6)
    ax.set_ylabel("Daily rentals")
    ax.set_title(title, loc="left", fontweight="bold", pad=3)


def legend_handles():
    hs = [Patch(label=label, linewidth=0.5, **{"edgecolor": SWATCH_EDGE, **style}) for label, style in PERIODS]
    hs.append(Line2D([], [], linestyle="none", label=IMPUTED_LABEL, **MARKER))
    return hs


def report(city, h, d, p):
    names = ["Not used", "First lookback window", "Tuning period", "Evaluation period"]
    keys = ["start", "look", "tune", "eval", "end"]
    print(f"\n{city}: {h['ts'].min():%d %b %Y %H:%M} – {h['ts'].max():%d %b %Y %H:%M}, {len(h):,} h, "
          f"{int(h['imputed'].sum())} imputed h on {int((d['n_imputed'] > 0).sum())} days")
    for name, a, b in zip(names, keys[:-1], keys[1:]):
        a, b = p[a], p[b]
        if b <= a:
            print(f"  {name:<22} none")
            continue
        m = (h["ts"] >= a) & (h["ts"] < b)
        print(f"  {name:<22} {a:%d %b %Y} – {b - pd.Timedelta(hours=1):%d %b %Y}  "
              f"{(b - a).days:>3} days, imputed h: {int(h.loc[m, 'imputed'].sum())}")


def main():
    loaded = []
    for city, file, tcol, ycol, title in CITIES:
        h = load_city(file, tcol, ycol)
        loaded.append((city, title, h, daily_before_imputation(h), periods(h)))

    with mpl.rc_context(RC):
        fig, axes = plt.subplots(len(loaded), 1, figsize=(FIG_W, FIG_H), layout="constrained")
        for ax, (city, title, h, d, p) in zip(axes, loaded):
            draw_panel(ax, d, p, title)
        fig.align_ylabels(axes)

        # Legend once, in the empty left part of the Seoul panel (Seoul covers only the right half).
        ax0, (_, _, _, _, p0) = axes[0], loaded[0]
        x0, x1 = ax0.get_xlim()
        free = (num(p0["start"]) - x0) / (x1 - x0)
        leg = ax0.legend(handles=legend_handles(), loc="center", bbox_to_anchor=(free / 2, 0.5),
                         frameon=True, framealpha=1, edgecolor="0.75", fancybox=False, borderpad=0.6,
                         handlelength=1.8, handleheight=1.0, labelspacing=0.45, handletextpad=0.6)
        leg.get_frame().set_linewidth(0.5)

        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_FILE, format="pdf")
    print(f"written {OUT_FILE.resolve()}")

    for city, title, h, d, p in loaded:
        report(city, h, d, p)

    # Washington, late October 2012 (Hurricane Sandy)
    d_wa = dict((c, dd) for c, _, _, dd, _ in loaded)["Washington"]
    print("\nWashington, 26 Oct – 2 Nov 2012 (total = recorded hours only):")
    print(d_wa.loc["2012-10-26":"2012-11-02", ["total", "n_imputed"]].astype(int).to_string())
    return fig


if __name__ == "__main__":
    fig = main()

"""Rebuild data/LondonBikeData.csv and data/WashingtonBikeData.csv.

Both files contain visibility from Visual Crossing, which may not be
redistributed (see README_WEATHER.md), so they are not in the repository.
This script runs the recorded steps (provenance/README.md) unchanged, in the
folder layout they were written for:

  London:     provenance/london/saving_london_data.py
              provenance/london/saving_london_data_missing_values.py
              provenance/london/fill_missing_london_weather.py
  Washington: provenance/washington/load_weather_washington.py
  both:       add_visibility.py (Visual Crossing, needs VISUALCROSSING_KEY)
              build_weather_v7.py (London and Washington only; Seoul is
              already a v7 file)

Inputs: provenance/raw/ (bike data and Open-Meteo downloads) and open_meteo/.
The Open-Meteo APIs are not called (their downloads are in provenance/raw/);
Visual Crossing is called once per month and city (49 requests).

Run from the repository root, with your own key:
    VISUALCROSSING_KEY=... python data/rebuild_london_washington.py
(or put VISUALCROSSING_KEY in .env). Existing files in data/ are not
overwritten unless --overwrite is given.

Checked on 21 Oct 2026 with the Visual Crossing answers replaced by the
visibility values of the files used in the paper: both files come out
identical in every value; only solar_radiation_mjm2 (not a model input)
differs in the 16th digit in 81 (London) and 192 (Washington) cells.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent
PROV = DATA / "provenance"
FILES = ["LondonBikeData.csv", "WashingtonBikeData.csv"]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: Path, cwd: Path) -> None:
    print(f"\n>>> {script.relative_to(DATA.parent)}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--overwrite", action="store_true", help="replace existing files in data/")
    args = parser.parse_args()

    existing = [f for f in FILES if (DATA / f).exists()]
    if existing and not args.overwrite:
        sys.exit(f"data/{', data/'.join(existing)} already exist(s); use --overwrite to replace.")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Folder layout of the repository the scripts were written in
        (work / "data" / "weather_data").mkdir(parents=True)
        for f in ["london_og.csv", "washington_hour.csv"]:
            shutil.copy(PROV / "raw" / f, work / "data" / f)
        for f in ["london_weather_2015-2017.csv", "washington_dc_weather_2011-2012.csv"]:
            shutil.copy(PROV / "raw" / f, work / "data" / "weather_data" / f)

        # Bike data + Open-Meteo weather
        for script in ["london/saving_london_data.py",
                       "london/saving_london_data_missing_values.py",
                       "london/fill_missing_london_weather.py",
                       "washington/load_weather_washington.py"]:
            _run(PROV / script, work)

        # Visibility (Visual Crossing); TEST_MODE = False as for the paper's files
        print("\n>>> data/add_visibility.py", flush=True)
        cwd = os.getcwd()
        os.chdir(work)               # its paths are relative (data/...)
        try:
            vis = _load(DATA / "add_visibility.py")
            if not vis.VISUALCROSSING_KEY:
                sys.exit("VISUALCROSSING_KEY not set (environment or .env).")
            vis.TEST_MODE = False
            for name, cfg in vis.DATASETS.items():
                vis.add_visibility(name, cfg["path"], cfg["location"])
        finally:
            os.chdir(cwd)            # the temporary folder can only be removed when we are not in it

        # v7 weather columns and daylight-saving re-alignment
        print("\n>>> data/build_weather_v7.py (london, washington)", flush=True)
        v7 = _load(DATA / "build_weather_v7.py")
        out = work / "out"
        out.mkdir()
        for city in ["london", "washington"]:
            v7.build(city, str(work / "data"), str(DATA / "open_meteo"), str(out))

        for f in FILES:
            shutil.copy(out / f, DATA / f)
            print(f"written data/{f}")


if __name__ == "__main__":
    main()

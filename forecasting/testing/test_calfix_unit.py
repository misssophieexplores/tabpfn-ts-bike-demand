"""
Corrected TabPFN-TS calendar features (models/calendar_fix.py): every hour, weekday
and month gets its own sin/cos code (the library's CalendarFeature maps hour 23 to
hour 0, Sunday to Monday, December to January), the columns are those of the
library, and TabPFN / TabPFN_NoWeather pass them to the pipeline.
Needs tabpfn-time-series (skipped without it). Run: pytest forecasting/testing/test_calfix_unit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # forecasting/


def _frame(ts):
    return pd.DataFrame({"target": 0.0}, index=pd.MultiIndex.from_arrays(
        [["s"] * len(ts), ts], names=["item_id", "timestamp"]))


def test_fixed_calendar_feature_has_no_collisions():
    pytest.importorskip("tabpfn_time_series")
    from models.calendar_fix import FixedCalendarFeature, CALFIX_FEATURES
    from tabpfn_time_series.features import AutoSeasonalFeature, RunningIndexFeature
    from tabpfn_time_series.features.basic_features import CalendarFeature
    ts = pd.date_range("2017-12-25", "2019-01-08", freq="h")
    fixed, orig = FixedCalendarFeature().generate(_frame(ts)), CalendarFeature().generate(_frame(ts))
    assert list(fixed.columns) == list(orig.columns)
    for feat, n in [("hour_of_day", 24), ("day_of_week", 7), ("month_of_year", 12)]:
        codes = lambda t: pd.DataFrame(np.round(t[[f"{feat}_sin", f"{feat}_cos"]].to_numpy(), 9) + 0.0)
        assert len(codes(fixed).drop_duplicates()) == n
        assert len(codes(orig).drop_duplicates()) == n - 1          # the library bug this fixes
    assert [type(g) for g in CALFIX_FEATURES] == [RunningIndexFeature, FixedCalendarFeature, AutoSeasonalFeature]


def test_tabpfn_models_use_corrected_features(monkeypatch):
    pytest.importorskip("tabpfn_time_series")
    pytest.importorskip("torch")
    import models.tabpfn_pipeline_model as tp
    from models.calendar_fix import CALFIX_FEATURES
    seen = []
    monkeypatch.setattr(tp, "TabPFNTSPipeline", lambda **kwargs: seen.append(kwargs))
    idx = pd.date_range("2018-01-01", periods=48, freq="h")
    X = pd.DataFrame({"temp": np.arange(48, dtype=float)}, index=idx)
    for cls in (tp.TabPFNPipelineForecaster, tp.TabPFNPipelineForecaster_NoWeather):
        cls().fit(np.arange(48, dtype=float), X)
    assert len(seen) == 2 and all(kw["temporal_features"] is CALFIX_FEATURES for kw in seen)

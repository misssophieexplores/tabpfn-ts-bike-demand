"""
Corrected TabPFN-TS calendar features, used by TabPFN and TabPFN_NoWeather
(models/tabpfn_pipeline_model.py).

tabpfn-time-series (1.0.9 used here, unchanged up to 1.4.0) encodes every calendar
index in CalendarFeature.generate as sin/cos(2*pi*index / (period - 1)). The indices
start at 0 (gluonts *_index), so the last value of each cycle lands on angle 2*pi,
the same as the first: hour 23 = hour 0, Sunday = Monday, December = January,
31 Dec = 1 Jan, week 52 = week 1 (testing/tabpfn_ts_calendar_features.py,
results/tabpfn_ts_calendar_check/).

FixedCalendarFeature is the library's CalendarFeature (same components, same seasonal
features and periods) with sin/cos(2*pi*index / period), so every value of a cycle
gets its own angle. CALFIX_FEATURES = the library's default feature list with this
calendar feature: [RunningIndexFeature, FixedCalendarFeature, AutoSeasonalFeature].
"""
import gluonts.time_feature
import numpy as np
import pandas as pd

from tabpfn_time_series.features import AutoSeasonalFeature, RunningIndexFeature
from tabpfn_time_series.features.basic_features import CalendarFeature


class FixedCalendarFeature(CalendarFeature):
    """CalendarFeature with angle 2*pi*index/period (the library divides by period - 1)."""

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        timestamps = df.index.get_level_values("timestamp")
        for component in self.components:
            df[component] = getattr(timestamps, component)
        for feature_name, periods in self.seasonal_features.items():
            feature = getattr(gluonts.time_feature, f"{feature_name}_index")(timestamps).astype(np.int32)
            if periods is not None:
                for period in periods:
                    df[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                    df[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                df[feature_name] = feature
        return df


CALFIX_FEATURES = [RunningIndexFeature(), FixedCalendarFeature(), AutoSeasonalFeature()]

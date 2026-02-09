"""Transform utilities for derived macro metrics."""

from __future__ import annotations

import pandas as pd


def prepare_monthly_series(series: pd.Series) -> pd.Series:
    """Convert a time series to month-end frequency with the last value per month."""
    source_attrs = dict(series.attrs)
    monthly = series.copy()
    monthly.index = pd.to_datetime(monthly.index).tz_localize(None)
    monthly = monthly.sort_index()
    monthly = monthly.resample("M").last().dropna().astype(float)
    monthly.attrs.update(source_attrs)
    return monthly


def build_buffett_ratio(willshire: pd.Series, gdp: pd.Series) -> pd.Series:
    """
    Build Buffett ratio from monthly Wilshire and GDP.

    GDP is forward-filled to month-end timestamps so the two series align.
    """
    willshire_monthly = prepare_monthly_series(willshire)
    gdp_monthly = prepare_monthly_series(gdp)
    gdp_aligned = gdp_monthly.reindex(willshire_monthly.index, method="ffill")

    ratio = willshire_monthly / gdp_aligned
    ratio = ratio.dropna()
    ratio.name = "BUFFETT_RATIO"
    return ratio

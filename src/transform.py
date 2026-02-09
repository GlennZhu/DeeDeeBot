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


def build_buffett_ratio(
    market_cap: pd.Series,
    gdp: pd.Series,
    market_cap_unit_divisor: float = 1.0,
) -> pd.Series:
    """
    Build Buffett ratio from monthly total market value and GDP.

    `market_cap_unit_divisor` converts market cap values into GDP units before
    division (for example, market cap in millions and GDP in billions => 1000).
    GDP is forward-filled to month-end timestamps so the two series align.
    """
    if market_cap_unit_divisor <= 0:
        raise ValueError("market_cap_unit_divisor must be > 0.")

    market_cap_monthly = prepare_monthly_series(market_cap) / float(market_cap_unit_divisor)
    gdp_monthly = prepare_monthly_series(gdp)
    gdp_aligned = gdp_monthly.reindex(market_cap_monthly.index, method="ffill")

    ratio = market_cap_monthly / gdp_aligned
    ratio = ratio.dropna()
    ratio.name = "BUFFETT_RATIO"
    return ratio

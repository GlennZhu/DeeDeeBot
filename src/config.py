"""Central configuration for macro dashboard series and app defaults."""

from __future__ import annotations

SERIES_CONFIG = {
    "m2": {
        "metric_name": "M2 Money Supply",
        "source": "FRED",
        "series_id": "M2SL",
        "frequency": "monthly",
    },
    "hiring_rate": {
        "metric_name": "Hiring Rate",
        "source": "FRED",
        "series_id": "JTSHIR",
        "frequency": "monthly",
    },
    "ten_year_yield": {
        "metric_name": "10Y Treasury Yield",
        "source": "FRED",
        "series_id": "DGS10",
        "fallback_source": "YAHOO",
        "fallback_ticker": "^TNX",
        "frequency": "daily",
    },
    "buffett_ratio": {
        "metric_name": "Buffett Indicator",
        "source": "FRED",
        "willshire_series_id": "WILL5000PR",
        "fallback_willshire_series_id": "SP500",
        "gdp_series_id": "GDP",
        "frequency": "monthly",
    },
    "unemployment_rate": {
        "metric_name": "Unemployment Rate",
        "source": "FRED",
        "series_id": "UNRATE",
        "frequency": "monthly",
    },
}

METRIC_ORDER = [
    "m2",
    "hiring_rate",
    "ten_year_yield",
    "buffett_ratio",
    "unemployment_rate",
]

DEFAULT_LOOKBACK_YEARS = 15

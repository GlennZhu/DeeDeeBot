"""Data fetching utilities for FRED and Yahoo Finance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from pandas_datareader.data import DataReader


def _normalize_series(series: pd.Series, series_name: str) -> pd.Series:
    """Return a clean, sorted float series with a timezone-naive datetime index."""
    clean = series.copy()
    clean.index = pd.to_datetime(clean.index).tz_localize(None)
    clean = clean.sort_index()
    clean = clean.dropna().astype(float)
    clean.name = series_name
    return clean


def fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
    """Fetch a single FRED series as a normalized pandas Series."""
    frame = DataReader(series_id, "fred", start=start_date)
    if frame.empty or series_id not in frame.columns:
        raise RuntimeError(f"No data returned from FRED for series {series_id}.")
    return _normalize_series(frame[series_id], series_id)


def fetch_tnx_fallback(start_date: str) -> pd.Series:
    """
    Fetch 10Y Treasury yield from Yahoo (^TNX).

    Yahoo's ^TNX quotes are typically 10x the percentage yield. The values are
    scaled down by 10 to align with FRED DGS10 units.
    """
    frame = yf.download(
        "^TNX",
        start=start_date,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if frame.empty:
        raise RuntimeError("No data returned from Yahoo Finance for ^TNX.")

    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    normalized = _normalize_series(close / 10.0, "DGS10")
    return normalized


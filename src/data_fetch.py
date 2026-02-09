"""Data fetching utilities for FRED and Yahoo Finance."""

from __future__ import annotations

import time

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


def _extract_close_series(frame: pd.DataFrame) -> pd.Series:
    """Extract a close-price series from a yfinance/pandas_datareader frame."""
    if frame.empty:
        return pd.Series(dtype=float)

    close: pd.Series | pd.DataFrame | None = None
    if "Close" in frame.columns:
        close = frame["Close"]
    elif "Adj Close" in frame.columns:
        close = frame["Adj Close"]
    elif isinstance(frame.columns, pd.MultiIndex):
        top_level = frame.columns.get_level_values(0)
        if "Close" in set(top_level):
            close = frame["Close"]
        elif "Adj Close" in set(top_level):
            close = frame["Adj Close"]

    if close is None:
        return pd.Series(dtype=float)

    if isinstance(close, pd.DataFrame):
        for col in close.columns:
            candidate = close[col].dropna()
            if not candidate.empty:
                return candidate.astype(float)
        return pd.Series(dtype=float)

    return close.dropna().astype(float)


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

    close = _extract_close_series(frame)
    if close.empty:
        raise RuntimeError("No close data returned from Yahoo Finance for ^TNX.")

    normalized = _normalize_series(close / 10.0, "DGS10")
    return normalized


def fetch_stock_daily_history(ticker: str, start_date: str) -> pd.Series:
    """Fetch daily close history for a stock ticker, with Yahoo retry + Stooq fallback."""
    symbol = str(ticker).strip().upper()
    yahoo_attempts = [
        {"start": start_date, "interval": "1d"},
        {"period": "max", "interval": "1d"},
    ]

    for params in yahoo_attempts:
        for attempt in range(2):
            try:
                frame = yf.download(
                    symbol,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    **params,
                )
            except Exception:
                frame = pd.DataFrame()
            close = _extract_close_series(frame)
            if close.empty:
                if attempt < 1:
                    time.sleep(0.5)
                continue

            normalized = _normalize_series(close, symbol)
            if "period" in params:
                start_ts = pd.to_datetime(start_date, errors="coerce")
                if pd.notna(start_ts):
                    normalized = normalized[normalized.index >= start_ts]
            if normalized.empty:
                continue

            normalized.attrs["source"] = f"YAHOO:{symbol}"
            return normalized

    # Fallback: Stooq is resilient when Yahoo rate-limits.
    stooq_symbol = f"{symbol}.US"
    try:
        stooq_frame = DataReader(stooq_symbol, "stooq", start=start_date)
        close = _extract_close_series(stooq_frame)
        normalized = _normalize_series(close, symbol)
        if not normalized.empty:
            normalized.attrs["source"] = f"STOOQ:{stooq_symbol}"
            return normalized
    except Exception as exc:
        raise RuntimeError(
            f"No daily stock data returned for {symbol} from Yahoo Finance or Stooq: {exc}"
        ) from exc

    raise RuntimeError(f"No daily stock data returned for {symbol} from Yahoo Finance or Stooq.")


def fetch_stock_intraday_latest(ticker: str) -> float | None:
    """Fetch latest 5-minute close for a stock ticker (including pre/post market)."""
    frame = yf.download(
        ticker,
        period="1d",
        interval="5m",
        progress=False,
        auto_adjust=False,
        threads=False,
        prepost=True,
    )
    if frame.empty:
        return None

    close = _extract_close_series(frame)
    if close.empty:
        return None

    clean = close.dropna().astype(float)
    if clean.empty:
        return None

    value = float(clean.iloc[-1])
    if value <= 0:
        return None
    return value

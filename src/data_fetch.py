"""Data fetching utilities for FRED and Stooq."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Any
from urllib import request
from zoneinfo import ZoneInfo

import pandas as pd
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
    """Extract a close-price series from a dataframe."""
    if frame.empty:
        return pd.Series(dtype=float)

    if "Close" in frame.columns:
        return frame["Close"].dropna().astype(float)
    if "Adj Close" in frame.columns:
        return frame["Adj Close"].dropna().astype(float)
    return pd.Series(dtype=float)


def _stock_symbol_variants(raw_symbol: str) -> list[str]:
    variants: list[str] = []
    for variant in [raw_symbol, raw_symbol.replace(".", "-"), raw_symbol.replace("/", "-")]:
        candidate = variant.strip().upper()
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _stooq_symbol_candidates(raw_symbol: str) -> list[str]:
    symbols: list[str] = []
    for variant in _stock_symbol_variants(raw_symbol):
        us_symbol = f"{variant}.US"
        if us_symbol not in symbols:
            symbols.append(us_symbol)
        if variant not in symbols:
            symbols.append(variant)
    return symbols


def _parse_stooq_quote_timestamp(
    quote_date_raw: str,
    quote_time_raw: str,
) -> datetime | None:
    date_digits = quote_date_raw.strip()
    time_digits = "".join(ch for ch in quote_time_raw.strip() if ch.isdigit())
    if len(date_digits) != 8 or not date_digits.isdigit() or len(time_digits) < 6:
        return None

    try:
        naive = datetime.strptime(f"{date_digits}{time_digits[:6]}", "%Y%m%d%H%M%S")
    except ValueError:
        return None

    # Stooq quote endpoint timestamps are emitted in feed-local clock time.
    timezone_name = "Europe/Warsaw"
    try:
        localized = naive.replace(tzinfo=ZoneInfo(timezone_name))
    except Exception:
        localized = naive.replace(tzinfo=timezone.utc)
    return localized.astimezone(timezone.utc)


def fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
    """Fetch a single FRED series as a normalized pandas Series."""
    frame = DataReader(series_id, "fred", start=start_date)
    if frame.empty or series_id not in frame.columns:
        raise RuntimeError(f"No data returned from FRED for series {series_id}.")
    return _normalize_series(frame[series_id], series_id)


def fetch_stock_daily_history(ticker: str, start_date: str) -> pd.Series:
    """Fetch daily close history for a stock ticker from Stooq only."""
    raw_symbol = str(ticker).strip().upper()
    if not raw_symbol:
        raise RuntimeError("Ticker cannot be empty.")

    last_error: Exception | None = None
    for stooq_symbol in _stooq_symbol_candidates(raw_symbol):
        try:
            frame = DataReader(stooq_symbol, "stooq", start=start_date)
            close = _extract_close_series(frame)
            if close.empty:
                continue

            normalized = _normalize_series(close, raw_symbol)
            if normalized.empty:
                continue

            normalized.attrs["source"] = f"STOOQ:{stooq_symbol}"
            return normalized
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(
            f"No daily stock data returned from Stooq for {raw_symbol}: {last_error}"
        ) from last_error
    raise RuntimeError(f"No daily stock data returned from Stooq for {raw_symbol}.")


def fetch_stock_intraday_quote(ticker: str) -> dict[str, Any] | None:
    """Fetch latest intraday quote details from Stooq quote endpoint."""
    raw_symbol = str(ticker).strip().upper()
    if not raw_symbol:
        return None

    for stooq_symbol in _stooq_symbol_candidates(raw_symbol):
        quote_symbol = stooq_symbol.lower()
        url = f"https://stooq.com/q/l/?s={quote_symbol}&i=1"
        fetched_at_utc = datetime.now(timezone.utc)

        try:
            with request.urlopen(url, timeout=10) as response:
                payload = response.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            continue

        if not payload:
            continue

        try:
            row = next(csv.reader([payload]))
        except Exception:
            continue

        if len(row) < 7:
            continue

        close_raw = str(row[6]).strip()
        if not close_raw or close_raw in {"N/D", "-"}:
            continue

        try:
            price = float(close_raw)
        except ValueError:
            continue

        if price > 0:
            quote_timestamp_utc = _parse_stooq_quote_timestamp(row[1], row[2])
            quote_age_seconds: int | None = None
            quote_timestamp_iso: str | None = None
            if quote_timestamp_utc is not None:
                quote_timestamp_iso = quote_timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                quote_age_seconds = max(
                    0,
                    int((fetched_at_utc - quote_timestamp_utc).total_seconds()),
                )

            return {
                "price": float(price),
                "quote_timestamp_utc": quote_timestamp_iso,
                "quote_age_seconds": quote_age_seconds,
                "source": f"STOOQ_INTRADAY:{stooq_symbol}",
            }

    return None


def fetch_stock_intraday_latest(ticker: str) -> float | None:
    """Fetch latest intraday quote price from Stooq quote endpoint."""
    quote = fetch_stock_intraday_quote(ticker)
    if not quote:
        return None
    return float(quote["price"])

"""Data fetching utilities for FRED and Stooq."""

from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone
from typing import Any
from urllib import request
from zoneinfo import ZoneInfo

import pandas as pd
from pandas_datareader.data import DataReader

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency fallback
    yf = None  # type: ignore[assignment]

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

_OTHER_LISTED_EXCHANGE_MAP = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "V": "IEX",
    "Z": "BATS",
}


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


def _normalize_ohlc_frame(frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    """Return normalized OHLCV bars with timezone-naive datetime index."""
    if frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    canonical_name = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
    }
    rename_map: dict[str, str] = {}
    for raw_col in out.columns:
        key = str(raw_col).strip().lower()
        if key in canonical_name:
            rename_map[raw_col] = canonical_name[key]
    if rename_map:
        out = out.rename(columns=rename_map)

    keep_cols = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in out.columns]
    if not keep_cols:
        return pd.DataFrame()
    out = out[keep_cols].copy()

    for col in keep_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")
    if "Close" in out.columns:
        out = out[out["Close"].notna()]

    out.attrs["source"] = frame_name
    return out


def _extract_yfinance_batch_bars_frame(frame: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if isinstance(frame.columns, pd.MultiIndex):
        data: dict[str, pd.Series] = {}
        for price_col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            key = (price_col, yahoo_symbol)
            if key in frame.columns:
                data[price_col] = frame[key]
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data, index=frame.index)

    # Single-symbol download path usually uses flat columns.
    return frame.copy()


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


def _yfinance_symbol_candidates(raw_symbol: str) -> list[str]:
    return _stock_symbol_variants(raw_symbol)


def _preferred_yfinance_symbol(raw_symbol: str) -> str:
    variants = _yfinance_symbol_candidates(raw_symbol)
    for variant in variants:
        if "." not in variant and "/" not in variant:
            return variant
    return variants[0] if variants else raw_symbol.strip().upper()


def _parse_stooq_quote_timestamp(
    quote_date_raw: str,
    quote_time_raw: str,
) -> datetime | None:
    date_digits = "".join(ch for ch in quote_date_raw.strip() if ch.isdigit())
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


def _parse_stooq_float(raw_value: str) -> float | None:
    text = str(raw_value).strip()
    if not text or text in {"N/D", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_stooq_percent_to_decimal(raw_value: str) -> float | None:
    text = str(raw_value).strip()
    if not text or text in {"N/D", "-"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text) / 100.0
    except ValueError:
        return None


def _fetch_text_payload(url: str, timeout: int = 20) -> str:
    req = request.Request(
        url,
        headers={
            "Accept": "text/plain,application/json;q=0.9,*/*;q=0.8",
            "User-Agent": UNIVERSE_FETCH_USER_AGENT,
        },
        method="GET",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _read_pipe_table(raw_text: str) -> list[dict[str, str]]:
    lines = [line for line in raw_text.splitlines() if line.strip()]
    filtered_lines = [line for line in lines if not line.startswith("File Creation Time")]
    if not filtered_lines:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(filtered_lines)), delimiter="|")
    rows: list[dict[str, str]] = []
    for row in reader:
        clean_row = {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}
        if clean_row:
            rows.append(clean_row)
    return rows


def _normalize_universe_ticker(raw_value: str) -> str:
    ticker = str(raw_value).strip().upper()
    if not ticker:
        return ""
    return ticker


def _normalize_index_ticker(raw_value: str) -> str:
    ticker = _normalize_universe_ticker(raw_value)
    if not ticker:
        return ""
    return ticker.replace(".", "-").replace("/", "-")


def _coerce_bool_flag(raw_value: str) -> bool:
    return str(raw_value).strip().upper() in {"Y", "YES", "TRUE", "1"}


def _normalize_nasdaq_listed_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in raw_rows:
        ticker = _normalize_universe_ticker(row.get("Symbol", ""))
        if not ticker:
            continue
        if _coerce_bool_flag(row.get("Test Issue", "N")):
            continue
        exchange_code = str(row.get("Market Category", "")).strip().upper()
        exchange = "NASDAQ"
        if exchange_code == "Q":
            exchange = "NASDAQ Global Select"
        elif exchange_code == "G":
            exchange = "NASDAQ Global Market"
        elif exchange_code == "S":
            exchange = "NASDAQ Capital Market"

        out.append(
            {
                "ticker": ticker,
                "security_name": str(row.get("Security Name", "")).strip(),
                "exchange": exchange,
                "is_etf": _coerce_bool_flag(row.get("ETF", "N")),
                "source": "NASDAQ_TRADER_NASDAQ_LISTED",
            }
        )
    return out


def _normalize_other_listed_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in raw_rows:
        ticker = _normalize_universe_ticker(row.get("ACT Symbol", ""))
        if not ticker:
            continue
        if _coerce_bool_flag(row.get("Test Issue", "N")):
            continue

        exchange_code = str(row.get("Exchange", "")).strip().upper()
        exchange = _OTHER_LISTED_EXCHANGE_MAP.get(exchange_code, exchange_code or "OTHER")

        out.append(
            {
                "ticker": ticker,
                "security_name": str(row.get("Security Name", "")).strip(),
                "exchange": exchange,
                "is_etf": _coerce_bool_flag(row.get("ETF", "N")),
                "source": "NASDAQ_TRADER_OTHER_LISTED",
            }
        )
    return out


def fetch_stock_universe_snapshot() -> pd.DataFrame:
    """Fetch a broad U.S. ticker universe from Nasdaq Trader symbol directories."""
    nasdaq_text = _fetch_text_payload(NASDAQ_LISTED_URL)
    other_text = _fetch_text_payload(OTHER_LISTED_URL)

    nasdaq_rows = _normalize_nasdaq_listed_rows(_read_pipe_table(nasdaq_text))
    other_rows = _normalize_other_listed_rows(_read_pipe_table(other_text))
    merged = [*nasdaq_rows, *other_rows]

    if not merged:
        raise RuntimeError("Ticker universe source returned no rows.")

    frame = pd.DataFrame(merged)
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["security_name"] = frame["security_name"].astype(str).str.strip()
    frame["exchange"] = frame["exchange"].astype(str).str.strip()
    frame["is_etf"] = frame["is_etf"].astype(bool)
    frame["source"] = frame["source"].astype(str).str.strip()

    deduped = frame.drop_duplicates(subset=["ticker"], keep="first").sort_values("ticker").reset_index(drop=True)
    return deduped[["ticker", "security_name", "exchange", "is_etf", "source"]]


def fetch_sp500_constituents() -> pd.DataFrame:
    """Fetch the latest S&P 500 constituents list."""
    raw_html = _fetch_text_payload(SP500_CONSTITUENTS_URL)
    tables = pd.read_html(io.StringIO(raw_html))
    if not tables:
        raise RuntimeError("No tables found in S&P 500 constituents source.")

    source_table: pd.DataFrame | None = None
    for table in tables:
        if "Symbol" in table.columns:
            source_table = table
            break
    if source_table is None:
        raise RuntimeError("S&P 500 constituents table with 'Symbol' column not found.")

    tickers = source_table["Symbol"].map(_normalize_index_ticker)
    tickers = tickers[tickers != ""]
    out = pd.DataFrame({"ticker": tickers})
    out["source"] = "WIKIPEDIA_SP500"
    out = out.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("S&P 500 constituents source returned no tickers.")
    return out[["ticker", "source"]]


def _fetch_yfinance_daily_bars(raw_symbol: str, start_date: str) -> pd.DataFrame | None:
    if yf is None:
        return None

    for yahoo_symbol in _yfinance_symbol_candidates(raw_symbol):
        try:
            frame = yf.download(  # type: ignore[union-attr]
                tickers=yahoo_symbol,
                start=start_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            continue
        normalized = _normalize_ohlc_frame(frame, f"YFINANCE:{yahoo_symbol}")
        if normalized.empty:
            continue
        return normalized
    return None


def fetch_stock_daily_bars_batch_yfinance(
    tickers: list[str],
    start_date: str,
    *,
    batch_size: int = 100,
    pause_seconds: float = 0.4,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV history for many tickers from Yahoo in batches."""
    if yf is None:
        return {}
    if not tickers:
        return {}

    normalized_tickers: list[str] = []
    seen: set[str] = set()
    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized_tickers.append(ticker)
    if not normalized_tickers:
        return {}

    results: dict[str, pd.DataFrame] = {}
    symbol_map = {ticker: _preferred_yfinance_symbol(ticker) for ticker in normalized_tickers}
    items = list(symbol_map.items())
    step = max(1, int(batch_size))

    for offset in range(0, len(items), step):
        batch_items = items[offset : offset + step]
        batch_symbols = [symbol for _, symbol in batch_items]
        try:
            frame = yf.download(  # type: ignore[union-attr]
                tickers=" ".join(batch_symbols),
                start=start_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            frame = pd.DataFrame()

        for ticker, yahoo_symbol in batch_items:
            raw_bars = _extract_yfinance_batch_bars_frame(frame, yahoo_symbol)
            normalized = _normalize_ohlc_frame(raw_bars, f"YFINANCE:{yahoo_symbol}")
            if normalized.empty:
                continue
            results[ticker] = normalized

        if pause_seconds > 0 and (offset + step) < len(items):
            time.sleep(float(pause_seconds))

    return results


def fetch_stock_daily_bars(ticker: str, start_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars for a stock ticker from Stooq, with Yahoo fallback."""
    raw_symbol = str(ticker).strip().upper()
    if not raw_symbol:
        raise RuntimeError("Ticker cannot be empty.")

    last_error: Exception | None = None
    for stooq_symbol in _stooq_symbol_candidates(raw_symbol):
        try:
            frame = DataReader(stooq_symbol, "stooq", start=start_date)
            normalized = _normalize_ohlc_frame(frame, f"STOOQ:{stooq_symbol}")
            if normalized.empty:
                continue
            return normalized
        except Exception as exc:
            last_error = exc

    fallback = _fetch_yfinance_daily_bars(raw_symbol, start_date)
    if fallback is not None:
        return fallback

    if last_error is not None:
        raise RuntimeError(f"No daily OHLC data returned from Stooq or Yahoo for {raw_symbol}: {last_error}") from last_error
    raise RuntimeError(f"No daily OHLC data returned from Stooq or Yahoo for {raw_symbol}.")


def _fetch_yfinance_daily_history(raw_symbol: str, start_date: str) -> pd.Series | None:
    if yf is None:
        return None

    for yahoo_symbol in _yfinance_symbol_candidates(raw_symbol):
        try:
            frame = yf.download(  # type: ignore[union-attr]
                tickers=yahoo_symbol,
                start=start_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            continue
        close = _extract_close_series(frame)
        if close.empty:
            continue
        normalized = _normalize_series(close, raw_symbol)
        if normalized.empty:
            continue
        normalized.attrs["source"] = f"YFINANCE:{yahoo_symbol}"
        return normalized
    return None


def _extract_yfinance_batch_close_series(frame: pd.DataFrame, yahoo_symbol: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    if isinstance(frame.columns, pd.MultiIndex):
        for price_col in ["Close", "Adj Close"]:
            key = (price_col, yahoo_symbol)
            if key in frame.columns:
                return frame[key].dropna().astype(float)
        return pd.Series(dtype=float)

    return _extract_close_series(frame)


def fetch_stock_daily_history_batch_yfinance(
    tickers: list[str],
    start_date: str,
    *,
    batch_size: int = 100,
    pause_seconds: float = 0.4,
) -> dict[str, pd.Series]:
    """Fetch daily close history for many tickers from Yahoo in batches.

    Returns only successful ticker series.
    """
    if yf is None:
        return {}
    if not tickers:
        return {}

    normalized_tickers = []
    seen: set[str] = set()
    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized_tickers.append(ticker)

    if not normalized_tickers:
        return {}

    results: dict[str, pd.Series] = {}
    symbol_map = {ticker: _preferred_yfinance_symbol(ticker) for ticker in normalized_tickers}
    items = list(symbol_map.items())
    step = max(1, int(batch_size))

    for offset in range(0, len(items), step):
        batch_items = items[offset : offset + step]
        batch_symbols = [symbol for _, symbol in batch_items]
        try:
            frame = yf.download(  # type: ignore[union-attr]
                tickers=" ".join(batch_symbols),
                start=start_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            frame = pd.DataFrame()

        for ticker, yahoo_symbol in batch_items:
            close = _extract_yfinance_batch_close_series(frame, yahoo_symbol)
            if close.empty:
                continue
            normalized = _normalize_series(close, ticker)
            if normalized.empty:
                continue
            normalized.attrs["source"] = f"YFINANCE:{yahoo_symbol}"
            results[ticker] = normalized

        if pause_seconds > 0 and (offset + step) < len(items):
            time.sleep(float(pause_seconds))

    return results


def fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
    """Fetch a single FRED series as a normalized pandas Series."""
    frame = DataReader(series_id, "fred", start=start_date)
    if frame.empty or series_id not in frame.columns:
        raise RuntimeError(f"No data returned from FRED for series {series_id}.")
    return _normalize_series(frame[series_id], series_id)


def fetch_stock_daily_history(ticker: str, start_date: str) -> pd.Series:
    """Fetch daily close history for a stock ticker from Stooq, with Yahoo fallback."""
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

    fallback = _fetch_yfinance_daily_history(raw_symbol, start_date)
    if fallback is not None:
        return fallback

    if last_error is not None:
        raise RuntimeError(
            f"No daily stock data returned from Stooq or Yahoo for {raw_symbol}: {last_error}"
        ) from last_error
    raise RuntimeError(f"No daily stock data returned from Stooq or Yahoo for {raw_symbol}.")


def fetch_stock_intraday_quote(ticker: str) -> dict[str, Any] | None:
    """Fetch latest intraday quote details from Stooq quote endpoint, with Yahoo fallback."""
    raw_symbol = str(ticker).strip().upper()
    if not raw_symbol:
        return None

    for stooq_symbol in _stooq_symbol_candidates(raw_symbol):
        quote_symbol = stooq_symbol.lower()
        # m3 adds both absolute and percent daily change values.
        url = f"https://stooq.com/q/l/?s={quote_symbol}&i=1&f=sd2t2ohlcvpm3"
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

        price = _parse_stooq_float(row[6])
        if price is None:
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

            previous_close = _parse_stooq_float(row[8]) if len(row) > 8 else None
            day_change = _parse_stooq_float(row[9]) if len(row) > 9 else None
            day_change_pct = _parse_stooq_percent_to_decimal(row[10]) if len(row) > 10 else None

            # Fallback: derive missing values from prev close if available.
            if day_change is None and previous_close is not None:
                day_change = float(price) - float(previous_close)
            if day_change_pct is None and previous_close not in (None, 0):
                day_change_pct = (float(price) - float(previous_close)) / float(previous_close)

            return {
                "price": float(price),
                "previous_close": previous_close,
                "day_change": day_change,
                "day_change_pct": day_change_pct,
                "quote_timestamp_utc": quote_timestamp_iso,
                "quote_age_seconds": quote_age_seconds,
                "source": f"STOOQ_INTRADAY:{stooq_symbol}",
            }

    if yf is not None:
        for yahoo_symbol in _yfinance_symbol_candidates(raw_symbol):
            try:
                ticker_obj = yf.Ticker(yahoo_symbol)  # type: ignore[union-attr]
                fast_info = getattr(ticker_obj, "fast_info", None)
            except Exception:
                continue

            if not fast_info:
                continue
            try:
                price = fast_info.get("lastPrice")
                if price is None:
                    price = fast_info.get("regularMarketPrice")
                if price is None:
                    continue
                price_value = float(price)
            except Exception:
                continue
            if price_value <= 0:
                continue

            previous_close_raw = fast_info.get("previousClose")
            previous_close: float | None = None
            try:
                if previous_close_raw is not None:
                    previous_close = float(previous_close_raw)
            except Exception:
                previous_close = None

            day_change: float | None = None
            day_change_pct: float | None = None
            if previous_close not in (None, 0):
                day_change = price_value - float(previous_close)
                day_change_pct = day_change / float(previous_close)

            return {
                "price": price_value,
                "previous_close": previous_close,
                "day_change": day_change,
                "day_change_pct": day_change_pct,
                "quote_timestamp_utc": None,
                "quote_age_seconds": None,
                "source": f"YFINANCE_INTRADAY:{yahoo_symbol}",
            }

    return None


def fetch_stock_intraday_latest(ticker: str) -> float | None:
    """Fetch latest intraday quote price from Stooq quote endpoint."""
    quote = fetch_stock_intraday_quote(ticker)
    if not quote:
        return None
    return float(quote["price"])

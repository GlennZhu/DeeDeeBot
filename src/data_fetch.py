"""Data fetching utilities for FRED and market-data providers."""

from __future__ import annotations

import csv
import io
import json
import os
import base64
import gzip
import time
from datetime import datetime, timezone
from typing import Any
from urllib import error, parse, request
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

MARKET_DATA_PROVIDER_AUTO = "auto"
MARKET_DATA_PROVIDER_PUBLIC = "public"
MARKET_DATA_PROVIDER_SCHWAB = "schwab"
SCHWAB_BASE_URL = "https://api.schwabapi.com"
SCHWAB_OAUTH_TOKEN_PATH = "/v1/oauth/token"
SCHWAB_PRICEHISTORY_PATH = "/marketdata/v1/pricehistory"
SCHWAB_QUOTES_PATH = "/marketdata/v1/quotes"
SCHWAB_QUOTES_MAX_SYMBOLS_PER_REQUEST = 200
SCHWAB_REQUEST_TIMEOUT_SECONDS = 12
SCHWAB_YEARS_LOOKBACK = 20
SCHWAB_FETCH_MAX_ATTEMPTS = 3
SCHWAB_FETCH_RETRY_BACKOFF_SECONDS = 0.75
FRED_GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_REQUEST_TIMEOUT_SECONDS = 45
FRED_FETCH_MAX_ATTEMPTS = 4
FRED_FETCH_RETRY_BACKOFF_SECONDS = 1.0


class MarketDataError(RuntimeError):
    """Typed market-data fetch failure."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        error_type: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.provider = str(provider)
        self.error_type = str(error_type)
        self.retryable = bool(retryable)


_SCHWAB_TOKEN_CACHE: dict[str, Any] = {
    "access_token": "",
    "expires_at_utc": None,
}


def _decode_http_error_body(exc: error.HTTPError) -> str:
    raw_bytes = b""
    try:
        raw_bytes = exc.read()
    except Exception:
        raw_bytes = b""
    if not raw_bytes:
        return str(getattr(exc, "reason", "")).strip()

    decoded = raw_bytes.decode("utf-8", errors="replace")
    if decoded and "\ufffd" not in decoded and decoded.strip():
        return decoded.strip()

    try:
        inflated = gzip.decompress(raw_bytes)
        decoded = inflated.decode("utf-8", errors="replace")
        if decoded.strip():
            return decoded.strip()
    except Exception:
        pass

    return decoded.strip() or repr(raw_bytes[:200])


def _extract_api_error_summary(raw_text: str) -> str:
    clean = str(raw_text).strip()
    if not clean:
        return ""
    try:
        parsed = json.loads(clean)
    except Exception:
        return clean
    if not isinstance(parsed, dict):
        return clean
    err = str(parsed.get("error", "")).strip()
    desc = str(parsed.get("error_description", "")).strip()
    parts = [part for part in [err, desc] if part]
    if parts:
        return " | ".join(parts)
    return clean


def _schwab_auth_action(raw_text: str) -> str:
    summary = _extract_api_error_summary(raw_text)
    normalized = summary.lower()
    if not normalized:
        return ""
    if "refresh_token_authentication_error" in normalized or "unsupported_token_type" in normalized:
        return (
            "Rotate SCHWAB_REFRESH_TOKEN via the external SchwabTokenRotator helper "
            "and update the GitHub secret."
        )
    if "invalid_client" in normalized or "unauthorized_client" in normalized:
        return "Verify SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET."
    return ""


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


def _market_data_provider() -> str:
    configured = str(os.getenv("MARKET_DATA_PROVIDER", MARKET_DATA_PROVIDER_AUTO)).strip().lower()
    if configured in {"stooq", "yfinance", MARKET_DATA_PROVIDER_PUBLIC}:
        return MARKET_DATA_PROVIDER_PUBLIC
    if configured == MARKET_DATA_PROVIDER_SCHWAB:
        return MARKET_DATA_PROVIDER_SCHWAB
    if configured not in {MARKET_DATA_PROVIDER_AUTO, MARKET_DATA_PROVIDER_PUBLIC, MARKET_DATA_PROVIDER_SCHWAB}:
        configured = MARKET_DATA_PROVIDER_AUTO
    if configured == MARKET_DATA_PROVIDER_AUTO:
        access_token = str(os.getenv("SCHWAB_ACCESS_TOKEN", "")).strip()
        refresh_token = str(os.getenv("SCHWAB_REFRESH_TOKEN", "")).strip()
        client_id = str(os.getenv("SCHWAB_CLIENT_ID", "")).strip()
        client_secret = str(os.getenv("SCHWAB_CLIENT_SECRET", "")).strip()
        if access_token or (refresh_token and client_id and client_secret):
            return MARKET_DATA_PROVIDER_SCHWAB
        return MARKET_DATA_PROVIDER_PUBLIC
    return configured


def _schwab_public_fallback_enabled() -> bool:
    raw = str(os.getenv("SCHWAB_ALLOW_PUBLIC_FALLBACK", "")).strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "y", "on"}


def _schwab_timeout_seconds() -> int:
    raw = str(os.getenv("SCHWAB_REQUEST_TIMEOUT_SECONDS", "")).strip()
    if not raw:
        return SCHWAB_REQUEST_TIMEOUT_SECONDS
    try:
        parsed = int(raw)
    except ValueError:
        return SCHWAB_REQUEST_TIMEOUT_SECONDS
    return max(3, parsed)


def _schwab_max_attempts() -> int:
    raw = str(os.getenv("SCHWAB_FETCH_MAX_ATTEMPTS", "")).strip()
    if not raw:
        return SCHWAB_FETCH_MAX_ATTEMPTS
    try:
        parsed = int(raw)
    except ValueError:
        return SCHWAB_FETCH_MAX_ATTEMPTS
    return max(1, parsed)


def _schwab_retry_backoff_seconds() -> float:
    raw = str(os.getenv("SCHWAB_FETCH_RETRY_BACKOFF_SECONDS", "")).strip()
    if not raw:
        return SCHWAB_FETCH_RETRY_BACKOFF_SECONDS
    try:
        parsed = float(raw)
    except ValueError:
        return SCHWAB_FETCH_RETRY_BACKOFF_SECONDS
    return max(0.0, parsed)


def _fred_timeout_seconds() -> int:
    raw = str(os.getenv("FRED_REQUEST_TIMEOUT_SECONDS", "")).strip()
    if not raw:
        return FRED_REQUEST_TIMEOUT_SECONDS
    try:
        parsed = int(raw)
    except ValueError:
        return FRED_REQUEST_TIMEOUT_SECONDS
    return max(5, parsed)


def _fred_max_attempts() -> int:
    raw = str(os.getenv("FRED_FETCH_MAX_ATTEMPTS", "")).strip()
    if not raw:
        return FRED_FETCH_MAX_ATTEMPTS
    try:
        parsed = int(raw)
    except ValueError:
        return FRED_FETCH_MAX_ATTEMPTS
    return max(1, parsed)


def _fred_retry_backoff_seconds() -> float:
    raw = str(os.getenv("FRED_FETCH_RETRY_BACKOFF_SECONDS", "")).strip()
    if not raw:
        return FRED_FETCH_RETRY_BACKOFF_SECONDS
    try:
        parsed = float(raw)
    except ValueError:
        return FRED_FETCH_RETRY_BACKOFF_SECONDS
    return max(0.0, parsed)


def _fred_api_key() -> str:
    return str(os.getenv("FRED_API_KEY", "")).strip()


def _fred_api_observations_url() -> str:
    raw = str(os.getenv("FRED_API_OBSERVATIONS_URL", "")).strip()
    if not raw:
        return FRED_API_OBSERVATIONS_URL
    return raw


def _fred_graph_csv_url() -> str:
    raw = str(os.getenv("FRED_GRAPH_CSV_URL", "")).strip()
    if not raw:
        return FRED_GRAPH_CSV_URL
    return raw


def _fetch_fred_series_via_api(
    series_id: str,
    start_date: str,
    *,
    api_key: str,
    timeout_seconds: int,
) -> pd.Series:
    query = parse.urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "asc",
        }
    )
    url = f"{_fred_api_observations_url()}?{query}"
    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": UNIVERSE_FETCH_USER_AGENT,
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        if status in {400, 401, 403, 404}:
            raise RuntimeError(f"FRED API rejected series {series_id} request: HTTP {status} {body}") from exc
        raise

    try:
        parsed = json.loads(payload)
    except Exception as exc:
        raise RuntimeError(f"FRED API returned invalid JSON for series {series_id}: {exc}") from exc

    if isinstance(parsed, dict) and "error_code" in parsed:
        raise RuntimeError(
            f"FRED API error for series {series_id}: {parsed.get('error_message', 'unknown error')}"
        )

    observations = parsed.get("observations") if isinstance(parsed, dict) else None
    if not isinstance(observations, list):
        raise RuntimeError(f"FRED API observations payload missing for series {series_id}.")

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        raw_date = str(obs.get("date", "")).strip()
        raw_value = str(obs.get("value", "")).strip()
        if not raw_date or raw_value in {"", ".", "nan", "NaN"}:
            continue
        parsed_date = pd.to_datetime(raw_date, errors="coerce")
        parsed_value = pd.to_numeric(raw_value, errors="coerce")
        if pd.isna(parsed_date) or pd.isna(parsed_value):
            continue
        dates.append(pd.Timestamp(parsed_date))
        values.append(float(parsed_value))

    if not values:
        raise RuntimeError(f"No data returned from FRED API for series {series_id}.")

    frame = pd.DataFrame({"date": dates, "value": values})
    frame = frame.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"], keep="last").set_index("date")
    return _normalize_series(frame["value"], series_id)


def _fetch_fred_series_via_graph_csv(
    series_id: str,
    start_date: str,
    *,
    timeout_seconds: int,
) -> pd.Series:
    query = parse.urlencode(
        {
            "id": series_id,
            "cosd": start_date,
        }
    )
    url = f"{_fred_graph_csv_url()}?{query}"
    req = request.Request(
        url,
        headers={
            "Accept": "text/csv",
            "User-Agent": UNIVERSE_FETCH_USER_AGENT,
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        raise RuntimeError(f"FRED graph CSV request failed for {series_id}: HTTP {status} {body}") from exc

    if not payload.strip():
        raise RuntimeError(f"FRED graph CSV returned an empty payload for series {series_id}.")

    try:
        frame = pd.read_csv(io.StringIO(payload))
    except Exception as exc:
        raise RuntimeError(f"FRED graph CSV returned invalid CSV for series {series_id}: {exc}") from exc

    if frame.empty:
        raise RuntimeError(f"No data returned from FRED graph CSV for series {series_id}.")

    date_col = "observation_date" if "observation_date" in frame.columns else frame.columns[0]
    if series_id in frame.columns:
        value_col = series_id
    elif len(frame.columns) >= 2:
        value_col = frame.columns[1]
    else:
        raise RuntimeError(f"FRED graph CSV missing value column for series {series_id}.")

    parsed_dates = pd.to_datetime(frame[date_col], errors="coerce")
    parsed_values = pd.to_numeric(frame[value_col], errors="coerce")
    out = pd.Series(parsed_values.values, index=parsed_dates, name=series_id)
    out = out[out.index.notna()].dropna()
    if out.empty:
        raise RuntimeError(f"No data returned from FRED graph CSV for series {series_id}.")
    return _normalize_series(out, series_id)


def _schwab_base_url() -> str:
    raw = str(os.getenv("SCHWAB_BASE_URL", SCHWAB_BASE_URL)).strip()
    if not raw:
        return SCHWAB_BASE_URL
    return raw.rstrip("/")


def _schwab_token_is_fresh() -> bool:
    access_token = str(_SCHWAB_TOKEN_CACHE.get("access_token", "")).strip()
    expires_at_utc = _SCHWAB_TOKEN_CACHE.get("expires_at_utc")
    if not access_token or not isinstance(expires_at_utc, datetime):
        return False
    now = datetime.now(timezone.utc)
    return now < expires_at_utc


def _schwab_refresh_access_token() -> str:
    refresh_token = str(os.getenv("SCHWAB_REFRESH_TOKEN", "")).strip()
    client_id = str(os.getenv("SCHWAB_CLIENT_ID", "")).strip()
    client_secret = str(os.getenv("SCHWAB_CLIENT_SECRET", "")).strip()
    if not refresh_token or not client_id or not client_secret:
        raise MarketDataError(
            "SCHWAB_ACCESS_TOKEN is missing and refresh credentials are not fully configured.",
            provider="schwab",
            error_type="auth_error",
            retryable=False,
        )

    token_url = f"{_schwab_base_url()}{SCHWAB_OAUTH_TOKEN_PATH}"
    payload = parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode("utf-8")
    basic_auth = f"{client_id}:{client_secret}".encode("utf-8")
    auth_header = base64.b64encode(basic_auth).decode("ascii")
    req = request.Request(
        token_url,
        headers={
            "Authorization": f"Basic {auth_header}",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": UNIVERSE_FETCH_USER_AGENT,
        },
        data=payload,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=_schwab_timeout_seconds()) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = _extract_api_error_summary(_decode_http_error_body(exc))
        action = _schwab_auth_action(body)
        detail = body or str(getattr(exc, "reason", "")).strip() or "unknown_error"
        if action:
            detail = f"{detail} Action: {action}"
        raise MarketDataError(
            f"Failed to refresh Schwab access token (HTTP {exc.code}): {detail}",
            provider="schwab",
            error_type="auth_error",
            retryable=False,
        ) from exc
    except Exception as exc:
        raise MarketDataError(
            f"Failed to refresh Schwab access token: {exc}",
            provider="schwab",
            error_type="auth_error",
            retryable=False,
        ) from exc

    try:
        parsed = json.loads(raw_body)
    except Exception as exc:
        raise MarketDataError(
            f"Schwab token refresh returned invalid JSON: {exc}",
            provider="schwab",
            error_type="auth_error",
            retryable=False,
        ) from exc

    access_token = str(parsed.get("access_token", "")).strip()
    expires_in = pd.to_numeric(parsed.get("expires_in"), errors="coerce")
    if not access_token or pd.isna(expires_in):
        raise MarketDataError(
            f"Schwab token refresh response missing required fields: {parsed}",
            provider="schwab",
            error_type="auth_error",
            retryable=False,
        )

    ttl_seconds = max(60, int(float(expires_in)))
    _SCHWAB_TOKEN_CACHE["access_token"] = access_token
    _SCHWAB_TOKEN_CACHE["expires_at_utc"] = datetime.now(timezone.utc) + pd.Timedelta(seconds=ttl_seconds - 30)
    return access_token


def _schwab_access_token() -> str:
    env_access_token = str(os.getenv("SCHWAB_ACCESS_TOKEN", "")).strip()
    if env_access_token:
        return env_access_token
    if _schwab_token_is_fresh():
        return str(_SCHWAB_TOKEN_CACHE.get("access_token", "")).strip()
    return _schwab_refresh_access_token()


def _schwab_raise_http_error(exc: error.HTTPError, *, endpoint: str, symbol: str = "") -> None:
    status = int(getattr(exc, "code", 0) or 0)
    body = _extract_api_error_summary(_decode_http_error_body(exc))

    provider = "schwab"
    symbol_suffix = f" symbol={symbol}" if symbol else ""
    details = body or str(exc.reason) or str(exc)
    if status == 429:
        raise MarketDataError(
            f"Schwab rate limited request endpoint={endpoint}{symbol_suffix}: {details}",
            provider=provider,
            error_type="rate_limited",
            retryable=True,
        ) from exc
    if status in {401, 403}:
        raise MarketDataError(
            f"Schwab auth rejected request endpoint={endpoint}{symbol_suffix}: {details}",
            provider=provider,
            error_type="auth_error",
            retryable=False,
        ) from exc
    if status == 404:
        raise MarketDataError(
            f"Schwab symbol not found endpoint={endpoint}{symbol_suffix}: {details}",
            provider=provider,
            error_type="symbol_not_found",
            retryable=False,
        ) from exc
    if status in {408, 500, 502, 503, 504}:
        raise MarketDataError(
            f"Schwab upstream unavailable endpoint={endpoint}{symbol_suffix}: {details}",
            provider=provider,
            error_type="upstream_unreachable",
            retryable=True,
        ) from exc
    raise MarketDataError(
        f"Schwab API error endpoint={endpoint}{symbol_suffix} HTTP {status}: {details}",
        provider=provider,
        error_type="upstream_api_error",
        retryable=status >= 500,
    ) from exc


def _schwab_get_json(path: str, query: dict[str, Any]) -> Any:
    query_items = {k: v for k, v in query.items() if v is not None and str(v) != ""}
    encoded_query = parse.urlencode(query_items, doseq=True)
    url = f"{_schwab_base_url()}{path}"
    if encoded_query:
        url = f"{url}?{encoded_query}"
    max_attempts = _schwab_max_attempts()
    retry_backoff_seconds = _schwab_retry_backoff_seconds()

    for attempt in range(max_attempts):
        try:
            token = _schwab_access_token()
            req = request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": UNIVERSE_FETCH_USER_AGENT,
                },
                method="GET",
            )
            with request.urlopen(req, timeout=_schwab_timeout_seconds()) as response:
                payload = response.read().decode("utf-8", errors="replace")
            return json.loads(payload) if payload.strip() else {}
        except error.HTTPError as exc:
            try:
                _schwab_raise_http_error(exc, endpoint=path)
            except MarketDataError as raised:
                err = raised
            else:
                err = MarketDataError(
                    f"Schwab API error endpoint={path}: HTTP {getattr(exc, 'code', 'unknown')}",
                    provider="schwab",
                    error_type="upstream_api_error",
                    retryable=False,
                )
        except error.URLError as exc:
            err = MarketDataError(
                f"Schwab upstream unreachable endpoint={path}: {exc}",
                provider="schwab",
                error_type="upstream_unreachable",
                retryable=True,
            )
        except TimeoutError as exc:
            err = MarketDataError(
                f"Schwab request timed out endpoint={path}: {exc}",
                provider="schwab",
                error_type="upstream_unreachable",
                retryable=True,
            )
        except json.JSONDecodeError as exc:
            err = MarketDataError(
                f"Schwab returned invalid JSON for endpoint={path}: {exc}",
                provider="schwab",
                error_type="upstream_invalid_response",
                retryable=True,
            )
        except MarketDataError as exc:
            err = exc
        except Exception as exc:
            err = MarketDataError(
                f"Unexpected Schwab request error endpoint={path}: {exc}",
                provider="schwab",
                error_type="upstream_api_error",
                retryable=False,
            )

        is_last_attempt = attempt >= (max_attempts - 1)
        if is_last_attempt or not bool(err.retryable):
            raise err

        sleep_seconds = retry_backoff_seconds * (2**attempt)
        print(
            f"Warning: Schwab request failed endpoint={path} "
            f"(attempt {attempt + 1}/{max_attempts}), retrying: {err}"
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Unreachable retry state for Schwab endpoint={path}.")


def _schwab_candles_to_ohlc_frame(candles: Any, symbol: str) -> pd.DataFrame:
    if not isinstance(candles, list) or not candles:
        return pd.DataFrame()
    frame = pd.DataFrame(candles)
    if frame.empty or "datetime" not in frame.columns:
        return pd.DataFrame()

    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms", errors="coerce", utc=True)
    frame = frame[frame["datetime"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["datetime"] = frame["datetime"].dt.tz_localize(None)
    frame = frame.set_index("datetime").sort_index()
    keep_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in frame.columns]
    frame = frame[keep_cols].copy()
    for col in keep_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[frame["Close"].notna()] if "Close" in frame.columns else frame
    frame.attrs["source"] = f"SCHWAB:{symbol}"
    return frame


def _fetch_schwab_daily_bars(raw_symbol: str, start_date: str) -> pd.DataFrame:
    symbol = _preferred_yfinance_symbol(raw_symbol)
    start_ts = pd.Timestamp(start_date)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    now_ts = pd.Timestamp.utcnow()
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    query = {
        "symbol": symbol,
        "periodType": "year",
        "period": SCHWAB_YEARS_LOOKBACK,
        "frequencyType": "daily",
        "frequency": 1,
        "needExtendedHoursData": "false",
        "needPreviousClose": "false",
        "startDate": int(start_ts.timestamp() * 1000),
        "endDate": int(now_ts.timestamp() * 1000),
    }
    payload = _schwab_get_json(SCHWAB_PRICEHISTORY_PATH, query)
    candles = payload.get("candles", []) if isinstance(payload, dict) else []
    frame = _schwab_candles_to_ohlc_frame(candles, symbol)
    if frame.empty:
        raise MarketDataError(
            f"No daily OHLC candles returned from Schwab for {symbol}.",
            provider="schwab",
            error_type="symbol_not_found",
            retryable=False,
        )
    return frame


def _fetch_schwab_intraday_quotes_batch(tickers: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    normalized_tickers = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not normalized_tickers:
        return results

    max_symbols_raw = str(os.getenv("SCHWAB_QUOTES_MAX_SYMBOLS_PER_REQUEST", "")).strip()
    try:
        max_symbols = int(max_symbols_raw) if max_symbols_raw else SCHWAB_QUOTES_MAX_SYMBOLS_PER_REQUEST
    except ValueError:
        max_symbols = SCHWAB_QUOTES_MAX_SYMBOLS_PER_REQUEST
    max_symbols = max(1, min(500, max_symbols))

    for offset in range(0, len(normalized_tickers), max_symbols):
        batch = normalized_tickers[offset : offset + max_symbols]
        payload = _schwab_get_json(
            SCHWAB_QUOTES_PATH,
            {
                "symbols": ",".join(batch),
                "fields": "quote",
                "indicative": "false",
            },
        )
        if not isinstance(payload, dict):
            continue
        fetched_at_utc = datetime.now(timezone.utc)
        for symbol in batch:
            raw_entry = payload.get(symbol)
            if raw_entry is None:
                raw_entry = payload.get(symbol.upper())
            if raw_entry is None:
                raw_entry = payload.get(symbol.lower())
            if not isinstance(raw_entry, dict):
                continue
            quote_payload = raw_entry.get("quote") if isinstance(raw_entry.get("quote"), dict) else raw_entry
            if not isinstance(quote_payload, dict):
                continue

            price_candidates = [
                quote_payload.get("lastPrice"),
                quote_payload.get("mark"),
                quote_payload.get("closePrice"),
                quote_payload.get("bidPrice"),
            ]
            price_value: float | None = None
            for candidate in price_candidates:
                parsed = pd.to_numeric(candidate, errors="coerce")
                if pd.notna(parsed):
                    price_value = float(parsed)
                    break
            if price_value is None or price_value <= 0:
                continue

            previous_close_raw = quote_payload.get("closePrice")
            previous_close_num = pd.to_numeric(previous_close_raw, errors="coerce")
            previous_close = float(previous_close_num) if pd.notna(previous_close_num) else None
            day_change = pd.to_numeric(quote_payload.get("netChange"), errors="coerce")
            day_change_pct = pd.to_numeric(quote_payload.get("netPercentChangeInDouble"), errors="coerce")
            day_change_value = float(day_change) if pd.notna(day_change) else None
            day_change_pct_value = (float(day_change_pct) / 100.0) if pd.notna(day_change_pct) else None
            if day_change_value is None and previous_close not in (None, 0):
                day_change_value = float(price_value) - float(previous_close)
            if day_change_pct_value is None and previous_close not in (None, 0):
                day_change_pct_value = (float(price_value) - float(previous_close)) / float(previous_close)

            quote_time_raw = quote_payload.get("quoteTime") or quote_payload.get("tradeTime")
            quote_timestamp_utc: str | None = None
            quote_age_seconds: int | None = None
            quote_time_num = pd.to_numeric(quote_time_raw, errors="coerce")
            if pd.notna(quote_time_num):
                try:
                    quote_ts = pd.to_datetime(int(float(quote_time_num)), unit="ms", utc=True)
                    quote_timestamp_utc = quote_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                    quote_age_seconds = max(0, int((fetched_at_utc - quote_ts.to_pydatetime()).total_seconds()))
                except Exception:
                    quote_timestamp_utc = None
                    quote_age_seconds = None

            results[symbol] = {
                "price": float(price_value),
                "previous_close": previous_close,
                "day_change": day_change_value,
                "day_change_pct": day_change_pct_value,
                "quote_timestamp_utc": quote_timestamp_utc,
                "quote_age_seconds": quote_age_seconds,
                "source": f"SCHWAB_QUOTES:{symbol}",
            }
    return results


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


def _normalize_ticker_batch(tickers: list[str]) -> list[str]:
    normalized_tickers: list[str] = []
    seen: set[str] = set()
    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        normalized_tickers.append(ticker)
    return normalized_tickers


def _fetch_schwab_daily_bars_batch(
    tickers: list[str],
    start_date: str,
    *,
    batch_size: int = 100,
    pause_seconds: float = 0.4,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    normalized_tickers = _normalize_ticker_batch(tickers)
    if not normalized_tickers:
        return results

    step = max(1, int(batch_size))
    consecutive_rate_limited_batches = 0
    for offset in range(0, len(normalized_tickers), step):
        batch = normalized_tickers[offset : offset + step]
        batch_rate_limited = 0
        for ticker in batch:
            try:
                results[ticker] = _fetch_schwab_daily_bars(ticker, start_date)
            except MarketDataError as exc:
                if exc.error_type == "auth_error":
                    raise
                if exc.error_type == "rate_limited":
                    batch_rate_limited += 1
                continue
            except Exception:
                continue

        if batch and batch_rate_limited == len(batch):
            consecutive_rate_limited_batches += 1
        else:
            consecutive_rate_limited_batches = 0

        if consecutive_rate_limited_batches >= 2:
            print(
                "Warning: Schwab daily OHLC batch prefetch is repeatedly rate-limited; "
                "stopping prefetch early to limit churn."
            )
            break

        if pause_seconds > 0 and (offset + step) < len(normalized_tickers):
            time.sleep(float(pause_seconds))
    return results


def fetch_stock_daily_bars_batch_yfinance(
    tickers: list[str],
    start_date: str,
    *,
    batch_size: int = 100,
    pause_seconds: float = 0.4,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV history for many tickers from configured market-data provider."""
    provider = _market_data_provider()
    if provider == MARKET_DATA_PROVIDER_SCHWAB:
        try:
            return _fetch_schwab_daily_bars_batch(
                tickers,
                start_date,
                batch_size=batch_size,
                pause_seconds=pause_seconds,
            )
        except MarketDataError:
            if not _schwab_public_fallback_enabled():
                raise

    if yf is None:
        return {}
    if not tickers:
        return {}

    normalized_tickers = _normalize_ticker_batch(tickers)
    if not normalized_tickers:
        return {}

    results: dict[str, pd.DataFrame] = {}
    symbol_map = {ticker: _preferred_yfinance_symbol(ticker) for ticker in normalized_tickers}
    items = list(symbol_map.items())
    step = max(1, int(batch_size))
    consecutive_empty_large_batches = 0

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

        batch_hits = 0
        for ticker, yahoo_symbol in batch_items:
            raw_bars = _extract_yfinance_batch_bars_frame(frame, yahoo_symbol)
            normalized = _normalize_ohlc_frame(raw_bars, f"YFINANCE:{yahoo_symbol}")
            if normalized.empty:
                continue
            results[ticker] = normalized
            batch_hits += 1

        if batch_hits == 0 and len(batch_items) >= 25:
            consecutive_empty_large_batches += 1
        else:
            consecutive_empty_large_batches = 0
        if consecutive_empty_large_batches >= 2:
            print(
                "Warning: yfinance daily OHLC batch prefetch returned empty results for consecutive large batches; "
                "stopping early to avoid rate-limit churn."
            )
            break

        if pause_seconds > 0 and (offset + step) < len(items):
            time.sleep(float(pause_seconds))

    return results


def fetch_stock_daily_bars(ticker: str, start_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars for a stock ticker from configured market-data provider."""
    raw_symbol = str(ticker).strip().upper()
    if not raw_symbol:
        raise RuntimeError("Ticker cannot be empty.")

    provider = _market_data_provider()
    if provider == MARKET_DATA_PROVIDER_SCHWAB:
        try:
            return _fetch_schwab_daily_bars(raw_symbol, start_date)
        except MarketDataError:
            if not _schwab_public_fallback_enabled():
                raise

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
        raise MarketDataError(
            f"No daily OHLC data returned from Stooq or Yahoo for {raw_symbol}: {last_error}",
            provider="public",
            error_type="upstream_unreachable",
            retryable=True,
        ) from last_error
    raise MarketDataError(
        f"No daily OHLC data returned from Stooq or Yahoo for {raw_symbol}.",
        provider="public",
        error_type="symbol_not_found",
        retryable=False,
    )


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


def _fetch_schwab_daily_history_batch(
    tickers: list[str],
    start_date: str,
    *,
    batch_size: int = 100,
    pause_seconds: float = 0.4,
) -> dict[str, pd.Series]:
    bars_by_ticker = _fetch_schwab_daily_bars_batch(
        tickers,
        start_date,
        batch_size=batch_size,
        pause_seconds=pause_seconds,
    )
    out: dict[str, pd.Series] = {}
    for ticker, bars in bars_by_ticker.items():
        close = _extract_close_series(bars)
        if close.empty:
            continue
        normalized = _normalize_series(close, ticker)
        if normalized.empty:
            continue
        normalized.attrs["source"] = str(getattr(bars, "attrs", {}).get("source", f"SCHWAB:{ticker}"))
        out[ticker] = normalized
    return out


def fetch_stock_daily_history_batch_yfinance(
    tickers: list[str],
    start_date: str,
    *,
    batch_size: int = 100,
    pause_seconds: float = 0.4,
) -> dict[str, pd.Series]:
    """Fetch daily close history for many tickers from configured market-data provider.

    Returns only successful ticker series.
    """
    provider = _market_data_provider()
    if provider == MARKET_DATA_PROVIDER_SCHWAB:
        try:
            return _fetch_schwab_daily_history_batch(
                tickers,
                start_date,
                batch_size=batch_size,
                pause_seconds=pause_seconds,
            )
        except MarketDataError:
            if not _schwab_public_fallback_enabled():
                raise

    if yf is None:
        return {}
    if not tickers:
        return {}

    normalized_tickers = _normalize_ticker_batch(tickers)

    if not normalized_tickers:
        return {}

    results: dict[str, pd.Series] = {}
    symbol_map = {ticker: _preferred_yfinance_symbol(ticker) for ticker in normalized_tickers}
    items = list(symbol_map.items())
    step = max(1, int(batch_size))
    consecutive_empty_large_batches = 0

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

        batch_hits = 0
        for ticker, yahoo_symbol in batch_items:
            close = _extract_yfinance_batch_close_series(frame, yahoo_symbol)
            if close.empty:
                continue
            normalized = _normalize_series(close, ticker)
            if normalized.empty:
                continue
            normalized.attrs["source"] = f"YFINANCE:{yahoo_symbol}"
            results[ticker] = normalized
            batch_hits += 1

        if batch_hits == 0 and len(batch_items) >= 25:
            consecutive_empty_large_batches += 1
        else:
            consecutive_empty_large_batches = 0
        if consecutive_empty_large_batches >= 2:
            print(
                "Warning: yfinance daily history batch prefetch returned empty results for consecutive large batches; "
                "stopping early to avoid rate-limit churn."
            )
            break

        if pause_seconds > 0 and (offset + step) < len(items):
            time.sleep(float(pause_seconds))

    return results


def fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
    """Fetch a single FRED series as a normalized pandas Series."""
    max_attempts = _fred_max_attempts()
    timeout_seconds = _fred_timeout_seconds()
    retry_backoff_seconds = _fred_retry_backoff_seconds()
    api_key = _fred_api_key()
    source_name = "FRED API" if api_key else "FRED graph CSV"
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            if api_key:
                return _fetch_fred_series_via_api(
                    series_id,
                    start_date,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )

            return _fetch_fred_series_via_graph_csv(
                series_id,
                start_date,
                timeout_seconds=timeout_seconds,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            is_last_attempt = attempt >= (max_attempts - 1)
            if is_last_attempt:
                break
            print(
                f"Warning: {source_name} fetch failed for {series_id} "
                f"(attempt {attempt + 1}/{max_attempts}), retrying: {exc}"
            )
            sleep_seconds = retry_backoff_seconds * (2**attempt)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Failed to fetch FRED series {series_id} from {source_name} after {max_attempts} attempts."
    ) from last_error


def fetch_stock_daily_history(ticker: str, start_date: str) -> pd.Series:
    """Fetch daily close history for a stock ticker from configured market-data provider."""
    raw_symbol = str(ticker).strip().upper()
    if not raw_symbol:
        raise RuntimeError("Ticker cannot be empty.")

    provider = _market_data_provider()
    if provider == MARKET_DATA_PROVIDER_SCHWAB:
        try:
            bars = _fetch_schwab_daily_bars(raw_symbol, start_date)
            close = _extract_close_series(bars)
            if close.empty:
                raise MarketDataError(
                    f"No daily close returned from Schwab for {raw_symbol}.",
                    provider="schwab",
                    error_type="symbol_not_found",
                    retryable=False,
                )
            normalized = _normalize_series(close, raw_symbol)
            if normalized.empty:
                raise MarketDataError(
                    f"Schwab returned empty normalized close series for {raw_symbol}.",
                    provider="schwab",
                    error_type="symbol_not_found",
                    retryable=False,
                )
            normalized.attrs["source"] = str(getattr(bars, "attrs", {}).get("source", f"SCHWAB:{raw_symbol}"))
            return normalized
        except MarketDataError:
            if not _schwab_public_fallback_enabled():
                raise

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
        raise MarketDataError(
            f"No daily stock data returned from Stooq or Yahoo for {raw_symbol}: {last_error}",
            provider="public",
            error_type="upstream_unreachable",
            retryable=True,
        ) from last_error
    raise MarketDataError(
        f"No daily stock data returned from Stooq or Yahoo for {raw_symbol}.",
        provider="public",
        error_type="symbol_not_found",
        retryable=False,
    )


def _fetch_public_intraday_quote(raw_symbol: str) -> dict[str, Any] | None:
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


def fetch_stock_intraday_quotes_batch(tickers: list[str]) -> dict[str, dict[str, Any]]:
    provider = _market_data_provider()
    normalized_tickers = _normalize_ticker_batch(tickers)
    if not normalized_tickers:
        return {}

    if provider == MARKET_DATA_PROVIDER_SCHWAB:
        try:
            return _fetch_schwab_intraday_quotes_batch(normalized_tickers)
        except MarketDataError:
            if not _schwab_public_fallback_enabled():
                raise

    out: dict[str, dict[str, Any]] = {}
    for ticker in normalized_tickers:
        quote = _fetch_public_intraday_quote(ticker)
        if quote is not None:
            out[ticker] = quote
    return out


def fetch_stock_intraday_quote(ticker: str) -> dict[str, Any] | None:
    """Fetch latest intraday quote details from configured market-data provider."""
    raw_symbol = str(ticker).strip().upper()
    if not raw_symbol:
        return None

    provider = _market_data_provider()
    if provider == MARKET_DATA_PROVIDER_SCHWAB:
        try:
            batched = _fetch_schwab_intraday_quotes_batch([raw_symbol])
            quote = batched.get(raw_symbol)
            if quote is not None:
                return quote
        except MarketDataError:
            if not _schwab_public_fallback_enabled():
                raise
    return _fetch_public_intraday_quote(raw_symbol)


def fetch_stock_intraday_latest(ticker: str) -> float | None:
    """Fetch latest intraday quote price from Stooq quote endpoint."""
    quote = fetch_stock_intraday_quote(ticker)
    if not quote:
        return None
    return float(quote["price"])

"""Data pipeline for fetching macro series, computing signals, and writing CSV caches."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import DEFAULT_LOOKBACK_YEARS, METRIC_ORDER, SERIES_CONFIG
from src.data_fetch import (
    MarketDataError,
    fetch_fred_series,
    fetch_stock_daily_history,
    fetch_stock_daily_history_batch_yfinance,
    fetch_stock_intraday_quotes_batch,
    fetch_stock_intraday_quote,
)
from src.signals import compute_signals
from src.stock_signals import (
    BENCHMARK_RELATED_TRIGGER_COLUMNS,
    STOCK_SIGNAL_COLUMNS,
    STOCK_TRIGGER_COLUMNS,
    compute_stock_signal_row,
)
from src.transform import build_buffett_ratio, prepare_monthly_series

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVED_DATA_DIR = PROJECT_ROOT / "data" / "derived"
STOCK_WATCHLIST_PATH = DERIVED_DATA_DIR / "stock_watchlist.csv"
STOCK_SIGNALS_PATH = DERIVED_DATA_DIR / "stock_signals_latest.csv"
SIGNAL_EVENTS_PATH = DERIVED_DATA_DIR / "signal_events_7d.csv"
STOCK_FAIL_MAX_ERROR_RATIO_DEFAULT = 0.80
WATCHLIST_CIRCUIT_PREFETCH_MAX_COVERAGE_DEFAULT = 0.05
WATCHLIST_CIRCUIT_PROBE_COUNT_DEFAULT = 4

SIGNAL_EVENT_RETENTION_DAYS = 7
SIGNAL_EVENT_COLUMNS = [
    "event_timestamp_utc",
    "event_timestamp_et",
    "domain",
    "event_type",
    "subject_id",
    "subject_name",
    "benchmark_ticker",
    "signal_id",
    "signal_label",
    "as_of_date",
    "value",
    "price",
    "state_transition",
    "status",
    "details",
]
LEGACY_SIGNAL_EVENT_TIMESTAMP_COLUMN = "event_timestamp_pt"
DEFAULT_STOCK_BENCHMARK = "QQQ"
DEFAULT_STOCK_WATCHLIST: list[dict[str, str]] = [
    {"ticker": "GOOG", "benchmark": DEFAULT_STOCK_BENCHMARK},
    {"ticker": "AVGO", "benchmark": DEFAULT_STOCK_BENCHMARK},
    {"ticker": "NVDA", "benchmark": DEFAULT_STOCK_BENCHMARK},
    {"ticker": "MSFT", "benchmark": DEFAULT_STOCK_BENCHMARK},
    {"ticker": "QQQ", "benchmark": DEFAULT_STOCK_BENCHMARK},
]

THRESHOLD_TRIGGER_MAP: dict[str, dict[str, set[str]]] = {
    "m2": {
        "long_environment": {"m2_yoy_gt_0"},
        "caution_contraction": set(),
        "insufficient_data": set(),
    },
    "hiring_rate": {
        "recession_alert": {"hiring_rate_lte_3_4"},
        "normal": set(),
    },
    "ten_year_yield": {
        "normal": set(),
        "equity_pressure_zone": {"ten_year_yield_gte_4_4"},
        "extreme_pressure_bond_opportunity": {"ten_year_yield_gte_4_4", "ten_year_yield_gte_5_0"},
    },
    "buffett_ratio": {
        "overheat_peak_risk": {"buffett_ratio_gte_2_0"},
        "normal": set(),
    },
    "unemployment_rate": {
        "labor_weakening": {"unemployment_mom_gte_0_1", "unemployment_mom_gte_0_2"},
        "watch": {"unemployment_mom_gte_0_1"},
        "stable": set(),
        "insufficient_data": set(),
    },
}

THRESHOLD_LABELS: dict[str, str] = {
    "m2_yoy_gt_0": "M2 YoY > 0",
    "hiring_rate_lte_3_4": "Hiring rate <= 3.4",
    "ten_year_yield_gte_4_4": "10Y yield >= 4.4",
    "ten_year_yield_gte_5_0": "10Y yield >= 5.0",
    "buffett_ratio_gte_2_0": "Buffett ratio >= 2.0",
    "unemployment_mom_gte_0_1": "Unemployment MoM >= 0.1",
    "unemployment_mom_gte_0_2": "Unemployment MoM >= 0.2",
}

NEGATIVE_MACRO_THRESHOLD_IDS: set[str] = {
    "hiring_rate_lte_3_4",
    "ten_year_yield_gte_4_4",
    "ten_year_yield_gte_5_0",
    "buffett_ratio_gte_2_0",
    "unemployment_mom_gte_0_1",
    "unemployment_mom_gte_0_2",
}

STOCK_TRIGGER_LABELS: dict[str, str] = {
    "entry_bullish_alignment": "Entry: Trend alignment (SMA14 > SMA50 > SMA100/200)",
    "entry_rsi_extreme_oversold": "Entry: Extreme RSI oversold (< 25; QQQ < 30)",
    "exit_price_below_sma50": "Exit: Price Below SMA50",
    "exit_death_cross_50_lt_200": "Risk: Death cross (SMA50 < SMA200)",
    "exit_rsi_overbought": "Risk: RSI14 overbought (> 80)",
    "rsi_bullish_divergence": "Bullish RSI divergence",
    "rsi_bearish_divergence": "Risk: Bearish RSI divergence",
    "weekly_macd_bullish_divergence": "Bullish Weekly MACD divergence (confirmed)",
    "weekly_macd_bearish_divergence": "Risk: Bearish Weekly MACD divergence (confirmed)",
    "strong_sell_weak_strength": "Strong Sell: Underperforming vs benchmark",
    "squat_ambush_near_ma100_or_ma200": "🟢 Buy-Zone Watch: 2%-3% above MA100/200",
    "squat_dca_below_ma100": "🔵 DCA Zone: Broke below MA100",
    "squat_last_stand_ma200": "⚠️ Critical Support (MA200): Within +1% / -2%",
    "squat_breakdown_below_ma200": "🚨 Breakdown (MA200): More than 2% below",
}

NEGATIVE_STOCK_TRIGGER_IDS: set[str] = {
    "exit_price_below_sma50",
    "exit_death_cross_50_lt_200",
    "exit_rsi_overbought",
    "rsi_bearish_divergence",
    "weekly_macd_bearish_divergence",
    "strong_sell_weak_strength",
    "squat_dca_below_ma100",
    "squat_last_stand_ma200",
    "squat_breakdown_below_ma200",
}
ERROR_DIAGNOSTIC_COLUMNS = ["error_type", "error_provider", "error_retryable"]
ERROR_TYPE_RATE_LIMITED = "rate_limited"
ERROR_TYPE_UPSTREAM_UNREACHABLE = "upstream_unreachable"
ERROR_TYPE_SYMBOL_NOT_FOUND = "symbol_not_found"
ERROR_TYPE_AUTH_ERROR = "auth_error"
ERROR_TYPE_UPSTREAM_API_ERROR = "upstream_api_error"
ERROR_TYPE_UPSTREAM_INVALID_RESPONSE = "upstream_invalid_response"
ERROR_TYPE_FETCH_ERROR = "fetch_error"
ERROR_TYPE_COMPUTE_ERROR = "compute_error"
STOCK_YF_PREFETCH_DEGRADED_COVERAGE = 0.35
STOCK_BOOL_COLUMNS = [
    "rs_structural_divergence",
    "rs_trend_down",
    "rs_negative_alpha",
    "relative_strength_weak",
    *STOCK_TRIGGER_COLUMNS,
]
EASTERN_TZ = ZoneInfo("America/New_York")
INTRADAY_SESSION_START_MINUTES_ET = 4 * 60
INTRADAY_SESSION_END_MINUTES_ET = 20 * 60


def _default_start_date(lookback_years: int) -> str:
    now_utc = datetime.now(timezone.utc).date()
    return (pd.Timestamp(now_utc) - pd.DateOffset(years=lookback_years)).date().isoformat()


def _attach_source(series: pd.Series, source: str) -> pd.Series:
    out = series.copy()
    out.attrs["source"] = source
    return out


def _clean_series(series: pd.Series) -> pd.Series:
    attrs = dict(series.attrs)
    clean = series.sort_index().dropna().astype(float)
    clean.attrs.update(attrs)
    return clean


def _series_to_csv_frame(series: pd.Series, source: str) -> pd.DataFrame:
    clean = series.dropna().sort_index()
    return pd.DataFrame(
        {
            "date": pd.to_datetime(clean.index).date,
            "value": clean.astype(float).values,
            "source": source,
        }
    )


def _latest_snapshot(metric_key: str, series: pd.Series, now_iso: str) -> dict[str, object]:
    clean = series.dropna().sort_index()
    if clean.empty:
        raise ValueError(f"No data available for metric {metric_key}.")
    as_of = pd.Timestamp(clean.index[-1])
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("UTC").tz_localize(None)
    stale_days = (datetime.now(timezone.utc).date() - as_of.date()).days
    source = str(series.attrs.get("source", "unknown"))
    return {
        "metric_key": metric_key,
        "as_of_date": as_of.date().isoformat(),
        "value": float(clean.iloc[-1]),
        "source": source,
        "stale_days": stale_days,
        "last_updated_utc": now_iso,
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _empty_signal_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_EVENT_COLUMNS)


def _format_et_timestamp_from_utc(raw_value: str) -> str:
    ts = pd.Timestamp(raw_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S %Z")


def _is_trading_or_extended_hours_et(now_utc: datetime | None = None) -> bool:
    ts_utc = now_utc or datetime.now(timezone.utc)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    else:
        ts_utc = ts_utc.astimezone(timezone.utc)

    ts_et = ts_utc.astimezone(EASTERN_TZ)
    if ts_et.isoweekday() > 5:
        return False

    total_minutes = ts_et.hour * 60 + ts_et.minute
    return INTRADAY_SESSION_START_MINUTES_ET <= total_minutes <= INTRADAY_SESSION_END_MINUTES_ET


def _event_cell_value(raw_value: Any) -> Any:
    if pd.isna(raw_value):
        return ""
    return raw_value


def _load_signal_event_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_signal_events_frame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read signal events at {path}: {exc}; reinitializing empty history.")
        return _empty_signal_events_frame()

    out = frame.copy()
    if "event_timestamp_et" not in out.columns and LEGACY_SIGNAL_EVENT_TIMESTAMP_COLUMN in out.columns:
        out["event_timestamp_et"] = out[LEGACY_SIGNAL_EVENT_TIMESTAMP_COLUMN]
    for col in SIGNAL_EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[SIGNAL_EVENT_COLUMNS]


def _prune_signal_event_history(frame: pd.DataFrame, now_iso: str) -> pd.DataFrame:
    if frame.empty:
        return _empty_signal_events_frame()

    out = frame.copy()
    for col in SIGNAL_EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[SIGNAL_EVENT_COLUMNS]

    event_ts = pd.to_datetime(out["event_timestamp_utc"], errors="coerce", utc=True)
    cutoff = pd.Timestamp(now_iso)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    cutoff = cutoff - pd.Timedelta(days=SIGNAL_EVENT_RETENTION_DAYS)

    keep_mask = event_ts >= cutoff
    kept = out.loc[keep_mask].copy()
    kept_ts = event_ts.loc[keep_mask]
    if kept.empty:
        return _empty_signal_events_frame()

    kept["event_timestamp_utc"] = kept_ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    kept["_sort_ts"] = kept_ts
    kept = kept.sort_values("_sort_ts").drop(columns=["_sort_ts"]).reset_index(drop=True)
    return kept[SIGNAL_EVENT_COLUMNS]


def _build_macro_signal_event_rows(events: list[dict[str, Any]], now_iso: str) -> list[dict[str, Any]]:
    event_rows: list[dict[str, Any]] = []
    event_timestamp_et = _format_et_timestamp_from_utc(now_iso)

    for event in events:
        metric_key = str(event.get("metric_key", "")).strip()
        if not metric_key:
            continue

        metric_name = str(event.get("metric_name", metric_key)).strip() or metric_key
        as_of_date = str(event.get("as_of_date", ""))
        state_transition = f"{event.get('prev_signal_state', 'unknown')} -> {event.get('signal_state', 'unknown')}"

        for event_type, id_key, label_key in (
            ("triggered", "new_threshold_ids", "new_threshold_labels"),
            ("cleared", "cleared_threshold_ids", "cleared_threshold_labels"),
        ):
            threshold_ids = list(event.get(id_key, []))
            threshold_labels = list(event.get(label_key, []))
            for idx, threshold_id in enumerate(threshold_ids):
                signal_id = str(threshold_id)
                signal_label = (
                    str(threshold_labels[idx]) if idx < len(threshold_labels) else THRESHOLD_LABELS.get(signal_id, signal_id)
                )
                event_rows.append(
                    {
                        "event_timestamp_utc": now_iso,
                        "event_timestamp_et": event_timestamp_et,
                        "domain": "macro",
                        "event_type": event_type,
                        "subject_id": metric_key,
                        "subject_name": metric_name,
                        "benchmark_ticker": "",
                        "signal_id": signal_id,
                        "signal_label": signal_label,
                        "as_of_date": as_of_date,
                        "value": _event_cell_value(event.get("value")),
                        "price": "",
                        "state_transition": state_transition,
                        "status": "",
                        "details": "",
                    }
                )
    return event_rows


def _build_stock_signal_event_rows(events: list[dict[str, Any]], now_iso: str) -> list[dict[str, Any]]:
    event_rows: list[dict[str, Any]] = []
    event_timestamp_et = _format_et_timestamp_from_utc(now_iso)

    for event in events:
        ticker = _normalize_ticker(event.get("ticker", ""))
        if not ticker:
            continue
        event_type = str(event.get("event_type", "triggered")).strip().lower()
        if event_type not in {"triggered", "cleared"}:
            event_type = "triggered"

        details = ""
        if event.get("trigger_id") == "strong_sell_weak_strength":
            raw_details = event.get("relative_strength_reasons")
            if raw_details is not None and not pd.isna(raw_details):
                details = str(raw_details)

        event_rows.append(
            {
                "event_timestamp_utc": now_iso,
                "event_timestamp_et": event_timestamp_et,
                "domain": "stock",
                "event_type": event_type,
                "subject_id": ticker,
                "subject_name": ticker,
                "benchmark_ticker": _normalize_benchmark(event.get("benchmark_ticker", DEFAULT_STOCK_BENCHMARK)),
                "signal_id": str(event.get("trigger_id", "")),
                "signal_label": str(event.get("trigger_label", "")),
                "as_of_date": str(event.get("as_of_date", "")),
                "value": "",
                "price": _event_cell_value(event.get("price")),
                "state_transition": "",
                "status": str(event.get("status", "")),
                "details": details,
            }
        )
    return event_rows


def _update_signal_event_history(
    path: Path,
    *,
    macro_events: list[dict[str, Any]],
    stock_events: list[dict[str, Any]],
    now_iso: str,
) -> None:
    existing = _load_signal_event_history(path)
    new_rows = [
        *_build_macro_signal_event_rows(macro_events, now_iso),
        *_build_stock_signal_event_rows(stock_events, now_iso),
    ]
    if new_rows:
        new_frame = pd.DataFrame(new_rows, columns=SIGNAL_EVENT_COLUMNS)
        combined = new_frame if existing.empty else pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = existing.copy()

    pruned = _prune_signal_event_history(combined, now_iso=now_iso)
    _write_csv(path, pruned)


def _normalize_ticker(raw_value: Any) -> str:
    if pd.isna(raw_value):
        return ""
    return str(raw_value).strip().upper()


def _normalize_benchmark(raw_value: Any) -> str:
    clean = _normalize_ticker(raw_value)
    return clean or DEFAULT_STOCK_BENCHMARK


def _normalize_watchlist_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen_tickers: set[str] = set()
    for row in rows:
        ticker = _normalize_ticker(row.get("ticker", ""))
        benchmark = _normalize_benchmark(row.get("benchmark", DEFAULT_STOCK_BENCHMARK))
        if not ticker or ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        normalized.append({"ticker": ticker, "benchmark": benchmark})
    return normalized


def _normalize_optional_text(raw_value: Any) -> str:
    if pd.isna(raw_value):
        return ""
    text = str(raw_value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _write_watchlist(path: Path, rows: list[dict[str, Any]]) -> None:
    normalized = _normalize_watchlist_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(normalized, columns=["ticker", "benchmark"]).to_csv(path, index=False)


def _load_or_initialize_watchlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker", "benchmark"])

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read watchlist at {path}: {exc}; reinitializing defaults.")
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker", "benchmark"])

    if "ticker" not in frame.columns:
        print(f"Warning: watchlist at {path} has no 'ticker' column; reinitializing defaults.")
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker", "benchmark"])

    raw_rows = frame.to_dict(orient="records")
    for row in raw_rows:
        row.setdefault("benchmark", DEFAULT_STOCK_BENCHMARK)

    normalized_rows = _normalize_watchlist_rows(raw_rows)
    if not normalized_rows:
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker", "benchmark"])

    normalized_frame = pd.DataFrame(normalized_rows, columns=["ticker", "benchmark"])
    current_subset = frame.copy()
    missing_benchmark_column = "benchmark" not in current_subset.columns
    if missing_benchmark_column:
        current_subset["benchmark"] = DEFAULT_STOCK_BENCHMARK
    current_subset = current_subset[["ticker", "benchmark"]]
    current_subset["ticker"] = current_subset["ticker"].map(_normalize_ticker)
    current_subset["benchmark"] = current_subset["benchmark"].map(_normalize_benchmark)

    if missing_benchmark_column or not normalized_frame.equals(current_subset.reset_index(drop=True)):
        _write_watchlist(path, normalized_rows)

    return normalized_frame


def _env_int(name: str, default_value: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        parsed = int(raw_value)
    except ValueError:
        print(f"Warning: invalid integer for {name}={raw_value!r}; using default {default_value}.")
        return default_value
    if parsed < minimum:
        return default_value
    return parsed


def _env_float(name: str, default_value: float, *, minimum: float = 0.0) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    try:
        parsed = float(raw_value)
    except ValueError:
        print(f"Warning: invalid float for {name}={raw_value!r}; using default {default_value}.")
        return default_value
    if parsed < minimum:
        return default_value
    return parsed


def _env_optional_int(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        print(f"Warning: invalid integer for {name}={raw_value!r}; ignoring.")
        return None


def _env_bool(name: str, default_value: bool) -> bool:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default_value
    return raw_value.lower() in {"1", "true", "yes", "y", "on"}


def _coerce_bool_flag(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if pd.isna(raw_value):
        return False
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _classify_fetch_exception(exc: Exception) -> tuple[str, str, bool]:
    if isinstance(exc, MarketDataError):
        return (str(exc.error_type), str(exc.provider), bool(exc.retryable))

    message = str(exc)
    normalized = message.lower()
    provider = "unknown"
    if "schwab" in normalized or "schwabapi.com" in normalized:
        provider = "schwab"
    elif "yfratelimiterror" in normalized or "yfinance" in normalized or "yahoo" in normalized:
        provider = "yfinance"
    elif "stooq" in normalized:
        provider = "stooq"

    if "too many requests" in normalized or "rate limit" in normalized or "yfratelimiterror" in normalized or " 429" in normalized:
        return (ERROR_TYPE_RATE_LIMITED, provider, True)
    if "invalid access token" in normalized or "auth rejected" in normalized or "unauthorized" in normalized:
        return (ERROR_TYPE_AUTH_ERROR, provider, False)
    if (
        "failed to establish a new connection" in normalized
        or "max retries exceeded" in normalized
        or "connection refused" in normalized
        or "timed out" in normalized
        or "temporary failure" in normalized
        or "name resolution" in normalized
        or "connection reset" in normalized
    ):
        return (ERROR_TYPE_UPSTREAM_UNREACHABLE, provider, True)
    if (
        "symbol not found" in normalized
        or "no data returned" in normalized
        or "no daily" in normalized
        or "not found" in normalized
    ):
        return (ERROR_TYPE_SYMBOL_NOT_FOUND, provider, False)
    return (ERROR_TYPE_FETCH_ERROR, provider, False)


def _default_error_diagnostics() -> dict[str, Any]:
    return {
        "error_type": "",
        "error_provider": "",
        "error_retryable": False,
    }


def _apply_error_diagnostics(
    row: dict[str, Any],
    *,
    error_type: str,
    error_provider: str,
    error_retryable: bool,
) -> dict[str, Any]:
    row["error_type"] = str(error_type).strip()
    row["error_provider"] = str(error_provider).strip()
    row["error_retryable"] = bool(error_retryable)
    return row


def _is_rate_limited_error_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("status", "")).strip() == "fetch_error"
        and str(row.get("error_type", "")).strip() == ERROR_TYPE_RATE_LIMITED
    )


def _enforce_error_budget(frame: pd.DataFrame, *, name: str, max_error_ratio: float | None) -> None:
    if max_error_ratio is None:
        return
    if frame.empty or "status" not in frame.columns:
        return
    threshold = min(1.0, max(0.0, float(max_error_ratio)))
    status_series = frame["status"].fillna("").astype(str)
    error_mask = status_series.isin({"fetch_error", "compute_error"})
    error_count = int(error_mask.sum())
    ratio = float(error_count) / float(len(frame)) if len(frame) else 0.0
    if ratio > threshold:
        retryable_error_count = 0
        if "error_retryable" in frame.columns:
            retryable_series = frame["error_retryable"].map(_coerce_bool_flag)
            retryable_error_count = int((error_mask & retryable_series).sum())
        raise RuntimeError(
            f"{name} error ratio exceeded threshold: errors={error_count}/{len(frame)} "
            f"({ratio:.1%}) > allowed {threshold:.1%} (retryable_errors={retryable_error_count})"
        )


def _status_ratio(frame: pd.DataFrame, *, status_value: str) -> tuple[int, int, float]:
    total = int(len(frame))
    if total <= 0 or "status" not in frame.columns:
        return (0, total, 0.0)
    normalized = frame["status"].fillna("").astype(str).str.strip().str.lower()
    target = str(status_value).strip().lower()
    count = int((normalized == target).sum())
    ratio = (float(count) / float(total)) if total else 0.0
    return (count, total, ratio)


def _hard_error_row_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "error_type" in frame.columns:
        error_type = frame["error_type"].fillna("").astype(str).str.strip()
    else:
        error_type = pd.Series("", index=frame.index, dtype="string")
    if "error_provider" in frame.columns:
        error_provider = frame["error_provider"].fillna("").astype(str).str.strip()
    else:
        error_provider = pd.Series("", index=frame.index, dtype="string")
    return int(((error_type != "") | (error_provider != "")).sum())


def _empty_stock_signal_row(
    ticker: str,
    benchmark_ticker: str,
    now_iso: str,
    status: str,
    status_message: str,
    error_type: str = "",
    error_provider: str = "",
    error_retryable: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {column: pd.NA for column in [*STOCK_SIGNAL_COLUMNS, *ERROR_DIAGNOSTIC_COLUMNS]}
    row.update(
        {
            "ticker": ticker,
            "benchmark_ticker": benchmark_ticker,
            "as_of_date": "",
            "source": f"STOOQ:{ticker}",
            "status": status,
            "status_message": status_message,
            "last_updated_utc": now_iso,
            "relative_strength_reasons": "not_available",
        }
    )
    _apply_error_diagnostics(
        row,
        error_type=error_type,
        error_provider=error_provider,
        error_retryable=error_retryable,
    )

    for trigger_col in STOCK_BOOL_COLUMNS:
        row[trigger_col] = False
    return row


def _compute_watchlist_signals(
    watchlist: pd.DataFrame,
    start_date: str,
    now_iso: str,
    *,
    previous_stock_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    del previous_stock_signals
    if watchlist.empty:
        return pd.DataFrame(columns=[*STOCK_SIGNAL_COLUMNS, *ERROR_DIAGNOSTIC_COLUMNS, "last_updated_utc"])

    prefetch_tickers = sorted(
        {
            *watchlist["ticker"].map(_normalize_ticker).tolist(),
            *watchlist["benchmark"].map(_normalize_benchmark).tolist(),
        }
    )
    watchlist_circuit_prefetch_max_coverage = _env_float(
        "WATCHLIST_CIRCUIT_PREFETCH_MAX_COVERAGE",
        WATCHLIST_CIRCUIT_PREFETCH_MAX_COVERAGE_DEFAULT,
        minimum=0.0,
    )
    watchlist_circuit_probe_count = _env_int(
        "WATCHLIST_CIRCUIT_PROBE_COUNT",
        WATCHLIST_CIRCUIT_PROBE_COUNT_DEFAULT,
        minimum=1,
    )
    yfinance_daily_prefetch = fetch_stock_daily_history_batch_yfinance(prefetch_tickers, start_date)
    prefetch_total = len(prefetch_tickers)
    prefetch_hits = len(yfinance_daily_prefetch)
    prefetch_coverage = (float(prefetch_hits) / float(prefetch_total)) if prefetch_total else 1.0
    degraded_watchlist_prefetch = prefetch_total > 0 and prefetch_coverage < STOCK_YF_PREFETCH_DEGRADED_COVERAGE
    if degraded_watchlist_prefetch:
        print(
            "Warning: watchlist provider prefetch coverage is degraded "
            f"({prefetch_coverage:.1%}); proceeding with direct fetches and surfacing fetch errors."
        )

    def _get_watchlist_daily_history(ticker: str) -> pd.Series:
        prefetched = yfinance_daily_prefetch.get(ticker)
        if prefetched is not None:
            return prefetched.copy()
        return fetch_stock_daily_history(ticker, start_date)

    benchmark_history: dict[str, pd.Series | None] = {}
    benchmark_intraday: dict[str, dict[str, Any] | None] = {}
    batched_intraday_quotes: dict[str, dict[str, Any]] = {}
    intraday_session_active = _is_trading_or_extended_hours_et()
    if intraday_session_active:
        try:
            batched_intraday_quotes = fetch_stock_intraday_quotes_batch(prefetch_tickers)
        except Exception as exc:
            print(f"Warning: failed to fetch batched intraday quotes: {exc}")
            batched_intraday_quotes = {}
    else:
        print(
            "Info: outside ET trading/extended-hours window (04:00-20:00); "
            "skipping intraday quote fetch and using daily-close signal basis."
        )

    for benchmark in sorted(watchlist["benchmark"].astype(str).unique()):
        benchmark_symbol = _normalize_benchmark(benchmark)
        try:
            benchmark_history[benchmark_symbol] = _get_watchlist_daily_history(benchmark_symbol)
        except Exception as exc:
            print(f"Warning: failed to fetch benchmark history for {benchmark_symbol}: {exc}")
            benchmark_history[benchmark_symbol] = None

        benchmark_quote: dict[str, Any] | None = None
        if intraday_session_active:
            benchmark_quote = batched_intraday_quotes.get(benchmark_symbol)
            if benchmark_quote is None:
                try:
                    benchmark_quote = fetch_stock_intraday_quote(benchmark_symbol)
                except Exception as exc:
                    print(f"Warning: failed to fetch benchmark intraday quote for {benchmark_symbol}: {exc}")
                    benchmark_quote = None
        benchmark_intraday[benchmark_symbol] = benchmark_quote

    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    probe_outcomes: list[bool] = []
    circuit_triggered = False
    circuit_provider = "unknown"
    for _, watch_row in watchlist.iterrows():
        ticker = _normalize_ticker(watch_row.get("ticker", ""))
        benchmark_ticker = _normalize_benchmark(watch_row.get("benchmark", DEFAULT_STOCK_BENCHMARK))
        if not ticker:
            continue

        if circuit_triggered:
            rows.append(
                _empty_stock_signal_row(
                    ticker=ticker,
                    benchmark_ticker=benchmark_ticker,
                    now_iso=now_iso,
                    status="fetch_error",
                    status_message=(
                        "Circuit breaker: prefetch coverage was near-zero and initial direct fetches were "
                        "rate-limited; skipped remaining watchlist symbols."
                    ),
                    error_type=ERROR_TYPE_RATE_LIMITED,
                    error_provider=circuit_provider,
                    error_retryable=True,
                )
            )
            continue

        try:
            daily_close = _get_watchlist_daily_history(ticker)
        except Exception as exc:
            error_type, error_provider, error_retryable = _classify_fetch_exception(exc)
            fetch_error_row = _empty_stock_signal_row(
                ticker=ticker,
                benchmark_ticker=benchmark_ticker,
                now_iso=now_iso,
                status="fetch_error",
                status_message=f"Failed to fetch daily history: {exc}",
                error_type=error_type,
                error_provider=error_provider,
                error_retryable=error_retryable,
            )
            rows.append(fetch_error_row)
            if (
                prefetch_total > 0
                and prefetch_coverage <= watchlist_circuit_prefetch_max_coverage
                and len(probe_outcomes) < watchlist_circuit_probe_count
            ):
                probe_rows.append(fetch_error_row)
                probe_outcomes.append(_is_rate_limited_error_row(fetch_error_row))
                if (
                    len(probe_outcomes) >= watchlist_circuit_probe_count
                    and all(probe_outcomes)
                ):
                    providers = pd.Series(
                        [
                            str(probe_row.get("error_provider", "")).strip()
                            for probe_row in probe_rows
                            if str(probe_row.get("error_provider", "")).strip()
                        ]
                    )
                    circuit_provider = str(providers.value_counts().index[0]) if not providers.empty else "unknown"
                    circuit_triggered = True
                    print(
                        "Warning: watchlist circuit breaker triggered after repeated rate-limited fetch errors; "
                        f"remaining symbols will be marked fetch_error (provider={circuit_provider})."
                    )
            continue

        latest_quote: dict[str, Any] | None = None
        latest_price: float | None = None
        if intraday_session_active:
            latest_quote = batched_intraday_quotes.get(ticker)
            if latest_quote is None:
                try:
                    latest_quote = fetch_stock_intraday_quote(ticker)
                except Exception as exc:
                    print(f"Warning: failed to fetch intraday quote for {ticker}: {exc}")
                    latest_quote = None
            if latest_quote is not None:
                try:
                    latest_price = float(latest_quote["price"])
                except Exception:
                    latest_price = None
                quote_source = str(latest_quote.get("source", "")).strip()
                if quote_source:
                    daily_close.attrs["intraday_source"] = quote_source

        try:
            benchmark_close = benchmark_history.get(benchmark_ticker)
            benchmark_quote = benchmark_intraday.get(benchmark_ticker)
            benchmark_price = float(benchmark_quote["price"]) if benchmark_quote else None
            row = compute_stock_signal_row(
                ticker=ticker,
                daily_close=daily_close,
                latest_price=latest_price,
                benchmark_ticker=benchmark_ticker,
                benchmark_close=benchmark_close,
                benchmark_latest_price=benchmark_price,
            )
            if latest_quote is not None:
                row["intraday_quote_timestamp_utc"] = latest_quote.get("quote_timestamp_utc")
                row["intraday_quote_age_seconds"] = latest_quote.get("quote_age_seconds")
                row["intraday_quote_source"] = latest_quote.get("source")
                quote_day_change = latest_quote.get("day_change")
                if quote_day_change is not None and pd.notna(quote_day_change):
                    row["day_change"] = float(quote_day_change)
                quote_day_change_pct = latest_quote.get("day_change_pct")
                if quote_day_change_pct is not None and pd.notna(quote_day_change_pct):
                    row["day_change_pct"] = float(quote_day_change_pct)
            row["last_updated_utc"] = now_iso
            if str(row.get("status", "")).strip() == "fetch_error":
                _apply_error_diagnostics(
                    row,
                    error_type=ERROR_TYPE_FETCH_ERROR,
                    error_provider="unknown",
                    error_retryable=False,
                )
            elif str(row.get("status", "")).strip() == "compute_error":
                _apply_error_diagnostics(
                    row,
                    error_type=ERROR_TYPE_COMPUTE_ERROR,
                    error_provider="pipeline",
                    error_retryable=False,
                )
            else:
                _apply_error_diagnostics(row, **_default_error_diagnostics())
            rows.append(row)
            if (
                prefetch_total > 0
                and prefetch_coverage <= watchlist_circuit_prefetch_max_coverage
                and len(probe_outcomes) < watchlist_circuit_probe_count
            ):
                probe_outcomes.append(False)
        except Exception as exc:
            error_type, error_provider, error_retryable = _classify_fetch_exception(exc)
            rows.append(
                _empty_stock_signal_row(
                    ticker=ticker,
                    benchmark_ticker=benchmark_ticker,
                    now_iso=now_iso,
                    status="compute_error",
                    status_message=f"Failed to compute stock signals: {exc}",
                    error_type=ERROR_TYPE_COMPUTE_ERROR if error_type == ERROR_TYPE_FETCH_ERROR else error_type,
                    error_provider=error_provider or "pipeline",
                    error_retryable=error_retryable,
                )
            )
            if (
                prefetch_total > 0
                and prefetch_coverage <= watchlist_circuit_prefetch_max_coverage
                and len(probe_outcomes) < watchlist_circuit_probe_count
            ):
                probe_outcomes.append(False)

    frame = pd.DataFrame(rows)
    expected_cols = [*STOCK_SIGNAL_COLUMNS, *ERROR_DIAGNOSTIC_COLUMNS, "last_updated_utc"]
    for col in expected_cols:
        if col not in frame.columns:
            frame[col] = pd.NA
    if "error_retryable" in frame.columns:
        frame["error_retryable"] = frame["error_retryable"].map(_coerce_bool_flag)
    if "error_type" in frame.columns:
        frame["error_type"] = frame["error_type"].fillna("").astype(str)
    if "error_provider" in frame.columns:
        frame["error_provider"] = frame["error_provider"].fillna("").astype(str)
    return frame[expected_cols].sort_values("ticker").reset_index(drop=True)


def _load_previous_stock_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read previous stock signals at {path}: {exc}")
        return pd.DataFrame()

    required = {"ticker", *STOCK_TRIGGER_COLUMNS}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame


def _load_previous_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read previous signals at {path}: {exc}")
        return pd.DataFrame()

    required = {"metric_key", "signal_state"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame


def _thresholds_for_state(metric_key: str, signal_state: str) -> set[str]:
    metric_map = THRESHOLD_TRIGGER_MAP.get(metric_key, {})
    return set(metric_map.get(signal_state, set()))


def _value_for_message(raw_value: Any) -> str:
    if pd.isna(raw_value):
        return "n/a"
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return str(raw_value)
    return f"{numeric:.4f}"


def _detect_new_threshold_events(previous: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, Any]]:
    if previous.empty or current.empty:
        return []
    if not {"metric_key", "signal_state"}.issubset(current.columns):
        return []

    prev_thresholds: dict[str, set[str]] = {}
    prev_states: dict[str, str] = {}
    for _, row in previous.iterrows():
        metric_key = str(row.get("metric_key", ""))
        signal_state = str(row.get("signal_state", ""))
        if not metric_key:
            continue
        prev_thresholds[metric_key] = _thresholds_for_state(metric_key, signal_state)
        prev_states[metric_key] = signal_state

    events: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        metric_key = str(row.get("metric_key", ""))
        signal_state = str(row.get("signal_state", ""))
        if not metric_key or metric_key not in prev_states:
            continue

        current_thresholds = _thresholds_for_state(metric_key, signal_state)
        previous_metric_thresholds = prev_thresholds.get(metric_key, set())
        newly_triggered = sorted(current_thresholds - previous_metric_thresholds)
        newly_cleared_negative = sorted(
            (previous_metric_thresholds - current_thresholds) & NEGATIVE_MACRO_THRESHOLD_IDS
        )
        if not newly_triggered and not newly_cleared_negative:
            continue

        events.append(
            {
                "metric_key": metric_key,
                "metric_name": str(row.get("metric_name", metric_key)),
                "as_of_date": str(row.get("as_of_date", "unknown")),
                "value": row.get("value"),
                "signal_state": signal_state,
                "prev_signal_state": prev_states.get(metric_key, "unknown"),
                "new_threshold_ids": newly_triggered,
                "new_threshold_labels": [THRESHOLD_LABELS.get(tid, tid) for tid in newly_triggered],
                "cleared_threshold_ids": newly_cleared_negative,
                "cleared_threshold_labels": [THRESHOLD_LABELS.get(tid, tid) for tid in newly_cleared_negative],
            }
        )

    return events


def _as_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if pd.isna(raw_value):
        return False
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    normalized = str(raw_value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _detect_new_stock_trigger_events(previous: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, Any]]:
    if current.empty:
        return []
    if not {"ticker", *STOCK_TRIGGER_COLUMNS}.issubset(current.columns):
        return []

    previous_map: dict[tuple[str, str], bool] = {}
    if not previous.empty and {"ticker", *STOCK_TRIGGER_COLUMNS}.issubset(previous.columns):
        for _, row in previous.iterrows():
            ticker = _normalize_ticker(row.get("ticker", ""))
            if not ticker:
                continue
            for trigger_id in STOCK_TRIGGER_COLUMNS:
                previous_map[(ticker, trigger_id)] = _as_bool(row.get(trigger_id))

    events: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        ticker = _normalize_ticker(row.get("ticker", ""))
        if not ticker:
            continue
        is_benchmark_ticker = ticker == DEFAULT_STOCK_BENCHMARK

        for trigger_id in STOCK_TRIGGER_COLUMNS:
            if is_benchmark_ticker and trigger_id in BENCHMARK_RELATED_TRIGGER_COLUMNS:
                continue
            current_state = _as_bool(row.get(trigger_id))
            previous_state = previous_map.get((ticker, trigger_id), False)
            event_type: str | None = None
            if current_state and not previous_state:
                event_type = "triggered"
            elif previous_state and not current_state and trigger_id in NEGATIVE_STOCK_TRIGGER_IDS:
                event_type = "cleared"
            if event_type is None:
                continue

            raw_reasons = row.get("relative_strength_reasons", "")
            events.append(
                {
                    "ticker": ticker,
                    "benchmark_ticker": _normalize_benchmark(row.get("benchmark_ticker", DEFAULT_STOCK_BENCHMARK)),
                    "trigger_id": trigger_id,
                    "trigger_label": STOCK_TRIGGER_LABELS.get(trigger_id, trigger_id),
                    "event_type": event_type,
                    "as_of_date": str(row.get("as_of_date", "unknown")),
                    "price": row.get("price"),
                    "status": str(row.get("status", "")),
                    "relative_strength_reasons": "" if pd.isna(raw_reasons) else str(raw_reasons),
                }
            )

    return events


def _build_discord_message(events: list[dict[str, Any]], now_iso: str) -> str:
    lines = [f"Macro threshold state alert ({now_iso})"]
    for event in events:
        changes: list[str] = []
        if event["new_threshold_labels"]:
            changes.append(f"triggered: {', '.join(event['new_threshold_labels'])}")
        if event["cleared_threshold_labels"]:
            changes.append(f"cleared: {', '.join(event['cleared_threshold_labels'])}")
        threshold_changes = "; ".join(changes)
        lines.append(
            (
                f"- {event['metric_name']} ({event['metric_key']}): {threshold_changes}; "
                f"state {event['prev_signal_state']} -> {event['signal_state']}; "
                f"value={_value_for_message(event['value'])}; as_of={event['as_of_date']}"
            )
        )
    return "\n".join(lines)


def _build_stock_discord_message(events: list[dict[str, Any]], now_iso: str) -> str:
    lines = [f"Stock watchlist trigger alert ({now_iso})"]
    for event in events:
        reason_suffix = ""
        if (
            event.get("event_type") == "triggered"
            and event.get("trigger_id") == "strong_sell_weak_strength"
            and event.get("relative_strength_reasons")
        ):
            reason_suffix = f"; rs_reasons={event['relative_strength_reasons']}"
        action = "triggered" if event.get("event_type") == "triggered" else "cleared"
        lines.append(
            (
                f"- {event['ticker']} vs {event['benchmark_ticker']}: {event['trigger_label']} {action}; "
                f"price={_value_for_message(event['price'])}; "
                f"as_of={event['as_of_date']}; status={event['status']}{reason_suffix}"
            )
        )
    return "\n".join(lines)


def _post_discord_message(webhook_url: str, content: str) -> None:
    payload = json.dumps({"content": content}).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "BingBingBot/1.0 (GitHub Actions)",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10):
            return
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        if detail:
            raise RuntimeError(f"Discord webhook request failed: HTTP {exc.code}; body={detail}") from exc
        raise RuntimeError(f"Discord webhook request failed: HTTP {exc.code}") from exc


def _notify_threshold_events(events: list[dict[str, Any]], now_iso: str) -> None:
    if not events:
        return

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("Warning: DISCORD_WEBHOOK_URL is not set; skipping threshold alert notification.")
        return

    message = _build_discord_message(events, now_iso)
    try:
        _post_discord_message(webhook_url, message)
    except Exception as exc:
        print(f"Warning: failed to send Discord alert: {exc}")


def _notify_new_thresholds(previous: pd.DataFrame, current: pd.DataFrame, now_iso: str) -> None:
    events = _detect_new_threshold_events(previous, current)
    _notify_threshold_events(events, now_iso)


def _notify_stock_trigger_events(events: list[dict[str, Any]], now_iso: str) -> None:
    if not events:
        return

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("Warning: DISCORD_WEBHOOK_URL is not set; skipping stock trigger alert notification.")
        return

    message = _build_stock_discord_message(events, now_iso)
    try:
        _post_discord_message(webhook_url, message)
    except Exception as exc:
        print(f"Warning: failed to send stock Discord alert: {exc}")


def _notify_new_stock_triggers(previous: pd.DataFrame, current: pd.DataFrame, now_iso: str) -> None:
    events = _detect_new_stock_trigger_events(previous, current)
    _notify_stock_trigger_events(events, now_iso)


def _run_macro_pipeline(start_date: str, now_iso: str) -> None:
    previous_signals = _load_previous_signals(DERIVED_DATA_DIR / "signals_latest.csv")

    m2_cfg = SERIES_CONFIG["m2"]
    hiring_cfg = SERIES_CONFIG["hiring_rate"]
    ten_cfg = SERIES_CONFIG["ten_year_yield"]
    buffett_cfg = SERIES_CONFIG["buffett_ratio"]
    unrate_cfg = SERIES_CONFIG["unemployment_rate"]

    m2 = _attach_source(fetch_fred_series(m2_cfg["series_id"], start_date), "FRED:M2SL")
    hiring = _attach_source(fetch_fred_series(hiring_cfg["series_id"], start_date), "FRED:JTSHIR")
    unrate = _attach_source(fetch_fred_series(unrate_cfg["series_id"], start_date), "FRED:UNRATE")
    market_cap_id = buffett_cfg["market_cap_series_id"]
    gdp_id = buffett_cfg["gdp_series_id"]
    market_cap_source = f"FRED:{market_cap_id}"
    market_cap_unit_divisor = float(buffett_cfg.get("market_cap_unit_divisor", 1.0))
    try:
        market_cap = fetch_fred_series(market_cap_id, start_date)
    except Exception:
        fallback_market_cap_id = buffett_cfg.get("fallback_market_cap_series_id")
        if not fallback_market_cap_id:
            raise
        market_cap = fetch_fred_series(fallback_market_cap_id, start_date)
        market_cap_source = f"FRED:{fallback_market_cap_id} (fallback for {market_cap_id})"
    gdp = fetch_fred_series(gdp_id, start_date)
    market_cap = _attach_source(market_cap, market_cap_source)
    gdp = _attach_source(gdp, f"FRED:{gdp_id}")

    ten_year = _attach_source(fetch_fred_series(ten_cfg["series_id"], start_date), "FRED:DGS10")

    buffett_ratio = build_buffett_ratio(
        market_cap,
        gdp,
        market_cap_unit_divisor=market_cap_unit_divisor,
    )
    buffett_ratio = _attach_source(buffett_ratio, f"{market_cap_source}/FRED:{gdp_id}")

    metric_series = {
        "m2": _clean_series(prepare_monthly_series(m2)),
        "hiring_rate": _clean_series(prepare_monthly_series(hiring)),
        "ten_year_yield": _clean_series(ten_year),
        "buffett_ratio": _clean_series(buffett_ratio),
        "unemployment_rate": _clean_series(prepare_monthly_series(unrate)),
    }

    raw_frames = {
        key: _series_to_csv_frame(series, str(series.attrs.get("source", "unknown")))
        for key, series in metric_series.items()
    }

    signals = compute_signals(metric_series)
    signals["last_updated_utc"] = now_iso

    snapshot_rows = [_latest_snapshot(key, metric_series[key], now_iso) for key in METRIC_ORDER if key in metric_series]
    metric_snapshot = pd.DataFrame(snapshot_rows)

    for metric_key, frame in raw_frames.items():
        _write_csv(RAW_DATA_DIR / f"{metric_key}.csv", frame)

    _write_csv(DERIVED_DATA_DIR / "metric_snapshot.csv", metric_snapshot)
    _write_csv(DERIVED_DATA_DIR / "signals_latest.csv", signals)

    macro_events = _detect_new_threshold_events(previous_signals, signals)
    _update_signal_event_history(
        SIGNAL_EVENTS_PATH,
        macro_events=macro_events,
        stock_events=[],
        now_iso=now_iso,
    )
    _notify_threshold_events(macro_events, now_iso)


def _run_stock_pipeline(
    start_date: str,
    now_iso: str,
) -> None:
    watchlist = _load_or_initialize_watchlist(STOCK_WATCHLIST_PATH)
    configured_stock_fail_max_error_ratio = _env_float(
        "STOCK_FAIL_MAX_ERROR_RATIO",
        STOCK_FAIL_MAX_ERROR_RATIO_DEFAULT,
        minimum=0.0,
    )
    previous_stock_signals = _load_previous_stock_signals(STOCK_SIGNALS_PATH)
    stock_signals = _compute_watchlist_signals(
        watchlist=watchlist,
        start_date=start_date,
        now_iso=now_iso,
        previous_stock_signals=previous_stock_signals,
    )
    _write_csv(STOCK_SIGNALS_PATH, stock_signals)
    _enforce_error_budget(
        stock_signals,
        name="watchlist",
        max_error_ratio=configured_stock_fail_max_error_ratio,
    )
    stock_events = _detect_new_stock_trigger_events(previous_stock_signals, stock_signals)
    _update_signal_event_history(
        SIGNAL_EVENTS_PATH,
        macro_events=[],
        stock_events=stock_events,
        now_iso=now_iso,
    )
    _notify_stock_trigger_events(stock_events, now_iso)


def run_pipeline(
    start_date: str | None = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    run_macro: bool = True,
    run_stock: bool = True,
) -> None:
    """Run pipeline sections and persist CSV artifacts."""
    if not run_macro and not run_stock:
        raise ValueError("At least one pipeline section must be enabled.")

    if not start_date:
        start_date = _default_start_date(lookback_years)

    now_iso = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if run_macro:
        _run_macro_pipeline(start_date=start_date, now_iso=now_iso)
    if run_stock:
        _run_stock_pipeline(
            start_date=start_date,
            now_iso=now_iso,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run macro signal dashboard data pipeline.")
    parser.add_argument("--start-date", type=str, default=None, help="ISO date, e.g. 2011-01-01")
    parser.add_argument("--lookback-years", type=int, default=DEFAULT_LOOKBACK_YEARS, help="Years of history to pull")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--macro-only", action="store_true", help="Run only macro data fetch/signal steps")
    mode_group.add_argument("--stock-only", action="store_true", help="Run only stock watchlist steps")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_macro = not args.stock_only
    run_stock = not args.macro_only
    run_pipeline(
        start_date=args.start_date,
        lookback_years=args.lookback_years,
        run_macro=run_macro,
        run_stock=run_stock,
    )


if __name__ == "__main__":
    main()

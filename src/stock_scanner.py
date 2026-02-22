"""End-of-day scanner logic for the broad stock universe."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.stock_signals import compute_sma

SCANNER_SIGNAL_COLUMNS = [
    "ticker",
    "security_name",
    "exchange",
    "universe_source",
    "is_watchlist",
    "is_etf",
    "as_of_date",
    "price",
    "open",
    "sma14",
    "sma50",
    "sma100",
    "sma200",
    "gap_to_sma100_pct",
    "gap_to_sma200_pct",
    "bullish_alignment_active",
    "bullish_alignment_triggered_today",
    "recovery_close_cross_sma50_today",
    "recovery_three_bullish_candles_today",
    "recovery_momentum_triggered_today",
    "ambush_trend_bullish_active",
    "ambush_near_ma100_active",
    "ambush_near_ma200_active",
    "ambush_squat_active",
    "ambush_squat_triggered_today",
    "source",
    "stale_days",
    "status",
    "status_message",
]

_SCANNER_BOOL_COLUMNS = [
    "bullish_alignment_active",
    "bullish_alignment_triggered_today",
    "recovery_close_cross_sma50_today",
    "recovery_three_bullish_candles_today",
    "recovery_momentum_triggered_today",
    "ambush_trend_bullish_active",
    "ambush_near_ma100_active",
    "ambush_near_ma200_active",
    "ambush_squat_active",
    "ambush_squat_triggered_today",
]


def _is_valid_number(value: Any) -> bool:
    return pd.notna(value)


def _safe_gt(lhs: Any, rhs: Any) -> bool:
    return _is_valid_number(lhs) and _is_valid_number(rhs) and float(lhs) > float(rhs)


def _safe_lte(lhs: Any, rhs: Any) -> bool:
    return _is_valid_number(lhs) and _is_valid_number(rhs) and float(lhs) <= float(rhs)


def _gap_to_ma_pct(price: Any, ma_value: Any) -> float:
    if not _is_valid_number(price) or not _is_valid_number(ma_value):
        return float("nan")
    if float(ma_value) == 0:
        return float("nan")
    return (float(price) - float(ma_value)) / float(ma_value)


def _in_ambush_band(gap_pct: Any) -> bool:
    return _is_valid_number(gap_pct) and 0.0 <= float(gap_pct) <= 0.02


def _normalize_scanner_bars(daily_bars: pd.DataFrame) -> pd.DataFrame:
    if daily_bars is None or not isinstance(daily_bars, pd.DataFrame):
        return pd.DataFrame()
    if daily_bars.empty:
        return pd.DataFrame()

    out = daily_bars.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
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

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")

    return out


def _alignment_active(ma14: Any, ma50: Any, ma100: Any, ma200: Any) -> bool:
    return _safe_gt(ma14, ma50) and (_safe_gt(ma50, ma100) or _safe_gt(ma50, ma200))


def _three_bullish_candles(open_series: pd.Series, close_series: pd.Series) -> bool:
    if len(open_series) < 3 or len(close_series) < 3:
        return False
    recent_open = open_series.iloc[-3:]
    recent_close = close_series.iloc[-3:]
    if recent_open.isna().any() or recent_close.isna().any():
        return False
    return bool((recent_close > recent_open).all())


def _build_empty_row(
    *,
    ticker: str,
    security_name: str,
    exchange: str,
    universe_source: str,
    is_watchlist: bool,
    is_etf: bool,
    status: str,
    status_message: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {column: pd.NA for column in SCANNER_SIGNAL_COLUMNS}
    row.update(
        {
            "ticker": ticker,
            "security_name": security_name,
            "exchange": exchange,
            "universe_source": universe_source,
            "is_watchlist": bool(is_watchlist),
            "is_etf": bool(is_etf),
            "status": status,
            "status_message": status_message,
            "source": f"STOCK:{ticker}",
        }
    )
    for col in _SCANNER_BOOL_COLUMNS:
        row[col] = False
    return row


def compute_scanner_signal_row(
    ticker: str,
    daily_bars: pd.DataFrame,
    *,
    is_watchlist: bool = False,
    security_name: str = "",
    exchange: str = "",
    universe_source: str = "",
    is_etf: bool = False,
) -> dict[str, Any]:
    """Compute a simplified EOD scanner row from daily OHLC bars."""
    ticker_clean = str(ticker).strip().upper()
    security_name_clean = str(security_name).strip()
    exchange_clean = str(exchange).strip()
    universe_source_clean = str(universe_source).strip()

    bars = _normalize_scanner_bars(daily_bars)
    if bars.empty:
        return _build_empty_row(
            ticker=ticker_clean,
            security_name=security_name_clean,
            exchange=exchange_clean,
            universe_source=universe_source_clean,
            is_watchlist=is_watchlist,
            is_etf=is_etf,
            status="fetch_error",
            status_message="Daily OHLC bars are empty.",
        )

    if "Open" not in bars.columns or "Close" not in bars.columns:
        return _build_empty_row(
            ticker=ticker_clean,
            security_name=security_name_clean,
            exchange=exchange_clean,
            universe_source=universe_source_clean,
            is_watchlist=is_watchlist,
            is_etf=is_etf,
            status="insufficient_data",
            status_message="Daily bars must include Open and Close columns.",
        )

    close_series = bars["Close"].dropna().astype(float)
    if close_series.empty:
        return _build_empty_row(
            ticker=ticker_clean,
            security_name=security_name_clean,
            exchange=exchange_clean,
            universe_source=universe_source_clean,
            is_watchlist=is_watchlist,
            is_etf=is_etf,
            status="insufficient_data",
            status_message="Daily close series is empty after cleaning.",
        )

    open_series = pd.to_numeric(bars["Open"], errors="coerce").reindex(close_series.index)
    sma14_series = compute_sma(close_series, 14)
    sma50_series = compute_sma(close_series, 50)
    sma100_series = compute_sma(close_series, 100)
    sma200_series = compute_sma(close_series, 200)

    as_of_ts = pd.Timestamp(close_series.index[-1])
    source = str(getattr(daily_bars, "attrs", {}).get("source", "") or bars.attrs.get("source", "") or f"STOCK:{ticker_clean}")

    base_row = _build_empty_row(
        ticker=ticker_clean,
        security_name=security_name_clean,
        exchange=exchange_clean,
        universe_source=universe_source_clean,
        is_watchlist=is_watchlist,
        is_etf=is_etf,
        status="ok",
        status_message="Scanner signals computed successfully.",
    )
    base_row.update(
        {
            "as_of_date": as_of_ts.date().isoformat(),
            "price": float(close_series.iloc[-1]),
            "open": float(open_series.iloc[-1]) if pd.notna(open_series.iloc[-1]) else pd.NA,
            "source": source,
            "stale_days": int((datetime.now(timezone.utc).date() - as_of_ts.date()).days),
        }
    )

    if len(close_series) < 201:
        base_row["status"] = "insufficient_data"
        base_row["status_message"] = f"Need at least 201 daily bars; found {len(close_series)}."
        return base_row

    required_current_prev = [
        sma14_series.iloc[-1],
        sma50_series.iloc[-1],
        sma100_series.iloc[-1],
        sma200_series.iloc[-1],
        sma14_series.iloc[-2],
        sma50_series.iloc[-2],
        sma100_series.iloc[-2],
        sma200_series.iloc[-2],
        close_series.iloc[-1],
        close_series.iloc[-2],
    ]
    if any(pd.isna(v) for v in required_current_prev):
        base_row["status"] = "insufficient_data"
        base_row["status_message"] = "Insufficient data for scanner moving averages."
        return base_row

    if len(open_series) < 3 or open_series.iloc[-3:].isna().any():
        base_row["status"] = "insufficient_data"
        base_row["status_message"] = "Missing open prices for the last three bars."
        return base_row

    sma14 = float(sma14_series.iloc[-1])
    sma50 = float(sma50_series.iloc[-1])
    sma100 = float(sma100_series.iloc[-1])
    sma200 = float(sma200_series.iloc[-1])
    sma14_prev = float(sma14_series.iloc[-2])
    sma50_prev = float(sma50_series.iloc[-2])
    sma100_prev = float(sma100_series.iloc[-2])
    sma200_prev = float(sma200_series.iloc[-2])
    close_now = float(close_series.iloc[-1])
    close_prev = float(close_series.iloc[-2])
    open_now = float(open_series.iloc[-1])

    gap_to_sma100_pct = _gap_to_ma_pct(close_now, sma100)
    gap_to_sma200_pct = _gap_to_ma_pct(close_now, sma200)
    gap_to_sma100_prev_pct = _gap_to_ma_pct(close_prev, sma100_prev)
    gap_to_sma200_prev_pct = _gap_to_ma_pct(close_prev, sma200_prev)

    bullish_alignment_active = _alignment_active(sma14, sma50, sma100, sma200)
    bullish_alignment_prev = _alignment_active(sma14_prev, sma50_prev, sma100_prev, sma200_prev)
    bullish_alignment_triggered_today = bool(bullish_alignment_active and not bullish_alignment_prev)

    recovery_close_cross_sma50_today = bool(close_now > sma50 and close_prev <= sma50_prev)
    recovery_three_bullish_candles_today = _three_bullish_candles(open_series, close_series)
    recovery_momentum_triggered_today = bool(recovery_close_cross_sma50_today and recovery_three_bullish_candles_today)

    ambush_trend_bullish_active = bool(sma50 > sma200)
    ambush_near_ma100_active = _in_ambush_band(gap_to_sma100_pct)
    ambush_near_ma200_active = _in_ambush_band(gap_to_sma200_pct)
    ambush_squat_active = bool(ambush_trend_bullish_active and (ambush_near_ma100_active or ambush_near_ma200_active))

    ambush_trend_prev = bool(sma50_prev > sma200_prev)
    ambush_near_ma100_prev = _in_ambush_band(gap_to_sma100_prev_pct)
    ambush_near_ma200_prev = _in_ambush_band(gap_to_sma200_prev_pct)
    ambush_squat_prev = bool(ambush_trend_prev and (ambush_near_ma100_prev or ambush_near_ma200_prev))
    ambush_squat_triggered_today = bool(ambush_squat_active and not ambush_squat_prev)

    base_row.update(
        {
            "price": close_now,
            "open": open_now,
            "sma14": sma14,
            "sma50": sma50,
            "sma100": sma100,
            "sma200": sma200,
            "gap_to_sma100_pct": gap_to_sma100_pct,
            "gap_to_sma200_pct": gap_to_sma200_pct,
            "bullish_alignment_active": bool(bullish_alignment_active),
            "bullish_alignment_triggered_today": bool(bullish_alignment_triggered_today),
            "recovery_close_cross_sma50_today": bool(recovery_close_cross_sma50_today),
            "recovery_three_bullish_candles_today": bool(recovery_three_bullish_candles_today),
            "recovery_momentum_triggered_today": bool(recovery_momentum_triggered_today),
            "ambush_trend_bullish_active": bool(ambush_trend_bullish_active),
            "ambush_near_ma100_active": bool(ambush_near_ma100_active),
            "ambush_near_ma200_active": bool(ambush_near_ma200_active),
            "ambush_squat_active": bool(ambush_squat_active),
            "ambush_squat_triggered_today": bool(ambush_squat_triggered_today),
        }
    )

    for col in SCANNER_SIGNAL_COLUMNS:
        if col not in base_row:
            base_row[col] = pd.NA
    return base_row


"""Signal computation for stock watchlist monitoring."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

STOCK_TRIGGER_COLUMNS = [
    "entry_bullish_alignment",
    "exit_price_below_sma50",
    "exit_death_cross_50_lt_100",
    "exit_death_cross_50_lt_200",
    "exit_rsi_overbought",
    "rsi_bearish_divergence",
]

STOCK_SIGNAL_COLUMNS = [
    "ticker",
    "as_of_date",
    "price",
    "sma14",
    "sma50",
    "sma100",
    "sma200",
    "rsi14",
    *STOCK_TRIGGER_COLUMNS,
    "source",
    "stale_days",
    "status",
    "status_message",
]


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Return rolling simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi14(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI with Wilder smoothing."""
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def _is_valid_number(value: Any) -> bool:
    return pd.notna(value)


def _safe_gt(lhs: Any, rhs: Any) -> bool:
    return _is_valid_number(lhs) and _is_valid_number(rhs) and float(lhs) > float(rhs)


def _safe_lt(lhs: Any, rhs: Any) -> bool:
    return _is_valid_number(lhs) and _is_valid_number(rhs) and float(lhs) < float(rhs)


def _find_swing_high_positions(series: pd.Series, left: int = 3, right: int = 3) -> list[int]:
    """Find swing-high bar positions in a numeric series."""
    if series.empty:
        return []

    values = series.astype(float).tolist()
    positions: list[int] = []
    upper_bound = len(values) - right

    for idx in range(left, upper_bound):
        center = values[idx]
        if pd.isna(center):
            continue

        left_window = [v for v in values[idx - left : idx] if pd.notna(v)]
        right_window = [v for v in values[idx + 1 : idx + right + 1] if pd.notna(v)]
        if not left_window or not right_window:
            continue

        if center > max(left_window) and center >= max(right_window):
            positions.append(idx)

    return positions


def _rsi_peak_for_price_peak(
    peak_pos: int,
    rsi_series: pd.Series,
    rsi_peak_positions: list[int],
    max_distance: int = 3,
) -> float | None:
    close_peaks = [
        idx
        for idx in rsi_peak_positions
        if abs(idx - peak_pos) <= max_distance and pd.notna(rsi_series.iloc[idx])
    ]
    if close_peaks:
        selected = min(close_peaks, key=lambda idx: abs(idx - peak_pos))
        return float(rsi_series.iloc[selected])

    if 0 <= peak_pos < len(rsi_series) and pd.notna(rsi_series.iloc[peak_pos]):
        return float(rsi_series.iloc[peak_pos])
    return None


def detect_bearish_rsi_divergence(
    price_series: pd.Series,
    rsi_series: pd.Series,
    lookback: int = 120,
    left: int = 3,
    right: int = 3,
) -> bool:
    """Detect bearish RSI divergence using the two latest confirmed price highs."""
    if price_series.empty or rsi_series.empty:
        return False

    price_recent = price_series.iloc[-lookback:].astype(float)
    rsi_recent = rsi_series.reindex(price_recent.index).astype(float)
    if len(price_recent) < (left + right + 2):
        return False

    price_peaks = _find_swing_high_positions(price_recent, left=left, right=right)
    if len(price_peaks) < 2:
        return False

    p1_pos, p2_pos = price_peaks[-2], price_peaks[-1]
    p1 = float(price_recent.iloc[p1_pos])
    p2 = float(price_recent.iloc[p2_pos])

    rsi_peaks = _find_swing_high_positions(rsi_recent, left=left, right=right)
    r1 = _rsi_peak_for_price_peak(p1_pos, rsi_recent, rsi_peaks)
    r2 = _rsi_peak_for_price_peak(p2_pos, rsi_recent, rsi_peaks)
    if r1 is None or r2 is None:
        return False

    return p2 > p1 and r2 < r1


def compute_stock_signal_row(
    ticker: str,
    daily_close: pd.Series,
    latest_price: float | None = None,
) -> dict[str, Any]:
    """Compute one watchlist signal row from daily closes and optional intraday price."""
    clean = daily_close.copy()
    clean.index = pd.to_datetime(clean.index).tz_localize(None)
    clean = clean.sort_index().dropna().astype(float)
    if clean.empty:
        raise ValueError("Daily close series is empty after cleaning.")

    evaluated = clean.copy()
    price = float(clean.iloc[-1])
    price_basis = "daily_close"
    if latest_price is not None and pd.notna(latest_price) and float(latest_price) > 0:
        price = float(latest_price)
        evaluated.iloc[-1] = price
        price_basis = "intraday"

    sma14 = float(compute_sma(evaluated, 14).iloc[-1])
    sma50 = float(compute_sma(evaluated, 50).iloc[-1])
    sma100 = float(compute_sma(evaluated, 100).iloc[-1])
    sma200 = float(compute_sma(evaluated, 200).iloc[-1])
    rsi14 = float(compute_rsi14(evaluated, period=14).iloc[-1])

    entry_bullish_alignment = _safe_gt(sma14, sma50) and (_safe_gt(sma50, sma100) or _safe_gt(sma50, sma200))
    exit_price_below_sma50 = _safe_lt(price, sma50)
    exit_death_cross_50_lt_100 = _safe_lt(sma50, sma100)
    exit_death_cross_50_lt_200 = _safe_lt(sma50, sma200)
    exit_rsi_overbought = _is_valid_number(rsi14) and float(rsi14) > 80.0
    rsi_bearish_divergence = detect_bearish_rsi_divergence(evaluated, compute_rsi14(evaluated, period=14))

    status = "ok"
    status_message = f"Signals computed successfully (price basis: {price_basis})."
    if len(evaluated) < 200:
        status = "insufficient_data"
        status_message = f"Need at least 200 daily bars; found {len(evaluated)}."
    elif any(pd.isna(v) for v in [sma14, sma50, sma100, sma200, rsi14]):
        status = "insufficient_data"
        status_message = "Insufficient data for SMA/RSI calculations."

    if status != "ok":
        entry_bullish_alignment = False
        exit_price_below_sma50 = False
        exit_death_cross_50_lt_100 = False
        exit_death_cross_50_lt_200 = False
        exit_rsi_overbought = False
        rsi_bearish_divergence = False

    as_of = pd.Timestamp(clean.index[-1])
    now_utc = datetime.now(timezone.utc).date()
    stale_days = (now_utc - as_of.date()).days

    source = str(clean.attrs.get("source", f"STOOQ:{ticker}"))
    intraday_source = str(clean.attrs.get("intraday_source", "")).strip()
    if intraday_source:
        source = f"{source} + {intraday_source}"

    return {
        "ticker": ticker,
        "as_of_date": as_of.date().isoformat(),
        "price": price,
        "sma14": sma14,
        "sma50": sma50,
        "sma100": sma100,
        "sma200": sma200,
        "rsi14": rsi14,
        "entry_bullish_alignment": bool(entry_bullish_alignment),
        "exit_price_below_sma50": bool(exit_price_below_sma50),
        "exit_death_cross_50_lt_100": bool(exit_death_cross_50_lt_100),
        "exit_death_cross_50_lt_200": bool(exit_death_cross_50_lt_200),
        "exit_rsi_overbought": bool(exit_rsi_overbought),
        "rsi_bearish_divergence": bool(rsi_bearish_divergence),
        "source": source,
        "stale_days": int(stale_days),
        "status": status,
        "status_message": status_message,
    }

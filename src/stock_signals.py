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
    "strong_sell_weak_strength",
]

STOCK_SIGNAL_COLUMNS = [
    "ticker",
    "benchmark_ticker",
    "as_of_date",
    "price",
    "intraday_quote_timestamp_utc",
    "intraday_quote_age_seconds",
    "intraday_quote_source",
    "sma14",
    "sma50",
    "sma100",
    "sma200",
    "rsi14",
    "benchmark_price",
    "benchmark_sma50",
    "rs_ratio",
    "rs_ratio_ma20",
    "alpha_1m",
    "rs_structural_divergence",
    "rs_trend_down",
    "rs_negative_alpha",
    "relative_strength_weak",
    "relative_strength_reasons",
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


def _clean_price_series(series: pd.Series) -> pd.Series:
    clean = series.copy()
    clean.index = pd.to_datetime(clean.index).tz_localize(None)
    clean = clean.sort_index().dropna().astype(float)
    return clean


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


def _relative_strength_signals(
    stock_series: pd.Series,
    stock_price: float,
    stock_sma50: float,
    benchmark_series: pd.Series,
    benchmark_ticker: str,
    benchmark_latest_price: float | None = None,
) -> dict[str, Any]:
    benchmark_clean = _clean_price_series(benchmark_series)
    benchmark_price = float("nan")
    benchmark_sma50 = float("nan")
    rs_ratio = float("nan")
    rs_ratio_ma20 = float("nan")
    alpha_1m = float("nan")
    rs_structural_divergence = False
    rs_trend_down = False
    rs_negative_alpha = False
    reasons: list[str] = []

    if benchmark_clean.empty:
        return {
            "benchmark_ticker": benchmark_ticker,
            "benchmark_price": benchmark_price,
            "benchmark_sma50": benchmark_sma50,
            "rs_ratio": rs_ratio,
            "rs_ratio_ma20": rs_ratio_ma20,
            "alpha_1m": alpha_1m,
            "rs_structural_divergence": rs_structural_divergence,
            "rs_trend_down": rs_trend_down,
            "rs_negative_alpha": rs_negative_alpha,
            "relative_strength_weak": False,
            "relative_strength_reasons": "benchmark_history_missing",
        }

    benchmark_eval = benchmark_clean.copy()
    if benchmark_latest_price is not None and pd.notna(benchmark_latest_price) and float(benchmark_latest_price) > 0:
        benchmark_eval.iloc[-1] = float(benchmark_latest_price)

    benchmark_price = float(benchmark_eval.iloc[-1])
    benchmark_sma50 = float(compute_sma(benchmark_eval, 50).iloc[-1])

    rs_structural_divergence = _safe_gt(benchmark_price, benchmark_sma50) and _safe_lt(stock_price, stock_sma50)
    if rs_structural_divergence:
        reasons.append("STRUCTURAL_DIVERGENCE")

    aligned = pd.concat(
        [stock_series.rename("stock_close"), benchmark_eval.rename("benchmark_close")],
        axis=1,
        join="inner",
    ).dropna()
    if not aligned.empty:
        ratio_series = aligned["stock_close"] / aligned["benchmark_close"]
        rs_ratio = float(ratio_series.iloc[-1])
        rs_ratio_ma20 = float(compute_sma(ratio_series, 20).iloc[-1])
        rs_trend_down = _safe_lt(rs_ratio, rs_ratio_ma20)
        if rs_trend_down:
            reasons.append("RS_TREND_DOWN")

    stock_ret_21 = stock_series.pct_change(periods=21).iloc[-1]
    bench_ret_21 = benchmark_eval.pct_change(periods=21).asof(stock_series.index[-1])
    if pd.notna(stock_ret_21) and pd.notna(bench_ret_21):
        alpha_1m = float(stock_ret_21 - bench_ret_21)
        rs_negative_alpha = alpha_1m < -0.05
        if rs_negative_alpha:
            reasons.append("NEGATIVE_ALPHA_LT_-5PCT")

    relative_strength_weak = rs_structural_divergence or rs_negative_alpha
    if not reasons:
        reasons.append("KEEPING_PACE_OR_LEADING")

    return {
        "benchmark_ticker": benchmark_ticker,
        "benchmark_price": benchmark_price,
        "benchmark_sma50": benchmark_sma50,
        "rs_ratio": rs_ratio,
        "rs_ratio_ma20": rs_ratio_ma20,
        "alpha_1m": alpha_1m,
        "rs_structural_divergence": bool(rs_structural_divergence),
        "rs_trend_down": bool(rs_trend_down),
        "rs_negative_alpha": bool(rs_negative_alpha),
        "relative_strength_weak": bool(relative_strength_weak),
        "relative_strength_reasons": "; ".join(reasons),
    }


def compute_stock_signal_row(
    ticker: str,
    daily_close: pd.Series,
    latest_price: float | None = None,
    benchmark_ticker: str = "QQQ",
    benchmark_close: pd.Series | None = None,
    benchmark_latest_price: float | None = None,
) -> dict[str, Any]:
    """Compute one watchlist signal row from daily closes and optional intraday prices."""
    clean = _clean_price_series(daily_close)
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
    rsi14_series = compute_rsi14(evaluated, period=14)
    rsi14 = float(rsi14_series.iloc[-1])

    entry_bullish_alignment = _safe_gt(sma14, sma50) and (_safe_gt(sma50, sma100) or _safe_gt(sma50, sma200))
    exit_price_below_sma50 = _safe_lt(price, sma50)
    exit_death_cross_50_lt_100 = _safe_lt(sma50, sma100)
    exit_death_cross_50_lt_200 = _safe_lt(sma50, sma200)
    exit_rsi_overbought = _is_valid_number(rsi14) and float(rsi14) > 80.0
    rsi_bearish_divergence = detect_bearish_rsi_divergence(evaluated, rsi14_series)

    relative = {
        "benchmark_ticker": benchmark_ticker,
        "benchmark_price": float("nan"),
        "benchmark_sma50": float("nan"),
        "rs_ratio": float("nan"),
        "rs_ratio_ma20": float("nan"),
        "alpha_1m": float("nan"),
        "rs_structural_divergence": False,
        "rs_trend_down": False,
        "rs_negative_alpha": False,
        "relative_strength_weak": False,
        "relative_strength_reasons": "benchmark_not_available",
    }

    if benchmark_close is not None:
        relative = _relative_strength_signals(
            stock_series=evaluated,
            stock_price=price,
            stock_sma50=sma50,
            benchmark_series=benchmark_close,
            benchmark_ticker=benchmark_ticker,
            benchmark_latest_price=benchmark_latest_price,
        )

    strong_sell_weak_strength = bool(relative["relative_strength_weak"])

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
        strong_sell_weak_strength = False
        relative["rs_structural_divergence"] = False
        relative["rs_trend_down"] = False
        relative["rs_negative_alpha"] = False
        relative["relative_strength_weak"] = False

    as_of = pd.Timestamp(clean.index[-1])
    now_utc = datetime.now(timezone.utc).date()
    stale_days = (now_utc - as_of.date()).days

    source = str(clean.attrs.get("source", f"STOOQ:{ticker}"))
    intraday_source = str(clean.attrs.get("intraday_source", "")).strip()
    if intraday_source:
        source = f"{source} + {intraday_source}"

    return {
        "ticker": ticker,
        "benchmark_ticker": str(relative.get("benchmark_ticker", benchmark_ticker)).upper(),
        "as_of_date": as_of.date().isoformat(),
        "price": price,
        "intraday_quote_timestamp_utc": None,
        "intraday_quote_age_seconds": None,
        "intraday_quote_source": None,
        "sma14": sma14,
        "sma50": sma50,
        "sma100": sma100,
        "sma200": sma200,
        "rsi14": rsi14,
        "benchmark_price": relative["benchmark_price"],
        "benchmark_sma50": relative["benchmark_sma50"],
        "rs_ratio": relative["rs_ratio"],
        "rs_ratio_ma20": relative["rs_ratio_ma20"],
        "alpha_1m": relative["alpha_1m"],
        "rs_structural_divergence": bool(relative["rs_structural_divergence"]),
        "rs_trend_down": bool(relative["rs_trend_down"]),
        "rs_negative_alpha": bool(relative["rs_negative_alpha"]),
        "relative_strength_weak": bool(relative["relative_strength_weak"]),
        "relative_strength_reasons": str(relative["relative_strength_reasons"]),
        "entry_bullish_alignment": bool(entry_bullish_alignment),
        "exit_price_below_sma50": bool(exit_price_below_sma50),
        "exit_death_cross_50_lt_100": bool(exit_death_cross_50_lt_100),
        "exit_death_cross_50_lt_200": bool(exit_death_cross_50_lt_200),
        "exit_rsi_overbought": bool(exit_rsi_overbought),
        "rsi_bearish_divergence": bool(rsi_bearish_divergence),
        "strong_sell_weak_strength": bool(strong_sell_weak_strength),
        "source": source,
        "stale_days": int(stale_days),
        "status": status,
        "status_message": status_message,
    }

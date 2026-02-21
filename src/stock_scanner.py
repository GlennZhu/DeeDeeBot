"""Signal logic for scanning a broader stock universe beyond the watchlist."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.stock_signals import compute_rsi14, compute_sma, compute_stock_signal_row

LEADER_SCORE_THRESHOLD = 3.0

SCANNER_SIGNAL_COLUMNS = [
    "ticker",
    "security_name",
    "exchange",
    "universe_source",
    "is_watchlist",
    "is_etf",
    "thesis_tags",
    "thesis_score",
    "benchmark_ticker",
    "as_of_date",
    "price",
    "day_change",
    "day_change_pct",
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
    "relative_strength_weak",
    "relative_strength_reasons",
    "entry_bullish_alignment",
    "squat_ambush_near_ma100_or_ma200",
    "squat_dca_below_ma100",
    "squat_last_stand_ma200",
    "squat_breakdown_below_ma200",
    "trend_established",
    "crossback_ma14_over_ma50",
    "three_green_candles",
    "bullish_rsi_divergence",
    "crossback_three_green_confirmed",
    "pullback_order_zone_ma100_or_ma200",
    "leader_score",
    "scanner_score",
    "leader_candidate",
    "scanner_candidate",
    "scanner_reasons",
    "source",
    "stale_days",
    "status",
    "status_message",
]


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


def _find_swing_low_positions(series: pd.Series, left: int = 3, right: int = 3) -> list[int]:
    if series.empty:
        return []

    values = series.astype(float).tolist()
    positions: list[int] = []
    upper_bound = len(values) - right
    for idx in range(left, upper_bound):
        center = values[idx]
        if pd.isna(center):
            continue

        left_window = [value for value in values[idx - left : idx] if pd.notna(value)]
        right_window = [value for value in values[idx + 1 : idx + right + 1] if pd.notna(value)]
        if not left_window or not right_window:
            continue

        if center < min(left_window) and center <= min(right_window):
            positions.append(idx)
    return positions


def _rsi_trough_for_price_trough(
    trough_pos: int,
    rsi_series: pd.Series,
    rsi_trough_positions: list[int],
    max_distance: int = 3,
) -> float | None:
    nearby = [idx for idx in rsi_trough_positions if abs(idx - trough_pos) <= max_distance and pd.notna(rsi_series.iloc[idx])]
    if nearby:
        selected = min(nearby, key=lambda idx: abs(idx - trough_pos))
        return float(rsi_series.iloc[selected])

    if 0 <= trough_pos < len(rsi_series) and pd.notna(rsi_series.iloc[trough_pos]):
        return float(rsi_series.iloc[trough_pos])
    return None


def detect_bullish_rsi_divergence(
    price_series: pd.Series,
    rsi_series: pd.Series,
    lookback: int = 120,
    left: int = 3,
    right: int = 3,
) -> bool:
    """Detect bullish RSI divergence using the latest two confirmed price lows."""
    if price_series.empty or rsi_series.empty:
        return False

    price_recent = price_series.iloc[-lookback:].astype(float)
    rsi_recent = rsi_series.reindex(price_recent.index).astype(float)
    if len(price_recent) < (left + right + 2):
        return False

    price_troughs = _find_swing_low_positions(price_recent, left=left, right=right)
    if len(price_troughs) < 2:
        return False

    p1_pos, p2_pos = price_troughs[-2], price_troughs[-1]
    p1 = float(price_recent.iloc[p1_pos])
    p2 = float(price_recent.iloc[p2_pos])

    rsi_troughs = _find_swing_low_positions(rsi_recent, left=left, right=right)
    r1 = _rsi_trough_for_price_trough(p1_pos, rsi_recent, rsi_troughs)
    r2 = _rsi_trough_for_price_trough(p2_pos, rsi_recent, rsi_troughs)
    if r1 is None or r2 is None:
        return False

    return p2 < p1 and r2 > r1


def _crossback_ma14_over_ma50(price_series: pd.Series) -> bool:
    clean = _clean_price_series(price_series)
    ma14 = compute_sma(clean, 14)
    ma50 = compute_sma(clean, 50)
    if len(ma14) < 2 or len(ma50) < 2:
        return False

    current_up = _safe_gt(ma14.iloc[-1], ma50.iloc[-1])
    previous_below_or_equal = _is_valid_number(ma14.iloc[-2]) and _is_valid_number(ma50.iloc[-2]) and float(ma14.iloc[-2]) <= float(ma50.iloc[-2])
    return bool(current_up and previous_below_or_equal)


def _three_green_candles(price_series: pd.Series) -> bool:
    clean = _clean_price_series(price_series)
    if len(clean) < 4:
        return False
    return bool(clean.iloc[-1] > clean.iloc[-2] > clean.iloc[-3] > clean.iloc[-4])


def _build_scanner_reasons(
    *,
    leader_candidate: bool,
    trend_established: bool,
    crossback_three_green_confirmed: bool,
    pullback_order_zone: bool,
) -> str:
    reasons: list[str] = []
    if leader_candidate and trend_established:
        reasons.append("LEADER_TREND")
    if crossback_three_green_confirmed:
        reasons.append("CROSSBACK_3_GREEN")
    if pullback_order_zone:
        reasons.append("PULLBACK_MA100_MA200")
    if not reasons:
        reasons.append("NO_ACTIVE_SETUP")
    return "; ".join(reasons)


def _leader_score(stock_row: dict[str, Any]) -> float:
    score = 0.0
    if bool(stock_row.get("entry_bullish_alignment", False)):
        score += 1.0
    if not bool(stock_row.get("relative_strength_weak", False)):
        score += 1.0

    rs_ratio = stock_row.get("rs_ratio")
    rs_ratio_ma20 = stock_row.get("rs_ratio_ma20")
    if _safe_gt(rs_ratio, rs_ratio_ma20):
        score += 1.0

    alpha_1m = stock_row.get("alpha_1m")
    if _is_valid_number(alpha_1m) and float(alpha_1m) > 0:
        score += 1.0

    price = stock_row.get("price")
    sma50 = stock_row.get("sma50")
    sma200 = stock_row.get("sma200")
    if _safe_gt(price, sma50) and _safe_gt(sma50, sma200):
        score += 1.0

    return round(float(score), 2)


def compute_scanner_signal_row(
    ticker: str,
    daily_close: pd.Series,
    *,
    latest_price: float | None = None,
    benchmark_ticker: str = "QQQ",
    benchmark_close: pd.Series | None = None,
    benchmark_latest_price: float | None = None,
    is_watchlist: bool = False,
    security_name: str = "",
    exchange: str = "",
    universe_source: str = "",
    is_etf: bool = False,
    thesis_tags: str = "",
    thesis_score: float = 0.0,
) -> dict[str, Any]:
    """Compute scanner row from a stock history and optional benchmark/intraday values."""
    base_row = compute_stock_signal_row(
        ticker=ticker,
        daily_close=daily_close,
        latest_price=latest_price,
        benchmark_ticker=benchmark_ticker,
        benchmark_close=benchmark_close,
        benchmark_latest_price=benchmark_latest_price,
    )

    clean = _clean_price_series(daily_close)
    rsi14_series = compute_rsi14(clean, period=14)
    crossback_ma14_over_ma50 = _crossback_ma14_over_ma50(clean)
    three_green_candles = _three_green_candles(clean)
    bullish_rsi_divergence = detect_bullish_rsi_divergence(clean, rsi14_series)
    crossback_three_green_confirmed = bool(crossback_ma14_over_ma50 and three_green_candles and bullish_rsi_divergence)

    trend_established = bool(base_row.get("entry_bullish_alignment", False))
    pullback_order_zone = bool(
        base_row.get("squat_ambush_near_ma100_or_ma200", False)
        or base_row.get("squat_last_stand_ma200", False)
    )

    thesis_score_clean = max(0.0, float(thesis_score))
    leader_score = _leader_score(base_row)
    scanner_score = round(leader_score + thesis_score_clean, 2)
    leader_candidate = scanner_score >= LEADER_SCORE_THRESHOLD
    scanner_candidate = bool((leader_candidate and trend_established) or crossback_three_green_confirmed or pullback_order_zone)

    status = str(base_row.get("status", ""))
    if status != "ok":
        leader_score = 0.0
        scanner_score = 0.0
        leader_candidate = False
        scanner_candidate = False
        trend_established = False
        crossback_ma14_over_ma50 = False
        three_green_candles = False
        bullish_rsi_divergence = False
        crossback_three_green_confirmed = False
        pullback_order_zone = False

    scanner_reasons = _build_scanner_reasons(
        leader_candidate=leader_candidate,
        trend_established=trend_established,
        crossback_three_green_confirmed=crossback_three_green_confirmed,
        pullback_order_zone=pullback_order_zone,
    )
    row: dict[str, Any] = {
        "ticker": str(base_row.get("ticker", ticker)).upper(),
        "security_name": str(security_name).strip(),
        "exchange": str(exchange).strip(),
        "universe_source": str(universe_source).strip(),
        "is_watchlist": bool(is_watchlist),
        "is_etf": bool(is_etf),
        "thesis_tags": str(thesis_tags).strip(),
        "thesis_score": round(thesis_score_clean, 2),
        "benchmark_ticker": str(base_row.get("benchmark_ticker", benchmark_ticker)).upper(),
        "as_of_date": base_row.get("as_of_date"),
        "price": base_row.get("price"),
        "day_change": base_row.get("day_change"),
        "day_change_pct": base_row.get("day_change_pct"),
        "intraday_quote_timestamp_utc": base_row.get("intraday_quote_timestamp_utc"),
        "intraday_quote_age_seconds": base_row.get("intraday_quote_age_seconds"),
        "intraday_quote_source": base_row.get("intraday_quote_source"),
        "sma14": base_row.get("sma14"),
        "sma50": base_row.get("sma50"),
        "sma100": base_row.get("sma100"),
        "sma200": base_row.get("sma200"),
        "rsi14": base_row.get("rsi14"),
        "benchmark_price": base_row.get("benchmark_price"),
        "benchmark_sma50": base_row.get("benchmark_sma50"),
        "rs_ratio": base_row.get("rs_ratio"),
        "rs_ratio_ma20": base_row.get("rs_ratio_ma20"),
        "alpha_1m": base_row.get("alpha_1m"),
        "relative_strength_weak": bool(base_row.get("relative_strength_weak", False)),
        "relative_strength_reasons": str(base_row.get("relative_strength_reasons", "")),
        "entry_bullish_alignment": bool(base_row.get("entry_bullish_alignment", False)),
        "squat_ambush_near_ma100_or_ma200": bool(base_row.get("squat_ambush_near_ma100_or_ma200", False)),
        "squat_dca_below_ma100": bool(base_row.get("squat_dca_below_ma100", False)),
        "squat_last_stand_ma200": bool(base_row.get("squat_last_stand_ma200", False)),
        "squat_breakdown_below_ma200": bool(base_row.get("squat_breakdown_below_ma200", False)),
        "trend_established": bool(trend_established),
        "crossback_ma14_over_ma50": bool(crossback_ma14_over_ma50),
        "three_green_candles": bool(three_green_candles),
        "bullish_rsi_divergence": bool(bullish_rsi_divergence),
        "crossback_three_green_confirmed": bool(crossback_three_green_confirmed),
        "pullback_order_zone_ma100_or_ma200": bool(pullback_order_zone),
        "leader_score": float(leader_score),
        "scanner_score": float(scanner_score),
        "leader_candidate": bool(leader_candidate),
        "scanner_candidate": bool(scanner_candidate),
        "scanner_reasons": scanner_reasons,
        "source": base_row.get("source"),
        "stale_days": base_row.get("stale_days"),
        "status": base_row.get("status"),
        "status_message": base_row.get("status_message"),
    }

    for col in SCANNER_SIGNAL_COLUMNS:
        if col not in row:
            row[col] = pd.NA
    return row

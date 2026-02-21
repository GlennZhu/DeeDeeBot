from __future__ import annotations

import pandas as pd

from src import stock_scanner


def _series(values: list[float], start: str = "2024-01-01", freq: str = "D") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_compute_scanner_signal_row_marks_leader_trend_candidate() -> None:
    stock = _series([100.0 + i * 1.2 for i in range(260)])
    benchmark = _series([100.0 + i * 0.5 for i in range(260)])

    row = stock_scanner.compute_scanner_signal_row(
        ticker="TEST",
        daily_close=stock,
        benchmark_ticker="QQQ",
        benchmark_close=benchmark,
        thesis_score=1.0,
    )

    assert row["status"] == "ok"
    assert row["trend_established"] is True
    assert row["leader_score"] >= 4.0
    assert row["scanner_score"] > row["leader_score"]
    assert row["leader_candidate"] is True
    assert row["scanner_candidate"] is True
    assert "LEADER_TREND" in row["scanner_reasons"]


def test_compute_scanner_signal_row_marks_pullback_candidate() -> None:
    closes = _series([100.0] * 200 + [112.0] * 20 + [110.0] * 39 + [111.0])
    row = stock_scanner.compute_scanner_signal_row(
        ticker="TEST",
        daily_close=closes,
        latest_price=109.0,
        thesis_score=0.0,
    )

    assert row["status"] == "ok"
    assert row["squat_ambush_near_ma100_or_ma200"] is True
    assert row["pullback_order_zone_ma100_or_ma200"] is True
    assert row["scanner_candidate"] is True
    assert "PULLBACK_MA100_MA200" in row["scanner_reasons"]


def test_detect_bullish_rsi_divergence_true_and_false_controls() -> None:
    prices = _series(
        [
            120,
            118,
            116,
            114,
            112,
            109,
            112,
            115,
            118,
            121,
            118,
            114,
            110,
            106,
            103,
            100,
            104,
            108,
            112,
            116,
            114,
            112,
            110,
            108,
            106,
        ]
    )
    rsi_bullish = _series(
        [
            48,
            44,
            40,
            36,
            31,
            24,
            30,
            37,
            44,
            49,
            43,
            36,
            31,
            28,
            30,
            32,
            38,
            44,
            49,
            53,
            50,
            47,
            44,
            42,
            40,
        ]
    )
    assert stock_scanner.detect_bullish_rsi_divergence(prices, rsi_bullish, left=3, right=3)

    rsi_not_bullish = rsi_bullish.copy()
    rsi_not_bullish.iloc[15] = 22.0
    assert not stock_scanner.detect_bullish_rsi_divergence(prices, rsi_not_bullish, left=3, right=3)

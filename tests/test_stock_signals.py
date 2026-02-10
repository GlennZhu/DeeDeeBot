from __future__ import annotations

import pandas as pd

from src import stock_signals


def _series(values: list[float], start: str = "2024-01-01", freq: str = "D") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_entry_bullish_alignment_true_on_uptrend() -> None:
    closes = _series([100.0 + i for i in range(260)])
    row = stock_signals.compute_stock_signal_row("TEST", closes)

    assert row["status"] == "ok"
    assert row["entry_bullish_alignment"] is True


def test_exit_price_below_sma50_and_death_crosses() -> None:
    closes = _series([200.0] * 200 + [100.0] * 60)
    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=99.0)

    assert row["status"] == "ok"
    assert row["exit_price_below_sma50"] is True
    assert row["exit_death_cross_50_lt_100"] is True
    assert row["exit_death_cross_50_lt_200"] is True


def test_rsi_overbought_boundary(monkeypatch) -> None:
    closes = _series([100.0 + i for i in range(260)])

    def rsi_at_80(series: pd.Series, period: int = 14) -> pd.Series:
        del period
        out = pd.Series(50.0, index=series.index, dtype=float)
        out.iloc[-1] = 80.0
        return out

    monkeypatch.setattr(stock_signals, "compute_rsi14", rsi_at_80)
    row_80 = stock_signals.compute_stock_signal_row("TEST", closes)
    assert row_80["exit_rsi_overbought"] is False

    def rsi_above_80(series: pd.Series, period: int = 14) -> pd.Series:
        del period
        out = pd.Series(50.0, index=series.index, dtype=float)
        out.iloc[-1] = 80.1
        return out

    monkeypatch.setattr(stock_signals, "compute_rsi14", rsi_above_80)
    row_above = stock_signals.compute_stock_signal_row("TEST", closes)
    assert row_above["exit_rsi_overbought"] is True


def test_detect_bearish_rsi_divergence_true_and_false_controls() -> None:
    prices = _series(
        [
            100,
            102,
            104,
            108,
            112,
            116,
            112,
            108,
            104,
            102,
            104,
            107,
            110,
            113,
            116,
            118,
            121,
            117,
            113,
            110,
            108,
            106,
            104,
            103,
            102,
        ]
    )
    rsi_bearish = _series(
        [
            52,
            55,
            59,
            65,
            72,
            79,
            73,
            69,
            63,
            58,
            60,
            63,
            66,
            69,
            72,
            73,
            71,
            68,
            64,
            60,
            58,
            56,
            54,
            53,
            52,
        ]
    )

    assert stock_signals.detect_bearish_rsi_divergence(prices, rsi_bearish, left=3, right=3)

    rsi_not_bearish = rsi_bearish.copy()
    rsi_not_bearish.iloc[16] = 85.0
    assert not stock_signals.detect_bearish_rsi_divergence(prices, rsi_not_bearish, left=3, right=3)


def test_compute_stock_signal_row_insufficient_data() -> None:
    closes = _series([100.0 + i for i in range(150)])
    row = stock_signals.compute_stock_signal_row("TEST", closes)

    assert row["status"] == "insufficient_data"
    assert row["entry_bullish_alignment"] is False
    assert row["exit_price_below_sma50"] is False
    assert row["exit_death_cross_50_lt_100"] is False
    assert row["exit_death_cross_50_lt_200"] is False
    assert row["exit_rsi_overbought"] is False
    assert row["rsi_bearish_divergence"] is False


def test_relative_strength_triggers_strong_sell_on_weakness() -> None:
    stock = _series([200.0] * 200 + [150.0 - i for i in range(60)])
    benchmark = _series([100.0 + i * 0.6 for i in range(260)])

    row = stock_signals.compute_stock_signal_row(
        "TEST",
        daily_close=stock,
        benchmark_ticker="QQQ",
        benchmark_close=benchmark,
    )

    assert row["rs_structural_divergence"] is True
    assert row["relative_strength_weak"] is True
    assert row["strong_sell_weak_strength"] is True
    assert "STRUCTURAL_DIVERGENCE" in row["relative_strength_reasons"]


def test_relative_strength_not_weak_when_stock_outperforms() -> None:
    stock = _series([100.0 + i for i in range(260)])
    benchmark = _series([100.0 + i * 0.3 for i in range(260)])

    row = stock_signals.compute_stock_signal_row(
        "TEST",
        daily_close=stock,
        benchmark_ticker="QQQ",
        benchmark_close=benchmark,
    )

    assert row["relative_strength_weak"] is False
    assert row["strong_sell_weak_strength"] is False

from __future__ import annotations

import pandas as pd
import pytest

from src import stock_signals


def _series(values: list[float], start: str = "2024-01-01", freq: str = "D") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_entry_bullish_alignment_true_on_uptrend() -> None:
    closes = _series([100.0 + i for i in range(260)])
    row = stock_signals.compute_stock_signal_row("TEST", closes)

    assert row["status"] == "ok"
    assert row["entry_bullish_alignment"] is True


def test_exit_price_below_sma50_and_death_cross_50_lt_200() -> None:
    closes = _series([200.0] * 200 + [100.0] * 60)
    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=99.0)

    assert row["status"] == "ok"
    assert row["exit_price_below_sma50"] is True
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


def test_entry_rsi_extreme_oversold_thresholds_for_stock_and_qqq(monkeypatch) -> None:
    closes = _series([100.0 + i for i in range(260)])

    def rsi_at_29(series: pd.Series, period: int = 14) -> pd.Series:
        del period
        out = pd.Series(50.0, index=series.index, dtype=float)
        out.iloc[-1] = 29.0
        return out

    monkeypatch.setattr(stock_signals, "compute_rsi14", rsi_at_29)
    row_stock_29 = stock_signals.compute_stock_signal_row("TEST", closes)
    row_qqq_29 = stock_signals.compute_stock_signal_row("QQQ", closes)
    assert row_stock_29["entry_rsi_extreme_oversold"] is False
    assert row_qqq_29["entry_rsi_extreme_oversold"] is True

    def rsi_at_30(series: pd.Series, period: int = 14) -> pd.Series:
        del period
        out = pd.Series(50.0, index=series.index, dtype=float)
        out.iloc[-1] = 30.0
        return out

    monkeypatch.setattr(stock_signals, "compute_rsi14", rsi_at_30)
    row_qqq_30 = stock_signals.compute_stock_signal_row("QQQ", closes)
    assert row_qqq_30["entry_rsi_extreme_oversold"] is False

    def rsi_below_25(series: pd.Series, period: int = 14) -> pd.Series:
        del period
        out = pd.Series(50.0, index=series.index, dtype=float)
        out.iloc[-1] = 24.9
        return out

    monkeypatch.setattr(stock_signals, "compute_rsi14", rsi_below_25)
    row_stock_24_9 = stock_signals.compute_stock_signal_row("TEST", closes)
    assert row_stock_24_9["entry_rsi_extreme_oversold"] is True


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


def test_detect_bullish_rsi_divergence_true_and_false_controls() -> None:
    prices = _series(
        [
            120,
            118,
            116,
            112,
            108,
            104,
            108,
            112,
            116,
            118,
            114,
            110,
            106,
            102,
            98,
            95,
            99,
            103,
            107,
            110,
            112,
        ]
    )
    rsi_bullish = _series(
        [
            55,
            52,
            48,
            42,
            35,
            28,
            36,
            44,
            52,
            58,
            53,
            47,
            41,
            37,
            35,
            34,
            42,
            49,
            55,
            60,
            62,
        ]
    )

    assert stock_signals.detect_bullish_rsi_divergence(prices, rsi_bullish, left=3, right=3)

    rsi_not_bullish = rsi_bullish.copy()
    rsi_not_bullish.iloc[15] = 20.0
    assert not stock_signals.detect_bullish_rsi_divergence(prices, rsi_not_bullish, left=3, right=3)


def test_detect_bullish_weekly_macd_divergence_with_confirmation() -> None:
    weekly_prices = _series(
        [
            120,
            118,
            116,
            114,
            112,
            110,
            108,
            106,
            104,
            102,
            100,
            103,
            106,
            109,
            112,
            110,
            108,
            106,
            104,
            102,
            99,
            101,
            104,
            107,
            110,
            112,
        ],
        freq="W-FRI",
    )
    macd_line = _series(
        [
            0.8,
            0.5,
            0.2,
            -0.2,
            -0.5,
            -0.8,
            -1.0,
            -1.3,
            -1.6,
            -1.8,
            -2.0,
            -1.6,
            -1.2,
            -0.8,
            -0.4,
            -0.6,
            -0.8,
            -0.9,
            -1.0,
            -1.1,
            -1.0,
            -0.8,
            -0.5,
            -0.2,
            0.1,
            0.3,
        ],
        start="2024-01-05",
        freq="W-FRI",
    )
    signal_line = _series(
        [
            0.9,
            0.7,
            0.5,
            0.2,
            -0.1,
            -0.3,
            -0.5,
            -0.7,
            -0.9,
            -1.1,
            -1.3,
            -1.25,
            -1.1,
            -0.95,
            -0.8,
            -0.75,
            -0.8,
            -0.85,
            -0.9,
            -0.95,
            -0.98,
            -0.9,
            -0.7,
            -0.4,
            -0.1,
            0.1,
        ],
        start="2024-01-05",
        freq="W-FRI",
    )
    params = stock_signals.WeeklyMacdDivergenceParams(
        lookback_weeks=52,
        left=2,
        right=2,
        max_distance=2,
        min_price_move_pct=0.005,
        min_macd_delta=0.2,
        min_pivot_separation=4,
        max_pivot_separation=20,
        confirmation_window=4,
        require_macd_pivot_match=True,
        bearish_min_macd_peak=0.0,
        bullish_max_macd_trough=0.0,
    )

    assert stock_signals.detect_bullish_weekly_macd_divergence_with_params(
        weekly_prices, macd_line, signal_line, params=params
    )

    no_cross_signal = signal_line.copy()
    no_cross_signal.iloc[21:] = [-0.6, -0.3, 0.0, 0.2, 0.4]
    assert not stock_signals.detect_bullish_weekly_macd_divergence_with_params(
        weekly_prices, macd_line, no_cross_signal, params=params
    )


def test_detect_bearish_weekly_macd_divergence_with_confirmation() -> None:
    weekly_prices = _series(
        [
            100,
            102,
            104,
            106,
            108,
            110,
            112,
            114,
            116,
            118,
            120,
            117,
            114,
            111,
            108,
            110,
            113,
            116,
            119,
            121,
            123,
            120,
            117,
            114,
            111,
            109,
        ],
        start="2024-01-05",
        freq="W-FRI",
    )
    macd_line = _series(
        [
            -0.2,
            0.0,
            0.3,
            0.6,
            0.9,
            1.2,
            1.4,
            1.6,
            1.8,
            1.9,
            2.0,
            1.7,
            1.4,
            1.0,
            0.6,
            0.8,
            1.0,
            1.1,
            1.2,
            1.3,
            1.2,
            0.9,
            0.6,
            0.3,
            0.0,
            -0.2,
        ],
        start="2024-01-05",
        freq="W-FRI",
    )
    signal_line = _series(
        [
            -0.1,
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
            1.1,
            1.3,
            1.5,
            1.6,
            1.7,
            1.6,
            1.4,
            1.2,
            1.0,
            0.95,
            0.98,
            1.0,
            1.05,
            1.1,
            1.1,
            1.0,
            0.9,
            0.8,
            0.7,
            0.6,
        ],
        start="2024-01-05",
        freq="W-FRI",
    )
    params = stock_signals.WeeklyMacdDivergenceParams(
        lookback_weeks=52,
        left=2,
        right=2,
        max_distance=2,
        min_price_move_pct=0.005,
        min_macd_delta=0.2,
        min_pivot_separation=4,
        max_pivot_separation=20,
        confirmation_window=4,
        require_macd_pivot_match=True,
        bearish_min_macd_peak=0.0,
        bullish_max_macd_trough=0.0,
    )

    assert stock_signals.detect_bearish_weekly_macd_divergence_with_params(
        weekly_prices, macd_line, signal_line, params=params
    )

    no_cross_signal = signal_line.copy()
    no_cross_signal.iloc[21:] = [0.5, 0.4, 0.2, 0.0, -0.2]
    assert not stock_signals.detect_bearish_weekly_macd_divergence_with_params(
        weekly_prices, macd_line, no_cross_signal, params=params
    )


def test_v2_filters_out_weak_bearish_divergence_signal() -> None:
    prices = _series(
        [
            100,
            102,
            104,
            106,
            108,
            110,
            112,
            109,
            106,
            110,
            113,
            109,
            106,
            104,
            103,
            102,
            101,
            100,
        ]
    )
    rsi = _series(
        [
            45,
            50,
            55,
            60,
            65,
            70,
            74,
            68,
            62,
            66,
            72,
            64,
            58,
            55,
            53,
            50,
            48,
            46,
        ]
    )

    assert stock_signals.detect_bearish_rsi_divergence(prices, rsi, left=3, right=3)
    assert not stock_signals.detect_bearish_rsi_divergence_v2(prices, rsi)


def test_v2_filters_out_weak_bullish_divergence_signal() -> None:
    prices = _series(
        [
            120,
            118,
            116,
            114,
            112,
            110,
            108,
            111,
            114,
            110,
            107,
            111,
            114,
            116,
            118,
            119,
        ]
    )
    rsi = _series(
        [
            58,
            54,
            50,
            46,
            42,
            36,
            30,
            35,
            40,
            36,
            32,
            38,
            44,
            49,
            53,
            56,
        ]
    )

    assert stock_signals.detect_bullish_rsi_divergence(prices, rsi, left=3, right=3)
    assert not stock_signals.detect_bullish_rsi_divergence_v2(prices, rsi)


def test_compute_stock_signal_row_uses_v2_divergence_functions(monkeypatch) -> None:
    closes = _series([100.0 + i for i in range(260)])

    monkeypatch.setattr(stock_signals, "detect_bullish_rsi_divergence_v2", lambda price, rsi: True)
    monkeypatch.setattr(stock_signals, "detect_bearish_rsi_divergence_v2", lambda price, rsi: False)

    row = stock_signals.compute_stock_signal_row("TEST", closes)
    assert row["rsi_bullish_divergence"] is True
    assert row["rsi_bearish_divergence"] is False


def test_compute_stock_signal_row_uses_weekly_macd_divergence_functions_on_daily_close_only(
    monkeypatch,
) -> None:
    closes = _series([100.0 + i for i in range(260)])
    observed: dict[str, float] = {}

    def _bullish(series: pd.Series, params=stock_signals.WEEKLY_MACD_DIVERGENCE_V1_PARAMS) -> bool:
        del params
        observed["bullish_last"] = float(series.iloc[-1])
        return True

    def _bearish(series: pd.Series, params=stock_signals.WEEKLY_MACD_DIVERGENCE_V1_PARAMS) -> bool:
        del params
        observed["bearish_last"] = float(series.iloc[-1])
        return False

    monkeypatch.setattr(stock_signals, "detect_bullish_weekly_macd_divergence", _bullish)
    monkeypatch.setattr(stock_signals, "detect_bearish_weekly_macd_divergence", _bearish)

    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=400.0)

    assert row["weekly_macd_bullish_divergence"] is True
    assert row["weekly_macd_bearish_divergence"] is False
    assert observed["bullish_last"] == pytest.approx(float(closes.iloc[-1]))
    assert observed["bearish_last"] == pytest.approx(float(closes.iloc[-1]))


def test_compute_stock_signal_row_insufficient_data() -> None:
    closes = _series([100.0 + i for i in range(150)])
    row = stock_signals.compute_stock_signal_row("TEST", closes)

    assert row["status"] == "insufficient_data"
    assert row["entry_bullish_alignment"] is False
    assert row["entry_rsi_extreme_oversold"] is False
    assert row["exit_price_below_sma50"] is False
    assert row["exit_death_cross_50_lt_200"] is False
    assert row["exit_rsi_overbought"] is False
    assert row["rsi_bullish_divergence"] is False
    assert row["rsi_bearish_divergence"] is False
    assert row["weekly_macd_bullish_divergence"] is False
    assert row["weekly_macd_bearish_divergence"] is False
    assert row["squat_bull_market_precondition"] is False
    assert row["squat_price_dropping"] is False
    assert row["squat_ambush_near_ma100_or_ma200"] is False
    assert row["squat_dca_below_ma100"] is False
    assert row["squat_last_stand_ma200"] is False
    assert row["squat_breakdown_below_ma200"] is False


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


def test_benchmark_related_trigger_disabled_for_qqq() -> None:
    qqq = _series([220.0] * 200 + [150.0 - i for i in range(60)])
    benchmark = _series([100.0 + i * 0.8 for i in range(260)])

    row = stock_signals.compute_stock_signal_row(
        "QQQ",
        daily_close=qqq,
        benchmark_ticker="QQQ",
        benchmark_close=benchmark,
    )

    assert row["benchmark_ticker"] == "QQQ"
    assert row["relative_strength_weak"] is False
    assert row["strong_sell_weak_strength"] is False
    assert row["relative_strength_reasons"] == "self_benchmark_reference_skipped"


def test_squat_ambush_triggers_near_ma100_zone_in_bull_pullback() -> None:
    closes = _series([100.0] * 200 + [112.0] * 20 + [110.0] * 39 + [111.0])
    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=109.0)

    assert row["status"] == "ok"
    assert row["squat_bull_market_precondition"] is True
    assert row["squat_price_dropping"] is True
    assert 0.02 <= float(row["squat_gap_to_sma100_pct"]) <= 0.03
    assert row["squat_ambush_near_ma100_or_ma200"] is True


def test_squat_dca_triggers_on_cross_below_ma100() -> None:
    closes = _series([100.0] * 200 + [120.0] * 30 + [118.0] * 30)
    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=109.0)

    assert row["status"] == "ok"
    assert row["squat_bull_market_precondition"] is True
    assert row["squat_dca_below_ma100"] is True
    assert float(row["price"]) < float(row["sma100"])


def test_squat_last_stand_triggers_when_price_tests_ma200() -> None:
    closes = _series([100.0 + i * 0.5 for i in range(260)])
    baseline = stock_signals.compute_stock_signal_row("TEST", closes)
    near_support_price = float(baseline["sma200"]) * 0.995
    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=near_support_price)

    assert row["status"] == "ok"
    assert row["squat_bull_market_precondition"] is True
    assert row["squat_last_stand_ma200"] is True
    assert row["squat_breakdown_below_ma200"] is False
    assert -0.02 <= float(row["squat_gap_to_sma200_pct"]) <= 0.01


def test_squat_breakdown_triggers_when_price_moves_well_below_ma200() -> None:
    closes = _series([100.0 + i * 0.5 for i in range(260)])
    baseline = stock_signals.compute_stock_signal_row("TEST", closes)
    breakdown_price = float(baseline["sma200"]) * 0.97
    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=breakdown_price)

    assert row["status"] == "ok"
    assert row["squat_bull_market_precondition"] is True
    assert row["squat_last_stand_ma200"] is False
    assert row["squat_breakdown_below_ma200"] is True
    assert float(row["squat_gap_to_sma200_pct"]) < -0.02


def test_squat_signals_blocked_without_bull_market_precondition() -> None:
    closes = _series([300.0 - i for i in range(260)])
    row = stock_signals.compute_stock_signal_row("TEST", closes)

    assert row["status"] == "ok"
    assert row["squat_bull_market_precondition"] is False
    assert row["squat_ambush_near_ma100_or_ma200"] is False
    assert row["squat_dca_below_ma100"] is False
    assert row["squat_last_stand_ma200"] is False
    assert row["squat_breakdown_below_ma200"] is False


def test_day_change_uses_intraday_price_when_available() -> None:
    closes = _series([100.0 + i for i in range(260)])
    row = stock_signals.compute_stock_signal_row("TEST", closes, latest_price=400.0)

    previous_close = float(closes.iloc[-1])
    assert row["day_change"] == pytest.approx(400.0 - previous_close)
    assert row["day_change_pct"] == pytest.approx((400.0 - previous_close) / previous_close)


def test_day_change_uses_latest_two_daily_closes_without_intraday() -> None:
    closes = _series([100.0 + i for i in range(260)])
    row = stock_signals.compute_stock_signal_row("TEST", closes)

    latest_close = float(closes.iloc[-1])
    previous_close = float(closes.iloc[-2])
    assert row["day_change"] == pytest.approx(latest_close - previous_close)
    assert row["day_change_pct"] == pytest.approx((latest_close - previous_close) / previous_close)

from __future__ import annotations

import pandas as pd

from src import stock_scanner


def _bars(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(closes), freq="D")
    if opens is None:
        opens = [float(value) - 1.0 for value in closes]
    frame = pd.DataFrame({"Open": opens, "Close": closes}, index=idx)
    frame.attrs["source"] = "TEST"
    return frame


def test_bullish_alignment_trigger_fires_on_false_to_true_transition() -> None:
    closes = [80.0] * 225 + [120.0] * 20 + [100.0] * 14 + [140.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes))

    assert row["status"] == "ok"
    assert row["bullish_alignment_active"] is True
    assert row["bullish_alignment_triggered_today"] is True
    assert float(row["sma14"]) > float(row["sma50"])
    assert float(row["sma50"]) > float(row["sma200"])


def test_bullish_alignment_trigger_does_not_repeat_on_next_day() -> None:
    closes = [80.0] * 225 + [120.0] * 20 + [100.0] * 14 + [140.0, 140.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes))

    assert row["status"] == "ok"
    assert row["bullish_alignment_active"] is True
    assert row["bullish_alignment_triggered_today"] is False


def test_bullish_alignment_accepts_50_over_100_branch() -> None:
    # This path satisfies 14>50>100 while 50<=200, which validates the OR rule.
    closes = [60.0] * 78 + [100.0] * 68 + [60.0] * 113 + [120.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes))

    assert row["status"] == "ok"
    assert row["bullish_alignment_triggered_today"] is True
    assert float(row["sma14"]) > float(row["sma50"])
    assert float(row["sma50"]) > float(row["sma100"])
    assert not (float(row["sma50"]) > float(row["sma200"]))


def test_bullish_alignment_rejects_when_longer_stack_not_confirmed() -> None:
    closes = [150.0] * 210 + [100.0] * 49 + [140.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes))

    assert row["status"] == "ok"
    assert float(row["sma14"]) > float(row["sma50"])
    assert float(row["sma50"]) < float(row["sma100"])
    assert float(row["sma50"]) < float(row["sma200"])
    assert row["bullish_alignment_active"] is False
    assert row["bullish_alignment_triggered_today"] is False


def test_recovery_momentum_triggers_on_close_cross_above_sma50_with_three_bullish_candles() -> None:
    closes = [100.0] * 257 + [98.0, 99.0, 101.0]
    opens = [99.0] * 257 + [97.0, 98.0, 100.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes, opens=opens))

    assert row["status"] == "ok"
    assert row["recovery_close_cross_sma50_today"] is True
    assert row["recovery_three_bullish_candles_today"] is True
    assert row["recovery_momentum_triggered_today"] is True


def test_recovery_momentum_does_not_trigger_when_three_bullish_candles_fail() -> None:
    closes = [100.0] * 257 + [98.0, 99.0, 101.0]
    opens = [99.0] * 257 + [97.0, 100.0, 100.0]  # middle candle not bullish
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes, opens=opens))

    assert row["status"] == "ok"
    assert row["recovery_close_cross_sma50_today"] is True
    assert row["recovery_three_bullish_candles_today"] is False
    assert row["recovery_momentum_triggered_today"] is False


def test_recovery_momentum_does_not_trigger_when_price_was_already_above_sma50() -> None:
    closes = [100.0] * 257 + [101.0, 102.0, 103.0]
    opens = [99.0] * 257 + [100.0, 101.0, 102.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes, opens=opens))

    assert row["status"] == "ok"
    assert row["recovery_close_cross_sma50_today"] is False
    assert row["recovery_three_bullish_candles_today"] is True
    assert row["recovery_momentum_triggered_today"] is False


def test_ambush_squat_triggers_when_entering_band_with_bullish_trend() -> None:
    closes = [80.0] * 210 + [120.0] * 48 + [130.0, 101.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes))

    assert row["status"] == "ok"
    assert row["ambush_trend_bullish_active"] is True
    assert row["ambush_near_ma100_active"] is True
    assert row["ambush_squat_active"] is True
    assert row["ambush_squat_triggered_today"] is True


def test_ambush_squat_does_not_retrigger_while_staying_in_band() -> None:
    closes = [80.0] * 210 + [120.0] * 48 + [101.0, 101.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes))

    assert row["status"] == "ok"
    assert row["ambush_squat_active"] is True
    assert row["ambush_squat_triggered_today"] is False


def test_ambush_squat_retriggers_after_exiting_and_reentering_band() -> None:
    closes = [80.0] * 210 + [120.0] * 47 + [101.0, 130.0, 101.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes))

    assert row["status"] == "ok"
    assert row["ambush_squat_active"] is True
    assert row["ambush_squat_triggered_today"] is True


def test_ambush_squat_does_not_trigger_above_band_or_below_ma() -> None:
    above_band = [80.0] * 210 + [120.0] * 48 + [130.0, 103.0]
    below_ma = [80.0] * 210 + [120.0] * 48 + [130.0, 99.0]

    above_row = stock_scanner.compute_scanner_signal_row("TEST", _bars(above_band))
    below_row = stock_scanner.compute_scanner_signal_row("TEST", _bars(below_ma))

    assert above_row["status"] == "ok"
    assert above_row["ambush_squat_active"] is False
    assert above_row["ambush_squat_triggered_today"] is False

    assert below_row["status"] == "ok"
    assert below_row["ambush_squat_active"] is False
    assert below_row["ambush_squat_triggered_today"] is False


def test_scanner_marks_insufficient_data_when_less_than_201_bars() -> None:
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars([100.0 + i for i in range(200)]))

    assert row["status"] == "insufficient_data"
    for key in [
        "bullish_alignment_active",
        "bullish_alignment_triggered_today",
        "recovery_momentum_triggered_today",
        "ambush_squat_active",
        "ambush_squat_triggered_today",
    ]:
        assert row[key] is False


def test_scanner_marks_insufficient_data_when_recent_opens_missing() -> None:
    closes = [100.0] * 257 + [98.0, 99.0, 101.0]
    opens = [99.0] * 257 + [97.0, None, 100.0]
    row = stock_scanner.compute_scanner_signal_row("TEST", _bars(closes, opens=opens))

    assert row["status"] == "insufficient_data"
    assert "Missing open prices" in str(row["status_message"])
    assert row["recovery_momentum_triggered_today"] is False


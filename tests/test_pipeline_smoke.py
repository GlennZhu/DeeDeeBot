from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import pipeline
from src.stock_signals import STOCK_TRIGGER_COLUMNS


def _stock_row(ticker: str, **triggers: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": ticker,
        "as_of_date": "2026-02-02",
        "price": 100.0,
        "status": "ok",
    }
    for trigger_col in STOCK_TRIGGER_COLUMNS:
        row[trigger_col] = False
    row.update(triggers)
    return row


def test_detect_new_threshold_events_flags_only_new_trigger() -> None:
    previous = pd.DataFrame(
        [
            {
                "metric_key": "ten_year_yield",
                "metric_name": "10Y Treasury Yield",
                "signal_state": "equity_pressure_zone",
                "value": 4.5,
                "as_of_date": "2026-02-01",
            },
            {
                "metric_key": "hiring_rate",
                "metric_name": "Hiring Rate",
                "signal_state": "normal",
                "value": 3.6,
                "as_of_date": "2026-02-01",
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "metric_key": "ten_year_yield",
                "metric_name": "10Y Treasury Yield",
                "signal_state": "extreme_pressure_bond_opportunity",
                "value": 5.0,
                "as_of_date": "2026-02-02",
            },
            {
                "metric_key": "hiring_rate",
                "metric_name": "Hiring Rate",
                "signal_state": "recession_alert",
                "value": 3.4,
                "as_of_date": "2026-02-02",
            },
            {
                "metric_key": "buffett_ratio",
                "metric_name": "Buffett Indicator",
                "signal_state": "overheat_peak_risk",
                "value": 2.1,
                "as_of_date": "2026-02-02",
            },
        ]
    )

    events = pipeline._detect_new_threshold_events(previous, current)

    assert len(events) == 2
    by_metric = {event["metric_key"]: event for event in events}
    assert by_metric["ten_year_yield"]["new_threshold_ids"] == ["ten_year_yield_gte_5_0"]
    assert by_metric["hiring_rate"]["new_threshold_ids"] == ["hiring_rate_lte_3_4"]


def test_detect_new_threshold_events_includes_negative_clears_only() -> None:
    previous = pd.DataFrame(
        [
            {
                "metric_key": "ten_year_yield",
                "metric_name": "10Y Treasury Yield",
                "signal_state": "extreme_pressure_bond_opportunity",
                "value": 5.1,
                "as_of_date": "2026-02-01",
            },
            {
                "metric_key": "m2",
                "metric_name": "M2 Money Supply",
                "signal_state": "long_environment",
                "value": 100.0,
                "as_of_date": "2026-02-01",
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "metric_key": "ten_year_yield",
                "metric_name": "10Y Treasury Yield",
                "signal_state": "equity_pressure_zone",
                "value": 4.8,
                "as_of_date": "2026-02-02",
            },
            {
                "metric_key": "m2",
                "metric_name": "M2 Money Supply",
                "signal_state": "caution_contraction",
                "value": 99.0,
                "as_of_date": "2026-02-02",
            },
        ]
    )

    events = pipeline._detect_new_threshold_events(previous, current)

    assert len(events) == 1
    assert events[0]["metric_key"] == "ten_year_yield"
    assert events[0]["new_threshold_ids"] == []
    assert events[0]["cleared_threshold_ids"] == ["ten_year_yield_gte_5_0"]


def test_detect_new_stock_trigger_events_flags_false_to_true_only() -> None:
    previous = pd.DataFrame(
        [
            _stock_row(
                "GOOG",
                entry_bullish_alignment=False,
                exit_price_below_sma50=True,
                exit_death_cross_50_lt_200=False,
                exit_rsi_overbought=False,
                rsi_bearish_divergence=False,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _stock_row(
                "GOOG",
                entry_bullish_alignment=True,
                exit_price_below_sma50=True,
                exit_death_cross_50_lt_200=False,
                exit_rsi_overbought=False,
                rsi_bearish_divergence=True,
            )
        ]
    )

    events = pipeline._detect_new_stock_trigger_events(previous, current)
    trigger_ids = sorted(event["trigger_id"] for event in events)

    assert trigger_ids == ["entry_bullish_alignment", "rsi_bearish_divergence"]


def test_detect_new_stock_trigger_events_includes_negative_clear_only() -> None:
    previous = pd.DataFrame(
        [
            _stock_row(
                "GOOG",
                entry_bullish_alignment=True,
                exit_price_below_sma50=True,
                squat_ambush_near_ma100_or_ma200=True,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _stock_row(
                "GOOG",
                entry_bullish_alignment=False,
                exit_price_below_sma50=False,
                squat_ambush_near_ma100_or_ma200=False,
            )
        ]
    )

    events = pipeline._detect_new_stock_trigger_events(previous, current)

    assert len(events) == 1
    assert events[0]["trigger_id"] == "exit_price_below_sma50"
    assert events[0]["event_type"] == "cleared"


def test_detect_new_stock_trigger_events_skips_benchmark_related_alerts_for_qqq() -> None:
    previous = pd.DataFrame(
        [
            _stock_row(
                "QQQ",
                strong_sell_weak_strength=False,
                exit_price_below_sma50=False,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _stock_row(
                "QQQ",
                strong_sell_weak_strength=True,
                exit_price_below_sma50=True,
            )
        ]
    )

    events = pipeline._detect_new_stock_trigger_events(previous, current)
    trigger_ids = [event["trigger_id"] for event in events]

    assert trigger_ids == ["exit_price_below_sma50"]


def test_update_signal_event_history_writes_macro_rows_for_triggered_and_cleared(tmp_path: Path) -> None:
    path = tmp_path / "signal_events_7d.csv"
    macro_events = [
        {
            "metric_key": "ten_year_yield",
            "metric_name": "10Y Treasury Yield",
            "as_of_date": "2026-02-10",
            "value": 4.9,
            "signal_state": "equity_pressure_zone",
            "prev_signal_state": "extreme_pressure_bond_opportunity",
            "new_threshold_ids": ["ten_year_yield_gte_4_4"],
            "new_threshold_labels": ["10Y yield >= 4.4"],
            "cleared_threshold_ids": ["ten_year_yield_gte_5_0"],
            "cleared_threshold_labels": ["10Y yield >= 5.0"],
        }
    ]

    pipeline._update_signal_event_history(
        path,
        macro_events=macro_events,
        stock_events=[],
        now_iso="2026-02-10T00:00:00Z",
    )

    events = pd.read_csv(path, keep_default_na=False)

    assert len(events) == 2
    assert set(events["domain"].tolist()) == {"macro"}
    assert set(events["event_type"].tolist()) == {"triggered", "cleared"}
    assert set(events["signal_id"].tolist()) == {"ten_year_yield_gte_4_4", "ten_year_yield_gte_5_0"}
    assert set(events["state_transition"].tolist()) == {"extreme_pressure_bond_opportunity -> equity_pressure_zone"}
    assert set(events["subject_id"].tolist()) == {"ten_year_yield"}
    assert all(bool(value) for value in events["event_timestamp_et"].tolist())


def test_update_signal_event_history_writes_stock_rows_for_triggered_and_cleared(tmp_path: Path) -> None:
    path = tmp_path / "signal_events_7d.csv"
    stock_events = [
        {
            "ticker": "NVDA",
            "benchmark_ticker": "QQQ",
            "trigger_id": "strong_sell_weak_strength",
            "trigger_label": "Strong Sell: Underperforming vs benchmark",
            "event_type": "triggered",
            "as_of_date": "2026-02-10",
            "price": 120.5,
            "status": "ok",
            "relative_strength_reasons": "RS_TREND_DOWN",
        },
        {
            "ticker": "NVDA",
            "benchmark_ticker": "QQQ",
            "trigger_id": "exit_price_below_sma50",
            "trigger_label": "Exit: Price Below SMA50",
            "event_type": "cleared",
            "as_of_date": "2026-02-10",
            "price": 121.0,
            "status": "ok",
            "relative_strength_reasons": "",
        },
    ]

    pipeline._update_signal_event_history(
        path,
        macro_events=[],
        stock_events=stock_events,
        now_iso="2026-02-10T00:00:00Z",
    )

    events = pd.read_csv(path, keep_default_na=False)

    assert len(events) == 2
    assert set(events["domain"].tolist()) == {"stock"}
    assert set(events["event_type"].tolist()) == {"triggered", "cleared"}
    assert set(events["subject_id"].tolist()) == {"NVDA"}
    assert set(events["benchmark_ticker"].tolist()) == {"QQQ"}
    assert set(events["signal_id"].tolist()) == {"strong_sell_weak_strength", "exit_price_below_sma50"}
    strong_sell = events[events["signal_id"] == "strong_sell_weak_strength"].iloc[0]
    non_strong_sell = events[events["signal_id"] == "exit_price_below_sma50"].iloc[0]
    assert strong_sell["details"] == "RS_TREND_DOWN"
    assert non_strong_sell["details"] == ""


def test_update_signal_event_history_prunes_to_exact_7_day_window(tmp_path: Path) -> None:
    path = tmp_path / "signal_events_7d.csv"

    def _row(timestamp_utc: str) -> dict[str, object]:
        return {
            "event_timestamp_utc": timestamp_utc,
            "event_timestamp_et": "2026-02-01 03:00:00 EST",
            "domain": "macro",
            "event_type": "triggered",
            "subject_id": "hiring_rate",
            "subject_name": "Hiring Rate",
            "benchmark_ticker": "",
            "signal_id": "hiring_rate_lte_3_4",
            "signal_label": "Hiring rate <= 3.4",
            "as_of_date": "2026-02-01",
            "value": 3.4,
            "price": "",
            "state_transition": "normal -> recession_alert",
            "status": "",
            "details": "",
        }

    seed = pd.DataFrame(
        [
            _row("2026-02-02T23:59:59Z"),
            _row("2026-02-03T00:00:00Z"),
            _row("2026-02-09T00:00:00Z"),
        ],
        columns=pipeline.SIGNAL_EVENT_COLUMNS,
    )
    seed.to_csv(path, index=False)

    pipeline._update_signal_event_history(
        path,
        macro_events=[],
        stock_events=[],
        now_iso="2026-02-10T00:00:00Z",
    )

    events = pd.read_csv(path, keep_default_na=False)

    assert events["event_timestamp_utc"].tolist() == ["2026-02-03T00:00:00Z", "2026-02-09T00:00:00Z"]


def test_update_signal_event_history_creates_empty_csv_with_headers_when_no_events(tmp_path: Path) -> None:
    path = tmp_path / "signal_events_7d.csv"

    pipeline._update_signal_event_history(
        path,
        macro_events=[],
        stock_events=[],
        now_iso="2026-02-10T00:00:00Z",
    )

    assert path.exists()
    events = pd.read_csv(path)
    assert events.empty
    assert events.columns.tolist() == pipeline.SIGNAL_EVENT_COLUMNS

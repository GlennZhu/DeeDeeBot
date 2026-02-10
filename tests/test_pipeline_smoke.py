from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import pipeline
from src.stock_signals import STOCK_TRIGGER_COLUMNS


def _series(values: list[float], start: str, freq: str) -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


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


def test_pipeline_generates_expected_csv_contracts(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    derived_dir = tmp_path / "data" / "derived"
    monkeypatch.setattr(pipeline, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(pipeline, "DERIVED_DATA_DIR", derived_dir)
    monkeypatch.setattr(pipeline, "STOCK_WATCHLIST_PATH", derived_dir / "stock_watchlist.csv")
    monkeypatch.setattr(pipeline, "STOCK_SIGNALS_PATH", derived_dir / "stock_signals_latest.csv")

    def fake_fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
        del start_date
        mapping = {
            "M2SL": _series([100.0 + i for i in range(24)], "2024-01-31", "M"),
            "JTSHIR": _series([3.8, 3.7, 3.6], "2025-10-31", "M"),
            "UNRATE": _series([4.0, 4.1, 4.2], "2025-10-31", "M"),
            "NCBCEL": _series([50_000.0, 52_000.0, 54_000.0], "2025-10-31", "M"),
            "GDP": _series([25.0, 26.0], "2025-09-30", "Q"),
            "DGS10": _series([4.2, 4.3, 4.4], "2025-12-29", "D"),
        }
        return mapping[series_id]

    def fake_fetch_stock_daily_history(ticker: str, start_date: str) -> pd.Series:
        del start_date
        base = {
            "GOOG": 120.0,
            "AVGO": 140.0,
            "NVDA": 160.0,
            "MSFT": 180.0,
        }[ticker]
        return _series([base + i * 0.4 for i in range(260)], "2025-01-02", "D")

    monkeypatch.setattr(pipeline, "fetch_fred_series", fake_fetch_fred_series)
    monkeypatch.setattr(pipeline, "fetch_stock_daily_history", fake_fetch_stock_daily_history)
    monkeypatch.setattr(pipeline, "fetch_stock_intraday_latest", lambda ticker: 250.0 if ticker == "NVDA" else None)

    pipeline.run_pipeline(start_date="2024-01-01", lookback_years=15)

    raw_files = {
        "m2.csv",
        "hiring_rate.csv",
        "ten_year_yield.csv",
        "buffett_ratio.csv",
        "unemployment_rate.csv",
    }
    assert raw_files.issubset({f.name for f in raw_dir.glob("*.csv")})

    metric_snapshot = pd.read_csv(derived_dir / "metric_snapshot.csv")
    assert {
        "metric_key",
        "as_of_date",
        "value",
        "source",
        "stale_days",
        "last_updated_utc",
    }.issubset(set(metric_snapshot.columns))

    signals = pd.read_csv(derived_dir / "signals_latest.csv")
    assert {
        "metric_key",
        "metric_name",
        "as_of_date",
        "value",
        "signal_state",
        "threshold_rule",
        "message",
        "source",
        "stale_days",
        "last_updated_utc",
    }.issubset(set(signals.columns))

    watchlist = pd.read_csv(derived_dir / "stock_watchlist.csv")
    assert watchlist["ticker"].tolist() == ["GOOG", "AVGO", "NVDA", "MSFT"]

    stock_signals = pd.read_csv(derived_dir / "stock_signals_latest.csv")
    assert {
        "ticker",
        "as_of_date",
        "price",
        "sma14",
        "sma50",
        "sma100",
        "sma200",
        "rsi14",
        "entry_bullish_alignment",
        "exit_price_below_sma50",
        "exit_death_cross_50_lt_100",
        "exit_death_cross_50_lt_200",
        "exit_rsi_overbought",
        "rsi_bearish_divergence",
        "source",
        "stale_days",
        "status",
        "status_message",
        "last_updated_utc",
    }.issubset(set(stock_signals.columns))
    assert set(stock_signals["ticker"].tolist()) == {"GOOG", "AVGO", "NVDA", "MSFT"}


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


def test_detect_new_stock_trigger_events_flags_false_to_true_only() -> None:
    previous = pd.DataFrame(
        [
            _stock_row(
                "GOOG",
                entry_bullish_alignment=False,
                exit_price_below_sma50=True,
                exit_death_cross_50_lt_100=False,
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
                exit_death_cross_50_lt_100=False,
                exit_death_cross_50_lt_200=False,
                exit_rsi_overbought=False,
                rsi_bearish_divergence=True,
            )
        ]
    )

    events = pipeline._detect_new_stock_trigger_events(previous, current)
    trigger_ids = sorted(event["trigger_id"] for event in events)

    assert trigger_ids == ["entry_bullish_alignment", "rsi_bearish_divergence"]


def test_notify_new_thresholds_posts_to_discord(monkeypatch) -> None:
    previous = pd.DataFrame(
        [
            {
                "metric_key": "hiring_rate",
                "metric_name": "Hiring Rate",
                "signal_state": "normal",
                "value": 3.6,
                "as_of_date": "2026-02-01",
            }
        ]
    )
    current = pd.DataFrame(
        [
            {
                "metric_key": "hiring_rate",
                "metric_name": "Hiring Rate",
                "signal_state": "recession_alert",
                "value": 3.4,
                "as_of_date": "2026-02-02",
            }
        ]
    )

    payload: dict[str, str] = {}

    def fake_post(url: str, content: str) -> None:
        payload["url"] = url
        payload["content"] = content

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.com/webhook")
    monkeypatch.setattr(pipeline, "_post_discord_message", fake_post)

    pipeline._notify_new_thresholds(previous, current, "2026-02-02T00:00:00Z")

    assert payload["url"] == "https://example.com/webhook"
    assert "Hiring Rate" in payload["content"]
    assert "Hiring rate <= 3.4" in payload["content"]


def test_notify_new_stock_triggers_posts_to_discord(monkeypatch) -> None:
    previous = pd.DataFrame(
        [
            _stock_row(
                "NVDA",
                entry_bullish_alignment=False,
                exit_price_below_sma50=False,
                exit_death_cross_50_lt_100=False,
                exit_death_cross_50_lt_200=False,
                exit_rsi_overbought=False,
                rsi_bearish_divergence=False,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _stock_row(
                "NVDA",
                entry_bullish_alignment=True,
                exit_price_below_sma50=False,
                exit_death_cross_50_lt_100=False,
                exit_death_cross_50_lt_200=False,
                exit_rsi_overbought=False,
                rsi_bearish_divergence=False,
            )
        ]
    )

    payload: dict[str, str] = {}

    def fake_post(url: str, content: str) -> None:
        payload["url"] = url
        payload["content"] = content

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.com/webhook")
    monkeypatch.setattr(pipeline, "_post_discord_message", fake_post)

    pipeline._notify_new_stock_triggers(previous, current, "2026-02-02T00:00:00Z")

    assert payload["url"] == "https://example.com/webhook"
    assert "NVDA" in payload["content"]
    assert "Entry signal: bullish alignment" in payload["content"]

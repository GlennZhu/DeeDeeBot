from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import pipeline
from src.stock_signals import STOCK_TRIGGER_COLUMNS


def _series(values: list[float], start: str, freq: str) -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def _bars_frame(closes: list[float], start: str = "2025-01-02") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(closes), freq="D")
    frame = pd.DataFrame(
        {
            "Open": [float(value) - 1.0 for value in closes],
            "Close": [float(value) for value in closes],
        },
        index=idx,
    )
    frame.attrs["source"] = "TEST_BARS"
    return frame


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
    monkeypatch.setattr(pipeline, "STOCK_UNIVERSE_PATH", derived_dir / "stock_universe.csv")
    monkeypatch.setattr(pipeline, "SCANNER_SIGNALS_PATH", derived_dir / "scanner_signals_latest.csv")
    monkeypatch.setattr(pipeline, "SCANNER_THESIS_PATH", derived_dir / "scanner_thesis_tags.csv")
    monkeypatch.setattr(pipeline, "SIGNAL_EVENTS_PATH", derived_dir / "signal_events_7d.csv")

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
            "QQQ": 200.0,
            "META": 170.0,
            "AMZN": 130.0,
        }[ticker]
        return _series([base + i * 0.4 for i in range(260)], "2025-01-02", "D")

    intraday_calls: list[str] = []

    def fake_fetch_stock_intraday_quote(ticker: str) -> dict[str, object] | None:
        intraday_calls.append(ticker)
        if ticker == "NVDA":
            return {
                "price": 250.0,
                "quote_timestamp_utc": "2026-02-10T21:30:00Z",
                "quote_age_seconds": 30,
                "source": "STOOQ_INTRADAY:NVDA.US",
            }
        if ticker == "QQQ":
            return {
                "price": 400.0,
                "quote_timestamp_utc": "2026-02-10T21:30:00Z",
                "quote_age_seconds": 30,
                "source": "STOOQ_INTRADAY:QQQ.US",
            }
        return None

    def fake_fetch_stock_universe_snapshot() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": "META",
                    "security_name": "Meta Platforms, Inc.",
                    "exchange": "NASDAQ Global Select",
                    "is_etf": False,
                    "source": "TEST_UNIVERSE",
                },
                {
                    "ticker": "AMZN",
                    "security_name": "Amazon.com, Inc.",
                    "exchange": "NASDAQ Global Select",
                    "is_etf": False,
                    "source": "TEST_UNIVERSE",
                },
            ]
        )

    scanner_bars_by_ticker = {
        # Bullish alignment trigger today.
        "GOOG": _bars_frame([80.0] * 225 + [120.0] * 20 + [100.0] * 14 + [140.0]),
        # Recovery + momentum trigger today.
        "META": _bars_frame([100.0] * 257 + [98.0, 99.0, 101.0]),
        # Ambush / squat trigger today.
        "AMZN": _bars_frame([80.0] * 210 + [120.0] * 48 + [130.0, 101.0]),
    }

    def fake_fetch_stock_daily_bars(ticker: str, start_date: str) -> pd.DataFrame:
        del start_date
        if ticker in scanner_bars_by_ticker:
            return scanner_bars_by_ticker[ticker].copy()
        return _bars_frame([100.0] * 260)

    def fake_fetch_stock_daily_bars_batch_yfinance(tickers: list[str], start_date: str) -> dict[str, pd.DataFrame]:
        del start_date
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            ticker_key = str(ticker).upper()
            if ticker_key in scanner_bars_by_ticker:
                out[ticker_key] = scanner_bars_by_ticker[ticker_key].copy()
        return out

    monkeypatch.setattr(pipeline, "fetch_fred_series", fake_fetch_fred_series)
    monkeypatch.setattr(pipeline, "fetch_stock_daily_history", fake_fetch_stock_daily_history)
    monkeypatch.setattr(pipeline, "fetch_stock_daily_history_batch_yfinance", lambda tickers, start_date: {})
    monkeypatch.setattr(pipeline, "fetch_stock_daily_bars", fake_fetch_stock_daily_bars)
    monkeypatch.setattr(pipeline, "fetch_stock_daily_bars_batch_yfinance", fake_fetch_stock_daily_bars_batch_yfinance)
    monkeypatch.setattr(pipeline, "fetch_stock_intraday_quote", fake_fetch_stock_intraday_quote)
    monkeypatch.setattr(pipeline, "fetch_stock_universe_snapshot", fake_fetch_stock_universe_snapshot)
    monkeypatch.setattr(
        pipeline,
        "fetch_sp500_constituents",
        lambda: pd.DataFrame([{"ticker": "META", "source": "WIKIPEDIA_SP500"}]),
    )

    pipeline.run_pipeline(start_date="2024-01-01", lookback_years=15, scanner_max_tickers=7)

    assert "META" not in intraday_calls
    assert "AMZN" not in intraday_calls

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
    assert {"ticker", "benchmark"}.issubset(set(watchlist.columns))
    assert watchlist["ticker"].tolist() == ["GOOG", "AVGO", "NVDA", "MSFT", "QQQ"]
    assert watchlist["benchmark"].tolist() == ["QQQ", "QQQ", "QQQ", "QQQ", "QQQ"]

    stock_signals = pd.read_csv(derived_dir / "stock_signals_latest.csv")
    assert {
        "ticker",
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
        "squat_bull_market_precondition",
        "squat_price_dropping",
        "squat_gap_to_sma100_pct",
        "squat_gap_to_sma200_pct",
        "rsi14",
        "benchmark_ticker",
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
        "entry_bullish_alignment",
        "exit_price_below_sma50",
        "exit_death_cross_50_lt_100",
        "exit_death_cross_50_lt_200",
        "exit_rsi_overbought",
        "rsi_bearish_divergence",
        "strong_sell_weak_strength",
        "squat_ambush_near_ma100_or_ma200",
        "squat_dca_below_ma100",
        "squat_last_stand_ma200",
        "squat_breakdown_below_ma200",
        "source",
        "stale_days",
        "status",
        "status_message",
        "last_updated_utc",
    }.issubset(set(stock_signals.columns))
    assert set(stock_signals["ticker"].tolist()) == {"GOOG", "AVGO", "NVDA", "MSFT", "QQQ"}

    stock_universe = pd.read_csv(derived_dir / "stock_universe.csv")
    assert {
        "ticker",
        "security_name",
        "exchange",
        "is_etf",
        "universe_source",
        "is_watchlist",
        "last_refreshed_utc",
    }.issubset(set(stock_universe.columns))
    assert {"GOOG", "AVGO", "NVDA", "MSFT", "QQQ", "META", "AMZN"}.issubset(set(stock_universe["ticker"].tolist()))

    scanner_signals = pd.read_csv(derived_dir / "scanner_signals_latest.csv")
    assert {
        "ticker",
        "security_name",
        "exchange",
        "universe_source",
        "is_watchlist",
        "price",
        "open",
        "sma14",
        "sma50",
        "sma100",
        "sma200",
        "bullish_alignment_active",
        "bullish_alignment_triggered_today",
        "recovery_close_cross_sma50_today",
        "recovery_three_bullish_candles_today",
        "recovery_momentum_triggered_today",
        "ambush_trend_bullish_active",
        "ambush_squat_active",
        "ambush_squat_triggered_today",
        "status",
        "last_updated_utc",
    }.issubset(set(scanner_signals.columns))
    assert {"GOOG", "AVGO", "NVDA", "MSFT", "QQQ", "META", "AMZN"}.issubset(set(scanner_signals["ticker"].tolist()))
    assert bool(scanner_signals["bullish_alignment_triggered_today"].fillna(False).astype(bool).any())
    assert bool(scanner_signals["recovery_momentum_triggered_today"].fillna(False).astype(bool).any())
    assert bool(scanner_signals["ambush_squat_triggered_today"].fillna(False).astype(bool).any())

    thesis = pd.read_csv(derived_dir / "scanner_thesis_tags.csv")
    assert thesis.columns.tolist() == pipeline.SCANNER_THESIS_COLUMNS
    assert {"GOOG", "AVGO", "NVDA", "MSFT", "QQQ"}.issubset(set(thesis["ticker"].tolist()))

    signal_events = pd.read_csv(derived_dir / "signal_events_7d.csv")
    assert signal_events.columns.tolist() == pipeline.SIGNAL_EVENT_COLUMNS
    if not signal_events.empty:
        assert signal_events["event_timestamp_utc"].notna().all()
        assert set(signal_events["event_type"].dropna().tolist()).issubset({"triggered", "cleared"})
        assert "scanner" in set(signal_events["domain"].dropna().astype(str).str.lower().tolist())
        scanner_rows = signal_events[signal_events["domain"].astype(str).str.lower() == "scanner"]
        assert set(scanner_rows["event_type"].astype(str).str.lower().unique().tolist()) <= {"triggered"}
        assert {
            "bullish_alignment_trigger",
            "recovery_momentum_trigger",
            "ambush_squat_trigger",
        }.issubset(set(scanner_rows["signal_id"].astype(str).tolist()))


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


def test_select_scanner_universe_all_tickers_includes_etfs_when_enabled() -> None:
    universe = pd.DataFrame(
        [
            {
                "ticker": "QQQ",
                "security_name": "Invesco QQQ",
                "exchange": "NASDAQ",
                "is_etf": True,
                "universe_source": "test",
                "is_watchlist": True,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "AAPL",
                "security_name": "Apple Inc.",
                "exchange": "NASDAQ",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "SPY",
                "security_name": "SPDR S&P 500 ETF Trust",
                "exchange": "NYSE Arca",
                "is_etf": True,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
        ]
    )

    selected = pipeline._select_scanner_universe(
        universe,
        max_tickers=None,
        include_etfs=True,
    )

    assert set(selected["ticker"].tolist()) == {"QQQ", "AAPL", "SPY"}


def test_select_scanner_universe_capped_excludes_non_watchlist_etfs_by_default() -> None:
    universe = pd.DataFrame(
        [
            {
                "ticker": "QQQ",
                "security_name": "Invesco QQQ",
                "exchange": "NASDAQ",
                "is_etf": True,
                "universe_source": "test",
                "is_watchlist": True,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "AAPL",
                "security_name": "Apple Inc.",
                "exchange": "NASDAQ",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "SPY",
                "security_name": "SPDR S&P 500 ETF Trust",
                "exchange": "NYSE Arca",
                "is_etf": True,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
        ]
    )

    selected = pipeline._select_scanner_universe(
        universe,
        max_tickers=3,
        include_etfs=False,
    )

    assert set(selected["ticker"].tolist()) == {"QQQ", "AAPL"}


def test_select_scanner_universe_cap_applies_beyond_watchlist_not_total() -> None:
    universe = pd.DataFrame(
        [
            {
                "ticker": "W1",
                "security_name": "Watch 1",
                "exchange": "NASDAQ",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": True,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "W2",
                "security_name": "Watch 2",
                "exchange": "NASDAQ",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": True,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "AAPL",
                "security_name": "Apple Inc.",
                "exchange": "NASDAQ",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "MSFT",
                "security_name": "Microsoft",
                "exchange": "NASDAQ",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
        ]
    )

    selected = pipeline._select_scanner_universe(
        universe,
        max_tickers=1,
        include_etfs=False,
    )

    # Both watchlist names stay pinned, plus 1 extra non-watchlist row.
    assert {"W1", "W2"}.issubset(set(selected["ticker"].tolist()))
    assert len(selected) == 3


def test_apply_scanner_scope_limits_to_sp500_nasdaq500_and_watchlist(monkeypatch) -> None:
    universe = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "security_name": "Apple Inc.",
                "exchange": "NASDAQ Global Select",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "ABC",
                "security_name": "Abc Corp",
                "exchange": "NASDAQ Global Market",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "XYZ",
                "security_name": "Xyz Inc.",
                "exchange": "NYSE",
                "is_etf": False,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "IWM",
                "security_name": "ETF Example",
                "exchange": "NASDAQ Global Select",
                "is_etf": True,
                "universe_source": "test",
                "is_watchlist": False,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
            {
                "ticker": "WLT",
                "security_name": "Watchlist Name",
                "exchange": "",
                "is_etf": False,
                "universe_source": "watchlist_seed",
                "is_watchlist": True,
                "last_refreshed_utc": "2026-02-02T00:00:00Z",
            },
        ]
    )
    watchlist = pd.DataFrame([{"ticker": "WLT", "benchmark": "QQQ"}])
    monkeypatch.setattr(
        pipeline,
        "fetch_sp500_constituents",
        lambda: pd.DataFrame([{"ticker": "XYZ", "source": "WIKIPEDIA_SP500"}]),
    )

    scoped = pipeline._apply_scanner_scope_sp500_nasdaq500(universe=universe, watchlist=watchlist)

    assert set(scoped["ticker"].tolist()) == {"AAPL", "ABC", "XYZ", "WLT"}


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
        scanner_events=[],
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
        scanner_events=[],
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


def test_update_signal_event_history_writes_scanner_rows_for_triggered_events(tmp_path: Path) -> None:
    path = tmp_path / "signal_events_7d.csv"
    scanner_events = [
        {
            "ticker": "GOOG",
            "security_name": "Alphabet Inc.",
            "signal_id": "bullish_alignment_trigger",
            "signal_label": "Bullish Alignment Trigger",
            "event_type": "triggered",
            "as_of_date": "2026-02-10",
            "price": 140.0,
            "status": "ok",
            "details": "sma14=102.8; sma50=102.7; sma100=91.4; sma200=85.7",
        }
    ]

    pipeline._update_signal_event_history(
        path,
        macro_events=[],
        stock_events=[],
        scanner_events=scanner_events,
        now_iso="2026-02-10T00:00:00Z",
    )

    events = pd.read_csv(path, keep_default_na=False)

    assert len(events) == 1
    row = events.iloc[0]
    assert row["domain"] == "scanner"
    assert row["event_type"] == "triggered"
    assert row["subject_id"] == "GOOG"
    assert row["subject_name"] == "Alphabet Inc."
    assert row["signal_id"] == "bullish_alignment_trigger"
    assert row["details"] != ""


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
        scanner_events=[],
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
        scanner_events=[],
        now_iso="2026-02-10T00:00:00Z",
    )

    assert path.exists()
    events = pd.read_csv(path)
    assert events.empty
    assert events.columns.tolist() == pipeline.SIGNAL_EVENT_COLUMNS


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


def test_notify_new_thresholds_posts_clear_to_discord(monkeypatch) -> None:
    previous = pd.DataFrame(
        [
            {
                "metric_key": "hiring_rate",
                "metric_name": "Hiring Rate",
                "signal_state": "recession_alert",
                "value": 3.4,
                "as_of_date": "2026-02-01",
            }
        ]
    )
    current = pd.DataFrame(
        [
            {
                "metric_key": "hiring_rate",
                "metric_name": "Hiring Rate",
                "signal_state": "normal",
                "value": 3.6,
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
    assert "cleared: Hiring rate <= 3.4" in payload["content"]
    assert "state recession_alert -> normal" in payload["content"]


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
    assert "Entry: Trend alignment (SMA14 > SMA50 > SMA100/200)" in payload["content"]


def test_notify_new_stock_triggers_posts_clear_to_discord(monkeypatch) -> None:
    previous = pd.DataFrame(
        [
            _stock_row(
                "NVDA",
                exit_price_below_sma50=True,
            )
        ]
    )
    current = pd.DataFrame(
        [
            _stock_row(
                "NVDA",
                exit_price_below_sma50=False,
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
    assert "Exit: Price Below SMA50 cleared" in payload["content"]


def test_post_discord_message_sets_cloudflare_compatible_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    def fake_urlopen(req, timeout: int = 10):
        captured["req"] = req
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(pipeline.request, "urlopen", fake_urlopen)

    pipeline._post_discord_message("https://example.com/webhook", "hello")

    req = captured["req"]
    assert req.full_url == "https://example.com/webhook"
    assert req.get_method() == "POST"
    assert captured["timeout"] == 10

    header_map = {k.lower(): v for k, v in req.header_items()}
    assert header_map["content-type"] == "application/json"
    assert header_map["accept"] == "application/json"
    assert header_map["user-agent"].startswith("BingBingBot/")
    assert json.loads(req.data.decode("utf-8")) == {"content": "hello"}

"""Data pipeline for fetching macro series, computing signals, and writing CSV caches."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

import pandas as pd

from src.config import DEFAULT_LOOKBACK_YEARS, METRIC_ORDER, SERIES_CONFIG
from src.data_fetch import (
    fetch_fred_series,
    fetch_stock_daily_history,
    fetch_stock_intraday_latest,
)
from src.signals import compute_signals
from src.stock_signals import (
    STOCK_SIGNAL_COLUMNS,
    STOCK_TRIGGER_COLUMNS,
    compute_stock_signal_row,
)
from src.transform import build_buffett_ratio, prepare_monthly_series

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVED_DATA_DIR = PROJECT_ROOT / "data" / "derived"
STOCK_WATCHLIST_PATH = DERIVED_DATA_DIR / "stock_watchlist.csv"
STOCK_SIGNALS_PATH = DERIVED_DATA_DIR / "stock_signals_latest.csv"
DEFAULT_STOCK_WATCHLIST = ["GOOG", "AVGO", "NVDA", "MSFT"]

THRESHOLD_TRIGGER_MAP: dict[str, dict[str, set[str]]] = {
    "m2": {
        "long_environment": {"m2_yoy_gt_0"},
        "caution_contraction": set(),
        "insufficient_data": set(),
    },
    "hiring_rate": {
        "recession_alert": {"hiring_rate_lte_3_4"},
        "normal": set(),
    },
    "ten_year_yield": {
        "normal": set(),
        "equity_pressure_zone": {"ten_year_yield_gte_4_4"},
        "extreme_pressure_bond_opportunity": {"ten_year_yield_gte_4_4", "ten_year_yield_gte_5_0"},
    },
    "buffett_ratio": {
        "overheat_peak_risk": {"buffett_ratio_gte_2_0"},
        "normal": set(),
    },
    "unemployment_rate": {
        "labor_weakening": {"unemployment_mom_gte_0_1", "unemployment_mom_gte_0_2"},
        "watch": {"unemployment_mom_gte_0_1"},
        "stable": set(),
        "insufficient_data": set(),
    },
}

THRESHOLD_LABELS: dict[str, str] = {
    "m2_yoy_gt_0": "M2 YoY > 0",
    "hiring_rate_lte_3_4": "Hiring rate <= 3.4",
    "ten_year_yield_gte_4_4": "10Y yield >= 4.4",
    "ten_year_yield_gte_5_0": "10Y yield >= 5.0",
    "buffett_ratio_gte_2_0": "Buffett ratio >= 2.0",
    "unemployment_mom_gte_0_1": "Unemployment MoM >= 0.1",
    "unemployment_mom_gte_0_2": "Unemployment MoM >= 0.2",
}

STOCK_TRIGGER_LABELS: dict[str, str] = {
    "entry_bullish_alignment": "Entry signal: bullish alignment",
    "exit_price_below_sma50": "Hard sell: price < SMA50",
    "exit_death_cross_50_lt_100": "Death cross: SMA50 < SMA100",
    "exit_death_cross_50_lt_200": "Death cross: SMA50 < SMA200",
    "exit_rsi_overbought": "RSI overbought: RSI14 > 80",
    "rsi_bearish_divergence": "RSI bearish divergence",
}


def _default_start_date(lookback_years: int) -> str:
    now_utc = datetime.now(timezone.utc).date()
    return (pd.Timestamp(now_utc) - pd.DateOffset(years=lookback_years)).date().isoformat()


def _attach_source(series: pd.Series, source: str) -> pd.Series:
    out = series.copy()
    out.attrs["source"] = source
    return out


def _clean_series(series: pd.Series) -> pd.Series:
    attrs = dict(series.attrs)
    clean = series.sort_index().dropna().astype(float)
    clean.attrs.update(attrs)
    return clean


def _series_to_csv_frame(series: pd.Series, source: str) -> pd.DataFrame:
    clean = series.dropna().sort_index()
    return pd.DataFrame(
        {
            "date": pd.to_datetime(clean.index).date,
            "value": clean.astype(float).values,
            "source": source,
        }
    )


def _latest_snapshot(metric_key: str, series: pd.Series, now_iso: str) -> dict[str, object]:
    clean = series.dropna().sort_index()
    if clean.empty:
        raise ValueError(f"No data available for metric {metric_key}.")
    as_of = pd.Timestamp(clean.index[-1])
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("UTC").tz_localize(None)
    stale_days = (datetime.now(timezone.utc).date() - as_of.date()).days
    source = str(series.attrs.get("source", "unknown"))
    return {
        "metric_key": metric_key,
        "as_of_date": as_of.date().isoformat(),
        "value": float(clean.iloc[-1]),
        "source": source,
        "stale_days": stale_days,
        "last_updated_utc": now_iso,
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _normalize_ticker(raw_value: Any) -> str:
    return str(raw_value).strip().upper()


def _write_watchlist(path: Path, tickers: list[str]) -> None:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        clean = _normalize_ticker(ticker)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": normalized}).to_csv(path, index=False)


def _load_or_initialize_watchlist(path: Path) -> list[str]:
    if not path.exists():
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read watchlist at {path}: {exc}; reinitializing defaults.")
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    if "ticker" not in frame.columns:
        print(f"Warning: watchlist at {path} has no 'ticker' column; reinitializing defaults.")
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    tickers: list[str] = []
    seen: set[str] = set()
    for raw in frame["ticker"].tolist():
        ticker = _normalize_ticker(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)

    if not tickers:
        _write_watchlist(path, DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    if tickers != frame["ticker"].astype(str).tolist():
        _write_watchlist(path, tickers)
    return tickers


def _empty_stock_signal_row(ticker: str, now_iso: str, status: str, status_message: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ticker": ticker,
        "as_of_date": "",
        "price": float("nan"),
        "sma14": float("nan"),
        "sma50": float("nan"),
        "sma100": float("nan"),
        "sma200": float("nan"),
        "rsi14": float("nan"),
        "source": f"STOOQ:{ticker}",
        "stale_days": float("nan"),
        "status": status,
        "status_message": status_message,
        "last_updated_utc": now_iso,
    }
    for trigger_col in STOCK_TRIGGER_COLUMNS:
        row[trigger_col] = False
    return row


def _compute_watchlist_signals(watchlist: list[str], start_date: str, now_iso: str) -> pd.DataFrame:
    if not watchlist:
        return pd.DataFrame(columns=[*STOCK_SIGNAL_COLUMNS, "last_updated_utc"])

    rows: list[dict[str, Any]] = []
    for ticker in watchlist:
        try:
            daily_close = fetch_stock_daily_history(ticker, start_date)
        except Exception as exc:
            rows.append(
                _empty_stock_signal_row(
                    ticker=ticker,
                    now_iso=now_iso,
                    status="fetch_error",
                    status_message=f"Failed to fetch daily history: {exc}",
                )
            )
            continue

        latest_price: float | None = None
        try:
            latest_price = fetch_stock_intraday_latest(ticker)
            if latest_price is not None:
                daily_close.attrs["intraday_source"] = f"STOOQ_INTRADAY:{ticker}"
        except Exception as exc:
            print(f"Warning: failed to fetch intraday quote for {ticker}: {exc}")

        try:
            row = compute_stock_signal_row(ticker, daily_close=daily_close, latest_price=latest_price)
            row["last_updated_utc"] = now_iso
            rows.append(row)
        except Exception as exc:
            rows.append(
                _empty_stock_signal_row(
                    ticker=ticker,
                    now_iso=now_iso,
                    status="compute_error",
                    status_message=f"Failed to compute stock signals: {exc}",
                )
            )

    frame = pd.DataFrame(rows)
    expected_cols = [*STOCK_SIGNAL_COLUMNS, "last_updated_utc"]
    for col in expected_cols:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[expected_cols].sort_values("ticker").reset_index(drop=True)


def _load_previous_stock_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read previous stock signals at {path}: {exc}")
        return pd.DataFrame()

    required = {"ticker", *STOCK_TRIGGER_COLUMNS}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame


def _load_previous_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read previous signals at {path}: {exc}")
        return pd.DataFrame()

    required = {"metric_key", "signal_state"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame


def _thresholds_for_state(metric_key: str, signal_state: str) -> set[str]:
    metric_map = THRESHOLD_TRIGGER_MAP.get(metric_key, {})
    return set(metric_map.get(signal_state, set()))


def _value_for_message(raw_value: Any) -> str:
    if pd.isna(raw_value):
        return "n/a"
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return str(raw_value)
    return f"{numeric:.4f}"


def _detect_new_threshold_events(previous: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, Any]]:
    if previous.empty or current.empty:
        return []
    if not {"metric_key", "signal_state"}.issubset(current.columns):
        return []

    prev_thresholds: dict[str, set[str]] = {}
    prev_states: dict[str, str] = {}
    for _, row in previous.iterrows():
        metric_key = str(row.get("metric_key", ""))
        signal_state = str(row.get("signal_state", ""))
        if not metric_key:
            continue
        prev_thresholds[metric_key] = _thresholds_for_state(metric_key, signal_state)
        prev_states[metric_key] = signal_state

    events: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        metric_key = str(row.get("metric_key", ""))
        signal_state = str(row.get("signal_state", ""))
        if not metric_key or metric_key not in prev_states:
            continue

        current_thresholds = _thresholds_for_state(metric_key, signal_state)
        newly_triggered = sorted(current_thresholds - prev_thresholds.get(metric_key, set()))
        if not newly_triggered:
            continue

        events.append(
            {
                "metric_key": metric_key,
                "metric_name": str(row.get("metric_name", metric_key)),
                "as_of_date": str(row.get("as_of_date", "unknown")),
                "value": row.get("value"),
                "signal_state": signal_state,
                "prev_signal_state": prev_states.get(metric_key, "unknown"),
                "new_threshold_ids": newly_triggered,
                "new_threshold_labels": [THRESHOLD_LABELS.get(tid, tid) for tid in newly_triggered],
            }
        )

    return events


def _as_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if pd.isna(raw_value):
        return False
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    normalized = str(raw_value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _detect_new_stock_trigger_events(previous: pd.DataFrame, current: pd.DataFrame) -> list[dict[str, Any]]:
    if current.empty:
        return []
    if not {"ticker", *STOCK_TRIGGER_COLUMNS}.issubset(current.columns):
        return []

    previous_map: dict[tuple[str, str], bool] = {}
    if not previous.empty and {"ticker", *STOCK_TRIGGER_COLUMNS}.issubset(previous.columns):
        for _, row in previous.iterrows():
            ticker = _normalize_ticker(row.get("ticker", ""))
            if not ticker:
                continue
            for trigger_id in STOCK_TRIGGER_COLUMNS:
                previous_map[(ticker, trigger_id)] = _as_bool(row.get(trigger_id))

    events: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        ticker = _normalize_ticker(row.get("ticker", ""))
        if not ticker:
            continue

        for trigger_id in STOCK_TRIGGER_COLUMNS:
            current_state = _as_bool(row.get(trigger_id))
            previous_state = previous_map.get((ticker, trigger_id), False)
            if not current_state or previous_state:
                continue

            events.append(
                {
                    "ticker": ticker,
                    "trigger_id": trigger_id,
                    "trigger_label": STOCK_TRIGGER_LABELS.get(trigger_id, trigger_id),
                    "as_of_date": str(row.get("as_of_date", "unknown")),
                    "price": row.get("price"),
                    "status": str(row.get("status", "")),
                }
            )

    return events


def _build_discord_message(events: list[dict[str, Any]], now_iso: str) -> str:
    lines = [f"Macro threshold trigger alert ({now_iso})"]
    for event in events:
        threshold_labels = ", ".join(event["new_threshold_labels"])
        lines.append(
            (
                f"- {event['metric_name']} ({event['metric_key']}): {threshold_labels} triggered; "
                f"state {event['prev_signal_state']} -> {event['signal_state']}; "
                f"value={_value_for_message(event['value'])}; as_of={event['as_of_date']}"
            )
        )
    return "\n".join(lines)


def _build_stock_discord_message(events: list[dict[str, Any]], now_iso: str) -> str:
    lines = [f"Stock watchlist trigger alert ({now_iso})"]
    for event in events:
        lines.append(
            (
                f"- {event['ticker']}: {event['trigger_label']} triggered; "
                f"price={_value_for_message(event['price'])}; "
                f"as_of={event['as_of_date']}; status={event['status']}"
            )
        )
    return "\n".join(lines)


def _post_discord_message(webhook_url: str, content: str) -> None:
    payload = json.dumps({"content": content}).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10):
        return


def _notify_new_thresholds(previous: pd.DataFrame, current: pd.DataFrame, now_iso: str) -> None:
    events = _detect_new_threshold_events(previous, current)
    if not events:
        return

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("Warning: DISCORD_WEBHOOK_URL is not set; skipping threshold alert notification.")
        return

    message = _build_discord_message(events, now_iso)
    try:
        _post_discord_message(webhook_url, message)
    except Exception as exc:
        print(f"Warning: failed to send Discord alert: {exc}")


def _notify_new_stock_triggers(previous: pd.DataFrame, current: pd.DataFrame, now_iso: str) -> None:
    events = _detect_new_stock_trigger_events(previous, current)
    if not events:
        return

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("Warning: DISCORD_WEBHOOK_URL is not set; skipping stock trigger alert notification.")
        return

    message = _build_stock_discord_message(events, now_iso)
    try:
        _post_discord_message(webhook_url, message)
    except Exception as exc:
        print(f"Warning: failed to send stock Discord alert: {exc}")


def run_pipeline(start_date: str | None = None, lookback_years: int = DEFAULT_LOOKBACK_YEARS) -> None:
    """Run full fetch-transform-signal pipeline and persist CSV artifacts."""
    if not start_date:
        start_date = _default_start_date(lookback_years)
    previous_signals = _load_previous_signals(DERIVED_DATA_DIR / "signals_latest.csv")
    previous_stock_signals = _load_previous_stock_signals(STOCK_SIGNALS_PATH)
    watchlist = _load_or_initialize_watchlist(STOCK_WATCHLIST_PATH)

    m2_cfg = SERIES_CONFIG["m2"]
    hiring_cfg = SERIES_CONFIG["hiring_rate"]
    ten_cfg = SERIES_CONFIG["ten_year_yield"]
    buffett_cfg = SERIES_CONFIG["buffett_ratio"]
    unrate_cfg = SERIES_CONFIG["unemployment_rate"]

    m2 = _attach_source(fetch_fred_series(m2_cfg["series_id"], start_date), "FRED:M2SL")
    hiring = _attach_source(fetch_fred_series(hiring_cfg["series_id"], start_date), "FRED:JTSHIR")
    unrate = _attach_source(fetch_fred_series(unrate_cfg["series_id"], start_date), "FRED:UNRATE")
    market_cap_id = buffett_cfg["market_cap_series_id"]
    gdp_id = buffett_cfg["gdp_series_id"]
    market_cap_source = f"FRED:{market_cap_id}"
    market_cap_unit_divisor = float(buffett_cfg.get("market_cap_unit_divisor", 1.0))
    try:
        market_cap = fetch_fred_series(market_cap_id, start_date)
    except Exception:
        fallback_market_cap_id = buffett_cfg.get("fallback_market_cap_series_id")
        if not fallback_market_cap_id:
            raise
        market_cap = fetch_fred_series(fallback_market_cap_id, start_date)
        market_cap_source = f"FRED:{fallback_market_cap_id} (fallback for {market_cap_id})"
    gdp = fetch_fred_series(gdp_id, start_date)
    market_cap = _attach_source(market_cap, market_cap_source)
    gdp = _attach_source(gdp, f"FRED:{gdp_id}")

    ten_year = _attach_source(fetch_fred_series(ten_cfg["series_id"], start_date), "FRED:DGS10")

    buffett_ratio = build_buffett_ratio(
        market_cap,
        gdp,
        market_cap_unit_divisor=market_cap_unit_divisor,
    )
    buffett_ratio = _attach_source(buffett_ratio, f"{market_cap_source}/FRED:{gdp_id}")

    metric_series = {
        "m2": _clean_series(prepare_monthly_series(m2)),
        "hiring_rate": _clean_series(prepare_monthly_series(hiring)),
        "ten_year_yield": _clean_series(ten_year),
        "buffett_ratio": _clean_series(buffett_ratio),
        "unemployment_rate": _clean_series(prepare_monthly_series(unrate)),
    }

    now_iso = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    raw_frames = {
        key: _series_to_csv_frame(series, str(series.attrs.get("source", "unknown")))
        for key, series in metric_series.items()
    }

    signals = compute_signals(metric_series)
    signals["last_updated_utc"] = now_iso

    snapshot_rows = [_latest_snapshot(key, metric_series[key], now_iso) for key in METRIC_ORDER if key in metric_series]
    metric_snapshot = pd.DataFrame(snapshot_rows)

    for metric_key, frame in raw_frames.items():
        _write_csv(RAW_DATA_DIR / f"{metric_key}.csv", frame)

    _write_csv(DERIVED_DATA_DIR / "metric_snapshot.csv", metric_snapshot)
    _write_csv(DERIVED_DATA_DIR / "signals_latest.csv", signals)
    stock_signals = _compute_watchlist_signals(watchlist=watchlist, start_date=start_date, now_iso=now_iso)
    _write_csv(STOCK_SIGNALS_PATH, stock_signals)

    _notify_new_thresholds(previous_signals, signals, now_iso)
    _notify_new_stock_triggers(previous_stock_signals, stock_signals, now_iso)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run macro signal dashboard data pipeline.")
    parser.add_argument("--start-date", type=str, default=None, help="ISO date, e.g. 2011-01-01")
    parser.add_argument("--lookback-years", type=int, default=DEFAULT_LOOKBACK_YEARS, help="Years of history to pull")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(start_date=args.start_date, lookback_years=args.lookback_years)


if __name__ == "__main__":
    main()

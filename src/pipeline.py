"""Data pipeline for fetching macro series, computing signals, and writing CSV caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_LOOKBACK_YEARS, METRIC_ORDER, SERIES_CONFIG
from src.data_fetch import fetch_fred_series, fetch_tnx_fallback
from src.signals import compute_signals
from src.transform import build_buffett_ratio, prepare_monthly_series

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVED_DATA_DIR = PROJECT_ROOT / "data" / "derived"


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


def run_pipeline(start_date: str | None = None, lookback_years: int = DEFAULT_LOOKBACK_YEARS) -> None:
    """Run full fetch-transform-signal pipeline and persist CSV artifacts."""
    if not start_date:
        start_date = _default_start_date(lookback_years)

    m2_cfg = SERIES_CONFIG["m2"]
    hiring_cfg = SERIES_CONFIG["hiring_rate"]
    ten_cfg = SERIES_CONFIG["ten_year_yield"]
    buffett_cfg = SERIES_CONFIG["buffett_ratio"]
    unrate_cfg = SERIES_CONFIG["unemployment_rate"]

    m2 = _attach_source(fetch_fred_series(m2_cfg["series_id"], start_date), "FRED:M2SL")
    hiring = _attach_source(fetch_fred_series(hiring_cfg["series_id"], start_date), "FRED:JTSHIL")
    unrate = _attach_source(fetch_fred_series(unrate_cfg["series_id"], start_date), "FRED:UNRATE")
    willshire = fetch_fred_series(buffett_cfg["willshire_series_id"], start_date)
    gdp = fetch_fred_series(buffett_cfg["gdp_series_id"], start_date)
    willshire = _attach_source(willshire, "FRED:WILL5000PR")
    gdp = _attach_source(gdp, "FRED:GDP")

    try:
        ten_year = _attach_source(fetch_fred_series(ten_cfg["series_id"], start_date), "FRED:DGS10")
    except Exception:
        try:
            ten_year = _attach_source(fetch_tnx_fallback(start_date), "YAHOO:^TNX")
        except Exception as exc:
            raise RuntimeError("Failed to fetch 10Y yield from both FRED and Yahoo.") from exc

    buffett_ratio = build_buffett_ratio(willshire, gdp)
    buffett_ratio = _attach_source(buffett_ratio, "FRED:WILL5000PR/FRED:GDP")

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

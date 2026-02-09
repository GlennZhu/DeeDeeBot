from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import pipeline


def _series(values: list[float], start: str, freq: str) -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_pipeline_generates_expected_csv_contracts(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    derived_dir = tmp_path / "data" / "derived"
    monkeypatch.setattr(pipeline, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(pipeline, "DERIVED_DATA_DIR", derived_dir)

    def fake_fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
        del start_date
        mapping = {
            "M2SL": _series([100.0 + i for i in range(24)], "2024-01-31", "M"),
            "JTSHIL": _series([3.8, 3.7, 3.6], "2025-10-31", "M"),
            "UNRATE": _series([4.0, 4.1, 4.2], "2025-10-31", "M"),
            "WILL5000PR": _series([50.0, 52.0, 54.0], "2025-10-31", "M"),
            "GDP": _series([25.0, 26.0], "2025-09-30", "Q"),
            "DGS10": _series([4.2, 4.3, 4.4], "2025-12-29", "D"),
        }
        return mapping[series_id]

    monkeypatch.setattr(pipeline, "fetch_fred_series", fake_fetch_fred_series)
    monkeypatch.setattr(
        pipeline,
        "fetch_tnx_fallback",
        lambda start_date: _series([4.4], "2025-12-31", "D"),
    )

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


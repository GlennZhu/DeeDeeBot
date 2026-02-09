from __future__ import annotations

import pandas as pd

from src import data_fetch


def _frame_with_close(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx)


def test_fetch_stock_daily_history_falls_back_to_stooq(monkeypatch) -> None:
    monkeypatch.setattr(data_fetch.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    def fake_datareader(symbol: str, source: str, start: str) -> pd.DataFrame:
        assert symbol == "NVDA.US"
        assert source == "stooq"
        assert start == "2024-01-01"
        return _frame_with_close([100.0, 101.0, 103.0])

    monkeypatch.setattr(data_fetch, "DataReader", fake_datareader)

    series = data_fetch.fetch_stock_daily_history("NVDA", "2024-01-01")

    assert len(series) == 3
    assert float(series.iloc[-1]) == 103.0
    assert series.attrs["source"] == "STOOQ:NVDA.US"


def test_fetch_stock_daily_history_uses_yahoo_when_available(monkeypatch) -> None:
    monkeypatch.setattr(data_fetch.yf, "download", lambda *args, **kwargs: _frame_with_close([10.0, 12.0]))

    series = data_fetch.fetch_stock_daily_history("MSFT", "2024-01-01")

    assert len(series) == 2
    assert float(series.iloc[-1]) == 12.0
    assert series.attrs["source"] == "YAHOO:MSFT"

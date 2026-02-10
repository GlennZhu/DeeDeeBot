from __future__ import annotations

import pandas as pd

from src import data_fetch


def _frame_with_close(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx)


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        del exc_type, exc, tb
        return False


def test_fetch_stock_daily_history_uses_stooq_symbol_candidates(monkeypatch) -> None:
    calls: list[str] = []

    def fake_datareader(symbol: str, source: str, start: str) -> pd.DataFrame:
        calls.append(symbol)
        assert source == "stooq"
        assert start == "2024-01-01"
        if symbol == "BRK-B.US":
            return _frame_with_close([400.0, 410.0, 415.0])
        return pd.DataFrame()

    monkeypatch.setattr(data_fetch, "DataReader", fake_datareader)

    series = data_fetch.fetch_stock_daily_history("BRK.B", "2024-01-01")

    assert float(series.iloc[-1]) == 415.0
    assert series.attrs["source"] == "STOOQ:BRK-B.US"
    assert "BRK.B.US" in calls
    assert "BRK-B.US" in calls


def test_fetch_stock_intraday_latest_uses_stooq_quote(monkeypatch) -> None:
    def fake_urlopen(url: str, timeout: int = 10) -> _FakeResponse:
        del timeout
        assert "stooq.com/q/l/" in url
        return _FakeResponse("NVDA.US,20260210,003001,188.0,190.0,187.5,189.25,1000000,")

    monkeypatch.setattr(data_fetch.request, "urlopen", fake_urlopen)

    value = data_fetch.fetch_stock_intraday_latest("NVDA")

    assert value == 189.25


def test_fetch_stock_intraday_latest_returns_none_on_bad_payload(monkeypatch) -> None:
    monkeypatch.setattr(data_fetch.request, "urlopen", lambda *args, **kwargs: _FakeResponse(""))

    assert data_fetch.fetch_stock_intraday_latest("NVDA") is None

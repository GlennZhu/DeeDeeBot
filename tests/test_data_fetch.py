from __future__ import annotations

import pandas as pd
import pytest

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


def test_fetch_stock_daily_history_falls_back_to_yfinance_when_stooq_fails(monkeypatch) -> None:
    def fake_datareader(symbol: str, source: str, start: str) -> pd.DataFrame:
        del symbol, source, start
        raise RuntimeError("stooq unavailable")

    class _FakeYF:
        @staticmethod
        def download(*, tickers: str, start: str, interval: str, auto_adjust: bool, progress: bool, threads: bool) -> pd.DataFrame:
            del start, interval, auto_adjust, progress, threads
            if tickers == "AAPL":
                return _frame_with_close([180.0, 181.5, 183.2])
            return pd.DataFrame()

    monkeypatch.setattr(data_fetch, "DataReader", fake_datareader)
    monkeypatch.setattr(data_fetch, "yf", _FakeYF())

    series = data_fetch.fetch_stock_daily_history("AAPL", "2024-01-01")

    assert float(series.iloc[-1]) == pytest.approx(183.2)
    assert series.attrs["source"] == "YFINANCE:AAPL"


def test_fetch_stock_intraday_quote_uses_stooq_quote(monkeypatch) -> None:
    def fake_urlopen(url: str, timeout: int = 10) -> _FakeResponse:
        del timeout
        assert "stooq.com/q/l/" in url
        assert "f=sd2t2ohlcvpm3" in url
        return _FakeResponse("NVDA.US,2026-02-10,22:00:18,191.38,192.48,188.12,188.595,136245955,190.04,-1.445,-0.76%")

    monkeypatch.setattr(data_fetch.request, "urlopen", fake_urlopen)

    quote = data_fetch.fetch_stock_intraday_quote("NVDA")
    value = data_fetch.fetch_stock_intraday_latest("NVDA")

    assert quote is not None
    assert quote["price"] == 188.595
    assert quote["previous_close"] == 190.04
    assert quote["day_change"] == -1.445
    assert quote["day_change_pct"] == pytest.approx(-0.0076)
    assert quote["source"] == "STOOQ_INTRADAY:NVDA.US"
    assert quote["quote_timestamp_utc"] is not None
    assert isinstance(quote["quote_age_seconds"], int)
    assert value == 188.595


def test_fetch_stock_intraday_latest_returns_none_on_bad_payload(monkeypatch) -> None:
    monkeypatch.setattr(data_fetch.request, "urlopen", lambda *args, **kwargs: _FakeResponse(""))

    assert data_fetch.fetch_stock_intraday_latest("NVDA") is None


def test_fetch_stock_universe_snapshot_parses_nasdaq_trader_files(monkeypatch) -> None:
    nasdaq_payload = "\n".join(
        [
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares",
            "AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N",
            "QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N",
            "ZTEST|Ignored Test Issue|Q|Y|N|100|N|N",
            "File Creation Time: 02012026",
        ]
    )
    other_payload = "\n".join(
        [
            "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
            "MSFT|Microsoft Corporation Common Stock|N|MSFT|N|100|N|MSFT",
            "SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY",
            "ATEST|Ignored Test Issue|A|ATEST|N|100|Y|ATEST",
            "File Creation Time: 02012026",
        ]
    )

    def fake_urlopen(req, timeout: int = 20):
        del timeout
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "nasdaqlisted.txt" in url:
            return _FakeResponse(nasdaq_payload)
        if "otherlisted.txt" in url:
            return _FakeResponse(other_payload)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(data_fetch.request, "urlopen", fake_urlopen)

    frame = data_fetch.fetch_stock_universe_snapshot()

    assert {"ticker", "security_name", "exchange", "is_etf", "source"}.issubset(set(frame.columns))
    assert {"AAPL", "QQQ", "MSFT", "SPY"} == set(frame["ticker"].tolist())
    assert "ZTEST" not in set(frame["ticker"].tolist())
    assert "ATEST" not in set(frame["ticker"].tolist())

    aapl = frame[frame["ticker"] == "AAPL"].iloc[0]
    spy = frame[frame["ticker"] == "SPY"].iloc[0]
    assert aapl["exchange"] == "NASDAQ Global Select"
    assert bool(spy["is_etf"]) is True


def test_fetch_sp500_constituents_parses_symbols(monkeypatch) -> None:
    html = """
    <html><body>
    <table>
      <tr><th>Symbol</th><th>Security</th></tr>
      <tr><td>AAPL</td><td>Apple Inc.</td></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
      <tr><td>BF/B</td><td>Brown-Forman</td></tr>
    </table>
    </body></html>
    """

    monkeypatch.setattr(data_fetch, "_fetch_text_payload", lambda url: html)

    frame = data_fetch.fetch_sp500_constituents()
    assert frame.columns.tolist() == ["ticker", "source"]
    assert set(frame["ticker"].tolist()) == {"AAPL", "BRK-B", "BF-B"}
    assert set(frame["source"].tolist()) == {"WIKIPEDIA_SP500"}


def test_fetch_stock_daily_bars_batch_yfinance_parses_multiindex_ohlc(monkeypatch) -> None:
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
            ("Open", "MSFT"),
            ("Close", "MSFT"),
        ]
    )
    frame = pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.5, 1_000_000, 200.0, 201.0],
            [101.0, 102.0, 100.0, 101.5, 1_100_000, 201.0, 202.0],
            [102.0, 103.0, 101.0, 102.5, 1_200_000, 202.0, 203.0],
        ],
        index=idx,
        columns=columns,
    )

    class _FakeYF:
        @staticmethod
        def download(*, tickers: str, start: str, interval: str, auto_adjust: bool, progress: bool, threads: bool) -> pd.DataFrame:
            del start, interval, auto_adjust, progress, threads
            assert tickers == "AAPL MSFT TSLA"
            return frame

    monkeypatch.setattr(data_fetch, "yf", _FakeYF())

    out = data_fetch.fetch_stock_daily_bars_batch_yfinance(["AAPL", "MSFT", "TSLA"], "2024-01-01", batch_size=10, pause_seconds=0.0)

    assert set(out.keys()) == {"AAPL", "MSFT"}
    assert {"Open", "Close"}.issubset(set(out["AAPL"].columns))
    assert float(out["AAPL"]["Close"].iloc[-1]) == pytest.approx(102.5)
    assert out["AAPL"].attrs["source"] == "YFINANCE:AAPL"
    assert float(out["MSFT"]["Open"].iloc[0]) == pytest.approx(200.0)
    assert out["MSFT"].attrs["source"] == "YFINANCE:MSFT"


def test_fetch_stock_daily_bars_batch_yfinance_parses_single_symbol_flat_columns(monkeypatch) -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="D")
    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.2, 11.2],
            "Volume": [1000, 1100],
        },
        index=idx,
    )

    class _FakeYF:
        @staticmethod
        def download(*, tickers: str, start: str, interval: str, auto_adjust: bool, progress: bool, threads: bool) -> pd.DataFrame:
            del start, interval, auto_adjust, progress, threads
            assert tickers == "AAPL"
            return frame

    monkeypatch.setattr(data_fetch, "yf", _FakeYF())

    out = data_fetch.fetch_stock_daily_bars_batch_yfinance(["AAPL"], "2024-01-01", batch_size=1, pause_seconds=0.0)

    assert set(out.keys()) == {"AAPL"}
    assert {"Open", "Close"}.issubset(set(out["AAPL"].columns))
    assert float(out["AAPL"]["Open"].iloc[-1]) == pytest.approx(11.0)
    assert float(out["AAPL"]["Close"].iloc[-1]) == pytest.approx(11.2)


def test_fetch_stock_daily_bars_falls_back_to_yfinance_when_stooq_fails(monkeypatch) -> None:
    def fake_datareader(symbol: str, source: str, start: str) -> pd.DataFrame:
        del symbol, source, start
        raise RuntimeError("stooq unavailable")

    idx = pd.date_range("2026-01-01", periods=2, freq="D")
    yf_frame = pd.DataFrame({"Open": [10.0, 11.0], "Close": [10.5, 11.5]}, index=idx)

    class _FakeYF:
        @staticmethod
        def download(*, tickers: str, start: str, interval: str, auto_adjust: bool, progress: bool, threads: bool) -> pd.DataFrame:
            del start, interval, auto_adjust, progress, threads
            assert tickers == "AAPL"
            return yf_frame

    monkeypatch.setattr(data_fetch, "DataReader", fake_datareader)
    monkeypatch.setattr(data_fetch, "yf", _FakeYF())

    bars = data_fetch.fetch_stock_daily_bars("AAPL", "2024-01-01")

    assert {"Open", "Close"}.issubset(set(bars.columns))
    assert float(bars["Close"].iloc[-1]) == pytest.approx(11.5)
    assert bars.attrs["source"] == "YFINANCE:AAPL"

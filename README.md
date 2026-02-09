# Macro + Stock Signal Monitor

Dashboard and pipeline that track:

- Macro regime signals (M2, Hiring, 10Y yield, Buffett indicator, Unemployment)
- A stock watchlist with technical monitoring and first-time Discord alerts

## Stack

- Python
- `pandas_datareader` (FRED macro data)
- `yfinance` (10Y fallback + stock watchlist data)
- GitHub Actions (scheduled cache refresh)
- Streamlit (dashboard UI)

## Project Structure

- `app.py`: Streamlit dashboard with tabs (`Macro Monitor`, `Stock Watchlist`)
- `src/pipeline.py`: fetch/transform/signal/cache pipeline + Discord first-trigger alerts
- `src/signals.py`: macro signal logic and thresholds
- `src/stock_signals.py`: stock signal logic (SMA/RSI/divergence)
- `src/data_fetch.py`: FRED and Yahoo fetchers
- `data/raw/*.csv`: per-metric historical macro series cache
- `data/derived/metric_snapshot.csv`: latest macro values
- `data/derived/signals_latest.csv`: latest macro signal states
- `data/derived/stock_watchlist.csv`: tracked watchlist symbols
- `data/derived/stock_signals_latest.csv`: latest stock signal states
- `.github/workflows/update_data.yml`: weekday scheduled refresh (Pacific time guard)

## Stock Watchlist Metrics

For each watched ticker, the pipeline checks:

1. Entry signal:
- `SMA14 > SMA50 > (SMA100 or SMA200)`

2. Exit and risk signals:
- `price < SMA50`
- `SMA50 < SMA100`
- `SMA50 < SMA200`
- `RSI14 > 80`

3. Bearish RSI divergence:
- Latest two confirmed price highs `P1`, `P2` with corresponding RSI highs `R1`, `R2`
- Trigger when `P2 > P1` and `R2 < R1`

Alerts are sent to Discord only on first trigger (`false -> true` versus previous run).

## Default Watchlist Seed

If `data/derived/stock_watchlist.csv` does not exist, it is initialized with:

- `GOOG`
- `AVGO`
- `NVDA`
- `MSFT`

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Data Pipeline

```bash
python -m src.pipeline
```

Optional flags:

```bash
python -m src.pipeline --start-date 2011-01-01 --lookback-years 15
```

## Run Dashboard

```bash
streamlit run app.py
```

The dashboard uses cached CSV files only (no live API calls on page load).

## Tests

```bash
pytest
```

## GitHub Actions Schedule

Workflow `update_data.yml` is triggered by two UTC cron candidates and guarded at runtime to run only when local Pacific time is exactly **4:30 PM on weekdays**:

- `30 23 * * 1-5`
- `30 0 * * 2-6`

`workflow_dispatch` remains available for manual runs.

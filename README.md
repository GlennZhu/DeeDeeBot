# Macro Signal Monitor

Macro dashboard that tracks five independent market/regime signals:

- M2 money supply trend (`M2SL`)
- Hiring rate recession alert (`JTSHIL`)
- 10Y Treasury yield pressure levels (`DGS10`, fallback `^TNX/10`)
- Buffett indicator (`WILL5000PR / GDP`, with `SP500` proxy fallback if `WILL5000PR` is unavailable)
- Unemployment MoM change (`UNRATE`)

## Stack

- Python
- `pandas_datareader` (FRED macro data)
- `yfinance` (10Y fallback)
- GitHub Actions (scheduled cache refresh)
- Streamlit (dashboard UI)

## Project Structure

- `app.py`: Streamlit dashboard (reads cached CSV only)
- `src/pipeline.py`: fetch/transform/signal/cache pipeline
- `src/signals.py`: signal logic and thresholds
- `data/raw/*.csv`: per-metric historical series cache
- `data/derived/metric_snapshot.csv`: latest values
- `data/derived/signals_latest.csv`: latest signal states
- `.github/workflows/update_data.yml`: weekday scheduled refresh

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

The dashboard intentionally uses only cached CSV files. It does not query live APIs at page load.

## Tests

```bash
pytest
```

## GitHub Actions

Workflow `update_data.yml` runs weekdays (`30 23 * * 1-5` UTC), refreshes CSV cache, and commits changes only when data files differ.

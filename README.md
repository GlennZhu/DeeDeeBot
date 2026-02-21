# Macro + Stock Signal Monitor

Dashboard and pipeline that track:

- Macro regime signals (M2, Hiring, 10Y yield, Buffett indicator, Unemployment)
- A stock watchlist with technical monitoring and first-time Discord alerts
- A broader market scanner with automatic ticker sourcing beyond watchlist

## Stack

- Python
- `pandas_datareader` (FRED macro data)
- `pandas_datareader` Stooq feed (stock daily history)
- Stooq quote endpoint (stock intraday quote at run time)
- GitHub Actions (scheduled cache refresh)
- Streamlit (dashboard UI)

## Project Structure

- `app.py`: Streamlit dashboard with tabs (`Macro Monitor`, `Stock Watchlist`, `Market Scanner`, `Signal History (7D)`)
- `src/pipeline.py`: fetch/transform/signal/cache pipeline + Discord alerts
- `src/signals.py`: macro signal logic and thresholds
- `src/stock_signals.py`: stock signal logic (SMA/RSI/divergence/relative strength)
- `src/stock_scanner.py`: broad-universe scanner logic (leader/trend/crossback/pullback setups)
- `src/data_fetch.py`: FRED and Stooq fetchers (daily + intraday quote endpoint)
- `data/raw/*.csv`: per-metric historical macro series cache
- `data/derived/metric_snapshot.csv`: latest macro values
- `data/derived/signals_latest.csv`: latest macro signal states
- `data/derived/stock_watchlist.csv`: tracked watchlist symbols
- `data/derived/stock_signals_latest.csv`: latest stock signal states (includes intraday quote freshness fields)
- `data/derived/stock_universe.csv`: auto-sourced scanner universe (watchlist + Nasdaq Trader directories)
- `data/derived/scanner_signals_latest.csv`: latest scanner outputs and setup scores
- `data/derived/scanner_thesis_tags.csv`: optional narrative tags (`pain_point`, `solution`, `conviction`)
- `data/derived/signal_events_7d.csv`: rolling 7-day signal event history (`triggered` and `cleared`)
- `.github/workflows/update_data.yml`: weekday scheduled refresh (Eastern time guard)
- `.github/workflows/update_stock_intraday.yml`: stock-only 15-minute intraday refresh (Eastern extended-hours guard)

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

4. Comparative relative strength vs benchmark (`benchmark` per ticker in watchlist):
- Structural divergence: benchmark `> MA50` while stock `< MA50`
- RS trend: `RS_Ratio = Stock / Benchmark`, warning when `RS_Ratio < MA20(RS_Ratio)`
- 1M alpha: `stock_21d_return - benchmark_21d_return`
- **Strong sell trigger** when the stock is underperforming its benchmark (`structural_divergence` or `alpha_1m < -5%`)
- For `QQQ` itself, benchmark-relative alerting is skipped (no benchmark-related trigger alerts).

5. "Squat" buy-zone alerts (bull-market pullback logic):
- Precondition: `MA200 rising` OR `SMA50 > SMA200`
- Gap tracking: `gap_to_ma = (price - ma) / ma` for MA100 and MA200
- **Ambush alert** (`🟢 Approaching Buy Zone`) when price is dropping and sits `2%-3% above` MA100 or MA200
- **DCA alert** (`🔵 Price Broken MA100`) when price crosses below MA100
- **Critical support alert** (`⚠️ Critical Support (MA200)`) when price is near MA200 (`+1% to -2%` gap)
- **Breakdown alert** (`🚨 Breakdown Below MA200`) when price falls more than `2%` below MA200
- The MA200 trend precondition uses daily-close SMA trend (not intraday-adjusted SMA) to reduce alert flicker.

Alerts are sent to Discord on first trigger (`false -> true` versus previous run), and when negative signals clear (`true -> false` for risk/exit macro and stock conditions).
Daily indicator history (SMA/RSI) comes from Stooq; run-time `price` uses Stooq intraday quote when available.
`stock_signals_latest.csv` also includes `intraday_quote_timestamp_utc` and `intraday_quote_age_seconds` so quote staleness is explicit.
`signal_events_7d.csv` captures both `triggered` and `cleared` transitions and is pruned to the last 7 days by event timestamp on every pipeline run.
You can browse this history in the Streamlit `Signal History (7D)` tab.

## Default Watchlist Seed

If `data/derived/stock_watchlist.csv` does not exist, it is initialized with:

- `GOOG,QQQ`
- `AVGO,QQQ`
- `NVDA,QQQ`
- `MSFT,QQQ`
- `QQQ,QQQ`

Watchlist schema:

- `ticker`
- `benchmark`

Watchlist editing is disabled in the Streamlit UI; edit `data/derived/stock_watchlist.csv` directly.

## Market Scanner Logic

Ticker sourcing beyond watchlist:

- Pulls universe from Nasdaq Trader symbol directories (`nasdaqlisted.txt`, `otherlisted.txt`)
- Caches universe to `data/derived/stock_universe.csv`
- Keeps watchlist symbols pinned even if external source fetch fails
- Refreshes universe at most once per 24 hours
- Default breadth is tuned to scan up to 60 tickers per run
- Intraday quote calls are prioritized for watchlist symbols (non-watchlist intraday is off by default)

Scanner setups:

1. Leader + right-side trend:
- Trend established via `SMA14 > SMA50 > (SMA100 or SMA200)`
- Relative-strength-aware leader score (`rs_ratio`, `alpha_1m`, weak-strength flags)

2. Cross-back confirmation:
- Requires MA cross-back (`SMA14` crossing above `SMA50`)
- Requires bullish RSI divergence at lows
- Requires "three green candles" confirmation (implemented as three consecutive higher closes)

3. Bull-market pullback zones:
- Reuses the watchlist MA100/MA200 squat zone logic for ambush/support setups

Manual thesis hooks:

- `scanner_thesis_tags.csv` lets you add `pain_point`, `solution`, and `conviction` (0-2 score)
- Conviction contributes to scanner ranking (`scanner_score`)

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
python -m src.pipeline --macro-only
python -m src.pipeline --stock-only
python -m src.pipeline --stock-only --scanner-max-tickers 200
python -m src.pipeline --stock-only --scanner-all-tickers --scanner-include-etfs
python -m src.pipeline --stock-only --skip-scanner
```

Optional environment knobs:

- `SCANNER_MAX_TICKERS` (default `60`)
- `SCANNER_NON_WATCHLIST_INTRADAY_QUOTES` (default `0`)
- `SCANNER_ALL_TICKERS` (default `false`)
- `SCANNER_INCLUDE_ETFS` (default `false`)

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

Workflow `update_data.yml` is triggered by two UTC cron candidates and guarded at runtime to run only when local Eastern time is exactly **4:30 PM on weekdays**:

- `30 20 * * 1-5`
- `30 21 * * 1-5`

It executes `python -m src.pipeline --macro-only`.

Notifications:

- Set `DISCORD_WEBHOOK_URL` as a GitHub repository secret to receive enriched Discord notifications after each refresh run.

Workflow `update_stock_intraday.yml` runs every 15 minutes in UTC and executes `python -m src.pipeline --stock-only --skip-scanner` to keep watchlist alerts fresh intraday.

Workflow `update_stock_scanner_daily.yml` runs once per weekday after regular U.S. market close (**5:10 PM Eastern**, DST-safe via dual UTC cron + ET runtime guard). It executes `python -m src.pipeline --stock-only --scanner-all-tickers --scanner-include-etfs`.

`workflow_dispatch` remains available for both workflows.

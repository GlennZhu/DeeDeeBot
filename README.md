# Macro + Stock Signal Monitor

Dashboard and pipeline that track:

- Macro regime signals (M2, Hiring, 10Y yield, Buffett indicator, Unemployment)
- A stock watchlist with technical monitoring and first-time Discord alerts
- A broader market EOD scanner with automatic ticker sourcing beyond watchlist

## Stack

- Python
- `pandas_datareader` (FRED macro data)
- Charles Schwab Market Data API (daily history + batched quotes)
- `pandas_datareader` Stooq + Yahoo (`yfinance`) fallback provider
- GitHub Actions (scheduled cache refresh)
- Streamlit (dashboard UI)

## Project Structure

- `app.py`: Streamlit dashboard with tabs (`Macro Monitor`, `Stock Watchlist`, `Market Scanner`, `Signal History (7D)`)
- `src/pipeline.py`: fetch/transform/signal/cache pipeline + Discord alerts
- `src/signals.py`: macro signal logic and thresholds
- `src/stock_signals.py`: stock signal logic (SMA/RSI/divergence/relative strength)
- `src/stock_scanner.py`: broad-universe EOD scanner logic (3 exact signal triggers)
- `src/data_fetch.py`: FRED + market-data fetchers (Schwab, Stooq, Yahoo fallback)
- `data/raw/*.csv`: per-metric historical macro series cache
- `data/derived/metric_snapshot.csv`: latest macro values
- `data/derived/signals_latest.csv`: latest macro signal states
- `data/derived/stock_watchlist.csv`: tracked watchlist symbols
- `data/derived/stock_signals_latest.csv`: latest stock signal states (includes intraday quote freshness fields)
- `data/derived/stock_universe.csv`: auto-sourced scanner universe (watchlist + Nasdaq Trader directories)
- `data/derived/scanner_signals_latest.csv`: latest scanner outputs (3-signal EOD states + trigger flags)
- `data/derived/scanner_thesis_tags.csv`: optional narrative tags (retained for compatibility; no longer used for scanner ranking)
- `data/derived/signal_events_7d.csv`: rolling 7-day signal event history (`triggered` and `cleared`)
- `.github/workflows/update_data.yml`: weekday scheduled refresh (Eastern time guard)
- `.github/workflows/update_stock_intraday.yml`: stock-only 15-minute intraday refresh (Eastern extended-hours guard)

## Stock Watchlist Metrics

For each watched ticker, the pipeline checks:

1. Entry signal:
- `SMA14 > SMA50 > (SMA100 or SMA200)`

2. Exit and risk signals:
- `price < SMA50`
- `SMA50 < SMA200`
- `RSI14 > 80`

3. Bearish RSI divergence:
- Latest two confirmed price highs `P1`, `P2` with corresponding RSI highs `R1`, `R2`
- Trigger when `P2 > P1` and `R2 < R1`
- Live watchlist now uses tuned **v2** divergence filters:
  - requires nearby RSI pivots (no same-bar fallback),
  - minimum pivot separation,
  - minimum price/RSI delta between pivots,
  - RSI regime guardrails (bearish peaks should be elevated, bullish troughs should be depressed).

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
Daily indicator history and run-time prices use the configured market-data provider (`MARKET_DATA_PROVIDER`), with optional public fallback.
`stock_signals_latest.csv` also includes `intraday_quote_timestamp_utc` and `intraday_quote_age_seconds` so quote staleness is explicit.
Both stock and scanner outputs now include `error_type`, `error_provider`, and `error_retryable` so hard failures can be routed explicitly.
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
- Scanner scope is restricted to `S&P 500 ∪ Nasdaq 500 proxy ∪ watchlist`
  - S&P 500 source: Wikipedia constituents table
  - Nasdaq 500 proxy: top 500 non-ETF Nasdaq-listed names (ranked by Nasdaq market tier, then symbol)
- Default breadth is tuned to scan up to 60 scoped tickers per run
- Scanner execution is bounded-parallel with shared request pacing to reduce provider throttling

Scanner signals (exact EOD implementations):

1. Bullish Alignment Trigger:
- Active when `SMA14 > SMA50` and (`SMA50 > SMA100` or `SMA50 > SMA200`)
- Triggered only on the transition day (active today, not active yesterday)

2. Moving Average Recovery + Momentum Confirmation:
- Triggered when `Close` crosses above `SMA50` today (`today > SMA50` and `yesterday <= SMA50`)
- Requires 3 bullish candles (`Close > Open`) across the last 3 trading days

3. Ambush / Squat Alert:
- Trend filter: `SMA50 > SMA200`
- Active when close is within `0% to +2%` above `SMA100` or `SMA200`
- Triggered only on the transition day (active today, not active yesterday)

Scanner alerts:
- Scanner trigger events are written into `signal_events_7d.csv` with `domain=scanner`
- Newly triggered scanner signals are posted to Discord using the same `DISCORD_WEBHOOK_URL`

Legacy thesis hooks (deprecated for scanner ranking):

- `scanner_thesis_tags.csv` lets you add `pain_point`, `solution`, and `conviction` (0-2 score)
- The file is retained, but conviction no longer affects scanner ranking

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

## Evaluate RSI Divergence Variants

Use backtesting to compare v1 baseline against v2 candidates and emit a recommendation:

```bash
.venv/bin/python scripts/evaluate_rsi_divergence.py
```

Outputs:
- `data/derived/rsi_divergence_eval_summary.csv` (train/holdout metrics per candidate)
- `data/derived/rsi_divergence_eval_events.csv` (event-level backtest rows)
- `data/derived/rsi_divergence_recommendation.json` (recommended params + rationale)

Optional flags:

```bash
python -m src.pipeline --start-date 2011-01-01 --lookback-years 15
python -m src.pipeline --macro-only
python -m src.pipeline --stock-only
python -m src.pipeline --stock-only --scanner-max-tickers 200
python -m src.pipeline --stock-only --scanner-all-tickers --scanner-include-etfs
python -m src.pipeline --stock-only --scanner-max-tickers 200 --scanner-workers 8 --scanner-daily-rps 4.0
python -m src.pipeline --stock-only --scanner-max-tickers 200 --scanner-progress-log-every 10
python -m src.pipeline --stock-only --skip-scanner
python -m src.pipeline --scanner-only --scanner-all-tickers --scanner-shard-index 0 --scanner-shard-count 6
# tuned local scanner runner
./scripts/run_scanner_tuned.sh
```

Optional environment knobs:

- `SCANNER_MAX_TICKERS` (default `60`)
  - Applies to non-watchlist breadth only; watchlist symbols are always included on top
- `SCANNER_ALL_TICKERS` (default `false`)
- `SCANNER_INCLUDE_ETFS` (default `false`)
- `SCANNER_PARALLEL_WORKERS` (default `8`)
- `SCANNER_DAILY_REQUESTS_PER_SECOND` (default `4.0`)
- `SCANNER_PROGRESS_LOG_EVERY` (default `25`)
- `SCANNER_SHARD_INDEX` (optional; requires `SCANNER_SHARD_COUNT > 1`)
- `SCANNER_SHARD_COUNT` (default `1`)
- `SCANNER_YF_PREFETCH_BATCH_SIZE_FAST` (default `100`)
- `SCANNER_YF_PREFETCH_PAUSE_FAST` (default `0.4`)
- `SCANNER_YF_PREFETCH_BATCH_SIZE_SLOW` (default `25`)
- `SCANNER_YF_PREFETCH_PAUSE_SLOW` (default `2.0`)
- `MARKET_DATA_PROVIDER` (`auto`, `schwab`, `public`; default `auto`)
- `FRED_API_KEY` (optional; if set, uses `api.stlouisfed.org`; otherwise uses keyless `fredgraph.csv`)
- `FRED_REQUEST_TIMEOUT_SECONDS` (default `45`)
- `FRED_FETCH_MAX_ATTEMPTS` (default `4`)
- `FRED_FETCH_RETRY_BACKOFF_SECONDS` (default `1.0`)
- `SCHWAB_ACCESS_TOKEN` (optional local override; CI scanner workflow uses refresh-token preflight)
- `SCHWAB_REFRESH_TOKEN`, `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET` (optional token refresh flow)
- `SCHWAB_QUOTES_MAX_SYMBOLS_PER_REQUEST` (default `200`)
- `SCANNER_FETCH_MAX_ATTEMPTS` (default `3`)
- `SCANNER_FETCH_BACKOFF_SECONDS` (default `0.4`)
- `SCANNER_CIRCUIT_PREFETCH_MAX_COVERAGE` (default `0.05`)
- `SCANNER_CIRCUIT_PROBE_COUNT` (default `6`)
- `WATCHLIST_CIRCUIT_PREFETCH_MAX_COVERAGE` (default `0.05`)
- `WATCHLIST_CIRCUIT_PROBE_COUNT` (default `4`)
- `STOCK_FAIL_MAX_ERROR_RATIO` (default `0.80`)
- `SCANNER_FAIL_MAX_ERROR_RATIO` (default `0.60`)
- `SCANNER_INSUFFICIENT_DATA_ALERT_RATIO` (default `0.10`; alert-only threshold, not a hard failure)

## Weekly Schwab Token Rotation

Use the semi-automated helper to rotate `SCHWAB_REFRESH_TOKEN` (expected cadence is about every 7 days):

```bash
./scripts/rotate_schwab_token.sh --repo <owner>/<repo>
```

The script auto-loads `.env.schwab.local` (ignored by git because `.env.*` is in `.gitignore`).
Create that file locally:

```bash
cat > .env.schwab.local <<'EOF'
SCHWAB_CLIENT_ID='your_client_id'
SCHWAB_CLIENT_SECRET='your_client_secret'
SCHWAB_REDIRECT_URI='https://127.0.0.1'
GH_REPO='owner/repo'
EOF
```

You can also point to a custom file:

```bash
./scripts/rotate_schwab_token.sh --env-file /path/to/local.env
```

The helper will:

- Open/print the Schwab authorize URL
- Prompt for the full browser redirect URL
- Exchange code for tokens and validate `refresh_token` exists
- Update repository secret `SCHWAB_REFRESH_TOKEN` using `gh secret set`

One-time prerequisites:

- Install GitHub CLI (`gh`)
- Authenticate once with secret-management permissions: `gh auth login`

If token auth is broken, `update_stock_scanner_daily.yml` now runs a Schwab preflight check and fails fast before shard work, with a Discord alert when `DISCORD_WEBHOOK_URL` is configured.

## Run Dashboard

```bash
streamlit run app.py
```

The dashboard uses cached CSV files only (no live API calls on page load).
The `Market Scanner` tab shows latest scanner quality counts (`rows total`, `ok`, `insufficient_data`, and hard-error rows from `error_type` / `error_provider`) plus last update time in ET.

## Tests

```bash
pytest
```

## GitHub Actions Schedule

Workflow `update_data.yml` is triggered by two UTC cron candidates. A runtime guard picks the cron expression that matches the current Eastern UTC offset (EST/EDT), so macro refresh runs once per weekday around **4:30 PM ET** without depending on exact job start minute:

- `30 20 * * 1-5`
- `30 21 * * 1-5`

It executes `python -m src.pipeline --macro-only`.

Notifications:

- Set `DISCORD_WEBHOOK_URL` as a GitHub repository secret to receive enriched Discord notifications after each refresh run.
- Optionally set `FRED_API_KEY` as a GitHub repository secret to use official FRED API responses instead of keyless graph CSV.

Workflow `update_stock_intraday.yml` runs every 15 minutes in UTC and executes `python -m src.pipeline --stock-only --skip-scanner` to keep watchlist alerts fresh intraday.

Workflow `update_stock_scanner_daily.yml` runs once per weekday at fixed UTC time (`10 22 * * 1-5`), with no DST guard. This is `5:10 PM` Eastern during standard time and `6:10 PM` Eastern during daylight time. It uses a 3-phase sharded flow:

- Pre-shard baseline refresh (`--stock-only --skip-scanner`)
- Six sequential scanner shards (`--scanner-only --scanner-all-tickers --scanner-shard-index i --scanner-shard-count 6`) with lower per-run RPS/workers and inter-shard wait windows
- Merge step (`scripts/merge_scanner_shards.py`) that rebuilds `scanner_signals_latest.csv` and updates scanner events in `signal_events_7d.csv`

The workflow is wired to fail fast when stock/scanner error ratios exceed configured budgets and now expects Schwab credentials in repository secrets:

- `SCHWAB_REFRESH_TOKEN`
- `SCHWAB_CLIENT_ID`
- `SCHWAB_CLIENT_SECRET`

`workflow_dispatch` remains available for both workflows.

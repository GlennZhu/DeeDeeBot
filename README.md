# Macro + Stock Signal Monitor

Dashboard and pipeline that track:

- Macro regime signals (M2, Hiring, 10Y yield, Buffett indicator, Unemployment)
- A stock watchlist with technical monitoring and first-time Discord alerts

## Stack

- Python
- `pandas_datareader` (FRED macro data)
- Charles Schwab Market Data API (daily history + batched quotes)
- `pandas_datareader` Stooq + Yahoo (`yfinance`) fallback provider
- GitHub Actions (scheduled cache refresh)
- Streamlit (dashboard UI)

## Project Structure

- `app.py`: Streamlit dashboard with tabs (`Macro Monitor`, `Stock Watchlist`, `Signal History (7D)`)
- `src/pipeline.py`: fetch/transform/signal/cache pipeline + Discord alerts
- `src/signals.py`: macro signal logic and thresholds
- `src/stock_signals.py`: stock signal logic (SMA/RSI/divergence/relative strength)
- `src/data_fetch.py`: FRED + market-data fetchers (Schwab, Stooq, Yahoo fallback)
- `data/raw/*.csv`: per-metric historical macro series cache
- `data/derived/metric_snapshot.csv`: latest macro values
- `data/derived/signals_latest.csv`: latest macro signal states
- `data/derived/stock_watchlist.csv`: tracked watchlist symbols
- `data/derived/stock_signals_latest.csv`: latest stock signal states (includes intraday quote freshness fields)
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
Stock outputs include `error_type`, `error_provider`, and `error_retryable` so hard failures can be routed explicitly.
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
```

Optional environment knobs:

- `MARKET_DATA_PROVIDER` (`auto`, `schwab`, `public`; default `auto`)
- `FRED_API_KEY` (optional; if set, uses `api.stlouisfed.org`; otherwise uses keyless `fredgraph.csv`)
- `FRED_REQUEST_TIMEOUT_SECONDS` (default `45`)
- `FRED_FETCH_MAX_ATTEMPTS` (default `4`)
- `FRED_FETCH_RETRY_BACKOFF_SECONDS` (default `1.0`)
- `SCHWAB_ACCESS_TOKEN` (optional local override)
- `SCHWAB_REFRESH_TOKEN`, `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET` (optional token refresh flow)
- `SCHWAB_QUOTES_MAX_SYMBOLS_PER_REQUEST` (default `200`)
- `WATCHLIST_CIRCUIT_PREFETCH_MAX_COVERAGE` (default `0.05`)
- `WATCHLIST_CIRCUIT_PROBE_COUNT` (default `4`)
- `STOCK_FAIL_MAX_ERROR_RATIO` (default `0.80`)

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

If token auth is broken, stock refresh workflows fail with provider-auth errors; rotate Schwab credentials and rerun.

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

Workflow `update_data.yml` is triggered by two UTC cron candidates. A runtime guard picks the cron expression that matches the current Eastern UTC offset (EST/EDT), so macro refresh runs once per weekday around **4:30 PM ET** without depending on exact job start minute:

- `30 20 * * 1-5`
- `30 21 * * 1-5`

It executes `python -m src.pipeline --macro-only`.

Notifications:

- Set `DISCORD_WEBHOOK_URL` as a GitHub repository secret to receive enriched Discord notifications after each refresh run.
- Optionally set `FRED_API_KEY` as a GitHub repository secret to use official FRED API responses instead of keyless graph CSV.

Workflow `update_stock_intraday.yml` is triggered every 15 minutes in UTC, then a runtime Eastern-time guard allows execution only on weekdays during the intraday session window (**4:00 AM ET through 8:00 PM ET**, inclusive end-of-session slot). It executes `python -m src.pipeline --stock-only` to keep watchlist alerts fresh intraday.

#!/usr/bin/env bash
set -euo pipefail

SCANNER_MAX_TICKERS="${SCANNER_MAX_TICKERS:-60}"
SCANNER_PARALLEL_WORKERS="${SCANNER_PARALLEL_WORKERS:-8}"
SCANNER_DAILY_REQUESTS_PER_SECOND="${SCANNER_DAILY_REQUESTS_PER_SECOND:-4.0}"
SCANNER_PROGRESS_LOG_EVERY="${SCANNER_PROGRESS_LOG_EVERY:-25}"
SCANNER_SHARD_INDEX="${SCANNER_SHARD_INDEX:-}"
SCANNER_SHARD_COUNT="${SCANNER_SHARD_COUNT:-}"
SCANNER_ONLY="${SCANNER_ONLY:-false}"
MARKET_DATA_PROVIDER="${MARKET_DATA_PROVIDER:-schwab}"
SCANNER_FETCH_MAX_ATTEMPTS="${SCANNER_FETCH_MAX_ATTEMPTS:-2}"
SCANNER_FETCH_BACKOFF_SECONDS="${SCANNER_FETCH_BACKOFF_SECONDS:-0.2}"
STOCK_FAIL_MAX_ERROR_RATIO="${STOCK_FAIL_MAX_ERROR_RATIO:-0.95}"
SCANNER_FAIL_MAX_ERROR_RATIO="${SCANNER_FAIL_MAX_ERROR_RATIO:-0.65}"

export MARKET_DATA_PROVIDER
export SCANNER_FETCH_MAX_ATTEMPTS
export SCANNER_FETCH_BACKOFF_SECONDS
export STOCK_FAIL_MAX_ERROR_RATIO
export SCANNER_FAIL_MAX_ERROR_RATIO

log() {
  printf '%s [scanner-runner] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

if [[ ! -x ".venv/bin/python" ]]; then
  log "ERROR: missing .venv/bin/python. Activate/create your venv first."
  exit 1
fi

start_epoch="$(date +%s)"
log "STATE=START"
log "CONFIG mode=eod_3_signal_scanner provider=${MARKET_DATA_PROVIDER} scanner_only=${SCANNER_ONLY} max_non_watchlist_tickers=${SCANNER_MAX_TICKERS} workers=${SCANNER_PARALLEL_WORKERS} daily_rps=${SCANNER_DAILY_REQUESTS_PER_SECOND} progress_log_every=${SCANNER_PROGRESS_LOG_EVERY} fetch_attempts=${SCANNER_FETCH_MAX_ATTEMPTS} backoff_s=${SCANNER_FETCH_BACKOFF_SECONDS} stock_fail_ratio=${STOCK_FAIL_MAX_ERROR_RATIO} scanner_fail_ratio=${SCANNER_FAIL_MAX_ERROR_RATIO} shard_index=${SCANNER_SHARD_INDEX:-none} shard_count=${SCANNER_SHARD_COUNT:-none}"

cmd=(
  .venv/bin/python -m src.pipeline
  --scanner-max-tickers "${SCANNER_MAX_TICKERS}"
  --scanner-workers "${SCANNER_PARALLEL_WORKERS}"
  --scanner-daily-rps "${SCANNER_DAILY_REQUESTS_PER_SECOND}"
  --scanner-progress-log-every "${SCANNER_PROGRESS_LOG_EVERY}"
  "$@"
)

if [[ "${SCANNER_ONLY}" == "true" ]]; then
  cmd+=(--scanner-only)
else
  cmd+=(--stock-only)
fi

if [[ -n "${SCANNER_SHARD_INDEX}" ]]; then
  cmd+=(--scanner-shard-index "${SCANNER_SHARD_INDEX}")
fi
if [[ -n "${SCANNER_SHARD_COUNT}" ]]; then
  cmd+=(--scanner-shard-count "${SCANNER_SHARD_COUNT}")
fi

log "STATE=RUNNING"
if "${cmd[@]}"; then
  end_epoch="$(date +%s)"
  log "STATE=SUCCESS elapsed_seconds=$((end_epoch - start_epoch))"
  exit 0
else
  rc="$?"
  end_epoch="$(date +%s)"
  log "STATE=FAILED exit_code=${rc} elapsed_seconds=$((end_epoch - start_epoch))"
  exit "${rc}"
fi

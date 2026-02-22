#!/usr/bin/env bash
set -euo pipefail

SCANNER_MAX_TICKERS="${SCANNER_MAX_TICKERS:-60}"
SCANNER_PARALLEL_WORKERS="${SCANNER_PARALLEL_WORKERS:-8}"
SCANNER_DAILY_REQUESTS_PER_SECOND="${SCANNER_DAILY_REQUESTS_PER_SECOND:-4.0}"
SCANNER_PROGRESS_LOG_EVERY="${SCANNER_PROGRESS_LOG_EVERY:-25}"

log() {
  printf '%s [scanner-runner] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

if [[ ! -x ".venv/bin/python" ]]; then
  log "ERROR: missing .venv/bin/python. Activate/create your venv first."
  exit 1
fi

start_epoch="$(date +%s)"
log "STATE=START"
log "CONFIG mode=eod_3_signal_scanner max_non_watchlist_tickers=${SCANNER_MAX_TICKERS} workers=${SCANNER_PARALLEL_WORKERS} daily_rps=${SCANNER_DAILY_REQUESTS_PER_SECOND} progress_log_every=${SCANNER_PROGRESS_LOG_EVERY}"

cmd=(
  .venv/bin/python -m src.pipeline
  --stock-only
  --scanner-max-tickers "${SCANNER_MAX_TICKERS}"
  --scanner-workers "${SCANNER_PARALLEL_WORKERS}"
  --scanner-daily-rps "${SCANNER_DAILY_REQUESTS_PER_SECOND}"
  --scanner-progress-log-every "${SCANNER_PROGRESS_LOG_EVERY}"
  "$@"
)

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

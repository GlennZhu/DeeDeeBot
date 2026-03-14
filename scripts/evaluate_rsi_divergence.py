#!/usr/bin/env python3
"""Backtest RSI divergence variants and recommend a tuned v2 configuration."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error, parse, request

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_fetch import fetch_stock_daily_history, fetch_stock_daily_history_batch_yfinance
from src.stock_signals import (
    RSI_DIVERGENCE_V1_PARAMS,
    RsiDivergenceParams,
    compute_rsi14,
    detect_bearish_rsi_divergence_with_params,
    detect_bullish_rsi_divergence_with_params,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RSI divergence v1/v2 variants with walk-forward backtesting.")
    parser.add_argument(
        "--watchlist-path",
        type=Path,
        default=Path("data/derived/stock_watchlist.csv"),
        help="CSV with watchlist tickers (column: ticker).",
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=Path("data/derived/stock_signals_latest.csv"),
        help="CSV used to source extra liquid symbols (column: ticker).",
    )
    parser.add_argument(
        "--max-candidate-tickers",
        type=int,
        default=30,
        help="How many non-watchlist candidate tickers to include.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Optional comma-separated tickers to append.",
    )
    parser.add_argument(
        "--include-etfs",
        action="store_true",
        help="Include ETF symbols from candidate list.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Daily history start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--horizon-bars",
        type=int,
        default=10,
        help="Forward bars used to score post-signal move.",
    )
    parser.add_argument(
        "--target-move-pct",
        type=float,
        default=0.03,
        help="Directional return threshold for a hit (e.g., 0.03 = 3%%).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Per-ticker chronological split ratio for train/holdout.",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=40,
        help="Minimum event count before score receives full confidence weight.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for daily-history fetches.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
        help="Pause between yfinance batches.",
    )
    parser.add_argument(
        "--single-fetch-pause-seconds",
        type=float,
        default=0.15,
        help="Pause between single-symbol fallback fetch calls.",
    )
    parser.add_argument(
        "--market-data-provider",
        type=str,
        default="public",
        help="MARKET_DATA_PROVIDER override used for this script run.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/derived/rsi_divergence_eval_summary.csv"),
        help="Where to write candidate summary metrics.",
    )
    parser.add_argument(
        "--recommendation-output",
        type=Path,
        default=Path("data/derived/rsi_divergence_recommendation.json"),
        help="Where to write the recommended params payload.",
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=Path("data/derived/rsi_divergence_eval_events.csv"),
        help="Where to write all generated events.",
    )
    return parser.parse_args()


def _normalize_ticker(raw: Any) -> str:
    ticker = str(raw).strip().upper()
    return ticker


def _load_watchlist_tickers(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if "ticker" not in frame.columns:
        return []
    out: list[str] = []
    for raw in frame["ticker"].tolist():
        ticker = _normalize_ticker(raw)
        if ticker and ticker not in out:
            out.append(ticker)
    return out


def _load_candidate_tickers(path: Path, *, max_tickers: int, include_etfs: bool) -> list[str]:
    if max_tickers <= 0 or not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if "ticker" not in frame.columns:
        return []

    out = frame.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower() == "ok"]
    if not include_etfs and "is_etf" in out.columns:
        etf_mask = out["is_etf"].astype(str).str.lower().isin({"1", "true", "yes"})
        out = out[~etf_mask]
    if "is_watchlist" in out.columns:
        watchlist_mask = out["is_watchlist"].astype(str).str.lower().isin({"1", "true", "yes"})
        out = out[~watchlist_mask]

    out["ticker"] = out["ticker"].map(_normalize_ticker)
    out = out[out["ticker"] != ""]
    out = out.drop_duplicates(subset=["ticker"]).sort_values("ticker")
    return out["ticker"].head(max_tickers).tolist()


def _parse_extra_tickers(raw: str) -> list[str]:
    out: list[str] = []
    for piece in str(raw).split(","):
        ticker = _normalize_ticker(piece)
        if ticker and ticker not in out:
            out.append(ticker)
    return out


def _candidate_params() -> dict[str, RsiDivergenceParams]:
    base = RsiDivergenceParams(
        lookback=120,
        left=3,
        right=3,
        max_distance=2,
        min_price_move_pct=0.015,
        min_rsi_delta=3.0,
        min_pivot_separation=6,
        require_rsi_pivot_match=True,
        bearish_min_rsi_peak=55.0,
        bullish_max_rsi_trough=45.0,
    )
    return {
        "v1_baseline": RSI_DIVERGENCE_V1_PARAMS,
        "v2_candidate_balanced": base,
        "v2_candidate_early": replace(
            base,
            lookback=100,
            min_price_move_pct=0.01,
            min_rsi_delta=2.5,
            min_pivot_separation=5,
            bearish_min_rsi_peak=54.0,
            bullish_max_rsi_trough=46.0,
        ),
        "v2_candidate_trend_aware": replace(
            base,
            lookback=140,
            min_price_move_pct=0.012,
            min_rsi_delta=3.5,
            min_pivot_separation=7,
            bearish_min_rsi_peak=57.0,
            bullish_max_rsi_trough=43.0,
        ),
        "v2_candidate_conservative": replace(
            base,
            max_distance=1,
            min_price_move_pct=0.02,
            min_rsi_delta=5.0,
            min_pivot_separation=8,
            bearish_min_rsi_peak=60.0,
            bullish_max_rsi_trough=40.0,
        ),
        "v2_candidate_wide_swings": replace(
            base,
            lookback=160,
            left=4,
            right=4,
            min_price_move_pct=0.02,
            min_rsi_delta=4.0,
            min_pivot_separation=8,
            bearish_min_rsi_peak=58.0,
            bullish_max_rsi_trough=42.0,
        ),
    }


def _parse_nasdaq_price(raw: Any) -> float | None:
    text = str(raw).strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fetch_nasdaq_daily_close(ticker: str, start_date: str) -> pd.Series | None:
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return None

    to_date = datetime.now(timezone.utc).date().isoformat()
    base_url = f"https://api.nasdaq.com/api/quote/{parse.quote(symbol)}/historical"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nasdaq.com/",
    }

    for asset_class in ["stocks", "etf"]:
        url = (
            f"{base_url}?assetclass={asset_class}"
            f"&fromdate={parse.quote(start_date)}&todate={parse.quote(to_date)}&limit=5000"
        )
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except error.URLError:
            continue
        except Exception:
            continue

        data_payload = payload.get("data")
        if not isinstance(data_payload, dict):
            continue
        trades_table = data_payload.get("tradesTable")
        if not isinstance(trades_table, dict):
            continue
        rows = trades_table.get("rows", [])
        if not isinstance(rows, list) or not rows:
            continue

        parsed_rows: list[tuple[pd.Timestamp, float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            date_raw = row.get("date")
            close_raw = row.get("close")
            close = _parse_nasdaq_price(close_raw)
            timestamp = pd.to_datetime(date_raw, format="%m/%d/%Y", errors="coerce")
            if close is None or pd.isna(timestamp):
                continue
            parsed_rows.append((pd.Timestamp(timestamp), float(close)))

        if not parsed_rows:
            continue
        parsed_rows = sorted(parsed_rows, key=lambda pair: pair[0])
        out = pd.Series([price for _, price in parsed_rows], index=[idx for idx, _ in parsed_rows], dtype=float)
        out.name = symbol
        out.attrs["source"] = f"NASDAQ:{symbol}:{asset_class}"
        return out

    return None


def _history_min_length(params: RsiDivergenceParams, horizon_bars: int) -> int:
    return max(
        220,
        int(params.lookback) + int(params.right) + int(horizon_bars) + 2,
        int(params.left) + int(params.right) + int(horizon_bars) + 20,
    )


def _build_events_for_ticker(
    ticker: str,
    close_series: pd.Series,
    params: RsiDivergenceParams,
    *,
    horizon_bars: int,
    train_ratio: float,
) -> pd.DataFrame:
    clean = close_series.copy()
    clean.index = pd.to_datetime(clean.index).tz_localize(None)
    clean = clean.sort_index().dropna().astype(float)
    clean = clean[~clean.index.duplicated(keep="last")]
    if len(clean) < _history_min_length(params, horizon_bars):
        return pd.DataFrame()

    rsi = compute_rsi14(clean, period=14)
    split_index = max(0, min(len(clean) - 1, int(len(clean) * train_ratio) - 1))

    records: list[dict[str, Any]] = []
    prev_bull = False
    prev_bear = False
    start_idx = max(30, int(params.lookback), int(params.left) + int(params.right) + 2)
    stop_idx = len(clean) - max(1, horizon_bars)
    if start_idx >= stop_idx:
        return pd.DataFrame()

    for end_idx in range(start_idx, stop_idx):
        hist_price = clean.iloc[: end_idx + 1]
        hist_rsi = rsi.iloc[: end_idx + 1]
        bull = detect_bullish_rsi_divergence_with_params(hist_price, hist_rsi, params=params)
        bear = detect_bearish_rsi_divergence_with_params(hist_price, hist_rsi, params=params)

        current_price = float(clean.iloc[end_idx])
        if current_price <= 0:
            prev_bull = bool(bull)
            prev_bear = bool(bear)
            continue
        forward_price = float(clean.iloc[end_idx + horizon_bars])
        forward_return = (forward_price / current_price) - 1.0

        split = "train" if end_idx <= split_index else "holdout"
        as_of_date = pd.Timestamp(clean.index[end_idx]).date().isoformat()

        if bull and not prev_bull:
            records.append(
                {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "split": split,
                    "direction": "bullish",
                    "forward_return": forward_return,
                    "directional_return": forward_return,
                }
            )
        if bear and not prev_bear:
            records.append(
                {
                    "ticker": ticker,
                    "as_of_date": as_of_date,
                    "split": split,
                    "direction": "bearish",
                    "forward_return": forward_return,
                    "directional_return": -forward_return,
                }
            )

        prev_bull = bool(bull)
        prev_bear = bool(bear)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _summarize_events(events: pd.DataFrame, *, target_move_pct: float, min_events: int) -> dict[str, Any]:
    if events.empty:
        return {
            "events": 0,
            "bullish_events": 0,
            "bearish_events": 0,
            "hit_rate": float("nan"),
            "win_rate": float("nan"),
            "mean_directional_return": float("nan"),
            "median_directional_return": float("nan"),
            "mean_bullish_forward_return": float("nan"),
            "mean_bearish_forward_return": float("nan"),
            "score": float("-inf"),
        }

    directional = events["directional_return"].astype(float)
    forward = events["forward_return"].astype(float)
    hits = directional >= float(target_move_pct)
    wins = directional > 0.0

    events_count = int(len(events))
    mean_directional = float(directional.mean())
    hit_rate = float(hits.mean())
    win_rate = float(wins.mean())
    confidence = min(1.0, events_count / max(1, int(min_events)))
    score = confidence * (
        (mean_directional * 100.0)
        + ((hit_rate - 0.5) * 6.0)
        + ((win_rate - 0.5) * 2.0)
    )

    bullish_slice = events[events["direction"] == "bullish"]
    bearish_slice = events[events["direction"] == "bearish"]
    mean_bullish_forward = float(bullish_slice["forward_return"].mean()) if not bullish_slice.empty else float("nan")
    mean_bearish_forward = float(bearish_slice["forward_return"].mean()) if not bearish_slice.empty else float("nan")

    return {
        "events": events_count,
        "bullish_events": int((events["direction"] == "bullish").sum()),
        "bearish_events": int((events["direction"] == "bearish").sum()),
        "hit_rate": hit_rate,
        "win_rate": win_rate,
        "mean_directional_return": mean_directional,
        "median_directional_return": float(directional.median()),
        "mean_bullish_forward_return": mean_bullish_forward,
        "mean_bearish_forward_return": mean_bearish_forward,
        "score": float(score),
        "mean_forward_return": float(forward.mean()),
    }


def _pick_recommendation(summary: pd.DataFrame, *, min_events: int) -> tuple[str, str]:
    holdout = summary[summary["split"] == "holdout"].copy()
    if holdout.empty:
        return "v1_baseline", "No holdout events available; keeping v1 baseline."

    holdout = holdout.sort_values("score", ascending=False)
    baseline = holdout[holdout["candidate"] == "v1_baseline"]
    baseline_score = float(baseline.iloc[0]["score"]) if not baseline.empty else float("-inf")
    baseline_events = int(baseline.iloc[0]["events"]) if not baseline.empty else 0

    contenders = holdout[holdout["candidate"] != "v1_baseline"].copy()
    if contenders.empty:
        return "v1_baseline", "No v2 contenders produced holdout events; keeping v1 baseline."

    contenders = contenders.sort_values("score", ascending=False)
    best = contenders.iloc[0]
    best_name = str(best["candidate"])
    best_events = int(best["events"])
    best_score = float(best["score"])

    min_required_events = max(8, int(min_events * 0.30), int(baseline_events * 0.30))
    if best_events < min_required_events:
        return "v1_baseline", (
            f"Best contender ({best_name}) had too few holdout events ({best_events} < {min_required_events}); "
            "keeping v1 baseline."
        )

    if best_score <= baseline_score:
        return "v1_baseline", (
            f"Best contender ({best_name}) did not beat v1 on holdout score "
            f"({best_score:.4f} <= {baseline_score:.4f}); keeping v1 baseline."
        )

    return best_name, (
        f"Promoting {best_name}: holdout score improved from {baseline_score:.4f} to {best_score:.4f} "
        f"with {best_events} events."
    )


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def main() -> None:
    args = _parse_args()
    os.environ["MARKET_DATA_PROVIDER"] = str(args.market_data_provider).strip().lower() or "public"

    watchlist_tickers = _load_watchlist_tickers(args.watchlist_path)
    candidate_tickers = _load_candidate_tickers(
        args.candidate_path,
        max_tickers=max(0, int(args.max_candidate_tickers)),
        include_etfs=bool(args.include_etfs),
    )
    extra_tickers = _parse_extra_tickers(args.tickers)

    tickers: list[str] = []
    for source in [watchlist_tickers, candidate_tickers, extra_tickers]:
        for ticker in source:
            if ticker and ticker not in tickers:
                tickers.append(ticker)

    if not tickers:
        raise SystemExit("No tickers selected. Provide watchlist/candidate files or --tickers.")

    print(
        f"Fetching daily history for {len(tickers)} tickers "
        f"(watchlist={len(watchlist_tickers)}, candidates={len(candidate_tickers)}, extra={len(extra_tickers)})."
    )
    history_map = fetch_stock_daily_history_batch_yfinance(
        tickers,
        args.start_date,
        batch_size=max(1, int(args.batch_size)),
        pause_seconds=max(0.0, float(args.pause_seconds)),
    )
    missing_after_batch = [ticker for ticker in tickers if ticker not in history_map]
    if missing_after_batch:
        print(f"Batch fetch missed {len(missing_after_batch)} tickers; trying single-symbol fallback fetch.")
        pause_seconds = max(0.0, float(args.single_fetch_pause_seconds))
        for idx, ticker in enumerate(missing_after_batch, start=1):
            try:
                single = fetch_stock_daily_history(ticker, args.start_date)
            except Exception:
                single = None
            if single is None or single.empty:
                single = _fetch_nasdaq_daily_close(ticker, args.start_date)
            if single is not None and not single.empty:
                history_map[ticker] = single
            if pause_seconds > 0 and idx < len(missing_after_batch):
                time.sleep(pause_seconds)

    usable_history: dict[str, pd.Series] = {}
    for ticker, series in history_map.items():
        clean = series.dropna().astype(float)
        if len(clean) >= 220:
            usable_history[_normalize_ticker(ticker)] = clean

    missing = [ticker for ticker in tickers if ticker not in usable_history]
    if missing:
        print(f"Warning: {len(missing)} tickers had no usable history and were skipped.")
    if not usable_history:
        raise SystemExit("No usable history available for evaluation.")

    print(f"Running backtest on {len(usable_history)} tickers.")
    candidates = _candidate_params()
    all_events: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate_name, params in candidates.items():
        candidate_events: list[pd.DataFrame] = []
        for ticker, series in usable_history.items():
            events = _build_events_for_ticker(
                ticker,
                series,
                params,
                horizon_bars=max(1, int(args.horizon_bars)),
                train_ratio=min(0.95, max(0.50, float(args.train_ratio))),
            )
            if events.empty:
                continue
            events["candidate"] = candidate_name
            candidate_events.append(events)

        merged = pd.concat(candidate_events, ignore_index=True) if candidate_events else pd.DataFrame()
        if not merged.empty:
            all_events.append(merged)

        for split in ["train", "holdout", "all"]:
            split_events = merged if split == "all" else merged[merged["split"] == split]
            summary = _summarize_events(
                split_events,
                target_move_pct=max(0.0, float(args.target_move_pct)),
                min_events=max(1, int(args.min_events)),
            )
            summary_rows.append(
                {
                    "candidate": candidate_name,
                    "split": split,
                    **summary,
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    if summary_frame.empty:
        raise SystemExit("Backtest produced no summary rows.")

    recommended_name, rationale = _pick_recommendation(summary_frame, min_events=max(1, int(args.min_events)))
    recommended_params = candidates[recommended_name]

    summary_output = args.summary_output
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_output, index=False)

    events_output = args.events_output
    events_output.parent.mkdir(parents=True, exist_ok=True)
    events_frame = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    events_frame.to_csv(events_output, index=False)

    recommendation_payload = {
        "recommended_candidate": recommended_name,
        "rationale": rationale,
        "params": asdict(recommended_params),
        "context": {
            "start_date": args.start_date,
            "horizon_bars": int(args.horizon_bars),
            "target_move_pct": float(args.target_move_pct),
            "train_ratio": float(args.train_ratio),
            "evaluated_tickers": len(usable_history),
        },
    }
    recommendation_output = args.recommendation_output
    recommendation_output.parent.mkdir(parents=True, exist_ok=True)
    recommendation_output.write_text(json.dumps(recommendation_payload, indent=2), encoding="utf-8")

    holdout = summary_frame[summary_frame["split"] == "holdout"].sort_values("score", ascending=False)
    print("\nHoldout ranking:")
    print(
        holdout[
            [
                "candidate",
                "events",
                "hit_rate",
                "win_rate",
                "mean_directional_return",
                "score",
            ]
        ].to_string(
            index=False,
            justify="left",
            formatters={
                "hit_rate": _format_pct,
                "win_rate": _format_pct,
                "mean_directional_return": _format_pct,
                "score": lambda v: f"{float(v):.4f}",
            },
        )
    )
    print(f"\nRecommendation: {recommended_name}")
    print(rationale)
    print(f"Summary CSV: {summary_output}")
    print(f"Events CSV: {events_output}")
    print(f"Recommendation JSON: {recommendation_output}")


if __name__ == "__main__":
    main()

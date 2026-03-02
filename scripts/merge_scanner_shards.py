#!/usr/bin/env python3
"""Merge scanner shard CSV outputs into the canonical scanner artifacts."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge scanner shard CSV outputs.")
    parser.add_argument(
        "--shard-glob",
        type=str,
        required=True,
        help="Glob pattern for shard CSV files (supports ** with --recursive).",
    )
    parser.add_argument(
        "--previous-scanner",
        type=Path,
        default=Path("data/derived/scanner_signals_latest.csv"),
        help="Path to previous full scanner CSV used for trigger-event diffing.",
    )
    parser.add_argument(
        "--output-scanner",
        type=Path,
        default=Path("data/derived/scanner_signals_latest.csv"),
        help="Output path for merged scanner CSV.",
    )
    parser.add_argument(
        "--signal-events",
        type=Path,
        default=Path("data/derived/signal_events_7d.csv"),
        help="Signal-event history CSV path to update with merged scanner events.",
    )
    parser.add_argument(
        "--min-shards",
        type=int,
        default=1,
        help="Minimum number of shard CSV files required.",
    )
    return parser.parse_args()


def _load_shard(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_cols = [*pipeline.SCANNER_SIGNAL_COLUMNS, *pipeline.ERROR_DIAGNOSTIC_COLUMNS, "last_updated_utc"]
    for col in expected_cols:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame["ticker"] = frame["ticker"].map(pipeline._normalize_ticker)
    frame = frame[frame["ticker"] != ""]
    return frame[expected_cols]


def _merge_shards(shard_paths: list[Path]) -> pd.DataFrame:
    frames = [_load_shard(path) for path in shard_paths]
    if not frames:
        return pd.DataFrame(columns=[*pipeline.SCANNER_SIGNAL_COLUMNS, *pipeline.ERROR_DIAGNOSTIC_COLUMNS, "last_updated_utc"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker"], keep="last").sort_values("ticker").reset_index(drop=True)
    return merged


def main() -> None:
    args = _parse_args()
    shard_paths = sorted(Path(path) for path in glob.glob(args.shard_glob, recursive=True))
    if len(shard_paths) < max(1, int(args.min_shards)):
        raise SystemExit(
            f"Found {len(shard_paths)} shard files with {args.shard_glob!r}, "
            f"but --min-shards={args.min_shards} requires more."
        )

    merged = _merge_shards(shard_paths)
    args.output_scanner.parent.mkdir(parents=True, exist_ok=True)
    pipeline._write_csv(args.output_scanner, merged)

    previous = pipeline._load_previous_scanner_signals(args.previous_scanner)
    scanner_events = pipeline._detect_new_scanner_trigger_events(previous, merged)
    now_iso = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    pipeline._update_signal_event_history(
        args.signal_events,
        macro_events=[],
        stock_events=[],
        scanner_events=scanner_events,
        now_iso=now_iso,
    )

    print(
        "merged_scanner_shards "
        f"files={len(shard_paths)} rows={len(merged)} emitted_events={len(scanner_events)} "
        f"output={args.output_scanner}"
    )


if __name__ == "__main__":
    main()

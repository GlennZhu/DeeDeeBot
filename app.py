"""Streamlit dashboard for macro and stock watchlist monitoring from cached CSV files."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import METRIC_ORDER, SERIES_CONFIG

RAW_DATA_DIR = Path("data/raw")
DERIVED_DATA_DIR = Path("data/derived")
STOCK_WATCHLIST_PATH = DERIVED_DATA_DIR / "stock_watchlist.csv"
STOCK_SIGNALS_PATH = DERIVED_DATA_DIR / "stock_signals_latest.csv"
DEFAULT_STOCK_WATCHLIST: list[dict[str, str]] = [
    {"ticker": "GOOG"},
    {"ticker": "AVGO"},
    {"ticker": "NVDA"},
    {"ticker": "MSFT"},
]

HISTORY_OPTIONS = {
    "5Y": 5,
    "10Y": 10,
    "15Y": 15,
    "All available": None,
}

PST = timezone(timedelta(hours=-8), name="PST")

STATE_COLORS = {
    "long_environment": "#188038",
    "caution_contraction": "#b3261e",
    "recession_alert": "#b3261e",
    "normal": "#1f6feb",
    "equity_pressure_zone": "#b26a00",
    "extreme_pressure_bond_opportunity": "#b3261e",
    "overheat_peak_risk": "#b3261e",
    "labor_weakening": "#b3261e",
    "watch": "#b26a00",
    "stable": "#188038",
    "insufficient_data": "#6b7280",
}

STOCK_TRIGGER_LABELS = {
    "entry_bullish_alignment": "Entry: Bullish Alignment",
    "exit_price_below_sma50": "Exit: Price < SMA50",
    "exit_death_cross_50_lt_100": "Risk: SMA50 < SMA100",
    "exit_death_cross_50_lt_200": "Risk: SMA50 < SMA200",
    "exit_rsi_overbought": "Risk: RSI14 > 80",
    "rsi_bearish_divergence": "Bearish RSI Divergence",
    "strong_sell_weak_strength": "Strong Sell: Weak Relative Strength",
}

STOCK_TRIGGER_COLORS = {
    "entry_bullish_alignment": "#188038",
    "exit_price_below_sma50": "#b3261e",
    "exit_death_cross_50_lt_100": "#b3261e",
    "exit_death_cross_50_lt_200": "#b3261e",
    "exit_rsi_overbought": "#b26a00",
    "rsi_bearish_divergence": "#b26a00",
    "strong_sell_weak_strength": "#b3261e",
}


@st.cache_data
def _load_csv(path: Path, date_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for col in date_columns:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def _load_metric_series(metric_key: str) -> pd.DataFrame:
    return _load_csv(RAW_DATA_DIR / f"{metric_key}.csv", ["date"])


def _format_value(metric_key: str, value: float) -> str:
    if metric_key in {"hiring_rate", "unemployment_rate", "ten_year_yield"}:
        return f"{value:.2f}%"
    if metric_key == "buffett_ratio":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def _format_float(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _format_pst_timestamp(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return str(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert(PST).strftime("%Y-%m-%d %H:%M:%S PST")


def _format_age_seconds(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        total_seconds = max(0, int(float(value)))
    except Exception:
        return "N/A"
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _signal_badge(label: str, color: str) -> str:
    return (
        f"<span style='background-color:{color};color:white;padding:0.2rem 0.45rem;"
        "border-radius:0.4rem;font-size:0.78rem;'>"
        f"{label}</span>"
    )


def _apply_history_window(frame: pd.DataFrame, years: int | None) -> pd.DataFrame:
    if frame.empty or years is None or "date" not in frame.columns:
        return frame
    cutoff = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.DateOffset(years=years)
    return frame[frame["date"] >= cutoff]


def _to_m2_yoy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "value" not in frame.columns:
        return pd.DataFrame(columns=["date", "value"])
    yoy = frame[["date", "value"]].dropna().sort_values("date").copy()
    yoy["value"] = yoy["value"].pct_change(periods=12) * 100
    return yoy.dropna(subset=["value"])


def _latest_mom_change(frame: pd.DataFrame) -> float | None:
    if frame.empty or "value" not in frame.columns:
        return None
    clean = frame[["date", "value"]].dropna().sort_values("date")
    if len(clean) < 2:
        return None
    return float(clean["value"].iloc[-1] - clean["value"].iloc[-2])


def _normalize_ticker(raw_value: object) -> str:
    if pd.isna(raw_value):
        return ""
    return str(raw_value).strip().upper()


def _normalize_watchlist_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        ticker = _normalize_ticker(str(row.get("ticker", "")))
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append({"ticker": ticker})
    return out


def _write_watchlist(rows: list[dict[str, object]]) -> None:
    STOCK_WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_watchlist_rows(rows)
    pd.DataFrame(normalized, columns=["ticker"]).to_csv(STOCK_WATCHLIST_PATH, index=False)


def _load_or_initialize_watchlist() -> pd.DataFrame:
    if not STOCK_WATCHLIST_PATH.exists():
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker"])

    try:
        frame = pd.read_csv(STOCK_WATCHLIST_PATH)
    except Exception:
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker"])

    if "ticker" not in frame.columns:
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker"])

    records = frame.to_dict(orient="records")
    normalized = _normalize_watchlist_rows(records)
    if not normalized:
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return pd.DataFrame(DEFAULT_STOCK_WATCHLIST, columns=["ticker"])

    normalized_frame = pd.DataFrame(normalized, columns=["ticker"])
    current_subset = frame[["ticker"]].copy()
    current_subset["ticker"] = current_subset["ticker"].map(_normalize_ticker)

    if set(frame.columns) != {"ticker"} or not normalized_frame.equals(current_subset.reset_index(drop=True)):
        _write_watchlist(normalized)

    return normalized_frame


def _as_bool(raw_value: object) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if pd.isna(raw_value):
        return False
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    return str(raw_value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _render_macro_tab(selected_window: str) -> None:
    signals = _load_csv(DERIVED_DATA_DIR / "signals_latest.csv", ["as_of_date", "last_updated_utc"])
    snapshot = _load_csv(DERIVED_DATA_DIR / "metric_snapshot.csv", ["as_of_date", "last_updated_utc"])

    if signals.empty and snapshot.empty:
        st.warning("No cached data found yet. Run `python -m src.pipeline` first.")
        return

    if "last_updated_utc" in snapshot.columns and not snapshot["last_updated_utc"].dropna().empty:
        last_update = snapshot["last_updated_utc"].dropna().max()
        st.info(f"Last successful macro update (PST): {_format_pst_timestamp(last_update)}")
    elif "last_updated_utc" in signals.columns and not signals["last_updated_utc"].dropna().empty:
        last_update = signals["last_updated_utc"].dropna().max()
        st.info(f"Last successful macro update (PST): {_format_pst_timestamp(last_update)}")

    available_keys = set(snapshot["metric_key"].unique()) if "metric_key" in snapshot.columns else set()
    missing = [key for key in METRIC_ORDER if key not in available_keys]
    if missing:
        st.warning(f"Partial data: missing metrics {', '.join(missing)}")

    series_by_metric = {metric_key: _load_metric_series(metric_key) for metric_key in METRIC_ORDER}

    cols = st.columns(5)
    for idx, metric_key in enumerate(METRIC_ORDER):
        metric_name = SERIES_CONFIG[metric_key]["metric_name"]
        if metric_key == "m2":
            metric_name = "M2 YoY"
        card = cols[idx]
        card.subheader(metric_name)

        metric_snapshot = snapshot[snapshot["metric_key"] == metric_key] if not snapshot.empty else pd.DataFrame()
        metric_signal = signals[signals["metric_key"] == metric_key] if not signals.empty else pd.DataFrame()
        metric_series = series_by_metric.get(metric_key, pd.DataFrame())

        if metric_snapshot.empty and metric_signal.empty:
            card.caption("No data")
            continue

        signal_row = metric_signal.iloc[0] if not metric_signal.empty else None
        snapshot_row = metric_snapshot.iloc[0] if not metric_snapshot.empty else None

        if metric_key == "m2":
            m2_yoy = _to_m2_yoy(metric_series)
            if m2_yoy.empty:
                card.metric("M2 YoY", "N/A")
            else:
                latest_yoy = m2_yoy.iloc[-1]
                card.metric("M2 YoY", f"{float(latest_yoy['value']):.2f}%")
        elif metric_key == "unemployment_rate" and snapshot_row is not None:
            value = float(snapshot_row["value"])
            mom_change = _latest_mom_change(metric_series)
            if mom_change is None:
                card.metric("Value", _format_value(metric_key, value))
            else:
                card.metric(
                    "Value",
                    _format_value(metric_key, value),
                    delta=f"{mom_change:+.2f} pp MoM",
                    delta_color="inverse",
                )
        elif snapshot_row is not None:
            value = float(snapshot_row["value"])
            card.metric("Value", _format_value(metric_key, value))

        if snapshot_row is not None:
            card.caption(f"As of: {pd.Timestamp(snapshot_row['as_of_date']).date().isoformat()}")
            card.caption(f"Stale days: {int(snapshot_row['stale_days'])}")

        if signal_row is not None:
            state = str(signal_row["signal_state"])
            card.markdown(_signal_badge(state.replace("_", " ").title(), STATE_COLORS.get(state, "#6b7280")), unsafe_allow_html=True)
            if "threshold_rule" in metric_signal.columns and pd.notna(signal_row["threshold_rule"]):
                card.caption(f"Rule: {signal_row['threshold_rule']}")
            if "message" in metric_signal.columns and pd.notna(signal_row["message"]):
                card.caption(f"Interpretation: {signal_row['message']}")
            if "source" in metric_signal.columns and pd.notna(signal_row["source"]):
                card.caption(f"Source: {signal_row['source']}")
        elif snapshot_row is not None and "source" in metric_snapshot.columns and pd.notna(snapshot_row["source"]):
            card.caption(f"Source: {snapshot_row['source']}")

    st.subheader("Historical Series")
    history_years = HISTORY_OPTIONS[selected_window]

    for metric_key in METRIC_ORDER:
        metric_name = SERIES_CONFIG[metric_key]["metric_name"]
        series = series_by_metric.get(metric_key, pd.DataFrame())
        if series.empty:
            continue

        series = _apply_history_window(series, history_years)
        if series.empty:
            continue

        chart_frame = series[["date", "value"]].dropna().sort_values("date")
        if metric_key == "m2":
            metric_name = "M2 YoY (%)"
            chart_frame = _to_m2_yoy(chart_frame)
            if chart_frame.empty:
                continue
        st.markdown(f"**{metric_name}**")
        st.line_chart(chart_frame.set_index("date")["value"])


def _render_stock_tab() -> None:
    watchlist = _load_or_initialize_watchlist()
    st.caption("Watchlist editing is disabled in UI. Update `data/derived/stock_watchlist.csv` to change tickers.")

    stock_signals = _load_csv(
        STOCK_SIGNALS_PATH,
        ["as_of_date", "last_updated_utc", "intraday_quote_timestamp_utc"],
    )
    st.subheader("Latest Watchlist Signals")

    if stock_signals.empty:
        st.info("No stock signals found yet. Run `python -m src.pipeline` first.")
        return

    if "last_updated_utc" in stock_signals.columns and not stock_signals["last_updated_utc"].dropna().empty:
        st.info(
            "Last successful stock update (PST): "
            f"{_format_pst_timestamp(stock_signals['last_updated_utc'].dropna().max())}"
        )

    stock_signals["ticker"] = stock_signals["ticker"].astype(str).str.upper()
    if "benchmark_ticker" in stock_signals.columns:
        stock_signals["benchmark_ticker"] = stock_signals["benchmark_ticker"].astype(str).str.upper()
    signal_by_ticker = {row["ticker"]: row for _, row in stock_signals.iterrows()}

    card_cols = st.columns(2)
    for idx, watch_row in watchlist.iterrows():
        ticker = str(watch_row["ticker"])
        card = card_cols[idx % 2]
        card.subheader(ticker)
        row = signal_by_ticker.get(ticker)
        if row is None:
            card.caption("No computed data for this ticker yet.")
            continue

        card.metric("Price", _format_float(row.get("price"), 2))
        card.caption(
            (
                f"As of: {row.get('as_of_date', 'N/A')} | "
                f"SMA14: {_format_float(row.get('sma14'), 2)} | "
                f"SMA50: {_format_float(row.get('sma50'), 2)} | "
                f"SMA100: {_format_float(row.get('sma100'), 2)} | "
                f"SMA200: {_format_float(row.get('sma200'), 2)} | "
                f"RSI14: {_format_float(row.get('rsi14'), 2)}"
            )
        )
        card.caption(
            (
                f"RS ratio: {_format_float(row.get('rs_ratio'), 4)} | "
                f"RS MA20: {_format_float(row.get('rs_ratio_ma20'), 4)} | "
                f"Alpha 1M: {_format_float(row.get('alpha_1m'), 4)}"
            )
        )
        card.caption(
            (
                f"Intraday quote (PST): {_format_pst_timestamp(row.get('intraday_quote_timestamp_utc'))} | "
                f"Quote age: {_format_age_seconds(row.get('intraday_quote_age_seconds'))}"
            )
        )
        if "relative_strength_reasons" in row and pd.notna(row.get("relative_strength_reasons")):
            card.caption(f"RS reasons: {row.get('relative_strength_reasons')}")

        active_badges: list[str] = []
        for trigger_col, label in STOCK_TRIGGER_LABELS.items():
            if _as_bool(row.get(trigger_col)):
                active_badges.append(_signal_badge(label, STOCK_TRIGGER_COLORS.get(trigger_col, "#1f6feb")))

        if active_badges:
            card.markdown(" ".join(active_badges), unsafe_allow_html=True)
        else:
            card.caption("No active watchlist triggers.")

        status = str(row.get("status", ""))
        if status and status != "ok":
            card.caption(f"Status: {status} ({row.get('status_message', '')})")

    display_columns = [
        "ticker",
        "benchmark_ticker",
        "as_of_date",
        "price",
        "intraday_quote_timestamp_utc",
        "intraday_quote_age_seconds",
        "intraday_quote_source",
        "sma14",
        "sma50",
        "sma100",
        "sma200",
        "rsi14",
        "benchmark_price",
        "benchmark_sma50",
        "rs_ratio",
        "rs_ratio_ma20",
        "alpha_1m",
        "rs_structural_divergence",
        "rs_trend_down",
        "rs_negative_alpha",
        "relative_strength_weak",
        "relative_strength_reasons",
        "entry_bullish_alignment",
        "exit_price_below_sma50",
        "exit_death_cross_50_lt_100",
        "exit_death_cross_50_lt_200",
        "exit_rsi_overbought",
        "rsi_bearish_divergence",
        "strong_sell_weak_strength",
        "status",
        "status_message",
        "last_updated_utc",
    ]

    existing_columns = [col for col in display_columns if col in stock_signals.columns]
    preview = stock_signals[existing_columns].copy()
    for col in ["price", "sma14", "sma50", "sma100", "sma200", "rsi14", "benchmark_price", "benchmark_sma50"]:
        if col in preview.columns:
            preview[col] = preview[col].apply(lambda v: None if pd.isna(v) else round(float(v), 2))
    for col in ["rs_ratio", "rs_ratio_ma20", "alpha_1m"]:
        if col in preview.columns:
            preview[col] = preview[col].apply(lambda v: None if pd.isna(v) else round(float(v), 4))
    if "intraday_quote_age_seconds" in preview.columns:
        preview["intraday_quote_age_seconds"] = preview["intraday_quote_age_seconds"].apply(_format_age_seconds)
    if "intraday_quote_timestamp_utc" in preview.columns:
        preview["intraday_quote_timestamp_utc"] = preview["intraday_quote_timestamp_utc"].apply(_format_pst_timestamp)
    if "last_updated_utc" in preview.columns:
        preview["last_updated_utc"] = preview["last_updated_utc"].apply(_format_pst_timestamp)

    st.caption("Signals table")
    st.dataframe(preview.sort_values("ticker"), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Macro + Stock Signal Monitor", layout="wide")
    st.title("Macro + Stock Signal Monitor")
    st.caption("Data source: cached CSV generated by GitHub Actions. No live API calls at page load.")

    selected_window = "15Y"
    if st.button("Refresh Cached Data"):
        st.cache_data.clear()
        st.rerun()

    macro_tab, stock_tab = st.tabs(["Macro Monitor", "Stock Watchlist"])
    with macro_tab:
        _render_macro_tab(selected_window)
    with stock_tab:
        _render_stock_tab()


if __name__ == "__main__":
    main()

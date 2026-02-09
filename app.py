"""Streamlit dashboard for macro and stock watchlist monitoring from cached CSV files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import METRIC_ORDER, SERIES_CONFIG

RAW_DATA_DIR = Path("data/raw")
DERIVED_DATA_DIR = Path("data/derived")
STOCK_WATCHLIST_PATH = DERIVED_DATA_DIR / "stock_watchlist.csv"
STOCK_SIGNALS_PATH = DERIVED_DATA_DIR / "stock_signals_latest.csv"
DEFAULT_STOCK_WATCHLIST = ["GOOG", "AVGO", "NVDA", "MSFT"]

HISTORY_OPTIONS = {
    "5Y": 5,
    "10Y": 10,
    "15Y": 15,
    "All available": None,
}

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
}

STOCK_TRIGGER_COLORS = {
    "entry_bullish_alignment": "#188038",
    "exit_price_below_sma50": "#b3261e",
    "exit_death_cross_50_lt_100": "#b3261e",
    "exit_death_cross_50_lt_200": "#b3261e",
    "exit_rsi_overbought": "#b26a00",
    "rsi_bearish_divergence": "#b26a00",
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


def _normalize_ticker(raw_value: str) -> str:
    return str(raw_value).strip().upper()


def _write_watchlist(tickers: list[str]) -> None:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in tickers:
        ticker = _normalize_ticker(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        ordered.append(ticker)

    STOCK_WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": ordered}).to_csv(STOCK_WATCHLIST_PATH, index=False)


def _load_or_initialize_watchlist() -> list[str]:
    if not STOCK_WATCHLIST_PATH.exists():
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    try:
        frame = pd.read_csv(STOCK_WATCHLIST_PATH)
    except Exception:
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    if "ticker" not in frame.columns:
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    tickers: list[str] = []
    seen: set[str] = set()
    for raw in frame["ticker"].tolist():
        ticker = _normalize_ticker(raw)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)

    if not tickers:
        _write_watchlist(DEFAULT_STOCK_WATCHLIST)
        return list(DEFAULT_STOCK_WATCHLIST)

    if tickers != frame["ticker"].astype(str).tolist():
        _write_watchlist(tickers)

    return tickers


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
        st.info(f"Last successful macro update (UTC): {last_update}")
    elif "last_updated_utc" in signals.columns and not signals["last_updated_utc"].dropna().empty:
        last_update = signals["last_updated_utc"].dropna().max()
        st.info(f"Last successful macro update (UTC): {last_update}")

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
    st.subheader("Watchlist Management")
    watchlist = _load_or_initialize_watchlist()

    with st.form("add_ticker_form", clear_on_submit=True):
        add_ticker = st.text_input("Add ticker", placeholder="e.g., AAPL")
        add_submitted = st.form_submit_button("Add")
    if add_submitted:
        ticker = _normalize_ticker(add_ticker)
        if not ticker:
            st.warning("Ticker cannot be empty.")
        elif ticker in watchlist:
            st.info(f"{ticker} is already in the watchlist.")
        else:
            watchlist.append(ticker)
            _write_watchlist(watchlist)
            st.success(f"Added {ticker}.")
            st.rerun()

    if watchlist:
        with st.form("remove_ticker_form"):
            remove_ticker = st.selectbox("Remove ticker", options=watchlist)
            remove_submitted = st.form_submit_button("Remove")
        if remove_submitted:
            updated = [ticker for ticker in watchlist if ticker != remove_ticker]
            _write_watchlist(updated)
            st.success(f"Removed {remove_ticker}.")
            st.rerun()

    st.caption("Current watchlist")
    st.dataframe(pd.DataFrame({"ticker": watchlist}), hide_index=True, use_container_width=True)

    stock_signals = _load_csv(STOCK_SIGNALS_PATH, ["as_of_date", "last_updated_utc"])
    st.subheader("Latest Watchlist Signals")

    if stock_signals.empty:
        st.info("No stock signals found yet. Run `python -m src.pipeline` first.")
        return

    if "last_updated_utc" in stock_signals.columns and not stock_signals["last_updated_utc"].dropna().empty:
        st.info(f"Last successful stock update (UTC): {stock_signals['last_updated_utc'].dropna().max()}")

    stock_signals["ticker"] = stock_signals["ticker"].astype(str).str.upper()
    signal_by_ticker = {row["ticker"]: row for _, row in stock_signals.iterrows()}

    card_cols = st.columns(2)
    for idx, ticker in enumerate(watchlist):
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
        "as_of_date",
        "price",
        "sma14",
        "sma50",
        "sma100",
        "sma200",
        "rsi14",
        "entry_bullish_alignment",
        "exit_price_below_sma50",
        "exit_death_cross_50_lt_100",
        "exit_death_cross_50_lt_200",
        "exit_rsi_overbought",
        "rsi_bearish_divergence",
        "status",
        "status_message",
        "last_updated_utc",
    ]

    existing_columns = [col for col in display_columns if col in stock_signals.columns]
    preview = stock_signals[existing_columns].copy()
    for col in ["price", "sma14", "sma50", "sma100", "sma200", "rsi14"]:
        if col in preview.columns:
            preview[col] = preview[col].apply(lambda v: None if pd.isna(v) else round(float(v), 2))

    st.caption("Signals table")
    st.dataframe(preview.sort_values("ticker"), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Macro + Stock Signal Monitor", layout="wide")
    st.title("Macro + Stock Signal Monitor")
    st.caption("Data source: cached CSV generated by GitHub Actions. No live API calls at page load.")

    st.sidebar.header("Controls")
    selected_window = st.sidebar.radio("Macro History Range", list(HISTORY_OPTIONS.keys()), index=2)
    if st.sidebar.button("Refresh Cached Data"):
        st.cache_data.clear()
        st.rerun()

    macro_tab, stock_tab = st.tabs(["Macro Monitor", "Stock Watchlist"])
    with macro_tab:
        _render_macro_tab(selected_window)
    with stock_tab:
        _render_stock_tab()


if __name__ == "__main__":
    main()

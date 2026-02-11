"""Streamlit dashboard for macro and stock watchlist monitoring from cached CSV files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.config import METRIC_ORDER, SERIES_CONFIG

RAW_DATA_DIR = Path("data/raw")
DERIVED_DATA_DIR = Path("data/derived")
STOCK_WATCHLIST_PATH = DERIVED_DATA_DIR / "stock_watchlist.csv"
STOCK_SIGNALS_PATH = DERIVED_DATA_DIR / "stock_signals_latest.csv"
SIGNAL_EVENTS_PATH = DERIVED_DATA_DIR / "signal_events_7d.csv"
DEFAULT_STOCK_WATCHLIST: list[dict[str, str]] = [
    {"ticker": "GOOG"},
    {"ticker": "AVGO"},
    {"ticker": "NVDA"},
    {"ticker": "MSFT"},
    {"ticker": "QQQ"},
]

HISTORY_OPTIONS = {
    "5Y": 5,
    "10Y": 10,
    "15Y": 15,
    "All available": None,
}

EASTERN_TZ = ZoneInfo("America/New_York")

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
    "squat_ambush_near_ma100_or_ma200": "🟢 Approaching Buy Zone",
    "squat_dca_below_ma100": "🔵 DCA Mode (MA100 Broken)",
    "squat_last_stand_ma200": "⚠️ Critical Support (MA200)",
}

STOCK_TRIGGER_COLORS = {
    "entry_bullish_alignment": "#188038",
    "exit_price_below_sma50": "#b3261e",
    "exit_death_cross_50_lt_100": "#b3261e",
    "exit_death_cross_50_lt_200": "#b3261e",
    "exit_rsi_overbought": "#b26a00",
    "rsi_bearish_divergence": "#b26a00",
    "strong_sell_weak_strength": "#b3261e",
    "squat_ambush_near_ma100_or_ma200": "#188038",
    "squat_dca_below_ma100": "#1f6feb",
    "squat_last_stand_ma200": "#b26a00",
}

MACRO_SIGNAL_LABELS = {
    "m2_yoy_gt_0": "M2 YoY > 0",
    "hiring_rate_lte_3_4": "Hiring rate <= 3.4",
    "ten_year_yield_gte_4_4": "10Y yield >= 4.4",
    "ten_year_yield_gte_5_0": "10Y yield >= 5.0",
    "buffett_ratio_gte_2_0": "Buffett ratio >= 2.0",
    "unemployment_mom_gte_0_1": "Unemployment MoM >= 0.1",
    "unemployment_mom_gte_0_2": "Unemployment MoM >= 0.2",
}

EVENT_TYPE_LABELS = {
    "triggered": "Triggered",
    "cleared": "Cleared",
}

EVENT_TYPE_COLORS = {
    "triggered": "#b26a00",
    "cleared": "#188038",
}

DOMAIN_LABELS = {
    "macro": "Macro",
    "stock": "Stock",
}

DOMAIN_COLORS = {
    "macro": "#1f6feb",
    "stock": "#0f766e",
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


def _format_ratio_pct(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def _format_price_delta(change: object, change_pct: object, digits: int = 2) -> str | None:
    if pd.isna(change) or pd.isna(change_pct):
        return None
    return f"{float(change):+.{digits}f} ({float(change_pct) * 100:+.{digits}f}%)"


def _format_et_timestamp(value: object) -> str:
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
    return ts.tz_convert(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


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


def _clean_event_text(raw_value: object) -> str:
    if pd.isna(raw_value):
        return ""
    return str(raw_value).strip()


def _format_event_number(raw_value: object, digits: int = 2) -> str | None:
    if pd.isna(raw_value):
        return None
    try:
        return f"{float(raw_value):.{digits}f}"
    except Exception:
        clean = str(raw_value).strip()
        return clean or None


def _format_event_price(raw_value: object) -> str | None:
    number = _format_event_number(raw_value, digits=2)
    if number is None:
        return None
    return f"${number}"


def _humanize_state_transition(raw_value: object) -> str:
    text = _clean_event_text(raw_value)
    if not text:
        return ""
    if "->" not in text:
        return text.replace("_", " ")
    left, right = [part.strip() for part in text.split("->", maxsplit=1)]
    left_clean = left.replace("_", " ").title() if left else "Unknown"
    right_clean = right.replace("_", " ").title() if right else "Unknown"
    return f"{left_clean} -> {right_clean}"


def _event_signal_label(row: pd.Series) -> str:
    domain = _clean_event_text(row.get("domain")).lower()
    signal_id = _clean_event_text(row.get("signal_id"))
    signal_label = _clean_event_text(row.get("signal_label"))

    if domain == "stock" and signal_id in STOCK_TRIGGER_LABELS:
        return STOCK_TRIGGER_LABELS[signal_id]
    if domain == "macro" and signal_id in MACRO_SIGNAL_LABELS:
        return MACRO_SIGNAL_LABELS[signal_id]
    if signal_label:
        return signal_label
    if signal_id:
        return signal_id.replace("_", " ").title()
    return "Unknown signal"


def _event_subject_label(row: pd.Series) -> str:
    subject_name = _clean_event_text(row.get("subject_name"))
    subject_id = _clean_event_text(row.get("subject_id"))
    return subject_name or subject_id or "Unknown subject"


def _event_headline(row: pd.Series) -> str:
    domain = _clean_event_text(row.get("domain")).lower()
    subject = _event_subject_label(row)
    signal_label = _event_signal_label(row)
    if domain == "stock":
        return f"{subject}: {signal_label}"
    return f"{subject}: {signal_label}"


def _event_message(row: pd.Series) -> str:
    domain = _clean_event_text(row.get("domain")).lower()
    as_of_date = _clean_event_text(row.get("as_of_date"))

    if domain == "macro":
        parts: list[str] = []
        if as_of_date:
            parts.append(f"As of {as_of_date}")
        value = _format_event_number(row.get("value"), digits=2)
        if value is not None:
            parts.append(f"Value {value}")
        state_transition = _humanize_state_transition(row.get("state_transition"))
        if state_transition:
            parts.append(f"State {state_transition}")
        details = _clean_event_text(row.get("details"))
        if details:
            parts.append(f"Details {details.replace('_', ' ')}")
        return " | ".join(parts)

    parts = []
    if as_of_date:
        parts.append(f"As of {as_of_date}")
    price = _format_event_price(row.get("price"))
    if price is not None:
        parts.append(f"Price {price}")
    status = _clean_event_text(row.get("status"))
    if status and status.lower() != "ok":
        parts.append(f"Status {status}")
    details = _clean_event_text(row.get("details"))
    if details:
        parts.append(f"Details {details.replace('_', ' ')}")
    return " | ".join(parts)


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
        st.info(f"Last successful macro update (ET): {_format_et_timestamp(last_update)}")
    elif "last_updated_utc" in signals.columns and not signals["last_updated_utc"].dropna().empty:
        last_update = signals["last_updated_utc"].dropna().max()
        st.info(f"Last successful macro update (ET): {_format_et_timestamp(last_update)}")

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
            "Last successful stock update (ET): "
            f"{_format_et_timestamp(stock_signals['last_updated_utc'].dropna().max())}"
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

        price_delta = _format_price_delta(row.get("day_change"), row.get("day_change_pct"))
        if price_delta is None:
            card.metric("Price", _format_float(row.get("price"), 2))
        else:
            card.metric("Price", _format_float(row.get("price"), 2), delta=price_delta)
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
                f"Squat trend-up: {'Yes' if _as_bool(row.get('squat_bull_market_precondition')) else 'No'} | "
                f"Price dropping: {'Yes' if _as_bool(row.get('squat_price_dropping')) else 'No'} | "
                f"Gap->MA100: {_format_ratio_pct(row.get('squat_gap_to_sma100_pct'))} | "
                f"Gap->MA200: {_format_ratio_pct(row.get('squat_gap_to_sma200_pct'))}"
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
                f"Intraday quote (ET): {_format_et_timestamp(row.get('intraday_quote_timestamp_utc'))} | "
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

def _render_signal_history_tab() -> None:
    signal_events = _load_csv(SIGNAL_EVENTS_PATH, ["event_timestamp_utc"])
    st.subheader("Signal Events (Past 7 Days)")

    if signal_events.empty:
        st.info("No signal event history found yet. Run the pipeline to begin tracking fired signals.")
        return

    required_columns = [
        "event_timestamp_utc",
        "event_timestamp_et",
        "domain",
        "event_type",
        "subject_id",
        "subject_name",
        "benchmark_ticker",
        "signal_id",
        "signal_label",
        "as_of_date",
        "value",
        "price",
        "state_transition",
        "status",
        "details",
    ]
    if "event_timestamp_et" not in signal_events.columns and "event_timestamp_pt" in signal_events.columns:
        signal_events["event_timestamp_et"] = signal_events["event_timestamp_pt"]
    for col in required_columns:
        if col not in signal_events.columns:
            signal_events[col] = ""

    signal_events = signal_events[required_columns].copy()
    event_timestamps = pd.to_datetime(signal_events["event_timestamp_utc"], errors="coerce", utc=True)
    now_utc = pd.Timestamp.utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")
    cutoff = now_utc - pd.Timedelta(days=7)
    in_window = event_timestamps >= cutoff
    signal_events = signal_events.loc[in_window].copy()
    event_timestamps = event_timestamps.loc[in_window]
    if signal_events.empty:
        st.info("No signal events in the last 7 days.")
        return

    signal_events["domain"] = signal_events["domain"].astype(str).str.strip().str.lower()
    signal_events["event_type"] = signal_events["event_type"].astype(str).str.strip().str.lower()
    signal_events["_event_ts"] = event_timestamps
    signal_events["event_timestamp_utc"] = event_timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    signal_events["event_timestamp_et"] = signal_events["event_timestamp_utc"].apply(_format_et_timestamp)
    signal_events["headline"] = signal_events.apply(_event_headline, axis=1)
    signal_events["message"] = signal_events.apply(_event_message, axis=1)
    signal_events["domain_display"] = signal_events["domain"].map(DOMAIN_LABELS).fillna(signal_events["domain"].str.title())
    signal_events["event_type_display"] = signal_events["event_type"].map(EVENT_TYPE_LABELS).fillna(
        signal_events["event_type"].str.title()
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total (7D)", len(signal_events))
    metric_cols[1].metric("Macro", int((signal_events["domain"] == "macro").sum()))
    metric_cols[2].metric("Stock", int((signal_events["domain"] == "stock").sum()))
    metric_cols[3].metric("Triggered", int((signal_events["event_type"] == "triggered").sum()))
    metric_cols[4].metric("Cleared", int((signal_events["event_type"] == "cleared").sum()))

    filter_cols = st.columns(3)
    selected_domain = filter_cols[0].selectbox("Domain", ["All", "Macro", "Stock"], index=0)
    selected_event_type = filter_cols[1].selectbox("Event Type", ["All", "Triggered", "Cleared"], index=0)
    search_term = filter_cols[2].text_input("Search", placeholder="Ticker, metric, signal, or detail")

    filtered = signal_events.copy()
    if selected_domain != "All":
        filtered = filtered[filtered["domain"] == selected_domain.lower()]
    if selected_event_type != "All":
        filtered = filtered[filtered["event_type"] == selected_event_type.lower()]
    if search_term.strip():
        query = search_term.strip().lower()
        search_columns = [
            "subject_id",
            "subject_name",
            "benchmark_ticker",
            "signal_id",
            "signal_label",
            "state_transition",
            "details",
            "headline",
            "message",
        ]
        search_mask = pd.Series(False, index=filtered.index)
        for col in search_columns:
            search_mask = search_mask | filtered[col].astype(str).str.lower().str.contains(query, na=False)
        filtered = filtered.loc[search_mask]

    if filtered.empty:
        st.info("No events match the selected filters.")
        return

    display = filtered.sort_values("_event_ts", ascending=False).copy()
    max_rows = 75
    if len(display) > max_rows:
        st.caption(f"Showing the latest {max_rows} events. Refine filters/search to narrow results.")

    for _, row in display.head(max_rows).iterrows():
        event_type = _clean_event_text(row.get("event_type")).lower()
        event_color = EVENT_TYPE_COLORS.get(event_type, "#6b7280")
        event_label = _clean_event_text(row.get("event_type_display")) or "Event"
        domain = _clean_event_text(row.get("domain")).lower()
        domain_color = DOMAIN_COLORS.get(domain, "#6b7280")
        domain_label = _clean_event_text(row.get("domain_display")) or "Domain"

        with st.container(border=True):
            header_cols = st.columns([1, 1, 5, 2])
            header_cols[0].markdown(_signal_badge(event_label, event_color), unsafe_allow_html=True)
            header_cols[1].markdown(_signal_badge(domain_label, domain_color), unsafe_allow_html=True)
            header_cols[2].markdown(f"**{_clean_event_text(row.get('headline'))}**")
            header_cols[3].caption(_clean_event_text(row.get("event_timestamp_et")) or "N/A")
            message = _clean_event_text(row.get("message"))
            if message:
                st.caption(message)


def main() -> None:
    st.set_page_config(page_title="Macro + Stock Signal Monitor", layout="wide")
    st.title("Macro + Stock Signal Monitor")
    st.caption("Data source: cached CSV generated by GitHub Actions. No live API calls at page load.")

    selected_window = "15Y"
    if st.button("Refresh Cached Data"):
        st.cache_data.clear()
        st.rerun()

    macro_tab, stock_tab, history_tab = st.tabs(["Macro Monitor", "Stock Watchlist", "Signal History (7D)"])
    with macro_tab:
        _render_macro_tab(selected_window)
    with stock_tab:
        _render_stock_tab()
    with history_tab:
        _render_signal_history_tab()


if __name__ == "__main__":
    main()

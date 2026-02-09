"""Signal computation logic for dashboard metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from src.config import SERIES_CONFIG
from src.transform import prepare_monthly_series

FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SignalRow:
    metric_key: str
    metric_name: str
    as_of_date: str
    value: float
    signal_state: str
    threshold_rule: str
    message: str
    source: str
    stale_days: int


def _latest(series: pd.Series) -> tuple[pd.Timestamp, float]:
    clean = series.dropna().sort_index()
    if clean.empty:
        raise ValueError("Series is empty after dropna.")
    ts = pd.Timestamp(clean.index[-1])
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts, float(clean.iloc[-1])


def _stale_days(as_of: pd.Timestamp) -> int:
    today_utc = datetime.now(timezone.utc).date()
    return (today_utc - as_of.date()).days


def _source(series: pd.Series, fallback: str = "unknown") -> str:
    return str(series.attrs.get("source", fallback))


def _m2_signal(series: pd.Series) -> SignalRow:
    monthly = prepare_monthly_series(series)
    as_of, current = _latest(monthly)
    yoy = monthly.pct_change(periods=12).iloc[-1]

    if pd.isna(yoy):
        signal_state = "insufficient_data"
        message = "Need at least 12 months of M2 history for YoY calculation."
    elif yoy > 0:
        signal_state = "long_environment"
        message = f"M2 YoY is {yoy:.2%} (> 0): long environment."
    else:
        signal_state = "caution_contraction"
        message = f"M2 YoY is {yoy:.2%} (<= 0): caution/contraction."

    return SignalRow(
        metric_key="m2",
        metric_name=SERIES_CONFIG["m2"]["metric_name"],
        as_of_date=as_of.date().isoformat(),
        value=current,
        signal_state=signal_state,
        threshold_rule="(current - m2_12m_ago) / m2_12m_ago > 0",
        message=message,
        source=_source(series, "FRED:M2SL"),
        stale_days=_stale_days(as_of),
    )


def _hiring_signal(series: pd.Series) -> SignalRow:
    as_of, current = _latest(series)
    is_alert = current <= 3.4

    return SignalRow(
        metric_key="hiring_rate",
        metric_name=SERIES_CONFIG["hiring_rate"]["metric_name"],
        as_of_date=as_of.date().isoformat(),
        value=current,
        signal_state="recession_alert" if is_alert else "normal",
        threshold_rule="current <= 3.4",
        message=(
            "Hiring rate is at or below 3.4: recession alert and defensive posture."
            if is_alert
            else "Hiring rate is above 3.4: no recession alert."
        ),
        source=_source(series, "FRED:JTSHIR"),
        stale_days=_stale_days(as_of),
    )


def _ten_year_signal(series: pd.Series) -> SignalRow:
    as_of, current = _latest(series)
    if current >= 5.0:
        state = "extreme_pressure_bond_opportunity"
        message = "10Y yield >= 5.0: extreme pressure zone, potential bond opportunity."
    elif current >= 4.4:
        state = "equity_pressure_zone"
        message = "10Y yield >= 4.4: equity pressure zone."
    else:
        state = "normal"
        message = "10Y yield < 4.4: normal zone."

    return SignalRow(
        metric_key="ten_year_yield",
        metric_name=SERIES_CONFIG["ten_year_yield"]["metric_name"],
        as_of_date=as_of.date().isoformat(),
        value=current,
        signal_state=state,
        threshold_rule=">=5.0 extreme; >=4.4 pressure; <4.4 normal",
        message=message,
        source=_source(series, "FRED:DGS10"),
        stale_days=_stale_days(as_of),
    )


def _buffett_signal(series: pd.Series) -> SignalRow:
    as_of, current = _latest(series)
    is_overheat = current >= 2.0

    return SignalRow(
        metric_key="buffett_ratio",
        metric_name=SERIES_CONFIG["buffett_ratio"]["metric_name"],
        as_of_date=as_of.date().isoformat(),
        value=current,
        signal_state="overheat_peak_risk" if is_overheat else "normal",
        threshold_rule="ratio >= 2.0",
        message=(
            "Buffett ratio >= 2.0: market overheat / long-term topping risk."
            if is_overheat
            else "Buffett ratio < 2.0: no overheat signal."
        ),
        source=_source(series, "FRED:NCBCEL/FRED:GDP"),
        stale_days=_stale_days(as_of),
    )


def _unemployment_signal(series: pd.Series) -> SignalRow:
    monthly = prepare_monthly_series(series)
    as_of, current = _latest(monthly)
    mom = monthly.diff().iloc[-1]

    if pd.isna(mom):
        state = "insufficient_data"
        message = "Need at least 2 months of unemployment data for MoM change."
    elif float(mom) >= 0.2 - FLOAT_TOLERANCE:
        state = "labor_weakening"
        message = f"Unemployment MoM change is {mom:+.2f}: labor weakening."
    elif float(mom) >= 0.1 - FLOAT_TOLERANCE:
        state = "watch"
        message = f"Unemployment MoM change is {mom:+.2f}: watch zone."
    else:
        state = "stable"
        message = f"Unemployment MoM change is {mom:+.2f}: stable."

    return SignalRow(
        metric_key="unemployment_rate",
        metric_name=SERIES_CONFIG["unemployment_rate"]["metric_name"],
        as_of_date=as_of.date().isoformat(),
        value=current,
        signal_state=state,
        threshold_rule="MoM >=0.2 weak; >=0.1 watch; <0.1 stable",
        message=message,
        source=_source(series, "FRED:UNRATE"),
        stale_days=_stale_days(as_of),
    )


def compute_signals(data: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compute independent metric signals from input series.

    Expected keys: m2, hiring_rate, ten_year_yield, buffett_ratio, unemployment_rate
    """
    rows: list[SignalRow] = []

    if "m2" in data:
        rows.append(_m2_signal(data["m2"]))
    if "hiring_rate" in data:
        rows.append(_hiring_signal(data["hiring_rate"]))
    if "ten_year_yield" in data:
        rows.append(_ten_year_signal(data["ten_year_yield"]))
    if "buffett_ratio" in data:
        rows.append(_buffett_signal(data["buffett_ratio"]))
    if "unemployment_rate" in data:
        rows.append(_unemployment_signal(data["unemployment_rate"]))

    frame = pd.DataFrame([row.__dict__ for row in rows])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "metric_key",
                "metric_name",
                "as_of_date",
                "value",
                "signal_state",
                "threshold_rule",
                "message",
                "source",
                "stale_days",
            ]
        )

    return frame[
        [
            "metric_key",
            "metric_name",
            "as_of_date",
            "value",
            "signal_state",
            "threshold_rule",
            "message",
            "source",
            "stale_days",
        ]
    ]

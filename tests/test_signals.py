from __future__ import annotations

import pandas as pd

from src.signals import compute_signals
from src.transform import build_buffett_ratio


def _series(values: list[float], start: str = "2024-01-31", freq: str = "M") -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


def _build_data(
    m2_values: list[float],
    hiring_value: float,
    ten_year_value: float,
    unrate_values: list[float],
    buffett_ratio_value: float,
) -> dict[str, pd.Series]:
    m2 = _series(m2_values)
    hiring = _series([hiring_value], start="2025-12-31")
    ten_year = _series([ten_year_value], start="2025-12-31")
    unrate = _series(unrate_values)

    willshire = _series([200.0, 210.0], start="2025-11-30")
    gdp = _series([100.0, 100.0], start="2025-10-31")
    buffett = build_buffett_ratio(willshire, gdp)
    buffett.iloc[-1] = buffett_ratio_value

    m2.attrs["source"] = "FRED:M2SL"
    hiring.attrs["source"] = "FRED:JTSHIL"
    ten_year.attrs["source"] = "FRED:DGS10"
    unrate.attrs["source"] = "FRED:UNRATE"
    buffett.attrs["source"] = "FRED:WILL5000PR/FRED:GDP"

    return {
        "m2": m2,
        "hiring_rate": hiring,
        "ten_year_yield": ten_year,
        "buffett_ratio": buffett,
        "unemployment_rate": unrate,
    }


def _state(df: pd.DataFrame, key: str) -> str:
    return str(df.loc[df["metric_key"] == key, "signal_state"].iloc[0])


def test_hiring_threshold_boundary_at_34() -> None:
    data = _build_data(
        m2_values=[100.0] * 13,
        hiring_value=3.4,
        ten_year_value=4.3,
        unrate_values=[4.0, 4.0],
        buffett_ratio_value=1.9,
    )
    result = compute_signals(data)
    assert _state(result, "hiring_rate") == "recession_alert"


def test_ten_year_threshold_boundaries() -> None:
    data_44 = _build_data(
        m2_values=[100.0] * 13,
        hiring_value=3.5,
        ten_year_value=4.4,
        unrate_values=[4.0, 4.0],
        buffett_ratio_value=1.9,
    )
    result_44 = compute_signals(data_44)
    assert _state(result_44, "ten_year_yield") == "equity_pressure_zone"

    data_50 = _build_data(
        m2_values=[100.0] * 13,
        hiring_value=3.5,
        ten_year_value=5.0,
        unrate_values=[4.0, 4.0],
        buffett_ratio_value=1.9,
    )
    result_50 = compute_signals(data_50)
    assert _state(result_50, "ten_year_yield") == "extreme_pressure_bond_opportunity"


def test_buffett_threshold_boundary_at_20() -> None:
    data = _build_data(
        m2_values=[100.0] * 13,
        hiring_value=3.5,
        ten_year_value=4.2,
        unrate_values=[4.0, 4.0],
        buffett_ratio_value=2.0,
    )
    result = compute_signals(data)
    assert _state(result, "buffett_ratio") == "overheat_peak_risk"


def test_unemployment_mom_threshold_boundaries() -> None:
    data_watch = _build_data(
        m2_values=[100.0] * 13,
        hiring_value=3.5,
        ten_year_value=4.2,
        unrate_values=[4.0, 4.1],
        buffett_ratio_value=1.8,
    )
    result_watch = compute_signals(data_watch)
    assert _state(result_watch, "unemployment_rate") == "watch"

    data_weak = _build_data(
        m2_values=[100.0] * 13,
        hiring_value=3.5,
        ten_year_value=4.2,
        unrate_values=[4.0, 4.2],
        buffett_ratio_value=1.8,
    )
    result_weak = compute_signals(data_weak)
    assert _state(result_weak, "unemployment_rate") == "labor_weakening"


def test_m2_yoy_boundary_at_zero() -> None:
    m2_values = [100.0] * 13
    data = _build_data(
        m2_values=m2_values,
        hiring_value=3.5,
        ten_year_value=4.2,
        unrate_values=[4.0, 4.0],
        buffett_ratio_value=1.8,
    )
    result = compute_signals(data)
    assert _state(result, "m2") == "caution_contraction"

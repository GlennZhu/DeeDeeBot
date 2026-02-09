from __future__ import annotations

import pandas as pd
import pytest

from src.transform import build_buffett_ratio


def _series(values: list[float], start: str, freq: str) -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def test_build_buffett_ratio_applies_market_cap_unit_divisor() -> None:
    market_cap_millions = _series([30_000.0, 33_000.0], "2025-06-30", "Q")
    gdp_billions = _series([15.0, 15.0], "2025-06-30", "Q")

    ratio = build_buffett_ratio(market_cap_millions, gdp_billions, market_cap_unit_divisor=1000.0)

    assert ratio.iloc[-1] == pytest.approx(2.2)


def test_build_buffett_ratio_rejects_non_positive_divisor() -> None:
    market_cap = _series([30_000.0], "2025-06-30", "Q")
    gdp = _series([15.0], "2025-06-30", "Q")

    with pytest.raises(ValueError, match="market_cap_unit_divisor must be > 0"):
        build_buffett_ratio(market_cap, gdp, market_cap_unit_divisor=0)

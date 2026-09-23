"""Comfort and time-of-use schedules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from thermalpilot.config import ComfortSchedule, TariffConfig


def comfort_bounds(
    index: pd.DatetimeIndex, config: ComfortSchedule | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return time-varying bounds; 06:00--22:00 is treated as occupied."""

    cfg = config or ComfortSchedule()
    hours = index.hour + index.minute / 60.0
    daytime = (hours >= cfg.day_start_hour) & (hours < cfg.night_start_hour)
    lower = np.where(daytime, cfg.day_lower_c, cfg.night_lower_c).astype(float)
    upper = np.full(len(index), cfg.upper_c, dtype=float)
    return lower, upper


def time_of_use_prices(
    index: pd.DatetimeIndex, config: TariffConfig | None = None
) -> NDArray[np.float64]:
    """Illustrative synthetic Zurich tariff in CHF/kWh (not a measured tariff)."""

    cfg = config or TariffConfig()
    hours = index.hour + index.minute / 60.0
    price = np.full(len(index), cfg.standard_chf_per_kwh, dtype=float)
    off_peak = (hours >= cfg.off_peak_start_hour) | (hours < cfg.off_peak_end_hour)
    peak = (hours >= cfg.peak_start_hour) & (hours < cfg.peak_end_hour)
    price[off_peak] = cfg.off_peak_chf_per_kwh
    price[peak] = cfg.peak_chf_per_kwh
    return price

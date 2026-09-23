"""Controller interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class ControlDecision:
    power_w: float
    status: str = "ok"
    predicted_states: FloatArray | None = None
    predicted_power_w: FloatArray | None = None


class Controller(Protocol):
    name: str
    horizon_steps: int

    def reset(self) -> None: ...

    def compute(
        self,
        state: FloatArray,
        disturbance_forecast: FloatArray,
        lower_bounds_c: FloatArray,
        upper_bounds_c: FloatArray,
        prices_chf_per_kwh: FloatArray,
        previous_power_w: float,
    ) -> ControlDecision: ...

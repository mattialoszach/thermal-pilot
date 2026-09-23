"""On/off thermostat benchmark."""

from __future__ import annotations

from thermalpilot.controllers.base import ControlDecision, FloatArray


class HysteresisThermostat:
    """Heat at full power below the lower bound, with an anti-chatter deadband."""

    name = "Hysteresis thermostat"
    horizon_steps = 1

    def __init__(self, p_max_w: float = 6000.0, deadband_c: float = 0.4) -> None:
        self.p_max_w = float(p_max_w)
        self.deadband_c = float(deadband_c)
        self._on = False

    def reset(self) -> None:
        self._on = False

    def compute(
        self,
        state: FloatArray,
        disturbance_forecast: FloatArray,
        lower_bounds_c: FloatArray,
        upper_bounds_c: FloatArray,
        prices_chf_per_kwh: FloatArray,
        previous_power_w: float,
    ) -> ControlDecision:
        del disturbance_forecast, upper_bounds_c, prices_chf_per_kwh, previous_power_w
        lower = float(lower_bounds_c[0])
        if state[0] <= lower:
            self._on = True
        elif state[0] >= lower + self.deadband_c:
            self._on = False
        power = self.p_max_w if self._on else 0.0
        return ControlDecision(power_w=power)

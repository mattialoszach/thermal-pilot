"""Closed-loop receding-horizon simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thermalpilot.controllers.base import Controller
from thermalpilot.model import DiscreteModel


@dataclass
class PredictionSnapshot:
    step: int
    timestamp: pd.Timestamp
    states: np.ndarray
    power_w: np.ndarray


@dataclass
class SimulationResult:
    controller_name: str
    frame: pd.DataFrame
    predictions: dict[int, PredictionSnapshot] = field(default_factory=dict)
    solver_failures: int = 0


def _noisy_forecast(
    forecast: np.ndarray, rng: np.random.Generator, temperature_std_c: float
) -> np.ndarray:
    if temperature_std_c <= 0:
        return forecast
    noisy = forecast.copy()
    errors = rng.normal(0.0, temperature_std_c, len(noisy))
    errors = np.convolve(errors, np.ones(7) / 7.0, mode="same")
    noisy[:, 0] += errors
    solar_factor = np.clip(rng.normal(1.0, 0.18, len(noisy)), 0.25, 1.75)
    noisy[:, 1] = np.maximum(0.0, noisy[:, 1] * solar_factor)
    return noisy


def run_simulation(
    controller: Controller,
    model: DiscreteModel,
    weather: pd.DataFrame,
    lower_bounds_c: np.ndarray,
    upper_bounds_c: np.ndarray,
    prices_chf_per_kwh: np.ndarray,
    steps: int,
    initial_state_c: np.ndarray | None = None,
    forecast_noise_std_c: float = 0.0,
    seed: int = 7,
    prediction_stride: int = 96,
) -> SimulationResult:
    """Run observe-optimize-apply-propagate for ``steps`` intervals."""

    horizon = controller.horizon_steps
    required = steps + horizon
    if len(weather) < required:
        raise ValueError(f"Weather needs at least {required} rows; got {len(weather)}")
    for values, label in [
        (lower_bounds_c, "comfort lower bounds"),
        (upper_bounds_c, "comfort upper bounds"),
        (prices_chf_per_kwh, "electricity prices"),
    ]:
        if len(values) < required:
            raise ValueError(f"Need at least {required} {label}")

    state = np.array(initial_state_c if initial_state_c is not None else [20.0, 18.5, 19.0])
    controller.reset()
    previous_power = 0.0
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | str]] = []
    snapshots: dict[int, PredictionSnapshot] = {}
    solver_failures = 0
    disturbances = weather[["temp_air_c", "solar_w_m2"]].to_numpy(dtype=float)

    for k in range(steps):
        forecast = disturbances[k : k + horizon]
        forecast = _noisy_forecast(forecast, rng, forecast_noise_std_c)
        # State x[k+1] is constrained by the comfort schedule at timestamp k+1.
        decision = controller.compute(
            state=state,
            disturbance_forecast=forecast,
            lower_bounds_c=lower_bounds_c[k + 1 : k + horizon + 1],
            upper_bounds_c=upper_bounds_c[k + 1 : k + horizon + 1],
            prices_chf_per_kwh=prices_chf_per_kwh[k : k + horizon],
            previous_power_w=previous_power,
        )
        if not decision.status.startswith("optimal") and decision.status != "ok":
            solver_failures += 1
        power = float(np.clip(decision.power_w, 0.0, np.inf))
        records.append(
            {
                "T_air_c": state[0],
                "T_wall_c": state[1],
                "T_mass_c": state[2],
                "T_ambient_c": disturbances[k, 0],
                "G_solar_w_m2": disturbances[k, 1],
                "P_heat_w": power,
                "comfort_lower_c": lower_bounds_c[k],
                "comfort_upper_c": upper_bounds_c[k],
                "price_chf_per_kwh": prices_chf_per_kwh[k],
                "solver_status": decision.status,
            }
        )
        if (
            decision.predicted_states is not None
            and decision.predicted_power_w is not None
            and (k == 0 or k % prediction_stride == 0)
        ):
            snapshots[k] = PredictionSnapshot(
                step=k,
                timestamp=weather.index[k],
                states=decision.predicted_states,
                power_w=decision.predicted_power_w,
            )
        state = model.step(state, power, disturbances[k])
        previous_power = power

    frame = pd.DataFrame.from_records(records, index=weather.index[:steps])
    frame.index.name = "timestamp"
    frame.attrs.update(weather.attrs)
    frame.attrs.update(
        {
            "controller": controller.name,
            "dt_hours": model.dt_seconds / 3600.0,
            "forecast_noise_std_c": forecast_noise_std_c,
        }
    )
    return SimulationResult(controller.name, frame, snapshots, solver_failures)

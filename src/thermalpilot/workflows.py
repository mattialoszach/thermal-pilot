"""High-level, reusable simulation workflows for the CLI and dashboard."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from thermalpilot.config import (
    BuildingParameters,
    ComfortSchedule,
    MPCConfig,
    SimulationConfig,
    TariffConfig,
)
from thermalpilot.controllers import HysteresisThermostat, MPCController
from thermalpilot.model import RCModel
from thermalpilot.schedules import comfort_bounds, time_of_use_prices
from thermalpilot.simulator import SimulationResult, run_simulation
from thermalpilot.weather import WeatherSource, load_weather

CONTROLLER_KEYS = ("thermostat", "energy-mpc", "price-mpc")


def simulate_project(
    simulation: SimulationConfig | None = None,
    building: BuildingParameters | None = None,
    comfort: ComfortSchedule | None = None,
    tariff: TariffConfig | None = None,
    mpc: MPCConfig | None = None,
    weather_source: WeatherSource = "auto",
    controller_keys: tuple[str, ...] = CONTROLLER_KEYS,
    forecast_noise_std_c: float = 0.0,
) -> tuple[list[SimulationResult], RCModel, pd.DataFrame]:
    """Prepare data and compare any subset of the built-in controllers."""

    sim = simulation or SimulationConfig()
    params = building or BuildingParameters()
    comfort_cfg = comfort or ComfortSchedule()
    tariff_cfg = tariff or TariffConfig()
    mpc_cfg = mpc or MPCConfig()
    if any(key not in CONTROLLER_KEYS for key in controller_keys):
        raise ValueError(f"controller_keys must be drawn from {CONTROLLER_KEYS}")

    start = pd.Timestamp(sim.start)
    if start.tzinfo is None:
        start = start.tz_localize(sim.timezone)
    else:
        start = start.tz_convert(sim.timezone)
    maximum_horizon = max(mpc_cfg.horizon_steps if "mpc" in " ".join(controller_keys) else 1, 1)
    end = start + pd.Timedelta(days=sim.days, minutes=sim.dt_minutes * maximum_horizon)
    weather = load_weather(
        start,
        end,
        source=weather_source,
        station=sim.station,
        interval_minutes=sim.dt_minutes,
        timezone=sim.timezone,
        seed=sim.seed,
    )
    lower, upper = comfort_bounds(weather.index, comfort_cfg)
    prices = time_of_use_prices(weather.index, tariff_cfg)

    rc_model = RCModel(params)
    discrete = rc_model.discretize(sim.dt_seconds)
    controllers = []
    if "thermostat" in controller_keys:
        controllers.append(HysteresisThermostat(params.p_max_w))
    if "energy-mpc" in controller_keys:
        controllers.append(MPCController(discrete, params.p_max_w, mpc_cfg, price_aware=False))
    if "price-mpc" in controller_keys:
        controllers.append(MPCController(discrete, params.p_max_w, mpc_cfg, price_aware=True))

    initial_state = np.array([sim.initial_air_c, sim.initial_wall_c, sim.initial_mass_c])
    results = [
        run_simulation(
            controller,
            discrete,
            weather,
            lower,
            upper,
            prices,
            steps=sim.steps,
            initial_state_c=initial_state,
            forecast_noise_std_c=forecast_noise_std_c,
            seed=sim.seed,
            prediction_stride=96,
        )
        for controller in controllers
    ]
    return results, rc_model, weather


def building_with_power(params: BuildingParameters, p_max_w: float) -> BuildingParameters:
    return replace(params, p_max_w=float(p_max_w))

import numpy as np

from thermalpilot.controllers import HysteresisThermostat
from thermalpilot.model import RCModel
from thermalpilot.schedules import comfort_bounds, time_of_use_prices
from thermalpilot.simulator import run_simulation
from thermalpilot.weather import synthetic_winter_weather


def test_closed_loop_result_has_expected_columns_and_bounds() -> None:
    model = RCModel().discretize(900)
    weather = synthetic_winter_weather("2024-01-15", "2024-01-15 06:00", seed=3)
    lower, upper = comfort_bounds(weather.index)
    prices = time_of_use_prices(weather.index)
    result = run_simulation(
        HysteresisThermostat(6000),
        model,
        weather,
        lower,
        upper,
        prices,
        steps=20,
    )
    assert len(result.frame) == 20
    assert {
        "T_air_c",
        "T_wall_c",
        "T_mass_c",
        "P_heat_w",
        "T_ambient_c",
        "G_solar_w_m2",
    }.issubset(result.frame.columns)
    assert result.frame["P_heat_w"].between(0, 6000).all()
    assert np.isfinite(result.frame.select_dtypes("number")).all().all()

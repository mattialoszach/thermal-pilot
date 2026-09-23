import numpy as np

from thermalpilot.config import MPCConfig
from thermalpilot.controllers import HysteresisThermostat, MPCController
from thermalpilot.model import RCModel


def test_hysteresis_respects_power_bounds_and_deadband() -> None:
    controller = HysteresisThermostat(p_max_w=4500, deadband_c=0.4)
    dummy_d = np.zeros((1, 2))
    prices = np.ones(1)
    on = controller.compute(np.array([19.5, 19, 19]), dummy_d, np.array([20]), np.array([22]), prices, 0)
    hold = controller.compute(np.array([20.2, 19, 19]), dummy_d, np.array([20]), np.array([22]), prices, on.power_w)
    off = controller.compute(np.array([20.5, 19, 19]), dummy_d, np.array([20]), np.array([22]), prices, hold.power_w)
    assert (on.power_w, hold.power_w, off.power_w) == (4500, 4500, 0)


def test_mpc_power_constraints_and_prediction_dynamics() -> None:
    discrete = RCModel().discretize(900)
    horizon = 8
    controller = MPCController(
        discrete,
        p_max_w=5000,
        config=MPCConfig(horizon_steps=horizon),
        price_aware=True,
    )
    disturbances = np.column_stack([np.full(horizon, -3.0), np.zeros(horizon)])
    decision = controller.compute(
        state=np.array([20.0, 18.5, 19.0]),
        disturbance_forecast=disturbances,
        lower_bounds_c=np.full(horizon, 20.0),
        upper_bounds_c=np.full(horizon, 22.0),
        prices_chf_per_kwh=np.linspace(0.18, 0.42, horizon),
        previous_power_w=0.0,
    )
    assert decision.status.startswith("optimal")
    assert 0 <= decision.power_w <= 5000
    assert decision.predicted_power_w is not None
    assert np.all(decision.predicted_power_w >= -1e-3)
    assert np.all(decision.predicted_power_w <= 5000 + 1e-3)
    expected = discrete.step(decision.predicted_states[:, 0], decision.predicted_power_w[0], disturbances[0])
    assert np.allclose(decision.predicted_states[:, 1], expected, atol=1e-4)


def test_mpc_slack_keeps_extreme_problem_feasible() -> None:
    discrete = RCModel().discretize(900)
    horizon = 4
    controller = MPCController(
        discrete,
        p_max_w=300,
        config=MPCConfig(horizon_steps=horizon),
        price_aware=False,
    )
    decision = controller.compute(
        state=np.array([15.0, 10.0, 12.0]),
        disturbance_forecast=np.column_stack([np.full(horizon, -25.0), np.zeros(horizon)]),
        lower_bounds_c=np.full(horizon, 24.0),
        upper_bounds_c=np.full(horizon, 25.0),
        prices_chf_per_kwh=np.ones(horizon),
        previous_power_w=0.0,
    )
    assert decision.status.startswith("optimal")
    assert 0 <= decision.power_w <= 300

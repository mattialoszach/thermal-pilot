"""Convex receding-horizon MPC with explicit comfort slack variables."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from thermalpilot.config import MPCConfig
from thermalpilot.controllers.base import ControlDecision, FloatArray
from thermalpilot.model import DiscreteModel


class MPCController:
    """Linear MPC for heating-only climate control.

    ``price_aware=False`` assigns a flat unit energy weight and therefore
    minimizes heating energy. ``price_aware=True`` uses the supplied time-of-use
    prices, allowing economically useful pre-heating inside the comfort band.
    """

    def __init__(
        self,
        model: DiscreteModel,
        p_max_w: float = 6000.0,
        config: MPCConfig | None = None,
        price_aware: bool = True,
    ) -> None:
        self.model = model
        self.p_max_w = float(p_max_w)
        self.p_max_kw = self.p_max_w / 1000.0
        self.config = config or MPCConfig()
        self.horizon_steps = self.config.horizon_steps
        self.price_aware = price_aware
        self.name = "Price-aware MPC" if price_aware else "Energy-minimizing MPC"
        self._build_problem()

    def _build_problem(self) -> None:
        n = self.horizon_steps
        self.x0 = cp.Parameter(3, name="x0")
        self.d = cp.Parameter((2, n), name="disturbance")
        self.lower = cp.Parameter(n, name="lower")
        self.upper = cp.Parameter(n, name="upper")
        self.price = cp.Parameter(n, nonneg=True, name="price")
        self.previous_power_kw = cp.Parameter(nonneg=True, name="previous_power_kw")

        self.x = cp.Variable((3, n + 1), name="state")
        # Optimize in kW rather than W so OSQP sees similarly scaled state,
        # slack, and input variables. The physical model itself remains in W.
        self.u_kw = cp.Variable(n, name="heat_power_kw")
        self.slack_low = cp.Variable(n, nonneg=True, name="slack_low")
        self.slack_high = cp.Variable(n, nonneg=True, name="slack_high")

        constraints: list[cp.Constraint] = [self.x[:, 0] == self.x0]
        for k in range(n):
            constraints += [
                self.x[:, k + 1]
                == self.model.A @ self.x[:, k]
                + self.model.B[:, 0] * (1000.0 * self.u_kw[k])
                + self.model.E @ self.d[:, k],
                self.u_kw[k] >= 0.0,
                self.u_kw[k] <= self.p_max_kw,
                self.x[0, k + 1] >= self.lower[k] - self.slack_low[k],
                self.x[0, k + 1] <= self.upper[k] + self.slack_high[k],
            ]

        dt_hours = self.model.dt_seconds / 3600.0
        energy_kwh = self.u_kw * dt_hours
        energy_cost = cp.sum(cp.multiply(self.price, energy_kwh))
        comfort_cost = self.config.comfort_weight * dt_hours * (
            cp.sum_squares(self.slack_low) + cp.sum_squares(self.slack_high)
        )
        ramps = cp.hstack(
            [
                self.u_kw[0] - self.previous_power_kw,
                self.u_kw[1:] - self.u_kw[:-1],
            ]
        )
        ramp_cost = self.config.ramp_weight * cp.sum_squares(ramps / self.p_max_kw)
        self.problem = cp.Problem(
            cp.Minimize(energy_cost + comfort_cost + ramp_cost), constraints
        )

    def reset(self) -> None:
        for variable in (self.x, self.u_kw, self.slack_low, self.slack_high):
            variable.value = None

    def compute(
        self,
        state: FloatArray,
        disturbance_forecast: FloatArray,
        lower_bounds_c: FloatArray,
        upper_bounds_c: FloatArray,
        prices_chf_per_kwh: FloatArray,
        previous_power_w: float,
    ) -> ControlDecision:
        n = self.horizon_steps
        if disturbance_forecast.shape != (n, 2):
            raise ValueError(f"Expected disturbance forecast shape {(n, 2)}")
        for values, label in [
            (lower_bounds_c, "lower bounds"),
            (upper_bounds_c, "upper bounds"),
            (prices_chf_per_kwh, "prices"),
        ]:
            if len(values) != n:
                raise ValueError(f"Expected {n} {label}")

        self.x0.value = np.asarray(state, dtype=float)
        self.d.value = np.asarray(disturbance_forecast, dtype=float).T
        self.lower.value = np.asarray(lower_bounds_c, dtype=float)
        self.upper.value = np.asarray(upper_bounds_c, dtype=float)
        if self.price_aware:
            self.price.value = np.asarray(prices_chf_per_kwh, dtype=float)
        else:
            self.price.value = np.ones(n, dtype=float)
        self.previous_power_kw.value = max(0.0, float(previous_power_w) / 1000.0)

        try:
            self.problem.solve(
                solver=self.config.solver,
                warm_start=True,
                verbose=False,
                max_iter=self.config.solver_max_iter,
                eps_abs=1e-4,
                eps_rel=1e-4,
                polishing=True,
            )
        except cp.error.SolverError as exc:
            return ControlDecision(power_w=0.0, status=f"solver_error:{exc}")
        if self.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return ControlDecision(power_w=0.0, status=str(self.problem.status))

        power = float(np.clip(self.u_kw.value[0] * 1000.0, 0.0, self.p_max_w))
        return ControlDecision(
            power_w=power,
            status=str(self.problem.status),
            predicted_states=np.asarray(self.x.value, dtype=float).copy(),
            predicted_power_w=np.asarray(self.u_kw.value, dtype=float).copy() * 1000.0,
        )

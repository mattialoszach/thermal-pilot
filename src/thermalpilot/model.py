"""Continuous 3R/3C-style model and exact zero-order-hold discretization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from thermalpilot.config import BuildingParameters

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DiscreteModel:
    """Discrete LTI model ``x+ = A x + B u + E d``."""

    A: FloatArray
    B: FloatArray
    E: FloatArray
    dt_seconds: float

    def step(self, x: FloatArray, u_w: float, disturbance: FloatArray) -> FloatArray:
        return self.A @ x + self.B[:, 0] * float(u_w) + self.E @ disturbance


class RCModel:
    """Physically interpretable air, envelope, and internal-mass model."""

    state_names = ("T_air", "T_wall", "T_mass")
    disturbance_names = ("T_ambient", "G_solar")

    def __init__(self, parameters: BuildingParameters | None = None) -> None:
        self.p = parameters or BuildingParameters()
        self.p.validate()

    def continuous_matrices(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Derive ``A_c, B_c, E_c`` directly from the energy balances.

        State temperatures are degrees C (temperature differences are K), input
        power is W, solar irradiance is W/m2, and time is seconds.
        """

        p = self.p
        ca, cw, cm = p.c_air_j_per_k, p.c_wall_j_per_k, p.c_mass_j_per_k
        raw, ram = p.r_air_wall_k_per_w, p.r_air_mass_k_per_w
        rao, rwo = p.r_air_out_k_per_w, p.r_wall_out_k_per_w

        a_c = np.array(
            [
                [-(1 / raw + 1 / ram + 1 / rao) / ca, 1 / (raw * ca), 1 / (ram * ca)],
                [1 / (raw * cw), -(1 / raw + 1 / rwo) / cw, 0.0],
                [1 / (ram * cm), 0.0, -1 / (ram * cm)],
            ],
            dtype=float,
        )
        b_c = np.array([[p.eta_heater / ca], [0.0], [0.0]], dtype=float)
        e_c = np.array(
            [
                [1 / (rao * ca), p.alpha_air_m2 / ca],
                [1 / (rwo * cw), p.alpha_wall_m2 / cw],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        return a_c, b_c, e_c

    def discretize(self, dt_seconds: float = 900.0) -> DiscreteModel:
        """Exact ZOH discretization through one augmented matrix exponential."""

        if dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        a_c, b_c, e_c = self.continuous_matrices()
        n_x, n_u, n_d = 3, 1, 2
        augmented = np.zeros((n_x + n_u + n_d, n_x + n_u + n_d))
        augmented[:n_x, :n_x] = a_c
        augmented[:n_x, n_x : n_x + n_u] = b_c
        augmented[:n_x, n_x + n_u :] = e_c
        transition = expm(augmented * dt_seconds)
        return DiscreteModel(
            A=transition[:n_x, :n_x],
            B=transition[:n_x, n_x : n_x + n_u],
            E=transition[:n_x, n_x + n_u :],
            dt_seconds=dt_seconds,
        )

    def plausibility_report(self, dt_seconds: float = 900.0) -> dict[str, float | bool]:
        """Return transparent order-of-magnitude and stability checks."""

        discrete = self.discretize(dt_seconds)
        eig = np.linalg.eigvals(discrete.A)
        p = self.p
        physical_air_capacity = 1.204 * 1006.0 * p.volume_m3
        return {
            "apartment_volume_m3": p.volume_m3,
            "physical_air_capacity_mj_per_k": physical_air_capacity / 1e6,
            "effective_air_capacity_mj_per_k": p.c_air_j_per_k / 1e6,
            "heat_loss_coefficient_w_per_k": p.heat_loss_coefficient_w_per_k,
            "specific_heat_loss_w_per_m2k": (
                p.heat_loss_coefficient_w_per_k / p.floor_area_m2
            ),
            "max_discrete_eigenvalue_magnitude": float(np.max(np.abs(eig))),
            "discrete_stable": bool(np.all(np.abs(eig) < 1.0)),
        }

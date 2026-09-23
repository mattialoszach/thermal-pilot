"""Typed, reproducible configuration for the synthetic apartment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildingParameters:
    """Assumed parameters for a synthetic 70 m2 Zurich apartment.

    Capacitances use J/K, resistances K/W, effective solar apertures m2,
    heater power W, and ``eta_heater`` is dimensionless. These values are
    engineering assumptions, not measurements from an actual apartment.
    """

    floor_area_m2: float = 70.0
    ceiling_height_m: float = 2.5
    c_air_j_per_k: float = 1.50e6
    c_wall_j_per_k: float = 12.0e6
    c_mass_j_per_k: float = 10.0e6
    r_air_wall_k_per_w: float = 0.015
    r_air_mass_k_per_w: float = 0.010
    r_air_out_k_per_w: float = 0.050
    r_wall_out_k_per_w: float = 0.025
    eta_heater: float = 0.98
    alpha_air_m2: float = 1.5
    alpha_wall_m2: float = 3.0
    p_max_w: float = 6000.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        positive = {
            key: value
            for key, value in self.as_dict().items()
            if key not in {"eta_heater"}
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("All building parameters must be positive")
        if not 0 < self.eta_heater <= 1.5:
            raise ValueError("eta_heater must be in (0, 1.5]")

    @property
    def volume_m3(self) -> float:
        return self.floor_area_m2 * self.ceiling_height_m

    @property
    def heat_loss_coefficient_w_per_k(self) -> float:
        """Steady conductance to outdoors, including the series wall path."""

        wall_path = 1.0 / (self.r_air_wall_k_per_w + self.r_wall_out_k_per_w)
        direct_path = 1.0 / self.r_air_out_k_per_w
        return wall_path + direct_path


@dataclass(frozen=True)
class ComfortSchedule:
    day_lower_c: float = 20.0
    night_lower_c: float = 17.0
    upper_c: float = 22.0
    day_start_hour: int = 6
    night_start_hour: int = 22


@dataclass(frozen=True)
class TariffConfig:
    off_peak_chf_per_kwh: float = 0.18
    standard_chf_per_kwh: float = 0.30
    peak_chf_per_kwh: float = 0.42
    off_peak_end_hour: int = 6
    peak_start_hour: int = 17
    peak_end_hour: int = 21
    off_peak_start_hour: int = 22


@dataclass(frozen=True)
class MPCConfig:
    horizon_steps: int = 96
    comfort_weight: float = 250.0
    ramp_weight: float = 0.08
    solver: str = "OSQP"
    solver_max_iter: int = 20_000


@dataclass(frozen=True)
class SimulationConfig:
    dt_minutes: int = 15
    start: str = "2024-01-15"
    days: int = 7
    timezone: str = "Europe/Zurich"
    station: str = "SMA"
    seed: int = 7
    initial_air_c: float = 20.0
    initial_wall_c: float = 18.5
    initial_mass_c: float = 19.0

    @property
    def dt_seconds(self) -> float:
        return self.dt_minutes * 60.0

    @property
    def steps(self) -> int:
        return self.days * 24 * 60 // self.dt_minutes


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "thermalpilot"

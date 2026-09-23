"""Controller performance metrics."""

from __future__ import annotations

import pandas as pd

from thermalpilot.simulator import SimulationResult


def compute_metrics(result: SimulationResult) -> dict[str, float | str]:
    frame = result.frame
    dt_h = float(frame.attrs.get("dt_hours", 0.25))
    energy_kwh = (frame["P_heat_w"] * dt_h / 1000.0).sum()
    interval_energy = frame["P_heat_w"] * dt_h / 1000.0
    lower_violation = (frame["comfort_lower_c"] - frame["T_air_c"]).clip(lower=0.0)
    upper_violation = (frame["T_air_c"] - frame["comfort_upper_c"]).clip(lower=0.0)
    # MPC commonly sits a few 1e-4 K below an active bound because of solver
    # tolerances. Count an interval as uncomfortable only beyond 0.05 K, while
    # the degree-hour metric below retains the full continuous magnitude.
    counting_tolerance_c = 0.05
    return {
        "controller": result.controller_name,
        "heating_energy_kwh": float(energy_kwh),
        "electricity_cost_chf": float((interval_energy * frame["price_chf_per_kwh"]).sum()),
        "comfort_violation_degree_hours": float(((lower_violation + upper_violation) * dt_h).sum()),
        "hours_below_comfort": float(
            lower_violation.gt(counting_tolerance_c).sum() * dt_h
        ),
        "hours_above_comfort": float(
            upper_violation.gt(counting_tolerance_c).sum() * dt_h
        ),
        "peak_heating_power_kw": float(frame["P_heat_w"].max() / 1000.0),
        "mean_indoor_temperature_c": float(frame["T_air_c"].mean()),
        "min_indoor_temperature_c": float(frame["T_air_c"].min()),
        "max_indoor_temperature_c": float(frame["T_air_c"].max()),
        "solver_failures": float(result.solver_failures),
    }


def comparison_table(results: list[SimulationResult]) -> pd.DataFrame:
    return pd.DataFrame([compute_metrics(result) for result in results]).set_index("controller")

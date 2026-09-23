"""Streamlit dashboard. Run with: streamlit run src/thermalpilot/dashboard.py"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from thermalpilot.config import (
    BuildingParameters,
    ComfortSchedule,
    MPCConfig,
    SimulationConfig,
    TariffConfig,
)
from thermalpilot.metrics import comparison_table
from thermalpilot.workflows import CONTROLLER_KEYS, simulate_project


def _interactive_figure(results):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    reference = results[0].frame
    fig.add_trace(
        go.Scatter(x=reference.index, y=reference["comfort_upper_c"], line={"width": 0}, showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reference.index,
            y=reference["comfort_lower_c"],
            fill="tonexty",
            fillcolor="rgba(80,145,190,0.18)",
            line={"width": 0},
            name="Comfort band",
        ),
        row=1,
        col=1,
    )
    for result in results:
        frame = result.frame
        fig.add_trace(go.Scatter(x=frame.index, y=frame["T_air_c"], name=result.controller_name), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["P_heat_w"] / 1000,
                name=result.controller_name,
                legendgroup=result.controller_name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title="Temperature [°C]", row=1, col=1)
    fig.update_yaxes(title="Heating [kW]", row=2, col=1)
    fig.update_layout(height=700, hovermode="x unified", legend={"orientation": "h"})
    return fig


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - only exercised without optional dependency
        raise SystemExit("Install dashboard dependencies: uv sync --extra dashboard") from exc

    st.set_page_config(page_title="ThermalPilot", layout="wide")
    st.title("ThermalPilot")
    st.caption("Single-zone 3-state RC model · Zurich weather · receding-horizon MPC")
    with st.sidebar:
        st.header("Simulation")
        controller_label = st.selectbox(
            "Controller", ["Compare all", "Thermostat", "Energy MPC", "Price-aware MPC"]
        )
        source = st.selectbox("Weather", ["auto", "meteoswiss", "synthetic"])
        start = st.date_input("Start date", value=date(2024, 1, 15))
        days = st.slider("Days", 1, 14, 7)
        horizon = st.slider("Prediction horizon [steps]", 16, 192, 96, step=8)
        p_max_kw = st.slider("Maximum heating power [kW]", 2.0, 10.0, 6.0, step=0.5)
        st.header("Comfort")
        day_lower = st.slider("Occupied lower [°C]", 18.0, 22.0, 20.0, step=0.5)
        night_lower = st.slider("Night lower [°C]", 14.0, 20.0, 17.0, step=0.5)
        upper = st.slider("Upper [°C]", 20.0, 25.0, 22.0, step=0.5)
        st.header("Tariff")
        off_peak = st.number_input("Off-peak [CHF/kWh]", 0.0, 1.0, 0.18, step=0.01)
        standard = st.number_input("Standard [CHF/kWh]", 0.0, 1.0, 0.30, step=0.01)
        peak = st.number_input("Peak [CHF/kWh]", 0.0, 1.0, 0.42, step=0.01)
        run = st.button("Run simulation", type="primary", use_container_width=True)

    if not run:
        st.info("Choose settings and run the simulation. MeteoSwiss data are cached automatically.")
        return
    if not (night_lower <= day_lower < upper):
        st.error("Comfort bounds must satisfy night lower ≤ occupied lower < upper.")
        return
    label_to_keys = {
        "Compare all": CONTROLLER_KEYS,
        "Thermostat": ("thermostat",),
        "Energy MPC": ("energy-mpc",),
        "Price-aware MPC": ("price-mpc",),
    }
    with st.spinner("Solving closed-loop control problems…"):
        results, model, weather = simulate_project(
            simulation=SimulationConfig(start=str(start), days=days),
            building=replace(BuildingParameters(), p_max_w=p_max_kw * 1000),
            comfort=ComfortSchedule(day_lower_c=day_lower, night_lower_c=night_lower, upper_c=upper),
            tariff=TariffConfig(
                off_peak_chf_per_kwh=off_peak,
                standard_chf_per_kwh=standard,
                peak_chf_per_kwh=peak,
            ),
            mpc=MPCConfig(horizon_steps=horizon),
            weather_source=source,
            controller_keys=label_to_keys[controller_label],
        )
    st.dataframe(comparison_table(results).round(3), use_container_width=True)
    st.plotly_chart(_interactive_figure(results), use_container_width=True)
    source_note = weather.attrs.get("source", "unknown")
    st.caption(f"Weather source: {source_note}. Building parameters are assumed/synthetic, not measured.")
    with st.expander("Physical plausibility checks"):
        st.json(model.plausibility_report())


if __name__ == "__main__":
    main()

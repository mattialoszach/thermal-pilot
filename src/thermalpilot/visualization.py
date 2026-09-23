"""Publication figures, interactive Plotly output, and apartment animation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from plotly.subplots import make_subplots

from thermalpilot.metrics import comparison_table
from thermalpilot.simulator import PredictionSnapshot, SimulationResult


def _style_time_axis(axes: list[plt.Axes]) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    for axis in axes:
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)
        axis.grid(True, color="#d7dce2", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)


def plot_timeseries(
    result: SimulationResult,
    output: str | Path | None = None,
    prediction: PredictionSnapshot | None = None,
    dpi: int = 180,
) -> Figure:
    """Create the main publication-quality thermal/control time series."""

    frame = result.frame
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0, 1.0, 0.8]},
        constrained_layout=True,
    )
    ax_t, ax_u, ax_s, ax_p = axes
    index = frame.index

    ax_t.fill_between(
        index,
        frame["comfort_lower_c"],
        frame["comfort_upper_c"],
        color="#80b1d3",
        alpha=0.18,
        label="Comfort band",
    )
    ax_t.plot(index, frame["T_air_c"], color="#d73027", linewidth=1.7, label="Indoor air")
    ax_t.plot(index, frame["T_wall_c"], color="#7b3294", linewidth=1.1, label="Wall")
    ax_t.plot(index, frame["T_mass_c"], color="#008837", linewidth=1.1, label="Internal mass")
    ax_t.plot(
        index,
        frame["T_ambient_c"],
        color="#4575b4",
        linewidth=1.0,
        alpha=0.9,
        label="Outdoor air",
    )
    if prediction is not None:
        dt = pd.Timedelta(hours=float(frame.attrs.get("dt_hours", 0.25)))
        predicted_index = pd.date_range(
            prediction.timestamp, periods=prediction.states.shape[1], freq=dt
        )
        ax_t.plot(
            predicted_index,
            prediction.states[0],
            color="#111111",
            linestyle="--",
            linewidth=1.3,
            label="MPC predicted air",
        )
    ax_t.set_ylabel("Temperature [°C]")
    ax_t.legend(ncol=3, loc="upper right", frameon=False)
    ax_t.set_title(f"ThermalPilot — {result.controller_name}", loc="left", weight="bold")

    ax_u.fill_between(index, 0, frame["P_heat_w"] / 1000.0, color="#e6550d", alpha=0.72)
    ax_u.set_ylabel("Heating [kW]")
    ax_u.set_ylim(bottom=0)

    ax_s.fill_between(index, 0, frame["G_solar_w_m2"], color="#fdbf11", alpha=0.62)
    ax_s.set_ylabel("Solar [W/m²]")
    ax_s.set_ylim(bottom=0)

    ax_p.step(index, frame["price_chf_per_kwh"], where="post", color="#4d4d4d", linewidth=1.3)
    ax_p.set_ylabel("Price\n[CHF/kWh]")
    ax_p.set_xlabel(f"Time [{index.tz}]" if index.tz is not None else "Time")
    ax_p.set_ylim(bottom=0)
    _style_time_axis(list(axes))

    source = frame.attrs.get("source", "unknown weather source")
    station = frame.attrs.get("station", "")
    fig.text(0.01, 0.002, f"Weather: {source} {station}".strip(), fontsize=8, color="#555555")
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_controller_comparison(
    results: list[SimulationResult], output: str | Path | None = None, dpi: int = 180
) -> Figure:
    """Compare indoor temperature, power, and cumulative cost."""

    colors = ["#444444", "#1b9e77", "#d95f02", "#7570b3"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True, constrained_layout=True)
    ax_t, ax_u, ax_c = axes
    reference = results[0].frame
    ax_t.fill_between(
        reference.index,
        reference["comfort_lower_c"],
        reference["comfort_upper_c"],
        color="#80b1d3",
        alpha=0.16,
        label="Comfort band",
    )
    for color, result in zip(colors, results, strict=False):
        frame = result.frame
        dt_h = float(frame.attrs.get("dt_hours", 0.25))
        ax_t.plot(frame.index, frame["T_air_c"], color=color, linewidth=1.25, label=result.controller_name)
        ax_u.plot(frame.index, frame["P_heat_w"] / 1000.0, color=color, linewidth=0.9)
        interval_cost = frame["P_heat_w"] * dt_h / 1000.0 * frame["price_chf_per_kwh"]
        ax_c.plot(frame.index, interval_cost.cumsum(), color=color, linewidth=1.35)
    ax_t.set_ylabel("Indoor [°C]")
    ax_u.set_ylabel("Heating [kW]")
    ax_c.set_ylabel("Cumulative cost [CHF]")
    ax_c.set_xlabel("Time")
    ax_u.set_ylim(bottom=0)
    ax_c.set_ylim(bottom=0)
    ax_t.legend(ncol=2, frameon=False, loc="upper right")
    ax_t.set_title("Controller comparison", loc="left", weight="bold")
    _style_time_axis(list(axes))
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def write_plotly_dashboard(results: list[SimulationResult], output: str | Path) -> Path:
    """Write a self-contained interactive controller comparison (Plotly HTML)."""

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.43, 0.21, 0.18, 0.18],
        subplot_titles=("Temperature", "Heating power", "Solar radiation", "Electricity price"),
    )
    reference = results[0].frame
    fig.add_trace(
        go.Scatter(
            x=reference.index,
            y=reference["comfort_upper_c"],
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name="Upper comfort",
        ),
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
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    for result in results:
        frame = result.frame
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["T_air_c"], mode="lines", name=result.controller_name),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["P_heat_w"] / 1000.0,
                mode="lines",
                name=result.controller_name,
                legendgroup=result.controller_name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=reference.index,
            y=reference["T_ambient_c"],
            mode="lines",
            line={"color": "#6699cc", "width": 1},
            name="Outdoor",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reference.index,
            y=reference["G_solar_w_m2"],
            mode="lines",
            fill="tozeroy",
            line={"color": "#e9ad00"},
            name="Solar",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=reference.index,
            y=reference["price_chf_per_kwh"],
            mode="lines",
            line={"shape": "hv", "color": "#555555"},
            name="Price",
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="W/m²", row=3, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="CHF/kWh", row=4, col=1, rangemode="tozero")
    fig.update_layout(
        title="ThermalPilot interactive simulation",
        template="plotly_white",
        height=900,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.04},
        margin={"l": 75, "r": 30, "t": 100, "b": 50},
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def animate_apartment(
    result: SimulationResult,
    output: str | Path,
    frame_stride: int = 2,
    fps: int = 15,
    dpi: int = 130,
) -> Path:
    """Animate room, wall, mass, heater, weather, and tariff state."""

    frame = result.frame
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    axis.set_xlim(0, 12.5)
    axis.set_ylim(0, 7.2)
    axis.set_aspect("equal")
    axis.axis("off")
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=14.0, vmax=24.0)

    wall = Rectangle((0.8, 0.8), 8.0, 5.6, facecolor=cmap(norm(18)), edgecolor="#333333", lw=2.0)
    air = Rectangle((1.25, 1.25), 7.1, 4.7, facecolor=cmap(norm(20)), edgecolor="none")
    mass = Rectangle((3.1, 2.15), 3.5, 1.8, facecolor=cmap(norm(19)), edgecolor="#555555", lw=1.3)
    heater = Rectangle((1.55, 1.38), 1.35, 0.35, facecolor="#d7301f", edgecolor="#7f0000", alpha=0.05)
    window = Rectangle((8.35, 2.35), 0.18, 2.25, facecolor="#b9e2f5", edgecolor="#3a6f89")
    for patch in (wall, air, mass, heater, window):
        axis.add_patch(patch)
    axis.text(4.85, 3.05, "INTERNAL MASS", ha="center", va="center", fontsize=9, weight="bold")
    axis.text(2.22, 1.96, "HEATER", ha="center", va="bottom", fontsize=8)
    axis.text(8.58, 4.75, "WINDOW", rotation=90, va="bottom", fontsize=8)

    sun = Circle((10.6, 5.7), 0.45, color="#fdb813", alpha=0.1)
    axis.add_patch(sun)
    rays = []
    for offset in (-0.55, 0.0, 0.55):
        line, = axis.plot([10.1, 8.55], [5.35 + offset, 4.15 + offset * 0.3], color="#fdb813", lw=3, alpha=0.1)
        rays.append(line)
    status = axis.text(9.35, 1.25, "", va="bottom", ha="left", family="monospace", fontsize=10)
    title = axis.text(0.8, 6.8, "", ha="left", va="center", fontsize=14, weight="bold")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, shrink=0.72, pad=0.02)
    colorbar.set_label("Component temperature [°C]")

    selected = np.arange(0, len(frame), max(1, frame_stride))
    p_scale = max(float(frame["P_heat_w"].max()), 1.0)
    solar_scale = max(float(frame["G_solar_w_m2"].max()), 1.0)

    def update(position: int):
        row = frame.iloc[int(selected[position])]
        stamp = frame.index[int(selected[position])]
        wall.set_facecolor(cmap(norm(row["T_wall_c"])))
        air.set_facecolor(cmap(norm(row["T_air_c"])))
        mass.set_facecolor(cmap(norm(row["T_mass_c"])))
        heater_alpha = 0.05 + 0.9 * row["P_heat_w"] / p_scale
        heater.set_alpha(float(np.clip(heater_alpha, 0.05, 0.95)))
        solar_alpha = 0.05 + 0.9 * row["G_solar_w_m2"] / solar_scale
        sun.set_alpha(float(np.clip(solar_alpha, 0.05, 0.95)))
        for ray in rays:
            ray.set_alpha(float(np.clip(solar_alpha, 0.05, 0.85)))
        title.set_text(f"{result.controller_name}  ·  {stamp:%a %d %b %H:%M}")
        status.set_text(
            f"INDOOR   {row['T_air_c']:5.1f} °C\n"
            f"OUTDOOR  {row['T_ambient_c']:5.1f} °C\n"
            f"HEATER   {row['P_heat_w'] / 1000:5.2f} kW\n"
            f"SOLAR    {row['G_solar_w_m2']:5.0f} W/m²\n"
            f"PRICE    {row['price_chf_per_kwh']:5.2f} CHF/kWh"
        )
        return wall, air, mass, heater, sun, *rays, status, title

    animation = FuncAnimation(fig, update, frames=len(selected), interval=1000 / fps, blit=False)
    try:
        if path.suffix.lower() == ".gif":
            animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        elif path.suffix.lower() == ".mp4":
            animation.save(path, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=dpi)
        else:
            raise ValueError("Animation output must end in .gif or .mp4")
    finally:
        plt.close(fig)
    return path


def metrics_figure(results: list[SimulationResult]) -> Figure:
    """Compact bar comparison useful in reports and notebooks."""

    table = comparison_table(results)
    columns = [
        "heating_energy_kwh",
        "electricity_cost_chf",
        "comfort_violation_degree_hours",
        "peak_heating_power_kw",
    ]
    labels = ["Energy [kWh]", "Cost [CHF]", "Violation [K·h]", "Peak [kW]"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), constrained_layout=True)
    for axis, column, label in zip(axes, columns, labels, strict=True):
        axis.barh(table.index, table[column], color=["#777777", "#1b9e77", "#d95f02"][: len(table)])
        axis.set_title(label, fontsize=10)
        axis.grid(axis="x", alpha=0.25)
        axis.spines[["top", "right", "left"]].set_visible(False)
        if axis is not axes[0]:
            axis.tick_params(labelleft=False)
    return fig

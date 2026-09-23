import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermalpilot.simulator import SimulationResult
from thermalpilot.visualization import plot_controller_comparison, plot_timeseries


def _result(name: str, offset: float = 0.0) -> SimulationResult:
    index = pd.date_range("2024-01-15", periods=16, freq="15min", tz="Europe/Zurich")
    phase = np.linspace(0, np.pi, len(index))
    frame = pd.DataFrame(
        {
            "T_air_c": 20.0 + offset + 0.2 * np.sin(phase),
            "T_wall_c": 18.5 + 0.1 * np.sin(phase),
            "T_mass_c": 19.0 + 0.1 * np.sin(phase),
            "T_ambient_c": np.linspace(-2.0, 2.0, len(index)),
            "G_solar_w_m2": 300.0 * np.sin(phase),
            "P_heat_w": np.linspace(0.0, 3000.0, len(index)),
            "comfort_lower_c": np.full(len(index), 20.0),
            "comfort_upper_c": np.full(len(index), 22.0),
            "price_chf_per_kwh": np.full(len(index), 0.3),
        },
        index=index,
    )
    frame.attrs["dt_hours"] = 0.25
    return SimulationResult(name, frame)


def _assert_legend_is_above_data(fig) -> None:
    fig.canvas.draw()
    axis = fig.axes[0]
    legend = axis.get_legend()
    assert legend is not None
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    assert legend_box.y0 >= axis.get_window_extent().y1


def test_timeseries_legend_does_not_cover_data() -> None:
    fig = plot_timeseries(_result("Price-aware MPC"))
    try:
        _assert_legend_is_above_data(fig)
    finally:
        plt.close(fig)


def test_comparison_legend_does_not_cover_data() -> None:
    results = [
        _result("Hysteresis thermostat", 0.2),
        _result("Energy-minimizing MPC"),
        _result("Price-aware MPC", 0.1),
    ]
    fig = plot_controller_comparison(results)
    try:
        _assert_legend_is_above_data(fig)
    finally:
        plt.close(fig)

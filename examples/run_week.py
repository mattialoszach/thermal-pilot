"""Minimal Python API example."""

from thermalpilot.metrics import comparison_table
from thermalpilot.visualization import plot_controller_comparison
from thermalpilot.workflows import simulate_project

results, model, weather = simulate_project(weather_source="auto")
print(comparison_table(results).round(3))
print(model.plausibility_report())
print(weather.attrs)
plot_controller_comparison(results, "outputs/example-comparison.png")

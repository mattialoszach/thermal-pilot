"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt

from thermalpilot.config import BuildingParameters, MPCConfig, SimulationConfig
from thermalpilot.metrics import comparison_table
from thermalpilot.visualization import (
    animate_apartment,
    plot_controller_comparison,
    plot_timeseries,
    write_plotly_dashboard,
)
from thermalpilot.workflows import CONTROLLER_KEYS, simulate_project


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermalpilot",
        description="Simulate thermostat and MPC climate control for a synthetic Zurich apartment.",
    )
    subparsers = parser.add_subparsers(dest="command")
    simulate = subparsers.add_parser("simulate", help="run a closed-loop winter simulation")
    simulate.add_argument(
        "--controller",
        choices=("all",) + CONTROLLER_KEYS,
        default="all",
    )
    simulate.add_argument("--weather", choices=("auto", "meteoswiss", "synthetic"), default="auto")
    simulate.add_argument("--start", default="2024-01-15")
    simulate.add_argument("--days", type=int, default=7)
    simulate.add_argument("--horizon", type=int, default=96, help="MPC horizon in 15-minute steps")
    simulate.add_argument("--p-max", type=float, default=6000.0, help="heater limit [W]")
    simulate.add_argument("--forecast-noise", type=float, default=0.0, help="forecast temperature error std [°C]")
    simulate.add_argument("--output-dir", type=Path, default=Path("outputs"))
    simulate.add_argument("--no-plots", action="store_true")
    simulate.add_argument("--interactive", action="store_true", help="write an interactive Plotly HTML")
    simulate.add_argument("--animate", choices=("gif", "mp4"), help="export the price-MPC apartment animation")

    subparsers.add_parser("model-info", help="print parameters, matrices, and plausibility checks")
    return parser


def _model_info() -> int:
    from thermalpilot.model import RCModel

    model = RCModel()
    a_c, b_c, e_c = model.continuous_matrices()
    discrete = model.discretize(900.0)
    print("Synthetic building parameters (not measurements):")
    print(json.dumps(model.p.as_dict(), indent=2))
    print("\nA_c [1/s]:\n", a_c)
    print("\nB_c [K/(W s)]:\n", b_c)
    print("\nE_c columns [ambient, solar]:\n", e_c)
    print("\nA (15 min):\n", discrete.A)
    print("\nB (15 min):\n", discrete.B)
    print("\nE (15 min):\n", discrete.E)
    print("\nPlausibility checks:")
    print(json.dumps(model.plausibility_report(), indent=2))
    return 0


def _simulate(args: argparse.Namespace) -> int:
    if args.days <= 0 or args.horizon <= 0 or args.p_max <= 0:
        raise SystemExit("days, horizon, and p-max must be positive")
    sim_cfg = SimulationConfig(start=args.start, days=args.days)
    building = replace(BuildingParameters(), p_max_w=args.p_max)
    mpc_cfg = MPCConfig(horizon_steps=args.horizon)
    keys = CONTROLLER_KEYS if args.controller == "all" else (args.controller,)
    results, model, weather = simulate_project(
        simulation=sim_cfg,
        building=building,
        mpc=mpc_cfg,
        weather_source=args.weather,
        controller_keys=keys,
        forecast_noise_std_c=args.forecast_noise,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = comparison_table(results)
    metrics.to_csv(output_dir / "metrics.csv")
    for result in results:
        slug = result.controller_name.lower().replace(" ", "-")
        result.frame.to_csv(output_dir / f"{slug}.csv")

    provenance = {
        "weather": weather.attrs,
        "building_parameters": {
            "classification": "assumed/synthetic; not measured",
            **building.as_dict(),
        },
        "simulation": asdict(sim_cfg),
        "mpc": asdict(mpc_cfg),
        "plausibility_checks": model.plausibility_report(sim_cfg.dt_seconds),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )

    if not args.no_plots:
        highlighted = next(
            (result for result in results if result.controller_name == "Price-aware MPC"), results[-1]
        )
        prediction = next(iter(highlighted.predictions.values()), None)
        plot_timeseries(highlighted, output_dir / "thermal-timeseries.png", prediction=prediction)
        if len(results) > 1:
            plot_controller_comparison(results, output_dir / "controller-comparison.png")
        plt.close("all")
    if args.interactive:
        write_plotly_dashboard(results, output_dir / "interactive-dashboard.html")
    if args.animate:
        highlighted = next(
            (result for result in results if result.controller_name == "Price-aware MPC"), results[-1]
        )
        animate_apartment(highlighted, output_dir / f"apartment-animation.{args.animate}")

    print(metrics.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nWeather: {weather.attrs.get('source')} ({weather.attrs.get('station')})")
    print(f"Outputs: {output_dir.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "model-info":
        return _model_info()
    if args.command == "simulate":
        return _simulate(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

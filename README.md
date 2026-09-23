# ThermalPilot

ThermalPilot is a reproducible educational simulation of heating control for a
**synthetic 70 m² apartment in Zurich**. It compares an on/off thermostat with
energy-minimizing and price-aware model-predictive control (MPC), using measured
MeteoSwiss weather when it is available and a deterministic winter generator
when it is not.

The distinction is deliberate:

- Outdoor temperature and global radiation are **measured observations** when
  the reported source is `MeteoSwiss SwissMetNet`.
- Every apartment parameter, occupancy schedule, comfort band, electricity
  tariff, and initial condition is **assumed/synthetic**, not measured from a
  real dwelling or utility customer.

The default example is a seven-day winter simulation at 15-minute resolution,
with a 24-hour (96-step) MPC horizon, nighttime setback, time-of-use prices,
and perfect weather forecasts.

## Quick start

```bash
uv sync --all-extras

uv run thermalpilot model-info
uv run thermalpilot simulate --controller all --weather auto --interactive
```

`uv sync` creates the local `.venv`, installs the project in editable mode,
installs the default development group, and reproduces versions from
`uv.lock`. `--all-extras` also installs the dashboard and animation exporters.

Outputs are written to `outputs/`: per-controller CSV files, `metrics.csv`, a
provenance JSON document, publication figures, and optionally an interactive
Plotly HTML. Export the apartment animation with:

```bash
uv run thermalpilot simulate --controller price-mpc --weather auto --animate gif
# MP4 requires an ffmpeg executable:
uv run thermalpilot simulate --controller price-mpc --weather auto --animate mp4
```

Launch the interactive parameter dashboard with:

```bash
uv run streamlit run src/thermalpilot/dashboard.py
```

The dashboard exposes controller, dates, weather source, comfort limits, MPC
horizon, heater rating, and tariff levels.

## Thermal model

The state, manipulated input, and disturbances are

\[
x = \begin{bmatrix}T_{air}&T_{wall}&T_{mass}\end{bmatrix}^{\mathsf T},\qquad
u=P_{heat},\qquad
d=\begin{bmatrix}T_{ambient}&G_{solar}\end{bmatrix}^{\mathsf T}.
\]

Temperatures are in °C (temperature differences are K), heat flow and heater
power in W, capacitance in J/K, resistance in K/W, global radiation in W/m²,
and time in seconds. The effective solar coefficients have units m² and turn
irradiance into admitted heat flow.

The continuous energy balances are

\[
C_a\dot T_a = \frac{T_w-T_a}{R_{aw}}
              +\frac{T_m-T_a}{R_{am}}
              +\frac{T_o-T_a}{R_{ao}}
              +\eta_h P_{heat}+\alpha_aG,
\]

\[
C_w\dot T_w = \frac{T_a-T_w}{R_{aw}}
              +\frac{T_o-T_w}{R_{wo}}+\alpha_wG,
\]

\[
C_m\dot T_m = \frac{T_a-T_m}{R_{am}}.
\]

Collecting like terms gives

\[
\dot x=A_cx+B_cu+E_cd,
\]

with

\[
A_c =
\begin{bmatrix}
-\frac{1/R_{aw}+1/R_{am}+1/R_{ao}}{C_a} &
 \frac{1}{R_{aw}C_a} & \frac{1}{R_{am}C_a}\\
 \frac{1}{R_{aw}C_w} &
-\frac{1/R_{aw}+1/R_{wo}}{C_w} & 0\\
 \frac{1}{R_{am}C_m} & 0 & -\frac{1}{R_{am}C_m}
\end{bmatrix},
\]

\[
B_c=\begin{bmatrix}\eta_h/C_a\\0\\0\end{bmatrix},\qquad
E_c=\begin{bmatrix}
1/(R_{ao}C_a)&\alpha_a/C_a\\
1/(R_{wo}C_w)&\alpha_w/C_w\\
0&0
\end{bmatrix}.
\]

Nothing in these matrices is fitted or hard-coded independently of the
parameters. `thermalpilot model-info` prints the parameters, matrices, and
plausibility checks used by the running code.

### Exact 15-minute discretization

ThermalPilot assumes the heater and disturbances are constant over each control
interval. It computes the exact zero-order-hold map with one augmented matrix
exponential:

\[
\exp\left(
\begin{bmatrix}
A_c&B_c&E_c\\0&0&0\\0&0&0
\end{bmatrix}\Delta t
\right)
=
\begin{bmatrix}
A&B&E\\0&I&0\\0&0&I
\end{bmatrix}, \qquad \Delta t=900\;s.
\]

Therefore

\[
x_{k+1}=Ax_k+Bu_k+Ed_k.
\]

This avoids the integration error of forward Euler and is tested against
SciPy's independent `cont2discrete(..., method="zoh")` implementation.

## Assumed apartment parameters

| Parameter | Default | Meaning and engineering rationale |
|---|---:|---|
| Floor area / height | 70 m² / 2.5 m | Synthetic single-zone apartment; volume 175 m³ |
| \(C_a\) | 1.50 MJ/K | Effective fast capacitance: air plus rapidly coupled contents/surfaces. Air alone is about 0.21 MJ/K using 1.204 kg/m³ and 1006 J/(kg·K). |
| \(C_w\) | 12.0 MJ/K | Effective exterior envelope storage |
| \(C_m\) | 10.0 MJ/K | Floor, furniture, and internal-wall storage |
| \(R_{aw}\) | 0.015 K/W | Air-to-envelope coupling |
| \(R_{am}\) | 0.010 K/W | Air-to-internal-mass coupling |
| \(R_{ao}\) | 0.050 K/W | Direct ventilation/window heat-loss path |
| \(R_{wo}\) | 0.025 K/W | Envelope-to-outdoor path |
| \(\eta_h\) | 0.98 | Resistive/electric heat delivered to the zone |
| \(\alpha_a,\alpha_w\) | 1.5, 3.0 m² | Effective solar apertures assigned to fast and wall nodes |
| \(P_{max}\) | 6.0 kW | Heating limit, leaving capacity for setback recovery |

These are reduced-order modeling choices, not a survey or calibration. Their
purpose is to retain plausible time scales and heat flows while making each
term interpretable. Useful cross-checks for the defaults are:

- steady outdoor heat-loss coefficient
  \(H=1/R_{ao}+1/(R_{aw}+R_{wo})=45\) W/K, or 0.64 W/(m²·K);
- total effective capacitance 23.5 MJ/K, or 336 kJ/(m²·K);
- about 1.26 kW steady heat loss at 20 °C inside and −8 °C outside, before
  solar gains;
- every continuous-time eigenvalue has negative real part and every 15-minute
  discrete eigenvalue lies inside the unit circle.

The thermal-mass range is informed by the lumped-capacitance approach used in
building energy standards such as [ISO 52016-1](https://www.iso.org/standard/65696.html),
but the specific numbers above are intentionally synthetic. Change them in
`BuildingParameters` and rerun `model-info` rather than treating them as
universal values.

## MPC formulation

At every 15-minute step the controller observes the three temperatures,
receives a disturbance forecast, solves a finite-horizon convex program,
applies only its first heater command, propagates the simulated building, and
then shifts/re-solves. The default horizon is 96 steps (24 hours).

For lower and upper comfort slacks \(s^-_k,s^+_k\ge0\), the constraints are

\[
x_{k+1}=Ax_k+Bu_k+Ed_k,\qquad 0\le u_k\le P_{max},
\]

\[
T_{min,k}-s^-_k\le T_{air,k}\le T_{max,k}+s^+_k.
\]

The default comfort band is 20–22 °C from 06:00 to 22:00 and 17–22 °C
overnight. Slack guarantees feasibility during initialization or extreme cold.
The objective is

\[
J=\sum_k \pi_k\frac{u_k\Delta t}{1000}
+w_s\Delta t\left((s^-_k)^2+(s^+_k)^2\right)
+w_{\Delta u}\left(\frac{u_k-u_{k-1}}{P_{max}}\right)^2.
\]

Here \(\Delta t\) is in hours, so the first term is CHF when \(\pi\) is in
CHF/kWh. `Energy-minimizing MPC` replaces \(\pi_k\) with a constant one, while
`Price-aware MPC` uses the time-varying tariff. Cheap 22:00–06:00 energy and an
expensive 17:00–21:00 peak are **illustrative synthetic prices**, not an actual
Zurich utility tariff. The upper comfort bound limits pre-heating.

CVXPY builds the problem once with parameters; OSQP then warm-starts each
receding-horizon solve. The default perfect forecast uses the future measured
or synthetic disturbance exactly. `--forecast-noise` adds repeatable errors to
the controller forecast while leaving the simulated weather unchanged.

## Weather data and provenance

`MeteoSwissLoader` uses the official Federal Spatial Data Infrastructure STAC
API for the SwissMetNet collection. The default station is **SMA — Zürich /
Fluntern**. It downloads the relevant decade's 10-minute historical CSV, caches
the raw file under `~/.cache/thermalpilot`, and reads:

- `tre200s0`: air temperature 2 m above ground [°C];
- `gre000z0`: global radiation, ten-minute mean [W/m²].

MeteoSwiss timestamps are UTC. They are converted to `Europe/Zurich`, then the
two measured series are averaged onto the 15-minute control grid. The
MeteoSwiss documentation describes the station files, identifiers, resolution,
and UTC convention: [Automatic weather stations — Open Data Documentation](https://opendatadocs.meteoswiss.ch/a-data-groundbased/a1-automatic-weather-stations).
The data may be used under MeteoSwiss's stated open-data terms; retain the
source attribution when redistributing results.

Modes are explicit:

- `--weather meteoswiss`: require measured data and fail clearly if unavailable;
- `--weather synthetic`: use the seeded offline winter generator;
- `--weather auto`: prefer measured data, then record the fallback reason if
  the deterministic generator is needed.

Every result CSV also has a neighboring `provenance.json` that labels weather
and building assumptions independently. The synthetic generator should not be
used for climatological claims.

## Metrics and visual outputs

Each controller reports:

- heating energy [kWh];
- electricity cost [CHF];
- comfort violation [degree-hours, K·h];
- hours more than 0.05 K below/above comfort (the tolerance avoids counting
  order-\(10^{-4}\) K optimizer residuals as whole uncomfortable intervals);
- peak heating power [kW];
- mean/minimum/maximum indoor air temperature;
- solver failure count.

The static report shows indoor, wall, mass, and outdoor temperatures; the
comfort band; heat power; solar radiation; electricity price; and an MPC
prediction trace. A controller comparison shows indoor temperature, heating,
and cumulative cost. The Plotly output supports linked zoom and hover.

The animated apartment is intentionally schematic rather than an architectural
model: room, envelope, and furniture colors encode the three states; heater and
sun intensity encode input and radiation; labels report indoor/outdoor
temperature, heater power, electricity price, time, and controller.

## Project layout

```text
src/thermalpilot/
├── config.py          assumed parameters and reproducible defaults
├── model.py           energy-balance matrices and exact ZOH
├── weather.py         MeteoSwiss STAC/cache and offline generator
├── schedules.py       comfort and synthetic tariff schedules
├── controllers/       thermostat and CVXPY/OSQP MPC
├── simulator.py       closed-loop receding-horizon loop
├── metrics.py         energy, cost, comfort, peak, temperature metrics
├── visualization.py   Matplotlib, Plotly, GIF/MP4 outputs
├── workflows.py       reusable end-to-end orchestration
├── dashboard.py       Streamlit controls
└── cli.py             command-line entry point
```

## Validation and tests

```bash
uv run pytest
```

The tests check matrix coefficients against the balances, exact discretization
against SciPy, isothermal equilibrium preservation, stability, deterministic
weather, thermostat hysteresis, MPC input constraints/dynamics, slack-enabled
feasibility, and closed-loop result integrity.

This remains a deliberately compact educational model. It omits humidity,
internal occupant/appliance gains, cooling, multi-zone airflow, state
estimation, actuator dynamics, forecast-provider uncertainty, and parameter
identification. Those are natural extensions, but they should not be hidden
inside unexplained matrices.

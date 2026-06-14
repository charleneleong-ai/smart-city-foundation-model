# The twin as a predictive world-model for Physical AI

Most "AI" today is digital — chatbots and dashboards. The frontier is **Physical (Embodied) AI**:
robots, drones, autonomous vehicles, and smart infrastructure that *sense and act in the real
world*. The hard part isn't the body — it's giving the body something reliable to plan against.

This project is that something: a **calibrated, verifiable predictive layer** — the world-model
an embodied-AI fleet acts on. *"Software is the brain, we build the body"* — this is the
predictive half of the brain, built so the body can be trusted on the street, in the sky, at sea.

## Every physical-AI use case runs on a prediction

| Embodied-AI deployment | Prediction the twin supplies | Channel |
| --- | --- | --- |
| Autonomous logistics / ports | demand & flow per zone; charging/energy load → route + schedule fleets | energy, EV charging |
| Smart urban infra ("Urban AI": cooling, water, waste, traffic) | consumption / load per cell, ahead of time → autonomous control | weather, energy |
| Mobility / AVs / EV charging | congestion + **EV-charging demand** surfaces → routing, depot siting | **EV charging** |
| Disaster (bushfire / flood) | hazard / spread surfaces → where to send scouting drones | (planned hazard layer) |
| Healthcare drone delivery | demand + weather-window → dispatch routing | weather, energy |

These are **not separate models** — they're channels of one multi-domain forecaster
([Chronos-2](https://hf.co/amazon/chronos-2), grounded 2026-06-14): one predictive substrate,
many embodied consumers, with weather conditioning energy via covariates.

## Macro *and* micro

H3 is multi-resolution, so every channel runs at both scales unchanged:
- **macro (res 4)** — region / grid scale: a fleet operator routing across a metro or a grid
  controller balancing a region.
- **micro (res 8)** — neighbourhood / charging-cluster scale: a depot planner siting chargers or
  a council scheduling a street-cleaning robot.

Same forecast, same calibrated interval — pick the resolution the actuator needs.

## What makes it safe enough for a body to act on

A wrong chatbot is annoying; a drone or AV acting on a wrong prediction is dangerous. Two layers
make this an embodied-AI substrate rather than just another forecast:

- **Split-conformal intervals (SP5)** — calibrated uncertainty, so an agent knows *when not to
  trust* a forecast and defers to a human (the "always human oversight where needed" requirement).
- **Verifiable reasoning / RLVR (SP7)** — predictions checked against held-out reality, not merely
  plausible, with a baseline established before training so any gain is measurable.

## Why Chronos-2, not Aurora

[Aurora](https://www.microsoft.com/en-us/research/project/aurora-forecasting/) is an Earth-system
FM — it forecasts *atmospheric* fields (weather, air quality, ocean waves, cyclones) on a grid,
and is the right upgrade for the **weather input channel**. It does **not** forecast energy / EV
demand. [Chronos-2](https://hf.co/amazon/chronos-2) is a general time-series FM — it forecasts
*any* city series (load, charging, footfall) with covariates. So Chronos-2 is the multi-domain
substrate; Aurora (or GenCast/WeatherNext) feeds *into* it as a high-fidelity weather covariate.

## First physical-AI channel: EV charging

[`sctwin.demand.ev_charging_load`](../src/sctwin/demand.py) turns the weather field into an
EV-charging demand surface per (cell, time) — evening-peaked, cold-amplified, fleet-scaled — then
the SP4 forecaster + SP5 conformal calibration produce a *forecast + interval* an operator can act
on. It renders on the twin under the **EV charging** layer group (demand · forecast · |error| ·
covered), at macro and micro.

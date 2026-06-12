# Urban Digital Twin — Foundation-Model Reference Architecture

**Date:** 2026-06-12
**Status:** Design (approved for decomposition into sub-project specs)
**Scope:** Full reference architecture, portable across cities, decomposed into independently-buildable sub-projects.

## Summary

A **verifiable urban digital twin** that uses modern foundation-model techniques —
LLMs, VLMs / geospatial foundation models, world models, and RL with verifiable
rewards (RLVR) — over **open data and open models** to make smart-city planning
predictions (energy first) more accurate, with continuous verification against
real measurements.

Two design commitments drive everything:

1. **Modular, not monolithic.** One black-box multimodal model cannot be verified
   layer-by-layer. Instead, a layered stack where each concern (perception, state,
   dynamics, scenario) is an independently testable, swappable unit with a typed
   interface.
2. **Verification is a first-class spine, not an afterthought.** A twin is only
   trustworthy if every prediction is continuously checked against reality. The
   verification layer is built *alongside* the first predictive vertical, not bolted
   on later.

**Portability:** the twin is instantiable for any city from globally-available open
layers (Sentinel-2, ERA5, OSM/Overture, Open-Meteo). Region-specific datasets (US
NAIP, EU permit registries, local smart-meter APIs) are **pluggable adapters**, never
hard dependencies.

## Architectural decision

Three styles were considered:

| Approach | Verdict |
|---|---|
| **A. Monolithic end-to-end multimodal model** — one model ingests everything, predicts everything | ❌ Black box; unverifiable layer-by-layer; data-hungry; reward-hackable |
| **B. Layered modular twin** — a foundation model per concern, typed interfaces | ✅ **Backbone.** Each layer independently testable, swappable, grounded |
| **C. Physics-simulator-grounded twin** — couple ML predictors with EnergyPlus/SUMO/etc. | ✅ Borrowed *into* the verification layer as a verifiable oracle |

**Decision: B as the backbone, with C's physical simulators used as the verifiable
oracle inside the verification spine.** ML provides priors and corrections; simulators
and real measurements provide checkable ground truth.

## The architecture — 6 layers + a verification spine

```
                    ┌──────────────────────────────────────────────┐
  L5 Scenario/Plan  │  LLM planner-agent: poses "what if we build X" │
                    │  → queries twin → returns plans + tradeoffs    │
                    └───────────────────┬──────────────────────────┘
  L4 Dynamics       │  WORLD MODEL: spatiotemporal GNN/transformer.  │
     (the twin core)│  Rolls out energy/load/heat/traffic over time, │
                    │  counterfactual scenarios.                     │
                    └───────────────────┬──────────────────────────┘
  L3 State fusion   │  LLM: permits/codes/reports + structured data  │
                    │  → typed, queryable CITY STATE GRAPH           │
                    │  (buildings, parcels, grid, roads)             │
                    └───────────────────┬──────────────────────────┘
  L2 Perception     │  VLM/geo-FM: satellite·aerial·street imagery   │
                    │  → footprints, heights, roof/solar, land use,  │
                    │  construction state, vegetation                │
                    └───────────────────┬──────────────────────────┘
  L1 Ingestion      │  Open-data adapters → canonical spatiotemporal │
                    │  schema (H3 hex grid + time). Region data =    │
                    │  pluggable adapters.                           │
                    └────────────────────────────────────────────────┘
        ╎ VERIFICATION SPINE (cross-cuts every layer) ╎
        ╎ RLVR rewards · backtest harness · conformal  ╎
        ╎ calibration · drift detection · re-grounding ╎
```

### L1 — Ingestion & canonical spatiotemporal schema

- **Job:** pull open data, normalize to one canonical schema keyed on an **H3 hex
  grid + time axis**, index spatially (STAC for rasters).
- **Open sources:** ERA5 / Open-Meteo (weather), Sentinel-2 / NAIP (imagery),
  OSM + Overture Maps (infrastructure/buildings), NREL ResStock/ComStock,
  Building Data Genome 2, OpenEI (energy).
- **Interface:** `get(cell, time_range, layer) -> tensor/records`. Region datasets
  register as adapters implementing this interface.

### L2 — Perception (VLM / geospatial foundation models)

- **Job:** imagery → physical attributes: building footprints, heights, roof area &
  solar potential, land use, construction progress, vegetation/imperviousness.
- **Open models:** geospatial FMs **Prithvi** (NASA/IBM), **Clay**, **SatMAE**,
  **Satlas**; **SAM 2** + open building-footprint models (Microsoft/Google open
  footprints); **Qwen2.5-VL / InternVL** for semantic extraction.
- **Interface:** writes attributes back to L1 cells with provenance + confidence.

### L3 — State fusion (LLM → city state graph)

- **Job:** fuse heterogeneous structured + **unstructured text** (permits, building
  codes, planning reports) into a **typed, queryable city state graph** — entities:
  buildings, parcels, grid nodes, road segments; relations: serves, adjacent-to,
  feeds.
- **Open models:** open LLMs for document extraction, entity resolution, schema mapping.
- **Interface:** graph query API; every node carries provenance + uncertainty.

### L4 — Dynamics (world model — the twin core)

- **Job:** learn the city's temporal dynamics and roll out forecasts &
  **counterfactuals** ("build a 40-storey tower on parcel P → district load Δ").
- **Model family:** spatiotemporal GNN / transformer / neural operator over the H3
  graph. Precedent: open ML weather models **GraphCast / Aurora / FourCastNet** show
  the spatiotemporal-rollout pattern at scale.
- **Interface:** `rollout(state, intervention, horizon) -> trajectory + uncertainty`.

### L5 — Scenario / planner agent

- **Job:** tool-using LLM agent that poses scenarios, queries L4, and returns plans
  with tradeoffs and confidence.
- **Interface:** natural-language goal → structured plan + predicted impacts + intervals.

## The verification spine (the crux)

A twin is only trustworthy if continuously checked against reality. Three
**independent** verifiable oracles:

1. **Real measurements** — held-out smart-meter / sensor data, predict-then-reveal.
2. **Physics simulators** as queryable oracles — **EnergyPlus / CityLearn** (building
   & district energy), **SUMO** (traffic), **UMEP** (urban heat / microclimate).
3. **Conservation constraints** — energy balance, mass balance, parts-sum-to-whole.

These oracles drive four mechanisms:

- **RLVR** — reward = agreement with an oracle, so model improvements are *real and
  checkable*, not reward-hacked by a learned critic.
- **Backtest harness** — temporal holdout scored continuously (financial-backtest
  analogy for a city): predict at *t*, reveal truth at *t+Δ*, score, log.
- **Conformal calibration** — prediction intervals with guaranteed coverage
  ("demand = X ± Y at 90% coverage"), independent of model family.
- **Drift detection** — when reality diverges from the twin, trigger **re-grounding**
  (refit / re-perceive the affected cells).

This is also the answer to "how to verify the digital twin": *verification is not a
one-time validation — it is a closed loop that re-grounds the twin against incoming
real data forever.*

## Decomposition into sub-projects

Each sub-project gets its own spec → plan → implementation cycle.

| # | Sub-project | Depends on | Notes |
|---|---|---|---|
| SP1 | Ingestion + canonical spatiotemporal schema | — | foundation |
| SP2 | VLM perception (imagery → attributes) | SP1 | parallel with SP3 |
| SP3 | LLM state-fusion → city state graph | SP1 | parallel with SP2 |
| SP4 | **Energy world model** (first vertical) | SP1–3 | matches project focus |
| SP5 | **Verification & calibration harness** | SP1 | built *alongside* SP4 |
| SP6 | Scenario / planner agent | SP4–5 | last |

**Build order:** SP1 → (SP2 ∥ SP3) → **SP4 + SP5 together** as the first end-to-end
*verifiable energy vertical* → SP6 → then generalize the same pattern to traffic / heat
/ water layers.

The first milestone worth defending is **SP4+SP5**: a district-level energy predictor
trained with verifiable rewards, backtested against held-out real meter data, emitting
calibrated intervals. Everything else is scaffolding around making that vertical real,
then replicating the pattern.

## Cross-cutting concerns

- **Spatiotemporal data model** — H3 hex grid + time axis is the universal key across
  layers; everything joins on `(cell, time)`.
- **Provenance & uncertainty** — every derived value carries source lineage and a
  confidence/interval; uncertainty propagates up the stack.
- **Open model zoo** — all named models/datasets are open-licence; region-specific or
  proprietary feeds are optional adapters.
- **Portability** — instantiating the twin for a new city = registering that city's
  bounding box + any region adapters; global layers work out of the box.

## Out-of-scope (deliberately deferred)

- Real-time streaming ingestion (batch first; streaming is an SP1 extension).
- Non-energy verticals (traffic, heat, water) — same pattern, after the energy
  vertical proves out.
- A production UI / dashboard for planners (the L5 agent API comes first).
- Multi-city federation / transfer learning across twins.

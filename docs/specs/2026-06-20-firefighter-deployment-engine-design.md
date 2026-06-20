# Firefighter Deployment Engine — Design

**Date:** 2026-06-20
**Status:** Approved (brainstorming) → ready for implementation plan
**Scope:** PoC of a personalised exposure→health deployment optimiser inside `sctwin`.

## Summary

Given a fire (size/type + live weather/wind/heat/smoke) and a **roster of distinct firefighters**
(each with their own age, comorbidities, fitness, role, and career smoke dose banked so far),
recommend a deployment **plan** that minimises crew health risk while still meeting the fire's
suppression requirement. The engine personalises by individual health profile — shielding the
high-acute-risk people and spreading cumulative carcinogen dose across the team — and ships every
risk score with an uncertainty interval.

It is built **on** the existing interventional machinery, not beside it: a deployment is the
analogue of [`Intervention`](../../src/sctwin/reason/intervention.py), scoring a deployment's risk
is the analogue of [`effect()`](../../src/sctwin/reason/intervention.py), and the constrained
search reuses the same "score a counterfactual, pick the best" shape.

## Approved decisions

| Decision | Choice |
|---|---|
| **Spine** | Personalised exposure→health scorer (the novel core) |
| **Output** | Multi-lever optimiser → a per-firefighter deployment **Plan** |
| **Risk objective** | Combined index: acute (heat/cardiac) + this-incident dose + career dose |
| **Optimiser** | **A — constrained greedy/grid search** (deterministic, explainable, testable) |
| **Anti-degeneracy** | Minimise risk **subject to a suppression-capacity constraint** |

## Architecture — `src/sctwin/deploy/`

Data flow: `FireScenario` + `Roster` + `Constraints` → optimiser searches deployments, each scored
by the exposure→risk core → returns the min-risk feasible `Plan` with per-firefighter risk + intervals.

| Module | Purpose |
|---|---|
| `hazard.py` | `FireScenario` — fire size/type→toxicity class, smoke/PM2.5, heat, wind (speed+dir), H3 cell, duration. `from_live(cell, time)` pulls live weather via the extended [`OpenMeteoForecastAdapter`](../../src/sctwin/adapters/open_meteo.py) (`WEATHER_VARS`) + a CAMS/FRP smoke stub — the realtime hook. |
| `roster.py` | `Firefighter` — id, age, sex, role, years-service, comorbidity flags (CV/respiratory), fitness, `career_dose`. `Roster` = the varied team. |
| `exposure.py` | `exposure_dose(scenario, firefighter, deployment)` = f(time × fire-toxicity × smoke × heat), reduced by PPE/BA. Pure function — the analogue of [`counterfactual()`](../../src/sctwin/reason/intervention.py). |
| `risk.py` | **The novel core.** Combined personalised index `w1·acute(heat,cardiac \| age,CV,fitness) + w2·incident_dose + w3·career_dose`, each term from a documented occupational-epi prior; returns score **+ uncertainty interval** (reusing the [`verify/`](../../src/sctwin/verify/) interval pattern). |
| `optimise.py` | Constrained greedy/grid search over `rotation_minutes × PPE_level × crew_assignment`, minimising aggregate **and** worst-individual risk subject to the suppression-capacity constraint. |
| `engine.py` | Orchestrator `recommend(scenario, roster, constraints) → Plan`. |
| `apps/deploy_demo.py` | Runnable demo on a fixed scenario (LA grassfire / London common) over a sample varied roster. |

## Core models

### Exposure dose (`exposure.py`)
Per firefighter, per deployment: `dose = time_on_scene × toxicity(fire_type) × smoke_factor(PM2.5)
× heat_factor(temp)`, attenuated by `ppe_factor(role, PPE)` (breathing apparatus sharply cuts
inhaled toxicant; turnout/PPE cuts heat+dermal). Monotone in each driver. Pure and unit-tested.

### Combined personalised risk (`risk.py`)
```
risk(ff) = w1·acute(heat, cardiac | age, CV, fitness)   # realtime-actionable, archetype-sensitive
         + w2·incident_dose(smoke × time)               # this fire's toxicant load
         + w3·career_dose(history + this increment)     # cumulative carcinogen accrual (IARC Grp 1)
```
Weights `w1..w3` are explicit config. Each term maps a dose to a relative risk via a **literature
prior**, returned with an uncertainty interval — never a claimed calibrated personal risk.

### The suppression-capacity constraint (`optimise.py`)
Pure risk-minimisation is degenerate ("deploy nobody"). The fire of size `S` defines a **required
effective on-task capacity** `K(S)` (firefighter-task-units). The optimiser searches only
deployments that meet `K`, then among those minimises risk. **Trucks / equipment / water supply
enter here** — as resources that *provide* capacity and *carry* crew slots (constraint inputs), not
a separately optimised logistics subsystem.

### Optimiser (Approach A)
Enumerate a bounded grid of `rotation_minutes × PPE_level`; for each grid point greedily assign the
varied roster to required roles — safest-fit person per slot, rotate to cap individual dose, prefer
the lowest-career-dose person for the smokiest roles (spreads carcinogen exposure). Return the
min-risk feasible `Plan`. Deterministic and explainable by construction.

## The honest seam

The **exposure physics** (dose from time × smoke × heat × PPE) is data-grounded. The
**health-risk coupling** (dose → cardiac/cancer relative risk) is **transferred from occupational
epidemiology**, not fitted — there is no firefighter outcome data. The engine surfaces this
explicitly, ships uncertainty intervals, and presents output as decision support (rotation / PPE /
crew-shielding priority), not a calibrated personal prognosis.

## Realtime, honestly

The engine runs on a `FireScenario` object. `FireScenario.from_live(cell, time)` wires the real
adapters (extended Open-Meteo forecast + a CAMS/FRP smoke stub) so it *can* run live; the demo and
tests use fixed scenarios for determinism. The hook is real and wired; nothing depends on a live
feed to run.

## Testing (TDD)

- **exposure** — monotonicity (more time/smoke/heat → more dose; BA/PPE → less).
- **risk** — archetype sensitivity (older + CV → higher acute risk at equal dose); interval present.
- **optimise** — feasibility (respects `K(S)`); shields high-acute-risk individuals; spreads career
  dose; degenerate-avoidance (won't return an empty plan when `K>0`).
- **hazard** — `from_live` calls the extended adapter and maps to a `FireScenario`.

## Out of scope (deliberate stubs / follow-ups)

- **Fire-spread model** — scenario provides the hazard; the LA wind-driven spread CA is separate.
- **RL deployment policy** — the natural next step (Approach B); the `risk.py` scorer built here is
  exactly its reward, plugged into [`InterventionEnvironment`](../../src/sctwin/reason/intervention.py).
- **Truck/water logistics optimiser** — resources are constraint inputs here, not separately optimised.

## Build order

1. `hazard.py` + `roster.py` (data structures, fixed-scenario + sample roster).
2. `exposure.py` (pure dose) — TDD.
3. `risk.py` (combined index + intervals) — TDD. **The core.**
4. `optimise.py` (constrained greedy/grid) — TDD.
5. `engine.py` + `apps/deploy_demo.py` (orchestrate + demo).
6. `FireScenario.from_live` realtime hook + test.

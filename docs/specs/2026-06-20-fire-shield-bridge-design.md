# Fire-Shield ↔ Deployment Engine Bridge — Design Note

**Date:** 2026-06-20
**Status:** Sketch (for discussion) — not yet an implementation plan
**Scope:** Map the variables in the `fire-shield-google-ai` live-monitoring app onto this engine's
firefighter **archetype** + risk model, and define the shared **risk-score threshold** vocabulary.

## Summary

`fire-shield-google-ai` and this deployment engine are the two ends of one problem.
Fire-Shield is the **RESPOND** end: live vitals + environment per firefighter, on-scene, producing a
bounded `0–100` risk score *right now*. This engine is the **PREVENT/PLAN** end: pre-deployment, it
takes a roster of static archetypes and a fire scenario and decides *who goes where* before anyone is
exposed. The piece that joins them is the **firefighter archetype** — Fire-Shield's per-person profile
is the live instantiation of our [`Firefighter`](../../src/sctwin/deploy/roster.py#L7) dataclass, and
both ends ought to speak one risk-band vocabulary so a plan made here can be policed by telemetry there.

This note maps the variables and pins down the **risk-score threshold** as the shared seam.

## A. Archetype ↔ Fire-Shield profile

Static, person-intrinsic fields — these are the archetype. Fire-Shield carries them on
`Firefighter.profile` (its [`HealthProfile`](../../../fire-shield-google-ai/src/lib/mock-data.ts));
we carry them on [`Firefighter`](../../src/sctwin/deploy/roster.py#L7).

| Our archetype field | Fire-Shield source | Note |
|---|---|---|
| `id` | `profile.id` | direct (Fire-Shield also has `callsign`, `name`) |
| `age` | `profile.age` | direct |
| `role` | `profile.role` | direct |
| `sex` | — | **genuinely ours-only** — no sex field in `HealthProfile` |
| `years_service` | — | **genuinely ours-only** — no tenure field (`prevShiftHours` is recent shift, not career) |
| `respiratory` (flag) | `profile.respiratoryRisk` + `conditions[]` | **present, needs map** — ordinal `none/mild/moderate/high` (+ free-text "mild asthma") → flag |
| `cardiovascular` (flag) | `profile.conditions[]` | **present, needs parse** — derivable from free-text (e.g. "mild hypertension"); no dedicated field |
| `fitness` (0..1) | `profile.fitness` | **present, needs ordinal→scalar** — `elite/high/moderate/recovering` → ~`1.0 / 0.8 / 0.5 / 0.25` |
| `career_dose` | — | **genuinely ours-only** — cumulative carcinogen accrual; see feedback loop below |

The seam is **narrower than demographics-vs-clinical**: Fire-Shield already carries a *categorical
clinical snapshot*. What it genuinely lacks is **`sex`, `years_service`, and `career_dose`** — tenure,
sex, and cumulative career dose. Those three must come from an occupational-health record, not the app.

**Fields Fire-Shield carries that our archetype is _missing_** (and should adopt — they make the
adapter lossless on the clinical side): `heatTolerance` (`low/avg/high` — a direct heat-susceptibility
input, currently only implicit in our `fitness`), `hrBaseline` (resting HR — a cardiovascular-fitness
proxy), the full `conditions[]` list (richer than two bool flags), and `prevShiftHours` / `hydrationStart`
(recent-shift state, between live and static). Widening our archetype to keep `heatTolerance` +
`conditions[]` beats lossily collapsing them to two booleans.

## B. Live vitals → acute strain (Fire-Shield only)

Fire-Shield's `vitals` and `location` blocks are **realtime tactical state** with no home in a
pre-deployment archetype — they are what telemetry *adds on top of* the archetype once a firefighter is
on scene. We do not model them; we model the archetype's *susceptibility* to the heat load that drives
them, via [`acute_risk`](../../src/sctwin/deploy/risk.py#L23).

| Fire-Shield field | Our analogue |
|---|---|
| `vitals.hr`, `bodyTempC`, `spo2`, `respRate`, `fatigue`, `hydration`, `movement`, `fall` | none — live only; our `acute_risk(ff, heat_load)` predicts *susceptibility*, telemetry *measures* the outcome |
| `location.distToExitM`, `collapseRisk`, `nearbyFireKw`, `roomTempC`, `floor`, `exitClear`, `buildingDensity` | none — live tactical; our `Constraints` model capacity/rotation, not micro-position |

This is the right asymmetry: a planner cannot know a firefighter's live heart rate, and a live monitor
should not be re-deriving career dose every tick.

## C. Environment → FireScenario

Fire-Shield's `Environment` block is the live-measured version of our (planned) `FireScenario`
hazard inputs.

| Fire-Shield `Environment` | Our hazard analogue |
|---|---|
| `smokeDensity`, `coPpm`, `hcnPpm` | toxicant load → feeds [`incident_dose_risk`](../../src/sctwin/deploy/risk.py#L32) |
| `ambientTempC` | heat load → feeds [`acute_risk`](../../src/sctwin/deploy/risk.py#L23) |
| `windKph`, `windDir`, `fireSpreadDir` | spread drivers (the wind-driven CA in `smart_city_foundation_model`), not risk terms here |
| `visibilityM` | tactical; no risk-model home |

## D. Risk score ↔ risk score

Both ends compute a scalar risk. They are **not** on the same scale, and that is the crux of the bridge.

| | Fire-Shield | This engine |
|---|---|---|
| Type | `RiskAssessment.score` | [`RiskScore.value`](../../src/sctwin/deploy/risk.py#L16) |
| Range | bounded **0–100** | **unbounded** (weighted sum of relative-risk priors) |
| Breakdown | `{physio, environment, location, profile}` | `drivers` (acute / incident / career) |
| Interval | none | `low`, `high` (transferred-prior uncertainty) |
| Horizon | now (live) | this deployment (planned) |

## The risk-score threshold — the shared seam

Fire-Shield already has the threshold vocabulary we lack. Its
[`Thresholds`](../../../fire-shield-google-ai/src/lib/thresholds-store.ts) — `{ safeMax: 30,
cautionMax: 60, highRiskMax: 80 }`, strictly ordered — partitions the `0–100` score into four
deployment-relevant bands. To reuse it we must do **two** things:

**1. Normalise our unbounded score to 0–100.** [`RiskScore.value`](../../src/sctwin/deploy/risk.py#L16)
is a weighted sum of relative-risk priors with no ceiling, so a raw threshold compare is meaningless.
Pick a saturating map (e.g. `100 · value / (value + k)`, `k` the calibration midpoint) so that "score
70" means the same band on both ends. This is a presentation/threshold transform only — the optimiser
keeps ranking on the raw unbounded value; bands are for the *gate*, not the *sort*.

**2. Lift the bands into config here.** Add a `RiskBands(safe_max, caution_max, high_risk_max)` config
to mirror Fire-Shield's `Thresholds`, with the same strict-ordering clamp, and use it as a
**deployment gate** in [`optimise.py`](../../src/sctwin/deploy/optimise.py) — a layer *on top of* the
existing min-risk ranking in [`recommend`](../../src/sctwin/deploy/optimise.py#L59):

| Normalised band | Gate in `_plan_for` / `recommend` |
|---|---|
| `≤ safe_max` | assign to BA / front-line freely |
| `≤ caution_max` | assignable, but **cap rotation** (shorter shifts, earlier swap-out) |
| `≤ high_risk_max` | **staging only** — support roles, no interior attack |
| `> high_risk_max` | **stand down** — exclude from the plan |

This makes the bands the contract: a plan produced here is expressed in the same four bands Fire-Shield
enforces live, so on-scene telemetry crossing `high_risk_max` is a recall signal against the very plan
that deployed the firefighter.

## Three bridge mechanisms

1. **Archetype adapter** — `from_fire_shield(profile) -> Firefighter`: map `id/age/role` directly;
   map `fitness` ordinal→scalar; map `respiratoryRisk` + parse `conditions[]` into the
   `respiratory`/`cardiovascular` flags (better: widen our archetype to keep the `conditions[]` list +
   `heatTolerance` rather than lossily collapsing to two bools). Only **`sex`, `years_service`,
   `career_dose`** must come from an occupational-health record — fail loud on *those three*, not on
   the clinical fields Fire-Shield already provides.
2. **`career_dose` feedback loop** — Fire-Shield measures realised incident dose live (`smokeDensity ×
   time`); after an incident that realised dose should be **banked back** into the archetype's
   `career_dose`, closing PREVENT→RESPOND→PREVENT. Today `career_dose` is a static input; this makes it
   accumulate from telemetry.
3. **Shared `RiskBands`** — one config object, same four-band semantics both ends. Plan-time gate here,
   live-alert thresholds there.

## The honest seam

The mapping is clean for **demographics** (age/role/id) and **hazard inputs** (C). It is deliberately
*empty* for **live vitals** (B) — and that emptiness is correct, not a gap to fill: a planner that
pretended to know live heart rate would be fabricating telemetry.

An earlier draft framed this as "Fire-Shield knows demographics, we own the clinical archetype." That
was **wrong** — Fire-Shield's `HealthProfile` already carries a categorical clinical snapshot
(`respiratoryRisk`, `heatTolerance`, `fitness`, `conditions[]`, `hrBaseline`). So the real seam is not
clinical-vs-demographic; it is **career-cumulative + tenure (ours: `sex`, `years_service`,
`career_dose`) vs live vitals (theirs)** — with the static clinical snapshot *shared* by both. Our
genuine value-add is the **career dose + tenure**; Fire-Shield's is the **live vitals** we cannot see.
The risk-band vocabulary is the contract that lets the two ends talk.

The same caveat as the parent engine holds: the dose→health-risk coupling is a transferred occupational-
epidemiology prior, not fitted to firefighter outcomes. Normalising to 0–100 to match Fire-Shield's
scale does **not** make it calibrated — it makes it *comparable*, which is a presentation guarantee, not
a statistical one.

## Out of scope (this note is a sketch)

- Implementing the normalisation map or `RiskBands` config — that is the follow-up plan.
- Calibrating `k` (the saturation midpoint) — needs a reference distribution of plausible scores.
- The Fire-Shield side's TypeScript changes — this note only specifies what our side must expose.
- Live wiring (websocket/telemetry ingest) of the `career_dose` feedback loop — design only here.

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
3. **Predicting ≠ advising.** Counterfactual planning queries are *causal* — the twin
   needs explicit causal structure and interventional validation, not just forecasting
   accuracy. And because smart-city data is surveillance-grade, **privacy/governance
   is designed in at ingestion**, not retrofitted.

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
- **Open models (as of 2026-06):** **AlphaEarth Foundations** (Google, 2025) is current
  SOTA — time-continuous embeddings over 9 sensors, shipped as the "Satellite Embedding"
  dataset on Google Earth Engine; open alternatives **Prithvi-EO-2.0** (NASA/IBM),
  **Clay v1.5**, SatMAE/Satlas. **SAM 2** + open building-footprint models (Microsoft/
  Google); **Qwen2.5-VL / InternVL** for semantic extraction. City-entity representations:
  **CityFM** (OSM), **GeoLink** (RS+OSM), **UrbanFusion** — reuse for L2/L3, don't reinvent.
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
- **Model family / menu (as of 2026-06):** spatiotemporal GNN / transformer / neural
  operator over the H3 graph. Precedent for spatiotemporal rollout: ML weather models
  **GenCast** (DeepMind diffusion ensemble — beats ECMWF ENS on ~97% of targets),
  **Aurora** (Microsoft foundation model), **WeatherNext 2** (Google), **GraphCast**.
  For the *temporal* backbone and a strong zero/few-shot baseline, pretrained
  **time-series foundation models — Chronos-2, TimesFM 2.0, Moirai-2, Chronos-Bolt** —
  lead recent energy-load benchmarks and must be evaluated before (and against) any
  bespoke model. Note **Chronos-2 emits well-calibrated intervals (~95% @ 90% nominal)** —
  it partly subsumes the SP5 conformal layer for the forecasting head.
- **Causal requirement (non-negotiable for L5):** counterfactual "what if we build X"
  is an **interventional** *do(X)* query, not a forecast. A model trained on
  observational data learns correlations and will give confidently wrong intervention
  answers. L4 must therefore carry **explicit causal structure** (causal graph /
  do-calculus over the intervention variables, or a structural world model) — *not* a
  pure autoregressive forecaster. See "Interventional validation" in the verification
  spine.
- **Interface:** `rollout(state, intervention, horizon) -> trajectory + uncertainty`.

### L5 — Scenario / planner agent

- **Job:** tool-using LLM agent that poses scenarios, queries L4, and returns plans
  with tradeoffs and confidence.
- **Interface:** natural-language goal → structured plan + predicted impacts + intervals.

## The verification spine (the crux)

A twin is only trustworthy if continuously checked against reality. Verification
draws on three independent sources — but they are **not equal**, so they form an
explicit **oracle hierarchy** (a higher tier always overrides a lower one):

1. **Real measurements** *(tier 1)* — held-out smart-meter / sensor data,
   predict-then-reveal. Ground truth.
2. **Calibrated physics simulators** *(tier 2)* — **EnergyPlus / CityLearn** (energy),
   **SUMO** (traffic), **UMEP** (urban heat). A simulator is a *model*, not truth:
   EnergyPlus routinely diverges 20–40% from metered reality (the **sim2real gap**).
   Simulators must themselves be calibrated against tier-1 data *before* they are
   allowed to score anything; an uncalibrated sim is tier 3.
3. **Conservation constraints** *(tier 3)* — energy balance, mass balance,
   parts-sum-to-whole. Always available, weakest signal.

These oracles drive five mechanisms:

- **RLVR** — reward = agreement with an oracle, so model improvements are *real and
  checkable*, not reward-hacked by a learned critic.
- **Backtest harness** — temporal holdout scored continuously (financial-backtest
  analogy for a city): predict at *t*, reveal truth at *t+Δ*, score, log.
- **Conformal calibration** — prediction intervals with guaranteed coverage
  ("demand = X ± Y at 90% coverage"), independent of model family.
- **Drift detection + active sensing** — when reality diverges from the twin, trigger
  **re-grounding** (refit / re-perceive the affected cells). And run the loop
  *forward*: use the twin's own uncertainty to recommend **where to place the next
  sensor / which cell to measure** (optimal experiment design / active learning), so
  verification is a closed control loop, not passive monitoring.
- **Interventional validation** — forecasting accuracy is *not* causal validity.
  Separately score the twin's **intervention predictions** against real before/after
  natural experiments (new construction, retrofits, tariff/policy changes). This is the
  only honest check on L4's causal structure and L5's advice.

This is also the answer to "how to verify the digital twin": *verification is not a
one-time validation — it is a closed loop that re-grounds the twin against incoming
real data forever, and that separates predicting-the-world from advising-on-changes.*

### Baselines & benchmarks (verification has teeth only against these)

Per standard ML discipline, a bespoke world model earns its place only by **beating
dumb baselines on public benchmarks**:

- **Benchmarks:** Building Data Genome 2, ASHRAE Great Energy Predictor III (GEPIII),
  NREL ResStock/ComStock.
- **Baselines to beat:** weather-normalized / degree-day regression, gradient-boosted
  trees (often the surprise winner on load forecasting), archetype-based **UBEM**, and
  the time-series FMs zero-shot (TimesFM / Chronos). SP4 reports against all of these
  before any "SOTA" claim.

## Decomposition into sub-projects

Each sub-project gets its own spec → plan → implementation cycle.

| # | Sub-project | Depends on | Notes |
|---|---|---|---|
| SP1 | Ingestion + canonical spatiotemporal schema | — | foundation |
| SP2 | VLM perception (imagery → attributes) | SP1 | parallel with SP3 |
| SP3 | LLM state-fusion → city state graph | SP1 | parallel with SP2 |
| SP4 | **Energy world model** (first vertical) | SP1–3 | matches project focus |
| SP5 | **Verification & calibration harness** (also: the verifiable-reasoning *environment*) | SP1 | built *alongside* SP4 |
| SP6 | Scenario / planner agent | SP4–5 | last |
| SP7 | **Urban reasoning model** — RLVR against the SP5 environment | SP4–5 | the research moat (see Novelty) |

**Build order:** SP1 → (SP2 ∥ SP3) → **SP4 + SP5 together** as the first end-to-end
*verifiable energy vertical* → SP6 → **SP7** (reasoning model trained against the SP5
verifiable-reward environment) → then generalize the same pattern to traffic / heat /
water layers.

The first milestone worth defending is **SP4+SP5**: a district-level energy predictor
trained with verifiable rewards, backtested against held-out real meter data, emitting
calibrated intervals. Everything else is scaffolding around making that vertical real,
then replicating the pattern.

## Novelty & research contributions

The perception/embedding/forecasting layers are *not* novel — geospatial FMs (Prithvi,
Clay), CityFM, and the urban spatiotemporal-forecasting line (UrbanGPT, LibCity) already
cover them, and SP2–SP4 should **reuse** rather than reinvent them. The contribution is
the **verifiable-reasoning** layer (SP5 as an environment + SP7 as the model):

> **Thesis.** Treat the city as a *verifiable-reward (RLVR) environment* — train a
> reasoning model whose multi-step intervention claims ("retrofit district D cuts load
> by X%") are scored by physics simulators (EnergyPlus / CityLearn / SUMO) and held-out
> real meter/sensor data, not by forecast error alone. RLVR works for math/code because
> the answer is checkable; a city is *also* checkable. The novelty is the **combination**
> (physics-sim-as-verifier + RLVR + urban + causal), not any single ingredient.

Five mechanisms, with literature status from a 2026-06 targeted survey (confidence tagged;
*not* adversarially verified — re-check before publication):

| # | Mechanism | Status (conf.) | Closest prior art to position against |
|---|---|---|---|
| 1 | RLVR where the verifier is a **physics sim + real meters** | ⚠️ **partially claimed** (med) | **RLVR-World** (2025) already applies RLVR to text & vision *world models* (F1/LPIPS verifiers). Open twist = *physics-simulator + real-measurement* verifier on an *urban* world model. Also *Outcome-based RL to Predict the Future* (RLVR forecasting, no sim) |
| 2 | Interleaved **simulator-grounded chain-of-thought** (AlphaGeometry-style) | **contested** (med) | OpenCity, Urban Generative Intelligence, LLMLight, PhysicsAgentABM — but they use the sim as a *sandbox*, not a *step-verifier for reward* |
| 3 | **Causal/interventional reward** (natural experiments) vs forecast accuracy | **unclaimed** (high) | LLM-agent papers do qualitative counterfactual *analysis*, not interventional reward training |
| 4 | **Conservation-law process rewards** for physical reasoning | **likely open** (med-high) | process-reward models + PINNs exist *separately*, not combined for reasoning steps |
| 5 | **Self-improving twin** via reasoning-discovered hypotheses + active sensing | **likely open** (med) | digital-twin+LLM closed loops do what-if; the hypothesis→active-sensing→update loop appears open |

**Honest framing (updated 2026-06).** **RLVR-World (2025) repositions mechanism 1** —
RLVR on world models is *no longer unclaimed*; our defensible contribution is the
*physics-simulator + real-measurement verifier on an urban world model*, not "RLVR on
world models" per se. So **mechanisms 3 (causal/interventional reward) and 4
(conservation-law process rewards) are now the strongest claims.** Mechanism 2 remains
contested (substantial "LLM-agent-in-city-simulator" work — frame SP7 as *physics-verified
reasoning reward*, not "LLM + simulator"). Caveat (Yue et al. 2025): RLVR mostly *sharpens
reasoning patterns already in the base model* rather than inventing new ones — temper
"teaches new reasoning" claims. Statuses are provisional (lighter survey, no adversarial
multi-vote verification).

**Related work to cite (2026-06):** geo-FMs **AlphaEarth Foundations**, Prithvi-EO-2.0,
Clay; urban FMs **CityFM**, **UrbanMFM**, **UrbanFusion**, **GeoLink**, ReFound;
**CityBench** (LLM-as-city-world-model — closest framing); TS-FMs Chronos-2 / TimesFM 2.0;
weather GenCast / Aurora; **RLVR-World** (RLVR on world models); urban ST forecasting
(UrbanGPT, LibCity, GPD); *Outcome-based RL to Predict the Future*; PhysicsAgentABM;
OpenCity / Urban Generative Intelligence; USTBench & STARK (ST-reasoning
benchmarks); Telecom World Models; *Digital Twin AI: from LLMs to World Models* (Jan 2026).

### SP7 — Urban reasoning model (the moat)

- **Job:** an LLM reasoner that answers interventional planning queries with a *grounded,
  verifiable* chain — each step may call the L4 world model / SP5 oracle, and the final
  claim is scored against simulator + real measurements.
- **Training:** RLVR (GRPO/PPO) against the **SP5 environment** as the reward source.
  Reuses the existing reasoning/RLVR stack; points it at the city instead of math/code.
- **Validation:** interventional validation (SP5) — score intervention predictions
  against real before/after natural experiments, not just forecast error.
- **Depends on:** SP4 (world model to call) + SP5 (the verifiable-reward environment).

## Privacy & governance (design-in, not retrofit)

High-resolution smart-meter / sensor data is **surveillance-grade** — it reveals
occupancy, behavior, even appliance-level activity. This is a hard blocker for real
deployment (GDPR, consent regimes) and must be designed in at L1, not bolted on:

- **k-anonymity at the H3 cell** — never release a prediction or fused value derived
  from fewer than *k* metered premises in a cell.
- **Differential privacy** on released predictions and any published aggregate.
- **Provenance-gated access** — sensitive layers carry access tiers; the L5 agent and
  external consumers see DP/aggregated views only.
- **Governance record** — consent basis and retention policy tracked per adapter; a
  region adapter that can't satisfy the privacy contract is rejected at registration.

## Cross-cutting concerns

- **Spatiotemporal data model** — H3 hex grid + time axis is the universal key across
  layers; everything joins on `(cell, time)`.
- **Provenance & uncertainty** — every derived value carries source lineage and a
  confidence/interval; uncertainty propagates up the stack. Where sources **conflict**
  (OSM vs satellite footprint), fuse with explicit uncertainty rather than picking one.
- **Reflexivity (observer effect)** — the twin's predictions drive policy that changes
  the city, so the system shifts under the model. Backtests and drift detection must
  attribute divergence to *model error vs twin-induced change*, or they will mistake
  successful interventions for failures.
- **Equity / distributional outcomes** — optimizing *aggregate* energy can worsen who
  bears heat, cost, or disruption. Report distributional metrics (per-population,
  per-income-tract) alongside aggregate accuracy; an equity regression is a failure
  even at higher aggregate accuracy.
- **Open model zoo** — all named models/datasets are open-licence; region-specific or
  proprietary feeds are optional adapters.
- **Portability** — instantiating the twin for a new city = registering that city's
  bounding box + any region adapters; global layers work out of the box.

## Compute budget & model sizing (single A100 80GB target)

**Design constraint:** every layer is sized to **train/fine-tune on one A100 80GB**.
The rule that makes this hold: *fine-tune pretrained checkpoints (LoRA/QLoRA), never
pretrain a foundation model from scratch; cap LLMs/VLMs at 7–8B, world models at ≤1B.*

| Layer | Model(s) | Params | Fits 1×A100 80GB? | Strategy |
|---|---|---|---|---|
| L2 perception | AlphaEarth (embeddings) / Prithvi-EO-2.0 / Clay v1.5 | 0.1–0.6B | ✅ | embeddings off-the-shelf / full fine-tune |
| L2 perception | SAM 2 | ~0.2B | ✅ | inference + decoder fine-tune |
| L2 / L3 / L5 | Qwen2.5-VL-7B, Llama-3.1-8B, Qwen2.5-7B | 7–8B | ✅ | **QLoRA** (train) / 4-bit (infer) |
| L4 world model | Chronos-2, TimesFM 2.0, Moirai-2 | 0.01–0.7B | ✅ | fine-tune (often zero-shot first) |
| L4 world model | spatiotemporal GNN / neural operator | <0.1B | ✅ | train from scratch (data-bound, not param-bound) |
| L5 agent | instruct LLM | 7–8B local *or* API | ✅ | **inference only**, no training |
| Spine | RLVR (GRPO/PPO) on the world model | <1B target | ✅ | LoRA + vLLM gen; tight but fits |

**What would need more than 1×A100 (and is therefore out of scope):**
- Full fine-tuning a 70B LLM, or RLVR/PPO on a 7B+ LLM with full precision → 2–8×A100
  or H100s. *Avoided* by keeping training on the ≤1B world model and using 7–8B models
  via QLoRA/API.
- Pretraining a geospatial or weather foundation model from scratch → TPU/H100 weeks.
  *Avoided* — we only ever fine-tune the open pretrained checkpoints.

**Cost estimate (cloud A100 80GB):** ~$1.5–2.5/hr spot (Lambda / RunPod / Vast),
~$3–4/hr on-demand.

| Work item | GPU-hours | Cost (spot) |
|---|---|---|
| L2 geo-FM fine-tune | 20–50 | ~$50–150 |
| L4 TS-FM fine-tune + GNN | 20–80 | ~$50–250 |
| SP5 RLVR loop on world model | 50–150 | ~$150–450 |
| L3 / L5 (inference-dominated) | <10 | ~$0–30 |
| **End-to-end energy vertical (SP1+SP2/3+SP4+SP5)** | **~200–500** | **~$500–1.5k** |

Non-GPU costs: object storage for Sentinel-2 / ERA5 tiles (read from cloud-hosted open
catalogs — AWS Open Data, Microsoft Planetary Computer — to avoid egress); negligible
LLM API spend if L5 uses a hosted instruct model instead of local 7–8B.

## Out-of-scope (deliberately deferred)

- Real-time streaming ingestion (batch first; streaming is an SP1 extension).
- Non-energy verticals (traffic, heat, water) — same pattern, after the energy
  vertical proves out.
- A production UI / dashboard for planners (the L5 agent API comes first).
- Multi-city federation / transfer learning across twins.

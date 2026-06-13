# Use Cases & Commercial Framing

**Date:** 2026-06-12
**Purpose:** Translate the urban digital twin into commercial value across sectors, and
use that to sharpen build order. Energy utilities are the lead deep-dive (the first
vertical); a cross-sector section follows. Each use case maps to the twin layer that
enables it and flags **commodity** (buyable today) vs **moat** (hard to buy, enabled by
our novel pieces).

> Layer / sub-project references: L1 ingestion, L2 VLM perception, L3 LLM state graph,
> L4 world model, L5 planner agent, SP5 verification spine, SP7 reasoning model. See
> [`specs/2026-06-12-urban-digital-twin-design.md`](specs/2026-06-12-urban-digital-twin-design.md).

**Platform thesis:** every sector below needs the same thing — *verifiable spatiotemporal
predictions over the built environment*. The moat (the verification spine, building-stock
intelligence from open satellite data, and counterfactual reasoning) is **sector-agnostic**.
This is one twin with many verticals, not many products. Energy proves the pattern; the
rest reuse it.

# Lead vertical: energy utilities

## The commercial through-line

Utilities can already *buy* load and price forecasting. The defensible value is in what
they **can't** easily buy, which is exactly what our novel pieces enable:

1. **Verifiable counterfactual planning** — electrification / EV "will-it-hold" answers
   with a *checkable* verdict (causal RLVR, SP7 + SP5).
2. **Building-stock intelligence from open satellite data** — solar potential, retrofit
   candidates, outage vegetation risk at scale, with no proprietary surveys (L2).
3. **Physics-verified personalized advice** — recommendations whose savings are
   EnergyPlus-grounded and explainable → customer trust + regulatory defensibility (SP5).

This splits the roadmap cleanly: **SP4 forecasting is revenue-now; SP5 + SP7 are the moat.**

## Use-case catalogue

### A. Forecasting & trading (revenue-now — mostly SP4)

| Use case | Value | Layers | Class |
|---|---|---|---|
| Load forecasting (system → feeder) | hour/day-ahead demand for ops & settlement | L4 + L1 weather | commodity |
| Energy trading / price prediction | day-ahead & intraday wholesale price, bidding, imbalance | L1 weather + L4 net load | commodity, *edge from granularity* |
| Renewable / solar forecasting | roof-solar & wind output for balancing | **L2 (roof/solar)** + L1 + L4 | **moat** |
| Net-load / duck-curve | demand minus behind-the-meter solar | L2 solar + L4 | **moat** |

**Edge on trading:** every utility buys price/load models, so a generic forecaster does
not differentiate. Our edge is **spatial + physical granularity** — net-load forecasts
that fold in *satellite-derived* behind-the-meter solar (L2) and weather (L1) at H3-cell
resolution, improving duck-curve and imbalance prediction over aggregate models. This is
buildable earliest: SP4 plus the Open-Meteo weather adapter already in SP1.

### B. Grid operations & asset management

| Use case | Value | Layers | Class |
|---|---|---|---|
| Peak / demand-response targeting | predict peaks, pick which customers to dispatch | L3 customer graph + L4 | semi-moat |
| Grid congestion forecasting | feeder / transformer overload, defer capex | L1 infra topology + L4 | semi-moat |
| Asset health / failure prediction | transformer & cable failure from load + weather + age | L1 infra + L4 | semi-moat |
| Outage prediction | storm-driven outages, tree-fall risk | L1 weather + **L2 vegetation** + L4 | **moat** |
| EV charging load | spatial EV adoption & charging patterns | L3 + L4 | semi-moat |

### C. Planning & investment (the core moat — SP7 reasoning)

| Use case | Value | Layers | Class |
|---|---|---|---|
| Capacity planning / electrification | "if 30% EV + heat pumps in district D, does the feeder hold?" | **L4 causal + SP7** | **★ core moat** |
| DER siting | where to place batteries / solar / substations for max benefit | L4 + active sensing | **moat** |
| Network loss reduction | predict & reduce technical losses | L4 | semi-moat |

These are **interventional** *do(X)* questions — precisely why the twin needs causal
structure and verifiable interventional validation, not just a forecaster. This is where
the project's research novelty becomes the commercial moat.

### D. Customer-facing / personalized (the better wedge — SP5 verification)

| Use case | Value | Layers | Class |
|---|---|---|---|
| Personalized energy insights | disaggregate bill (NILM), benchmark vs similar homes, tailored advice | **L2 attrs + L3 archetype + L4** | **moat** |
| Retrofit / solar-ROI advisor | "insulation saves you £X" / "your roof = Y kWp" | L2 + **SP5 EnergyPlus-verified** | **★ core moat** |
| Tariff / time-of-use coaching | shift load to cheaper / cleaner periods | L4 + L1 carbon | semi-moat |
| Fuel-poverty / vulnerable targeting | find at-risk households for support | L2/L3 + **equity layer** | **moat (regulatory)** |

**Why personalized insights is the stronger wedge:** the trust problem — "why should I
believe this £X saving?" — is exactly what the verification spine solves. A retrofit or
solar recommendation whose savings are **EnergyPlus-verified and physically checkable**
(SP5), explained by a **grounded reasoning chain** (SP7), is something a black-box
recommender cannot offer, and it is regulatory-defensible.

### E. Sustainability & regulatory

| Use case | Value | Layers | Class |
|---|---|---|---|
| Carbon-intensity forecasting | grid carbon by time → shift load to clean periods | L1 + L4 | growing commodity |
| Portfolio decarbonization reporting | Scope-2/3 tracking | L1 + L3 | commodity |
| Loss-of-load probability / reliability | regulatory reliability metrics | L4 + SP5 | semi-moat |

# Beyond utilities: cross-sector verticals

Same twin, same verification spine, different target layer. Ordered by twin-fit and how
directly the existing stack transfers.

### Highest-fit sectors (deep reuse of the energy stack)

**City government & urban planners** — the most natural buyer of a "smart city" twin.
Zoning / upzoning impact, infrastructure capacity planning, net-zero & climate-adaptation
strategy, permitting triage, heat-island mitigation siting. *Layers:* all. *Moat:* the
counterfactual reasoner (SP7) answers "what if we upzone district D / add this transit
line?" with a **verifiable** before/after-grounded verdict — exactly the interventional
question planners cannot get from a forecaster.

**Insurance & reinsurance** — property & catastrophe risk pricing (flood, heat, wildfire,
subsidence), portfolio exposure, parametric triggers. *Layers:* L1 weather/climate + **L2
(roof material, footprint, vegetation proximity)** + L4 hazard dynamics + SP5. *Moat:*
per-building risk from **open satellite data at scale** (no surveys) and **physics-grounded,
auditable** hazard estimates — regulatory-defensible pricing.

**Real estate & developers** — site selection, valuation, climate-risk disclosure (TCFD),
solar/retrofit ROI, stranded-asset screening. *Layers:* L2 building attrs + L3 + SP5. *Moat:*
building-stock intelligence (L2) plus **EnergyPlus-verified** retrofit/solar economics (SP5).

**Transport & mobility** — congestion forecasting, EV-charger siting, transit & active-travel
planning, low-emission-zone impact. *Layers:* L1 infra (road graph) + **L4 traffic world
model** (SUMO oracle in SP5) + SP7. *Moat:* counterfactual "add this bus lane / LTN" with a
verified mobility outcome. This is the **second vertical** — the energy pattern transplanted
to traffic, proving the platform thesis.

### Adjacent sectors (same pattern, new data layer)

| Sector | Headline use cases | Layers | Moat driver |
|---|---|---|---|
| **Water utilities** | demand, leakage detection, drought, stormwater/flood | L1 + L4 water layer + SP5 | verified scenario sim |
| **Emergency mgmt / public safety** | heat/flood early warning, evacuation, disaster response | L1 + L2 + L4 + SP5 | verifiable scenario rollout |
| **Public health** | heat & air-quality exposure, health-burden mapping | L1 + L2 + **equity layer** | exposure + equity, satellite-derived |
| **Climate / ESG / carbon MRV** | monitoring-reporting-verification for credits, heat-island | L1 + L2 + **SP5** | the verification spine *is* MRV — verifiable by construction |
| **Construction** | progress monitoring, infrastructure planning | **L2 VLM (construction state)** + L3 | build-state from imagery, no site visits |
| **Telecom / digital infra** | tower & 5G siting, coverage vs built form | L1 + L2 | built-environment intelligence (L2) |

### Why one platform, not six products

Each vertical swaps the **target layer** (load → traffic → water → risk) and the **physics
oracle** (EnergyPlus → SUMO → flood model), but reuses **SP1 ingestion, L2 perception, L3
state graph, the SP5 verification spine, and the SP7 reasoner unchanged**. The expensive,
differentiated parts are shared; only the thin vertical-specific head changes. That is the
commercial case for building the verification + reasoning moat once and amortizing it.

## Build-order implication

1. **SP4 first** — load / price / solar forecasting. Revenue-now, validates the world
   model, reuses the SP1 weather adapter. Beat the LibCity / UrbanGPT baselines.
2. **SP5 alongside SP4** — verification turns forecasts into *trustable* numbers and
   unlocks the personalized-advice wedge.
3. **SP7 reasoning** — verifiable counterfactual planning; the electrification and
   retrofit-advisor moat.
4. **Second vertical (mobility/traffic)** — transplant the proven energy pattern: swap the
   target layer (load → traffic) and oracle (EnergyPlus → SUMO), reuse everything else.
   This is the step that demonstrates the platform thesis rather than a single product.

Within energy, the two use cases originally asked about sit at opposite ends: **energy
trading** is the commodity entry point (build first, modest differentiation); **personalized
energy insights** is the differentiated wedge (depends on the verification + reasoning moat).

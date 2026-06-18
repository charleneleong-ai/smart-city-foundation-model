# Intervention oracles — real before/after datasets for interventional validity

The interventional environment ([`src/sctwin/reason/intervention.py`](../../src/sctwin/reason/intervention.py)) scores a reasoner's predicted intervention effect (Δ load) against an **oracle Δ**. Today that oracle is a transparent physics proxy (`counterfactual` — heating-degree retrofit / peak→off-peak tariff). The paper's headline result needs the Δ validated against **real natural experiments**. This is the dataset scout + what got wired.

Grounded against **June 2026**.

## Shortlist

| # | Source | Lever | Design | Res / period | Access |
|---|--------|-------|--------|--------------|--------|
| 1 | **Low Carbon London — dynamic-ToU trial 2013** | tariff | 1,122 treated (ToU) vs 4,545 control (Std), same period | ½-hourly, 2013 | **CC-BY, direct download** ([London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/)) |
| 2 | **UK NEED (DESNZ)** | retrofit | property-level consumption matched to installed measures, pre/post | **annual**, 2005–2024 | **open anonymised 50k & 4M samples** ([gov.uk](https://www.gov.uk/government/collections/national-energy-efficiency-data-need-framework)) |
| 3 | CER Ireland Smart Metering Trial 2010 | tariff | 4,225 homes, randomized control vs ToU/DSM arms | ½-hourly, 536 d | free but **gated** (ISSDA data request) |
| 4 | France "Hello Watt" retrofit ([arXiv 2603.26548](https://arxiv.org/abs/2603.26548), Jun 2026) | retrofit | true difference-in-differences, ~2,500 homes | daily elec+gas | ⚠️ availability/license **not stated** in the paper |

## Wired (this PR) — #1 LCL tariff + #2 NEED retrofit

Both slot in behind one seam: a measured before/after **pair of profiles** → [`measured_question`](../../src/sctwin/reason/intervention.py) → [`InterventionEnvironment.from_questions`](../../src/sctwin/reason/intervention.py). The reward/rollout machinery is identical to the physics-proxy path; only `true_delta` is now empirical.

- **Tariff** — [`LCLTariffAdapter`](../../src/sctwin/adapters/demand.py)`.profiles(cell)` → `(control Std, treated ToU)` mean half-hourly profiles ([`lcl_group_profile`](../../src/sctwin/demand.py)). Δ = `effect(Std, ToU, "peak")` = the measured peak-load cut.
- **Retrofit** — [`NEEDRetrofitAdapter`](../../src/sctwin/adapters/demand.py)`.split(cell)` → `(pre, post)` annual consumption for measure-homes ([`need_measure_split`](../../src/sctwin/demand.py)). Δ = `effect(pre, post, "mean")` = the measured annual saving.

### Box usage (the data files are large / need download)

```python
from sctwin.adapters.demand import LCLTariffAdapter, NEEDRetrofitAdapter
from sctwin.reason.intervention import InterventionEnvironment, measured_question

# tariff — the LCL Datastore CSV (~760 MB) downloaded to a local path
control, treated = LCLTariffAdapter(source="data/lcl.csv").profiles(cell="<h3>")
q_tariff = measured_question("tariff", control, treated, cell="<h3>", metric="peak")

# retrofit — the NEED anonymised sample (set cols to the release's data dictionary)
pre, post = NEEDRetrofitAdapter(source="data/need.csv", measure_col="LOFT_FLAG").split(cell="<h3>")
q_retrofit = measured_question("retrofit", pre, post, cell="<h3>", metric="mean")

env = InterventionEnvironment.from_questions([q_tariff, q_retrofit])  # real-oracle env
print(env.rollout(lambda question: question.true_delta))               # oracle policy -> 1.0
```

The adapters are box-ready (injectable `_read`), exercised offline via stubbed readers in [`tests/test_oracle.py`](../../tests/test_oracle.py).

## Difference-in-differences (the de-biased estimand)

A plain **treated − control** (tariff) / **post − pre** (retrofit) Δ conflates the intervention with the pre-existing treated/control gap (selection) and the common time trend (weather, economy). [`did_effect`](../../src/sctwin/reason/intervention.py) nets both out:

```
DiD = (treated_post − treated_pre) − (control_post − control_pre)
```

[`did_question(kind, treated_pre, treated_post, control_pre, control_post, *, cell, metric)`](../../src/sctwin/reason/intervention.py) builds the question with this de-biased `true_delta`. The adapters produce the four groups:

- **Tariff** — [`LCLTariffAdapter.did_profiles(cell, pre_year=2012, post_year=2013)`](../../src/sctwin/adapters/demand.py): ToU vs Std across the pre-trial and trial years (the dToU prices applied in 2013 only, so both groups are on standard tariffs in 2012 — the DiD baseline).
- **Retrofit** — [`NEEDRetrofitAdapter.did_split(cell)`](../../src/sctwin/adapters/demand.py): measure-homes vs non-measure homes across the pre/post years.

```python
tp_pre, tp_post, cp_pre, cp_post = LCLTariffAdapter("data/lcl.csv").did_profiles(cell="<h3>")
q = did_question("tariff", tp_pre, tp_post, cp_pre, cp_post, cell="<h3>", metric="peak")
```

## Caveats (verify before quoting results)

- **LCL schema** — the kWh column name (`KWH/hh (per half hour) `) and the `stdorToU` Std/ToU labels are the published LCL schema; the Datastore page didn't expose column names to the scout, so confirm on download. The repo's existing LCL feed is the Monash/Chronos parquet (consumption-only) — the trial labels come from the Datastore release.
- **NEED granularity** — annual, so it validates the retrofit *magnitude*, not its hourly shape. Column names vary by release (passed as params).
- **DiD assumption** — parallel trends (treated and control would have moved together absent the intervention). The DiD profiles use each group-year's mean profile (peak = its annual max); a diurnal (half-hour-of-day) aggregation is a finer refinement.
- **France Hello-Watt** dataset access is unconfirmed (not in the abstract).

## Follow-ups

- Wire an `--oracle real` path into `apps/eval_reasoner.py` / `apps/train_reasoner.py` (needs the downloaded files; left box-side).
- CER Ireland (richer tariff arms) once an ISSDA agreement is in place.

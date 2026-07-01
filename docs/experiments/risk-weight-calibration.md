# Risk-Weight Calibration — Firefighter Acute-Risk Model

**Date:** 2026-07-01
**Scope:** provenance for every multiplier in [`acute_risk`](../../src/sctwin/deploy/risk.py) — the
person-intrinsic heat + cardiac susceptibility term of the deployment risk model.

## What "calibrate" means here (and what it doesn't)

We have **no linked firefighter outcome dataset** in this repo — no roster joined to on-duty cardiac
events — so these weights are **not fitted**. Calibration here means the weaker thing, done honestly:
replace the first-pass guesses (`cv=1.5`, flat `0.05·len(conditions)`, `age=0.02/yr`) with **relative
multipliers grounded in published firefighter sudden-cardiac-death (SCD) epidemiology**, each number
traceable to a source below.

Two deliberate compressions, because the model is a *bounded relative susceptibility index*, not a
mortality probability:

1. The literature reports **odds ratios for the terminal SCD event** in case-control studies. Those
   don't compose as independent probabilities — stacking OR 12 × OR 6.89 × … would explode a
   multiplicative index. So the raw ORs set the **ranking and rough magnitude**, then are compressed
   (e.g. prior-CVD OR ≈ 6.89 [1] → a `cv` multiplier of 2.5, not 6.9).
2. Cardiac events are **~45 % of on-duty firefighter deaths** [2][6], which is why the acute term is
   cardiac-dominated rather than heat-dominated.

## Provenance

| Weight (in `acute_risk`) | Before | After | Source | Rationale |
|---|---|---|---|---|
| `cv_factor` (known-CVD flag) | 1.5 | **2.5** | [1] | Prior CVD history OR ≈ 6.89 for SCD; compressed for a bounded index. |
| `CONDITION_WEIGHTS` cardiac/hypertensive (`hypertension`, `prior mi`, `coronary`, `myocardial`, `cardiovascular`) | 0.05 flat | **0.40** | [1] | Hypertension + LVH OR ≈ 12 — the single strongest person-intrinsic predictor. |
| `CONDITION_WEIGHTS` metabolic (`diabetes` 0.25, `obesity` 0.20) | 0.05 flat | **0.20–0.25** | [1][5] | Metabolic-syndrome components; obesity/lifestyle named as the driver of *young*-firefighter SCD. |
| `CONDITION_WEIGHTS` respiratory (`copd` 0.25, `asthma`/`respiratory` 0.15) | 0.05 flat | **0.15–0.25** | [2] | Weaker cardiac link; matters more via smoke-inhalation sensitisation than SCD. |
| `_CONDITION_DEFAULT` (unrecognised) | 0.05 flat | **0.08** | — | A listed comorbidity still adds a little; kept small so it can't rival a cardiac one. |
| `fitness_factor` (low CRF → up to ~2×) | up to 2× | **up to 2×** (kept) | [3][4] | Low cardiorespiratory fitness is a dominant *modifiable* risk; meta-analysis ties low CRF to CVD risk factors. |
| `age_factor` (per year over 40) | 0.02 | **0.03** | [3][4] | CRF declines and SCD risk climbs past ~40; linear proxy. |
| `HEAT_TOLERANCE` band (1.25 / 1.0 / 0.85) | — | **kept** | — | **Least-grounded** — heat-strain physiology, not from an OR. Individual heat tolerance modulates core-temp rise; a follow-up should ground this against a heat-strain study. |

The flat penalty was the weakest link: it scored **hypertension (OR ≈ 12) and eczema identically**.
The keyword table now separates them (`condition_burden(("hypertension","prior MI")) = 0.80` vs
`("eczema",) = 0.08`), which is what [`test_condition_burden_weights_cardiac_above_minor`](../../tests/test_deploy_risk.py) pins.

## Known limitations (carried as follow-ups)

- **Flag ↔ ledger double-count.** A firefighter with `cardiovascular=True` *and* `"hypertension"`
  listed is counted by both `cv_factor` and `condition_burden`. Intentional for now (both signals are
  real), but a cleaner design derives the flag from the ledger. Tracked with the cross-repo
  `HealthProfile` reconciliation in [PR #47](https://github.com/charleneleong-ai/smart-city-foundation-model/pull/47).
- **No held-out validation / calibration run.** These are literature-shaped priors, not a fit — there
  is **no experimental run to reference yet**. A real calibration would need a linked-outcome dataset:
  the NIOSH firefighter-fatality registry [6] joined to rosters, or Fire-Shield telemetry outcomes as
  labels. Until then the `RiskScore` band stays an explicit *prior* uncertainty, not a conformal one.
- **Heat band** is the one factor with no citation — flagged above.

## References

1. Smith DL, Kales SN, et al. **Sudden Cardiac Death Among Firefighters ≤45 Years of Age in the United States.** *Am J Cardiol* 2013. — HTN+LVH OR ≈ 12; prior CVD OR ≈ 6.89; smoking OR 3.53. <https://pubmed.ncbi.nlm.nih.gov/24079519/>
2. Kales SN, et al. **Emergency Duties and Deaths from Heart Disease among Firefighters in the United States.** *N Engl J Med* 2007;356:1207–15. — exertion → CHD death; SCD ≈ 45 % of on-duty deaths. <https://www.ahajournals.org/doi/10.1161/circulationaha.117.027018>
3. **Association between Cardiovascular Disease Risk Factors and Cardiorespiratory Fitness in Firefighters: A Systematic Review and Meta-Analysis.** *IJERPH* 2023. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9957465/>
4. **Higher cardiorespiratory fitness is strongly associated with lower cardiovascular risk factors in firefighters.** *Sci Rep* 2021. <https://www.nature.com/articles/s41598-021-81921-1>
5. **Hypertension in the United States Fire Service.** *Int J Environ Res Public Health.* <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8160987/>
6. NIOSH. **Preventing Fire Fighter Fatalities Due to Heart Attacks and Other Sudden Cardiovascular Events.** Pub. 2007-133. <https://www.cdc.gov/niosh/docs/2007-133/pdfs/2007-133.pdf>

*Literature grounded via web search on 2026-07-01.*

from dataclasses import dataclass

from sctwin.deploy.exposure import heat_load, toxicant_dose
from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.roster import Firefighter


@dataclass(frozen=True)
class RiskWeights:
    acute: float = 1.0  # heat / cardiac event on the fireground
    incident: float = 1.0  # this fire's toxicant load
    career: float = 1.0  # cumulative carcinogen accrual


@dataclass(frozen=True)
class RiskScore:
    value: float
    low: float  # prior-uncertainty band (NOT conformal — transferred dose-response prior)
    high: float
    drivers: dict[str, float]  # unweighted per-term contributions


# Person-intrinsic acute-risk multipliers, calibrated to firefighter sudden-cardiac-death (SCD)
# epidemiology (Kales/Smith case-control, Am J Cardiol 2013; cardiorespiratory-fitness meta-analysis,
# IJERPH 2023). These are RELATIVE weights on a bounded susceptibility index, deliberately compressed
# from the raw death odds ratios (they don't compound as a mortality probability). Provenance of every
# number: docs/experiments/risk-weight-calibration.md.
HEAT_TOLERANCE = {"low": 1.25, "avg": 1.0, "high": 0.85}  # heat-strain susceptibility band

# Per-condition acute weight, keyword-matched. Cardiac/hypertensive conditions inherit the SCD odds
# ratios' dominance (hypertension+LVH OR~12, prior CVD OR~6.89); metabolic/respiratory are moderate;
# anything unrecognised still adds a little.
CONDITION_WEIGHTS = {
    "hypertension": 0.40,
    "prior mi": 0.40,
    "myocardial": 0.40,
    "coronary": 0.40,
    "cardiovascular": 0.40,
    "diabetes": 0.25,
    "copd": 0.25,
    "obesity": 0.20,
    "asthma": 0.15,
    "respiratory": 0.15,
}
_CONDITION_DEFAULT = 0.08


def condition_burden(conditions: tuple[str, ...]) -> float:
    """Summed acute weight of the listed comorbidities, keyword-matched — a high-risk cardiac
    condition weighs far more than a minor one, unlike a flat per-count penalty."""
    return sum(
        next((w for k, w in CONDITION_WEIGHTS.items() if k in c.lower()), _CONDITION_DEFAULT)
        for c in conditions
    )


def acute_risk(ff: Firefighter, hl: float) -> float:
    """Heat + cardiac susceptibility for thermal load `hl`, amplified by age, known CVD, low fitness,
    respiratory status, heat-tolerance band, and the comorbidity ledger. Weights are literature-shaped
    relative multipliers, not fitted to outcome data — see the calibration doc."""
    age_factor = 1.0 + max(ff.age - 40, 0) * 0.03  # SCD/CRF risk climbs with age past ~40
    cv_factor = 2.5 if ff.cardiovascular else 1.0  # known CVD: compressed from SCD OR~6.89
    resp_factor = 1.2 if ff.respiratory else 1.0
    fitness_factor = 1.0 + (1.0 - ff.fitness)  # low CRF is a dominant modifiable risk -> up to ~2x
    heat_factor = HEAT_TOLERANCE[ff.heat_tolerance]
    comorbidity_factor = 1.0 + condition_burden(ff.conditions)
    return hl * age_factor * cv_factor * resp_factor * fitness_factor * heat_factor * comorbidity_factor * 0.01


def incident_dose_risk(td: float, ff: Firefighter) -> float:
    """Acute toxicant risk from this incident's dose; respiratory comorbidity sensitises."""
    return td * (1.3 if ff.respiratory else 1.0) * 0.01


def career_risk(ff: Firefighter, td: float) -> float:
    """Cumulative carcinogen accrual (career bank + this increment), convex so late dose hurts more."""
    return (ff.career_dose + td) ** 1.2 * 0.001


def combined_risk(
    ff: Firefighter,
    scenario: FireScenario,
    time_on_scene_min: float,
    role: str,
    ppe: str,
    weights: RiskWeights = RiskWeights(),
    prior_band: float = 0.3,
) -> RiskScore:
    """The personalised combined index. The `prior_band` is symmetric prior uncertainty on the
    transferred dose-response coupling — it is NOT a calibrated/conformal interval."""
    td = toxicant_dose(scenario, time_on_scene_min, ppe)
    hl = heat_load(scenario, time_on_scene_min, role)
    drivers = {"acute": acute_risk(ff, hl), "incident": incident_dose_risk(td, ff), "career": career_risk(ff, td)}
    value = weights.acute * drivers["acute"] + weights.incident * drivers["incident"] + weights.career * drivers["career"]
    return RiskScore(value, value * (1.0 - prior_band), value * (1.0 + prior_band), drivers)

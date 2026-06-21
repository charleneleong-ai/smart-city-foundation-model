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


HEAT_TOLERANCE = {"low": 1.25, "avg": 1.0, "high": 0.85}


def acute_risk(ff: Firefighter, hl: float) -> float:
    """Heat + cardiac risk for thermal load `hl`, amplified by age, CV, low fitness, respiratory,
    heat-tolerance band, and the count of listed comorbidities."""
    age_factor = 1.0 + max(ff.age - 40, 0) * 0.02
    cv_factor = 1.5 if ff.cardiovascular else 1.0
    resp_factor = 1.2 if ff.respiratory else 1.0
    fitness_factor = 1.0 + (1.0 - ff.fitness)  # unfit -> up to ~2x
    heat_factor = HEAT_TOLERANCE[ff.heat_tolerance]
    comorbidity_factor = 1.0 + 0.05 * len(ff.conditions)  # fuller clinical ledger beyond cv/resp
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

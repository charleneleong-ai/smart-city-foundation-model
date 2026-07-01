import pytest

from sctwin.deploy.risk import RiskWeights, combined_risk, condition_burden
from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.roster import Firefighter

SCN = FireScenario("c", "dwelling", 4.0, 80.0, 32.0, 8.0, 70.0, 120.0)
FIT = Firefighter("fit", 28, "M", "ba", 5, False, False, 0.95, 5.0)
FRAIL = Firefighter("frail", 57, "M", "ba", 30, True, True, 0.50, 80.0)


def test_frailer_archetype_scores_higher_at_same_deployment():
    fit = combined_risk(FIT, SCN, 20, "ba", "ba")
    frail = combined_risk(FRAIL, SCN, 20, "ba", "ba")
    assert frail.value > fit.value  # age + CV + low fitness + high career dose all push up


def test_score_carries_a_prior_band_and_driver_breakdown():
    r = combined_risk(FIT, SCN, 20, "ba", "ba")
    assert r.low < r.value < r.high
    assert set(r.drivers) == {"acute", "incident", "career"}


def test_weights_zero_out_terms():
    only_career = combined_risk(FRAIL, SCN, 20, "ba", "ba", weights=RiskWeights(acute=0.0, incident=0.0))
    assert only_career.value == only_career.drivers["career"] * 1.0


def test_lower_heat_tolerance_raises_acute_risk():
    high = Firefighter("h", 30, "M", "ba", 5, False, False, 0.9, 5.0, heat_tolerance="high")
    low = Firefighter("l", 30, "M", "ba", 5, False, False, 0.9, 5.0, heat_tolerance="low")
    assert combined_risk(low, SCN, 20, "ba", "ba").drivers["acute"] > combined_risk(high, SCN, 20, "ba", "ba").drivers["acute"]


def test_listed_conditions_raise_acute_risk():
    clean = Firefighter("c", 30, "M", "ba", 5, False, False, 0.9, 5.0)
    burdened = Firefighter("b", 30, "M", "ba", 5, False, False, 0.9, 5.0, conditions=("hypertension", "diabetes"))
    assert combined_risk(burdened, SCN, 20, "ba", "ba").drivers["acute"] > combined_risk(clean, SCN, 20, "ba", "ba").drivers["acute"]


def test_condition_burden_weights_cardiac_above_minor():
    # calibrated to SCD odds ratios: hypertension/prior-MI dominate, unrecognised conditions add a little
    assert condition_burden(("hypertension", "prior MI")) == pytest.approx(0.80)
    assert condition_burden(("eczema",)) == pytest.approx(0.08)


def test_high_risk_condition_outweighs_a_minor_one():
    htn = Firefighter("h", 35, "M", "ba", 5, False, False, 0.9, 5.0, conditions=("hypertension",))
    minor = Firefighter("m", 35, "M", "ba", 5, False, False, 0.9, 5.0, conditions=("eczema",))
    a_htn = combined_risk(htn, SCN, 20, "ba", "ba").drivers["acute"]
    a_minor = combined_risk(minor, SCN, 20, "ba", "ba").drivers["acute"]
    assert a_htn > a_minor  # a flat per-count penalty would score these equally

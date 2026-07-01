import pytest

from sctwin.deploy.roster import Firefighter, sample_roster


def test_invalid_heat_tolerance_band_is_rejected():
    # a bridged HealthProfile field that isn't low/avg/high fails at construction, not in acute_risk
    with pytest.raises(ValueError, match="heat_tolerance"):
        Firefighter("x", 30, "M", "ba", 5, False, False, 0.9, 5.0, heat_tolerance="moderate")


def test_sample_roster_is_varied():
    roster = sample_roster()
    assert len(roster) >= 5
    assert all(isinstance(f, Firefighter) for f in roster)
    # genuinely varied health profiles, not clones
    assert len({f.age for f in roster}) >= 4
    assert any(f.cardiovascular for f in roster) and any(not f.cardiovascular for f in roster)
    assert len({round(f.career_dose, 1) for f in roster}) >= 4


def test_sample_roster_carries_clinical_detail():
    roster = sample_roster()
    assert {f.heat_tolerance for f in roster} >= {"low", "high"}  # varied heat tolerance
    assert any(f.conditions for f in roster) and any(not f.conditions for f in roster)

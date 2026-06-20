from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.optimise import Constraints, recommend
from sctwin.deploy.roster import Firefighter, sample_roster

SCN = FireScenario("c", "dwelling", 4.0, 80.0, 32.0, 8.0, 70.0, 120.0)


def test_meets_capacity_and_is_not_degenerate():
    plan = recommend(SCN, sample_roster(), Constraints(required_capacity=2.0))
    assert plan.feasible
    on_task = [a for a in plan.assignments if a.role == "ba"]
    assert len(on_task) >= 2  # K>0 -> crew deployed, not "send nobody"


def test_shields_the_frailest_member():
    # one clearly-frail member among fit ones; capacity needs only 2 of 4 -> frail should be spared
    roster = [
        Firefighter("fit1", 26, "M", "ba", 4, False, False, 0.95, 5.0),
        Firefighter("fit2", 30, "F", "ba", 7, False, False, 0.92, 10.0),
        Firefighter("fit3", 33, "M", "ba", 9, False, False, 0.90, 12.0),
        Firefighter("frail", 58, "M", "ba", 33, True, True, 0.45, 90.0),
    ]
    plan = recommend(SCN, roster, Constraints(required_capacity=2.0))
    frail = next(a for a in plan.assignments if a.firefighter_id == "frail")
    assert frail.role == "staging"  # the highest-risk member is held in reserve


def test_spreads_career_dose_between_equal_members():
    # identical except career dose; only 1 BA needed -> the lower-dose member goes in
    low = Firefighter("low", 30, "M", "ba", 8, False, False, 0.9, 5.0)
    high = Firefighter("high", 30, "M", "ba", 8, False, False, 0.9, 95.0)
    plan = recommend(SCN, [low, high], Constraints(required_capacity=1.0))
    ba = next(a for a in plan.assignments if a.role == "ba")
    assert ba.firefighter_id == "low"

from sctwin.deploy import deploy, explain
from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.optimise import Constraints
from sctwin.deploy.roster import sample_roster

SCN = FireScenario("c", "grass", 4.0, 60.0, 34.0, 9.0, 70.0, 120.0)


def test_deploy_and_explain_round_trip():
    roster = sample_roster()
    plan = deploy(SCN, roster, Constraints(required_capacity=2.0))
    text = explain(plan, roster)
    assert plan.feasible
    # every roster member appears by id in the human-readable plan
    assert all(f.id in text for f in roster)
    assert "rotate" in text.lower()

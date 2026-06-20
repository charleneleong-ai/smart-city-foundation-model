from sctwin.deploy.optimise import Constraints, Plan, recommend
from sctwin.deploy.risk import RiskWeights
from sctwin.deploy.roster import Roster
from sctwin.deploy.hazard import FireScenario


def deploy(
    scenario: FireScenario, roster: Roster, constraints: Constraints, weights: RiskWeights = RiskWeights()
) -> Plan:
    """Stable public entry point for the deployment engine."""
    return recommend(scenario, roster, constraints, weights)


def explain(plan: Plan, roster: Roster) -> str:
    """One line per firefighter: assignment + rotation + risk band, ordered by descending risk."""
    by_id = {f.id: f for f in roster}
    rows = sorted(plan.assignments, key=lambda a: plan.per_ff_risk[a.firefighter_id].value, reverse=True)
    lines = [f"Plan (feasible={plan.feasible}, total_risk={plan.total_risk:.2f}):"]
    for a in rows:
        ff = by_id[a.firefighter_id]
        s = plan.per_ff_risk[a.firefighter_id]
        flags = ",".join(k for k, v in {"CV": ff.cardiovascular, "resp": ff.respiratory}.items() if v) or "-"
        lines.append(
            f"  {a.firefighter_id} (age {ff.age}, {flags}, career {ff.career_dose:.0f}): "
            f"{a.role.upper()} ppe={a.ppe} rotate@{a.time_on_scene_min:.0f}min "
            f"risk={s.value:.2f} [{s.low:.2f},{s.high:.2f}]"
        )
    return "\n".join(lines)

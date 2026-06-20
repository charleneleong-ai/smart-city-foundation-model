from dataclasses import dataclass, field

from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.risk import RiskScore, RiskWeights, combined_risk
from sctwin.deploy.roster import Roster

# on-task role -> suppression capacity contributed (staging/command contribute none)
ROLE_CAPACITY: dict[str, float] = {"ba": 1.0, "pump": 1.0, "aerial": 1.5, "command": 0.0, "staging": 0.0}


@dataclass(frozen=True)
class Constraints:
    required_capacity: float  # K(S) — effective on-task units the fire needs
    rotation_grid: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0)
    ppe_levels: tuple[str, ...] = ("ba", "standard")


@dataclass(frozen=True)
class Assignment:
    firefighter_id: str
    role: str
    ppe: str
    time_on_scene_min: float


@dataclass(frozen=True)
class Plan:
    assignments: list[Assignment]
    total_risk: float
    max_individual_risk: float
    per_ff_risk: dict[str, RiskScore] = field(default_factory=dict)
    feasible: bool = True


def _plan_for(
    scenario: FireScenario, roster: Roster, c: Constraints, rotation_t: float, ppe: str, weights: RiskWeights
) -> Plan:
    # rank by risk of taking a BA entry slot at this (rotation_t, ppe): frail + high-career sort last
    ranked = sorted(roster, key=lambda f: combined_risk(f, scenario, rotation_t, "ba", ppe, weights).value)
    assignments: list[Assignment] = []
    per_ff: dict[str, RiskScore] = {}
    capacity = 0.0
    for ff in ranked:
        if capacity < c.required_capacity:
            role, posture = "ba", ppe
            capacity += ROLE_CAPACITY["ba"]
        else:
            role, posture = "staging", "staging"
        score = combined_risk(ff, scenario, rotation_t, role, posture, weights)
        assignments.append(Assignment(ff.id, role, posture, rotation_t))
        per_ff[ff.id] = score
    total = sum(s.value for s in per_ff.values())
    worst = max((s.value for s in per_ff.values()), default=0.0)
    return Plan(assignments, total, worst, per_ff, feasible=capacity >= c.required_capacity)


def recommend(
    scenario: FireScenario, roster: Roster, constraints: Constraints, weights: RiskWeights = RiskWeights()
) -> Plan:
    """Lowest-crew-health-risk feasible deployment over the rotation x PPE grid (greedy assignment)."""
    candidates = [
        _plan_for(scenario, roster, constraints, t, ppe, weights)
        for t in constraints.rotation_grid
        for ppe in constraints.ppe_levels
    ]
    feasible = [p for p in candidates if p.feasible]
    pool = feasible or candidates  # if none feasible (roster too small), still return the best effort
    return min(pool, key=lambda p: (p.total_risk, p.max_individual_risk))

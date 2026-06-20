# Firefighter Deployment Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a personalised exposure→health deployment optimiser that, given a fire scenario and a roster of distinct firefighters, recommends the lowest-crew-health-risk deployment plan that still meets the fire's suppression requirement.

**Architecture:** New `src/sctwin/deploy/` subpackage. Pure functions for the physics (`exposure.py`) and the transferred-prior risk index (`risk.py`); a constrained greedy/grid optimiser (`optimise.py`) over rotation × PPE × crew-assignment; a thin orchestrator (`engine.py`) and a runnable demo. A firefighter deployment is the analogue of the existing `reason/intervention.py` `Intervention` — scoring a deployment mirrors `effect()`.

**Tech Stack:** Python 3.12, dataclasses (frozen), polars (only at the live-hook boundary), pytest + respx (HTTP mocking), ruff. Reuses `sctwin.adapters.open_meteo` (`OpenMeteoForecastAdapter`, `WEATHER_VARS`) and `sctwin.geo`.

## Global Constraints

- Spec: `docs/specs/2026-06-20-firefighter-deployment-engine-design.md`. Every task implicitly serves it.
- **The honest seam:** exposure physics is data-grounded; the dose→risk coupling is a **transferred literature prior**. Risk scores ship an uncertainty band that represents *prior* uncertainty — it is NOT conformal (no calibration data exists). Never present output as a calibrated personal prognosis.
- **Anti-degeneracy:** the optimiser minimises risk **subject to** a suppression-capacity constraint `K(S)`. A feasible plan with `K>0` and a non-empty roster MUST place crew on-task.
- Pure functions stay pure (no I/O) except `FireScenario.from_live`. Frozen dataclasses with `__post_init__` validation. TDD: failing test → minimal impl → green → commit. Run `uv run pytest` and `uv run ruff check` per task.
- Determinism: no `Math.random`/wall-clock in core logic; the demo and all tests use fixed scenarios.

---

## File Structure

- `src/sctwin/deploy/__init__.py` — public exports.
- `src/sctwin/deploy/hazard.py` — `FireScenario`, `FIRE_TOXICITY`; `from_live` added in Task 7.
- `src/sctwin/deploy/roster.py` — `Firefighter`, `Roster`, `sample_roster()`.
- `src/sctwin/deploy/exposure.py` — `toxicant_dose()`, `heat_load()`, `PPE_ATTENUATION`, `ROLE_EXERTION`.
- `src/sctwin/deploy/risk.py` — `RiskWeights`, `RiskScore`, `acute_risk()`, `incident_dose_risk()`, `career_risk()`, `combined_risk()`.
- `src/sctwin/deploy/optimise.py` — `Constraints`, `Assignment`, `Plan`, `ROLE_CAPACITY`, `recommend()`.
- `src/sctwin/deploy/engine.py` — `deploy()` public entry + `explain()` summary.
- `apps/deploy_demo.py` — runnable demo (fixed LA grassfire scenario).
- Tests mirror under `tests/test_deploy_<module>.py`.

---

### Task 1: `hazard.py` — FireScenario

**Files:**
- Create: `src/sctwin/deploy/__init__.py`, `src/sctwin/deploy/hazard.py`
- Test: `tests/test_deploy_hazard.py`

**Interfaces:**
- Produces: `FIRE_TOXICITY: dict[str, float]`; `FireScenario(cell, fire_type, size, pm25, temp_c, wind_speed, wind_dir, duration_min)` frozen dataclass with `.toxicity() -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_hazard.py
import pytest
from sctwin.deploy.hazard import FIRE_TOXICITY, FireScenario


def _scn(**kw):
    base = dict(cell="8a1fb46622dffff", fire_type="grass", size=4.0, pm25=50.0,
               temp_c=30.0, wind_speed=8.0, wind_dir=70.0, duration_min=120.0)
    return FireScenario(**{**base, **kw})


def test_toxicity_looks_up_fire_type():
    assert _scn(fire_type="grass").toxicity() == FIRE_TOXICITY["grass"]
    assert _scn(fire_type="ev_lithium").toxicity() > _scn(fire_type="grass").toxicity()


@pytest.mark.parametrize("bad", [{"fire_type": "volcano"}, {"wind_dir": 360.0}, {"wind_dir": -1.0}])
def test_rejects_invalid_fields(bad):
    with pytest.raises(ValueError):
        _scn(**bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_hazard.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.deploy'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sctwin/deploy/__init__.py
```
(empty file)

```python
# src/sctwin/deploy/hazard.py
from dataclasses import dataclass

# fire type -> relative combustion-product toxicity of its smoke (dimensionless multiplier)
FIRE_TOXICITY: dict[str, float] = {
    "grass": 0.6,
    "dwelling": 1.0,
    "commercial": 1.3,
    "chemical": 2.0,
    "ev_lithium": 2.5,
}


@dataclass(frozen=True)
class FireScenario:
    """The fireground hazard state a deployment is scored against."""

    cell: str  # H3 cell of the incident
    fire_type: str  # key into FIRE_TOXICITY
    size: float  # fire size in suppression-demand units (drives required capacity K)
    pm25: float  # ambient smoke / PM2.5 at scene (ug/m3)
    temp_c: float  # ambient air temperature
    wind_speed: float  # 10 m wind speed (m/s)
    wind_dir: float  # 10 m wind direction, degrees [0, 360)
    duration_min: float  # expected incident duration

    def __post_init__(self) -> None:
        if self.fire_type not in FIRE_TOXICITY:
            raise ValueError(f"fire_type must be one of {tuple(FIRE_TOXICITY)}, got {self.fire_type!r}")
        if not 0.0 <= self.wind_dir < 360.0:
            raise ValueError(f"wind_dir must be in [0, 360), got {self.wind_dir}")

    def toxicity(self) -> float:
        return FIRE_TOXICITY[self.fire_type]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deploy_hazard.py -q` → Expected: PASS (2 tests, 3 cases).

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/deploy/__init__.py src/sctwin/deploy/hazard.py tests/test_deploy_hazard.py
git commit -m "feat(deploy): FireScenario hazard model"
```

---

### Task 2: `roster.py` — Firefighter + varied sample roster

**Files:**
- Create: `src/sctwin/deploy/roster.py`
- Test: `tests/test_deploy_roster.py`

**Interfaces:**
- Produces: `Firefighter(id, age, sex, role, years_service, cardiovascular, respiratory, fitness, career_dose)` frozen dataclass; `Roster = list[Firefighter]`; `sample_roster() -> Roster` (>=5 distinct profiles).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_roster.py
from sctwin.deploy.roster import Firefighter, sample_roster


def test_sample_roster_is_varied():
    roster = sample_roster()
    assert len(roster) >= 5
    assert all(isinstance(f, Firefighter) for f in roster)
    # genuinely varied health profiles, not clones
    assert len({f.age for f in roster}) >= 4
    assert any(f.cardiovascular for f in roster) and any(not f.cardiovascular for f in roster)
    assert len({round(f.career_dose, 1) for f in roster}) >= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_roster.py -q` → Expected: FAIL — `ImportError` (no `roster`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/sctwin/deploy/roster.py
from dataclasses import dataclass

ROLES = ("ba", "pump", "aerial", "command", "staging")  # ba = breathing-apparatus entry


@dataclass(frozen=True)
class Firefighter:
    """One firefighter's varied health profile — the unit the optimiser personalises around."""

    id: str
    age: int
    sex: str  # "M" / "F" / "X"
    role: str  # usual role; the optimiser may reassign
    years_service: int
    cardiovascular: bool  # CV comorbidity flag
    respiratory: bool  # respiratory comorbidity flag
    fitness: float  # 0..1 (1 = peak)
    career_dose: float  # cumulative smoke-dose units banked to date


Roster = list[Firefighter]


def sample_roster() -> Roster:
    """A deliberately varied demo watch — young/fit, veteran/CV, mid-career high-career-dose, etc."""
    return [
        Firefighter("FF-01", 27, "M", "ba", 4, False, False, 0.95, 8.0),
        Firefighter("FF-02", 34, "F", "ba", 9, False, False, 0.90, 22.0),
        Firefighter("FF-03", 45, "M", "pump", 20, False, True, 0.70, 51.0),
        Firefighter("FF-04", 52, "M", "ba", 27, True, False, 0.55, 73.0),
        Firefighter("FF-05", 39, "F", "aerial", 14, False, False, 0.80, 34.0),
        Firefighter("FF-06", 58, "M", "command", 33, True, True, 0.45, 88.0),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deploy_roster.py -q` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/deploy/roster.py tests/test_deploy_roster.py
git commit -m "feat(deploy): Firefighter profile + varied sample roster"
```

---

### Task 3: `exposure.py` — toxicant dose + heat load (pure physics)

**Files:**
- Create: `src/sctwin/deploy/exposure.py`
- Test: `tests/test_deploy_exposure.py`

**Interfaces:**
- Consumes: `FireScenario` (Task 1).
- Produces: `PPE_ATTENUATION: dict[str,float]`, `ROLE_EXERTION: dict[str,float]`; `toxicant_dose(scenario, time_on_scene_min, ppe) -> float`; `heat_load(scenario, time_on_scene_min, role, ppe) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_exposure.py
from sctwin.deploy.exposure import heat_load, toxicant_dose
from sctwin.deploy.hazard import FireScenario

SCN = FireScenario("c", "grass", 4.0, pm25=50.0, temp_c=30.0, wind_speed=8.0, wind_dir=70.0, duration_min=120.0)


def test_toxicant_dose_monotone_in_time_and_ppe():
    assert toxicant_dose(SCN, 30, "ba") > toxicant_dose(SCN, 20, "ba")  # more time -> more dose
    assert toxicant_dose(SCN, 20, "ba") < toxicant_dose(SCN, 20, "standard")  # BA cuts inhaled dose
    assert toxicant_dose(SCN, 20, "staging") < toxicant_dose(SCN, 20, "ba")  # reserve = least


def test_heat_load_rises_with_temp_and_exertion():
    hot = FireScenario("c", "grass", 4.0, 50.0, temp_c=38.0, wind_speed=8.0, wind_dir=70.0, duration_min=120.0)
    assert heat_load(hot, 20, "ba", "ba") > heat_load(SCN, 20, "ba", "ba")  # hotter -> more load
    assert heat_load(SCN, 20, "ba", "ba") > heat_load(SCN, 20, "command", "command")  # exertion
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_exposure.py -q` → Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/sctwin/deploy/exposure.py
from sctwin.deploy.hazard import FireScenario

# protective posture -> fraction of ambient toxicant actually taken on (lower = better protected)
PPE_ATTENUATION: dict[str, float] = {
    "ba": 0.15,  # breathing apparatus — large cut to inhaled toxicant
    "standard": 0.60,  # turnout gear, no BA
    "command": 0.30,  # mostly upwind / outside the smoke
    "staging": 0.05,  # held in reserve
}

# role -> physical exertion multiplier (drives heat load)
ROLE_EXERTION: dict[str, float] = {
    "ba": 1.5,
    "pump": 0.8,
    "aerial": 1.0,
    "command": 0.4,
    "staging": 0.2,
}

_PM25_REF = 50.0  # moderate-AQ reference for normalising smoke
_HEAT_BASE_C = 15.0  # thermal-neutral baseline


def toxicant_dose(scenario: FireScenario, time_on_scene_min: float, ppe: str) -> float:
    """Smoke/carcinogen dose over `time_on_scene_min` at protective posture `ppe`.
    Monotone up in time, fire toxicity, and ambient smoke; down in PPE protection."""
    smoke_factor = scenario.pm25 / _PM25_REF
    return time_on_scene_min * scenario.toxicity() * smoke_factor * PPE_ATTENUATION[ppe]


def heat_load(scenario: FireScenario, time_on_scene_min: float, role: str, ppe: str) -> float:
    """Thermal burden over time. Rises with ambient heat above baseline and role exertion;
    BA does not relieve heat (turnout retains it), so `ppe` enters only via the staging exemption."""
    over = max(scenario.temp_c - _HEAT_BASE_C, 0.0) / 10.0
    return time_on_scene_min * over * ROLE_EXERTION[role]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deploy_exposure.py -q` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/deploy/exposure.py tests/test_deploy_exposure.py
git commit -m "feat(deploy): exposure dose + heat load physics"
```

---

### Task 4: `risk.py` — combined personalised risk index (the core)

**Files:**
- Create: `src/sctwin/deploy/risk.py`
- Test: `tests/test_deploy_risk.py`

**Interfaces:**
- Consumes: `Firefighter` (T2), `FireScenario` (T1), `toxicant_dose`/`heat_load` (T3).
- Produces: `RiskWeights(acute=1.0, incident=1.0, career=1.0)`; `RiskScore(value, low, high, drivers)`; `combined_risk(ff, scenario, time_on_scene_min, role, ppe, weights=RiskWeights(), prior_band=0.3) -> RiskScore`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_risk.py
from sctwin.deploy.risk import RiskWeights, combined_risk
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_risk.py -q` → Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/sctwin/deploy/risk.py
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


def acute_risk(ff: Firefighter, hl: float) -> float:
    """Heat + cardiac risk for thermal load `hl`, amplified by age, CV, low fitness, respiratory."""
    age_factor = 1.0 + max(ff.age - 40, 0) * 0.02
    cv_factor = 1.5 if ff.cardiovascular else 1.0
    resp_factor = 1.2 if ff.respiratory else 1.0
    fitness_factor = 1.0 + (1.0 - ff.fitness)  # unfit -> up to ~2x
    return hl * age_factor * cv_factor * resp_factor * fitness_factor * 0.01


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
    hl = heat_load(scenario, time_on_scene_min, role, ppe)
    drivers = {"acute": acute_risk(ff, hl), "incident": incident_dose_risk(td, ff), "career": career_risk(ff, td)}
    value = weights.acute * drivers["acute"] + weights.incident * drivers["incident"] + weights.career * drivers["career"]
    return RiskScore(value, value * (1.0 - prior_band), value * (1.0 + prior_band), drivers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deploy_risk.py -q` → Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/deploy/risk.py tests/test_deploy_risk.py
git commit -m "feat(deploy): combined personalised risk index"
```

---

### Task 5: `optimise.py` — constrained greedy/grid optimiser

**Files:**
- Create: `src/sctwin/deploy/optimise.py`
- Test: `tests/test_deploy_optimise.py`

**Interfaces:**
- Consumes: `Roster`/`Firefighter` (T2), `FireScenario` (T1), `combined_risk`/`RiskScore`/`RiskWeights` (T4).
- Produces:
  - `ROLE_CAPACITY: dict[str,float]`
  - `Constraints(required_capacity, rotation_grid=(10.0,15.0,20.0,25.0,30.0), ppe_levels=("ba","standard"))`
  - `Assignment(firefighter_id, role, ppe, time_on_scene_min)`
  - `Plan(assignments, total_risk, max_individual_risk, per_ff_risk, feasible)`
  - `recommend(scenario, roster, constraints, weights=RiskWeights()) -> Plan`

**Algorithm (greedy/grid):** for each `(rotation_t, ppe)` in `rotation_grid × ppe_levels`: rank the roster by `combined_risk` for an on-task `ba` slot at that `(rotation_t, ppe)`, ascending; assign the lowest-risk members to `ba` until `sum ROLE_CAPACITY["ba"]` reaches `required_capacity`; the rest go to `staging` (role/ppe `"staging"`, same `rotation_t`). Score the whole plan. Return the feasible plan with the lowest `total_risk`, tie-broken by lowest `max_individual_risk`. Ranking by risk is what shields frail members (they sort last) and spreads career dose (the `career` term lifts high-dose members' risk).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_optimise.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_optimise.py -q` → Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/sctwin/deploy/optimise.py
from dataclasses import dataclass, field

from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.risk import RiskScore, RiskWeights, combined_risk
from sctwin.deploy.roster import Firefighter, Roster

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deploy_optimise.py -q` → Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/deploy/optimise.py tests/test_deploy_optimise.py
git commit -m "feat(deploy): constrained greedy/grid deployment optimiser"
```

---

### Task 6: `engine.py` + `apps/deploy_demo.py` — orchestrator + demo

**Files:**
- Create: `src/sctwin/deploy/engine.py`, `apps/deploy_demo.py`
- Modify: `src/sctwin/deploy/__init__.py` (export public API)
- Test: `tests/test_deploy_engine.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `deploy(scenario, roster, constraints, weights=RiskWeights()) -> Plan` (alias to `recommend`, the stable public entry); `explain(plan, roster) -> str` (human-readable, names each member's role + rotation + risk band).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_engine.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_engine.py -q` → Expected: FAIL — `ImportError` (no `deploy`/`explain`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/sctwin/deploy/engine.py
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
```

```python
# src/sctwin/deploy/__init__.py
from sctwin.deploy.engine import deploy, explain
from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.optimise import Assignment, Constraints, Plan, recommend
from sctwin.deploy.risk import RiskScore, RiskWeights, combined_risk
from sctwin.deploy.roster import Firefighter, Roster, sample_roster

__all__ = [
    "deploy",
    "explain",
    "FireScenario",
    "Assignment",
    "Constraints",
    "Plan",
    "recommend",
    "RiskScore",
    "RiskWeights",
    "combined_risk",
    "Firefighter",
    "Roster",
    "sample_roster",
]
```

```python
# apps/deploy_demo.py
"""Demo: personalised deployment for a hot, smoky grassfire over the sample watch."""
from sctwin.deploy import FireScenario, Constraints, deploy, explain, sample_roster


def main() -> None:
    scenario = FireScenario(
        cell="8a1fb46622dffff", fire_type="grass", size=4.0, pm25=120.0,
        temp_c=36.0, wind_speed=11.0, wind_dir=70.0, duration_min=180.0,
    )
    roster = sample_roster()
    plan = deploy(scenario, roster, Constraints(required_capacity=3.0))
    print(explain(plan, roster))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + demo to verify**

Run: `uv run pytest tests/test_deploy_engine.py -q` → Expected: PASS.
Run: `uv run python apps/deploy_demo.py` → Expected: prints a plan, frailest members (FF-04, FF-06) in `STAGING`, fit/low-dose members in `BA`.

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/deploy/engine.py src/sctwin/deploy/__init__.py apps/deploy_demo.py tests/test_deploy_engine.py
git commit -m "feat(deploy): engine orchestrator, public API, and demo"
```

---

### Task 7: `FireScenario.from_live` — realtime hazard hook

**Files:**
- Modify: `src/sctwin/deploy/hazard.py` (add `from_live` classmethod + helper)
- Test: `tests/test_deploy_live.py`

**Interfaces:**
- Consumes: `OpenMeteoForecastAdapter`, `WEATHER_VARS` from `sctwin.adapters.open_meteo`; `cell_of`/`Cell` from `sctwin.geo`.
- Produces: `FireScenario.from_live(cell_h3, res, *, fire_type, size, duration_min, pm25, when, adapter=None) -> FireScenario` — pulls live wind/temp from the forecast adapter for the cell; `pm25` is passed in (CAMS/FRP smoke wiring is a follow-up stub).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_live.py
from datetime import datetime

import httpx
import respx

from sctwin.adapters.open_meteo import OpenMeteoForecastAdapter, WEATHER_VARS
from sctwin.deploy.hazard import FireScenario
from sctwin.geo import cell_of

FORECAST = "https://api.open-meteo.com/v1/forecast"


@respx.mock
def test_from_live_pulls_wind_and_temp_from_adapter():
    respx.get(FORECAST).mock(
        return_value=httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2026-06-20T12:00"],
                    "temperature_2m": [36.0],
                    "wind_speed_10m": [11.0],
                    "wind_direction_10m": [70.0],
                    "precipitation": [0.0],
                    "relative_humidity_2m": [18.0],
                }
            },
        )
    )
    cell = cell_of(34.05, -118.24, res=7)
    scn = FireScenario.from_live(
        cell.h3, res=7, fire_type="grass", size=4.0, duration_min=180.0, pm25=120.0,
        when=datetime(2026, 6, 20), adapter=OpenMeteoForecastAdapter(variables=WEATHER_VARS),
    )
    assert scn.temp_c == 36.0 and scn.wind_speed == 11.0 and scn.wind_dir == 70.0
    assert scn.cell == cell.h3 and scn.pm25 == 120.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_live.py -q` → Expected: FAIL — `AttributeError: type object 'FireScenario' has no attribute 'from_live'`.

- [ ] **Step 3: Write minimal implementation** (append to `src/sctwin/deploy/hazard.py`)

```python
# --- add these imports at the top of hazard.py ---
from datetime import datetime

# --- add as a classmethod inside FireScenario ---
    @classmethod
    def from_live(
        cls,
        cell_h3: str,
        res: int,
        *,
        fire_type: str,
        size: float,
        duration_min: float,
        pm25: float,
        when: datetime,
        adapter,
    ) -> "FireScenario":
        """Build a scenario from live NWP. `adapter` is an OpenMeteoForecastAdapter configured with
        WEATHER_VARS; `pm25` (smoke) is supplied for now — CAMS/FRP wiring is a follow-up."""
        from sctwin.geo import Cell

        df = adapter.fetch([Cell(h3=cell_h3, res=res)], when, when)
        latest = df.sort("time").group_by("layer").last()
        vals = dict(zip(latest["layer"].to_list(), latest["value"].to_list()))
        return cls(
            cell=cell_h3,
            fire_type=fire_type,
            size=size,
            pm25=pm25,
            temp_c=vals["t2m"],
            wind_speed=vals["wind_speed"],
            wind_dir=vals["wind_dir"],
            duration_min=duration_min,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deploy_live.py -q` → Expected: PASS.

- [ ] **Step 5: Full suite + lint, then commit**

```bash
uv run pytest -q && uv run ruff check src/sctwin/deploy apps/deploy_demo.py tests/test_deploy_*.py
git add src/sctwin/deploy/hazard.py tests/test_deploy_live.py
git commit -m "feat(deploy): realtime FireScenario.from_live via Open-Meteo adapter"
```

---

## Self-Review

**Spec coverage:** spine (personalised exposure→health scorer) → T3+T4; multi-lever optimiser Plan → T5; combined index (acute+incident+career) → T4; Approach A greedy/grid → T5; suppression-capacity constraint → T5 (`Constraints.required_capacity`, `ROLE_CAPACITY`); varied roster → T2; honest seam / prior band → T4 (`prior_band`, not conformal); realtime hook → T7; demo → T6; out-of-scope (spread, RL, logistics) correctly absent. No gaps.

**Type consistency:** `FireScenario` fields and `.toxicity()` consistent T1→T3/T4/T7; `combined_risk(ff, scenario, time_on_scene_min, role, ppe, weights, prior_band)` identical T4→T5; `RiskScore(value, low, high, drivers)` identical T4→T5/T6; `Plan(assignments, total_risk, max_individual_risk, per_ff_risk, feasible)` and `Assignment(firefighter_id, role, ppe, time_on_scene_min)` identical T5→T6; `deploy`/`explain` T6 match `__init__` exports. `from_live` consumes the real `WEATHER_VARS` layer names (`t2m`/`wind_speed`/`wind_dir`) emitted by the committed adapter.

**Placeholders:** none — every code step is complete and runnable.

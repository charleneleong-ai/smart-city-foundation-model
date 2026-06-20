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
    heat_tolerance: str = "avg"  # "low" / "avg" / "high" — heat-susceptibility band
    conditions: tuple[str, ...] = ()  # fuller clinical ledger beyond the cv/resp flags


Roster = list[Firefighter]


def sample_roster() -> Roster:
    """A deliberately varied demo watch — young/fit, veteran/CV, mid-career high-career-dose, etc."""
    return [
        Firefighter("FF-01", 27, "M", "ba", 4, False, False, 0.95, 8.0, "high", ()),
        Firefighter("FF-02", 34, "F", "ba", 9, False, False, 0.90, 22.0, "high", ()),
        Firefighter("FF-03", 45, "M", "pump", 20, False, True, 0.70, 51.0, "avg", ("mild asthma",)),
        Firefighter("FF-04", 52, "M", "ba", 27, True, False, 0.55, 73.0, "low", ("hypertension",)),
        Firefighter("FF-05", 39, "F", "aerial", 14, False, False, 0.80, 34.0, "avg", ()),
        Firefighter("FF-06", 58, "M", "command", 33, True, True, 0.45, 88.0, "low", ("hypertension", "prior MI")),
    ]

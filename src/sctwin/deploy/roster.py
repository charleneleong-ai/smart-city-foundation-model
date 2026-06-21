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


_BASE_WATCH = [
    Firefighter("FF-01", 27, "M", "ba", 4, False, False, 0.95, 8.0, "high", ()),
    Firefighter("FF-02", 34, "F", "ba", 9, False, False, 0.90, 22.0, "high", ()),
    Firefighter("FF-03", 45, "M", "pump", 20, False, True, 0.70, 51.0, "avg", ("mild asthma",)),
    Firefighter("FF-04", 52, "M", "ba", 27, True, False, 0.55, 73.0, "low", ("hypertension",)),
    Firefighter("FF-05", 39, "F", "aerial", 14, False, False, 0.80, 34.0, "avg", ()),
    Firefighter("FF-06", 58, "M", "command", 33, True, True, 0.45, 88.0, "low", ("hypertension", "prior MI")),
]
_GEN_ROLES = ("ba", "ba", "ba", "ba", "pump", "aerial", "command", "staging")
_GEN_CONDITIONS = ((), (), ("mild asthma",), ("hypertension",), (), ("eczema",))


def sample_roster(n: int = 6) -> Roster:
    """A deliberately varied demo watch — young/fit, veteran/CV, mid-career high-career-dose, etc.
    `n` scales it toward a realistic sector-level deployment (the base 6 is one fire company; a fire
    this size is worked by many crews). Extras beyond the base 6 are generated deterministically."""
    if n <= len(_BASE_WATCH):
        return _BASE_WATCH[:n]
    out = list(_BASE_WATCH)
    for i in range(len(_BASE_WATCH), n):
        out.append(Firefighter(
            f"FF-{i + 1:02d}", 24 + (i * 7) % 36, "MFX"[i % 3], _GEN_ROLES[i % len(_GEN_ROLES)],
            (i * 3) % 30, i % 4 == 0, i % 5 == 0, round(0.45 + ((i * 13) % 11) / 20, 2),
            float((i * 11) % 90), ("high", "avg", "low")[i % 3], _GEN_CONDITIONS[i % len(_GEN_CONDITIONS)],
        ))
    return out

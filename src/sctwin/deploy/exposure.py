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


def heat_load(scenario: FireScenario, time_on_scene_min: float, role: str) -> float:
    """Thermal burden over time — rises with ambient heat above baseline and role exertion."""
    over = max(scenario.temp_c - _HEAT_BASE_C, 0.0) / 10.0
    return time_on_scene_min * over * ROLE_EXERTION[role]

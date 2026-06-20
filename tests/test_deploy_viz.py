from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.optimise import Constraints, recommend
from sctwin.deploy.roster import sample_roster
from sctwin.deploy.viz import crew_records, deploy_map, downwind_alignment, hazard_surface

SCN = FireScenario("8a1fb46622dffff", "grass", 4.0, 120.0, 36.0, 11.0, 70.0, 180.0)
PRESET = {"name": "Camden", "lat": 51.54, "lon": -0.14, "zoom": 12.5}


def test_downwind_alignment_peaks_downwind_zero_upwind():
    # wind FROM 70° -> smoke travels TOWARD 250°
    assert downwind_alignment(250.0, 70.0) == 1.0  # straight downwind
    assert downwind_alignment(70.0, 70.0) == 0.0  # straight upwind, clamped


def test_hazard_surface_has_three_layers_and_dose_tracks_smoke():
    surf = hazard_surface(SCN, rings=2)
    assert set(surf["layer"].unique().to_list()) == {"smoke", "heat", "dose"}
    smoke = surf.filter(surf["layer"] == "smoke").sort("cell")["value"].to_list()
    dose = surf.filter(surf["layer"] == "dose").sort("cell")["value"].to_list()
    # dose is monotone in smoke (same per-cell ordering)
    assert [s for _, s in sorted(zip(smoke, dose))] == sorted(dose)


def test_crew_records_cover_every_firefighter_and_carry_risk():
    roster = sample_roster()
    plan = recommend(SCN, roster, Constraints(required_capacity=3.0))
    recs = crew_records(plan, roster, SCN)
    assert {r["ff_id"] for r in recs} == {f.id for f in roster}
    assert all("risk" in r and "color" in r and set(r["drivers"]) == {"acute", "incident", "career"} for r in recs)
    # staging crew are displayed pulled back (north of the incident centre)
    staging = [r for r in recs if r["role"] == "staging"]
    assert all(r["lat"] > SCN_lat() for r in staging)


def SCN_lat():
    import h3
    return h3.cell_to_latlng(SCN.cell)[0]


def test_deploy_map_payload_is_render_ready():
    roster = sample_roster()
    plan = recommend(SCN, roster, Constraints(required_capacity=3.0))
    m = deploy_map(SCN, plan, roster, preset=PRESET)
    assert [L["group"] for L in m["layers"]] == ["Fire", "Fire", "Fire"]
    assert {r["ff_id"] for r in m["plan"]} == {f.id for f in roster}
    assert m["lat"] == PRESET["lat"] and "elevation_scale" in m

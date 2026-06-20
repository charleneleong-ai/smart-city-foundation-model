import sys
from datetime import datetime, timezone
from pathlib import Path

import h3
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from demo_fire import build_fire_map  # noqa: E402

PRESET = {"lat": 34.045, "lon": -118.526, "pitch": 50.0}
SEED = h3.latlng_to_cell(34.045, -118.526, 8)  # Palisades ignition
DISK = h3.grid_disk(SEED, 3)


def _frame(at: datetime, **overrides: float) -> pl.DataFrame:
    """Hot, dry, windy field over a 3-ring disk so the CA actually spreads (override per test)."""
    vals = {"t2m": 35.0, "rh": 10.0, "precip": 0.0, "wind_speed": 40.0, "wind_dir": 0.0, **overrides}
    rows = {"cell": [], "time": [], "layer": [], "value": []}
    for c in DISK:
        for layer, v in vals.items():
            rows["cell"].append(c), rows["time"].append(at), rows["layer"].append(layer), rows["value"].append(v)
    return pl.DataFrame(rows)


def test_build_fire_map_emits_dryness_and_arrival_layers_and_spreads():
    at = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    m = build_fire_map("LA", _frame(at), PRESET, zoom=10.0, res=8, seed_cell=SEED, steps=3)

    assert [layer["name"] for layer in m["layers"]] == ["fire spread", "fire arrival", "fuel dryness"]
    dry = m["layers"][2]["frames"][0]["records"]
    burned = {r["cell"] for r in m["layers"][1]["frames"][0]["records"]}
    assert len(dry) == len(DISK)  # dryness over every fetched cell
    assert SEED in burned and len(burned) > 1  # seed ignites and the fire spread downwind


def test_build_fire_map_does_not_spread_when_soaked():
    at = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    m = build_fire_map("LA", _frame(at, precip=20.0), PRESET, zoom=10.0, res=8, seed_cell=SEED, steps=3)
    burned = {r["cell"] for r in m["layers"][1]["frames"][0]["records"]}
    assert burned == {SEED}  # drenched fuel -> only the seed burns, no spread


def test_build_fire_map_overlays_deployment_at_ignition_when_roster_given():
    from sctwin.deploy import Constraints, sample_roster

    at = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    roster = sample_roster()
    m = build_fire_map("LA", _frame(at), PRESET, zoom=10.0, res=8, seed_cell=SEED, steps=3,
                       roster=roster, constraints=Constraints(required_capacity=3.0))
    assert {r["ff_id"] for r in m["plan"]} == {f.id for f in roster}  # every firefighter overlaid
    slat, slon = h3.cell_to_latlng(SEED)  # crew deploy at the ignition point, on top of the fire
    assert any(abs(r["lat"] - slat) < 0.02 and abs(r["lon"] - slon) < 0.02 for r in m["plan"])


def test_crew_advance_with_the_front_while_staging_holds():
    from sctwin.deploy import Constraints, sample_roster

    at = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    roster = sample_roster()
    m = build_fire_map("LA", _frame(at), PRESET, zoom=10.0, res=8, seed_cell=SEED, steps=3,
                       roster=roster, constraints=Constraints(required_capacity=3.0))
    pf = m["plan_frames"]
    assert len(pf) == len(m["layers"][0]["frames"])  # one crew-frame per fire-spread step

    def pos(frame: list[dict], role: str) -> tuple[float, float]:
        r = next(rec for rec in frame if rec["role"] == role)
        return r["lat"], r["lon"]

    assert pos(pf[0], "ba") != pos(pf[-1], "ba")  # on-task crew advance with the front
    assert pos(pf[0], "staging") == pos(pf[-1], "staging")  # staging crew hold at the rear


def test_build_fire_map_terrain_slope_changes_the_front():
    at = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    flat = build_fire_map("LA", _frame(at), PRESET, zoom=10.0, res=8, seed_cell=SEED, steps=6)
    # strong synthetic DEM: north (higher latitude) is much higher -> uphill biases the front north
    elev = {c: h3.cell_to_latlng(c)[0] * 100000 for c in DISK}
    sloped = build_fire_map("LA", _frame(at), PRESET, zoom=10.0, res=8, seed_cell=SEED, steps=6,
                            elevation=elev, slope_coeff=8.0)

    def arrival_steps(m: dict) -> dict:  # fire-arrival layer (index 1): {cell: CA step}
        return {r["cell"]: r["value"] for r in m["layers"][1]["frames"][0]["records"]}

    assert arrival_steps(flat) != arrival_steps(sloped)  # terrain changes arrival timing/extent


def test_build_fire_map_masks_ocean_cells():
    at = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    water = set(list(DISK)[1::2])  # mark every other cell as sea level
    water.discard(SEED)  # keep the ignition on land
    elev = {c: (0.0 if c in water else 100.0) for c in DISK}
    m = build_fire_map("LA", _frame(at), PRESET, zoom=10.0, res=8, seed_cell=SEED, steps=3, elevation=elev)

    rendered = {r["cell"] for layer in m["layers"] for fr in layer["frames"] for r in fr["records"]}
    assert rendered  # land cells remain in the model
    assert not (rendered & water)  # no ocean cell appears in any layer


def test_spread_frames_animate_growing_front_and_scar():
    from demo_fire import _spread_frames

    at = datetime(2025, 1, 7, tzinfo=timezone.utc)
    arrival = {"a": 0, "b": 1, "c": 2}  # ignite at steps 0, 1, 2
    cells = ["a", "b", "c", "d"]  # "d" never burns
    frames = _spread_frames(cells, arrival, n_steps=2, at=at)

    assert [f["label"] for f in frames] == ["step 0/2", "step 1/2", "step 2/2"]
    # burned area is monotone non-decreasing and strictly grows end-to-end
    burned = [sum(1 for r in f["records"] if r["value"] > 0.0) for f in frames]
    assert burned == sorted(burned) and burned[-1] > burned[0]

    by_step = [{r["cell"]: r["value"] for r in f["records"]} for f in frames]
    assert by_step[0]["a"] == 1.0  # active front at its ignition step
    assert by_step[1]["a"] == 0.35  # cooling scar one step later
    assert all(step["d"] == 0.0 for step in by_step)  # never-burned cell stays unlit

import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from viz_fire_3d import build_backtest_map, overlay_crew  # noqa: E402

from sctwin.geo import cell_of  # noqa: E402


def test_build_backtest_map_animates_steps_and_classifies_cells():
    cells = [cell_of(34.05 + 0.01 * i, -118.5, 8) for i in range(4)]
    hit, overreach, realburn, ctx = (c.h3 for c in cells)
    observed = {hit, realburn}
    arrival = {hit: 1, overreach: 2}  # maxstep 2 -> 3 frames (steps 0..2)
    m = build_backtest_map("t", cells, observed, arrival, {"wind_from": 20.0, "wind_speed": 30.0}, {"iou": 0.5, "recall": 0.5}, res=8)

    layer = m["layers"][0]
    assert len(layer["frames"]) == 3
    assert all(len(f["records"]) == 4 for f in layer["frames"])  # full cell universe every frame
    last = {r["cell"]: r["color"] for r in layer["frames"][-1]["records"]}
    assert last[hit][0] > 200 and last[hit][2] < 60  # model hit -> hot (yellow/red)
    assert last[overreach][0] > 180 and last[overreach][2] > 180  # over-reach -> magenta
    assert last[realburn][2] > 140 and last[realburn][0] < 60  # missed real burn -> blue
    assert last[ctx][3] < 60  # unburned context -> near-transparent


def test_build_backtest_map_carries_a_categorical_legend():
    m = build_backtest_map("t", [cell_of(34.05, -118.5, 8)], set(), {}, {"wind_from": 0.0, "wind_speed": 0.0}, {"iou": 0.0, "recall": 0.0}, res=8)
    labels = [e["label"] for e in m["legend"]]
    assert any("hit" in label for label in labels) and any("over-reach" in label for label in labels)
    assert all(len(e["color"]) == 3 for e in m["legend"])  # rgb swatches


def test_front_grows_monotonically_over_steps():
    cells = [cell_of(34.05 + 0.01 * i, -118.5, 8) for i in range(3)]
    a, b, _ = (c.h3 for c in cells)
    m = build_backtest_map("t", cells, {a, b}, {a: 1, b: 3}, {"wind_from": 0.0, "wind_speed": 10.0}, {"iou": 1.0, "recall": 1.0}, res=8)
    frames = m["layers"][0]["frames"]
    lit = [sum(1 for r in f["records"] if r["value"] > 0) for f in frames]  # cells ignited by step t
    assert lit == sorted(lit) and lit[0] == 0 and lit[-1] == 2  # monotonic; both ignited by the last step


def test_overlay_crew_advances_with_the_model_front_over_the_ground_truth():
    at = datetime(2025, 1, 7, 18, tzinfo=timezone.utc)
    cells = [cell_of(34.07 + 0.01 * i, -118.54, 8) for i in range(4)]
    ids = [c.h3 for c in cells]
    arrival = {ids[0]: 0, ids[1]: 1, ids[2]: 2, ids[3]: 3}  # model front marches north over 4 steps
    wx = pl.DataFrame({"cell": ids, "time": [at] * 4, "layer": ["t2m"] * 4, "value": [34.0] * 4})
    meta = {"at": at, "wind_from": 20.0, "wind_speed": 30.0}

    base = build_backtest_map("t", cells, {ids[0], ids[3]}, arrival, meta, {"iou": 0.5, "recall": 0.5}, res=8)
    m = overlay_crew(base, wx, ids[0], arrival, meta)

    assert {r["ff_id"] for r in m["plan"]}  # a deployment was scored
    pf = m["plan_frames"]
    assert len(pf) == len(m["layers"][0]["frames"])  # one crew-frame per CA step

    def pos(frame, role):
        r = next(rec for rec in frame if rec["role"] == role)
        return r["lat"], r["lon"]

    assert pos(pf[0], "ba") != pos(pf[-1], "ba")  # on-task crew advance with the front
    assert pos(pf[0], "staging") == pos(pf[-1], "staging")  # staging crew hold at the rear
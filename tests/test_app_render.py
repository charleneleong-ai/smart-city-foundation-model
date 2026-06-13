from datetime import datetime, timezone

import polars as pl

from sctwin.app.render import h3_layer_records


def _frame() -> pl.DataFrame:
    t = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "cell": ["a", "b", "c"],
            "time": [t, t, t],
            "layer": ["t2m"] * 3,
            "value": [0.0, 5.0, 10.0],
        }
    )


def test_records_one_per_cell_with_distinct_colors_by_value():
    recs = h3_layer_records(_frame(), at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert [r["cell"] for r in recs] == ["a", "b", "c"]
    assert all(len(r["color"]) == 4 for r in recs)
    assert recs[0]["color"] != recs[2]["color"]  # min vs max value map to different colors


def test_filters_to_requested_timestamp():
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2020, 1, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {"cell": ["a", "a"], "time": [t0, t1], "layer": ["t2m", "t2m"], "value": [1.0, 9.0]}
    )
    recs = h3_layer_records(df, at=t1)
    assert len(recs) == 1
    assert recs[0]["value"] == 9.0


def test_constant_field_does_not_divide_by_zero():
    t = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame({"cell": ["a", "b"], "time": [t, t], "layer": ["t2m"] * 2, "value": [3.0, 3.0]})
    recs = h3_layer_records(df, at=t)
    assert len(recs) == 2  # no crash on zero range

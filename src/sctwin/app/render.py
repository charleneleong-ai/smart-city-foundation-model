from datetime import datetime
from typing import cast

import polars as pl

RGBA = tuple[int, int, int, int]


def _ramp(t: float) -> RGBA:
    t = max(0.0, min(1.0, t))  # blue -> red over 0..1, fixed alpha
    return (int(255 * t), 40, int(255 * (1 - t)), 160)


def h3_layer_records(frame: pl.DataFrame, at: datetime) -> list[dict]:
    snap = frame.filter(pl.col("time") == at)
    if snap.height == 0:
        return []
    vmin = cast(float, snap["value"].min())
    span = (cast(float, snap["value"].max()) - vmin) or 1.0
    records = []
    for r in snap.iter_rows(named=True):
        t = (r["value"] - vmin) / span  # normalized 0..1 — drives both color and extrusion height
        records.append({"cell": r["cell"], "value": r["value"], "color": list(_ramp(t)), "height": t})
    return records

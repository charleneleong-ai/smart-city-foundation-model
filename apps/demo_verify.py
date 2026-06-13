"""Verify-view: render the SP5 verification results on the SP8 3D map, with a layer
dropdown (|error|, coverage, prediction, actual).

Forecasts a (synthetic, stationary) district load from a real Open-Meteo H3 grid, runs
the SP5 verification harness (split-conformal intervals), and renders the per-(cell,time)
results as a multi-layer 3D time-series map. Pick a layer; press Play to step through the
held-out test hours.

Run: uv run --extra forecast python apps/demo_verify.py --city london --radius 30 --res 8
(writes <city>_verify_3d.html — open it in a browser)
"""

import argparse
import math
from datetime import datetime, timedelta, timezone

import h3
import numpy as np
import polars as pl

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.geo import Cell, center_of
from sctwin.registry import Registry
from sctwin.verify.results import as_layer, verification_frame

from presets import PRESETS, bbox_and_zoom
from render_3d import to_self_contained_html

_ABOUT = (
    "Verify-view of the SP5 harness (GBM + split-conformal 90% intervals). Pick a layer: "
    "|error| (red = where the twin is least accurate), coverage, prediction, or actual. "
    "Demand is synthetic — diurnal cycle + a west→east noise gradient — over a real H3 "
    "grid. Press Play to step through the held-out test hours."
)
# field -> (display name, unit, force vmin)
_LAYERS = [
    ("abs_error", "|error|", "load", 0.0),
    ("covered", "covered", "in/out", None),
    ("y_pred", "prediction", "load", None),
    ("y_true", "actual", "load", None),
]


def _synth_load(wx: pl.DataFrame, res: int) -> pl.DataFrame:
    cell_ids = wx["cell"].unique().to_list()
    lons = {c: center_of(Cell(c, res))[1] for c in cell_ids}
    lo, hi = min(lons.values()), max(lons.values())
    noise = {c: 0.5 + 7.0 * ((lons[c] - lo) / ((hi - lo) or 1.0)) for c in cell_ids}
    rng = np.random.default_rng(0)
    rows = wx.select("cell", "time").to_dict(as_series=False)
    load = [
        100.0 + 18.0 * math.sin(2 * math.pi * t.hour / 24) + rng.normal(0, noise[c])
        for c, t in zip(rows["cell"], rows["time"], strict=True)
    ]
    return pl.DataFrame({"cell": rows["cell"], "time": rows["time"], "layer": "load", "value": load})


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the SP5 verification results map.")
    ap.add_argument("--city", default="london", choices=sorted(PRESETS))
    ap.add_argument("--start", default="2020-01-15")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--radius", type=float, default=None, help="km around the preset centre")
    ap.add_argument("--res", type=int, default=None, help="H3 resolution override")
    args = ap.parse_args()

    p = PRESETS[args.city]
    south, west, north, east, zoom, res = bbox_and_zoom(p, args.radius, args.res)
    cells = cells_in_bbox(south, west, north, east, res)
    if not 0 < len(cells) <= 400:
        raise SystemExit(f"{len(cells)} cells — keep 1..400 (one API call each); adjust --radius/--res")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=args.days - 1)
    reg = Registry()
    reg.register(OpenMeteoWeatherAdapter())
    print(f"fetching {len(cells)} cells × {args.days} d for {args.city} (res {res}) ...")
    wx = reg.get("weather.t2m", cells, start, end)

    supervised = build_supervised(_synth_load(wx, res), wx)
    results = verification_frame(GBMForecaster(), supervised, FEATURE_COLS, alpha=0.1)
    coverage = results["covered"].mean()

    # one map layer per result field; share the test-hour time axis
    res_num = results.with_columns(pl.col("covered").cast(pl.Float64))
    times = res_num.select(pl.col("time").unique().sort()).to_series().to_list()
    layers = []
    for field, name, unit, force_min in _LAYERS:
        flayer = as_layer(res_num, field)
        vmin = force_min if force_min is not None else float(flayer["value"].min())
        vmax = float(flayer["value"].max())
        frames = [
            {"label": t.strftime("%m-%d %H:%M"), "records": h3_layer_records(flayer, at=t, vmin=vmin, vmax=vmax)}
            for t in times
        ]
        layers.append({"name": name, "unit": unit, "frames": frames})

    html = to_self_contained_html(
        layers,
        lat=p["lat"],
        lon=p["lon"],
        zoom=zoom,
        pitch=p.get("pitch", 50.0),
        elevation_scale=4.0 * h3.average_hexagon_edge_length(res, unit="m"),
        title=f"{args.city.upper()} — load forecast verification",
        subtitle=f"SP5 · GBM · split-conformal · coverage {coverage:.0%} · {len(times)} test h",
        about=_ABOUT,
    )
    out = f"{args.city}_verify_3d.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} — {len(cells)} hexes × {len(times)} test h · empirical coverage {coverage:.1%}")


if __name__ == "__main__":
    main()

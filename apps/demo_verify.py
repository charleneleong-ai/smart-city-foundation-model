"""Verify-view: render the SP5 verification results on the SP8 3D map.

Forecasts a (synthetic) district load from real Open-Meteo weather, runs the SP5
verification harness (split-conformal intervals), and renders the per-(cell, time)
**prediction error** as a 3D time-series map — red where the twin is least accurate.
The load is synthetic (we don't have a real load adapter yet) with a west→east noise
gradient, so the error map has an interpretable spatial pattern.

Run: uv run --extra forecast python apps/demo_verify.py
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

from presets import PRESETS
from render_3d import to_self_contained_html

_ABOUT = (
    "Verify-view. Colour and height = |predicted − actual| district load from the SP5 "
    "verification harness (GBM + split-conformal 90% intervals). Red = where the twin is "
    "least accurate. Demand is synthetic (no real load adapter yet) — a diurnal cycle plus "
    "a west→east noise gradient — over London's real H3 grid. Press Play to step through "
    "the held-out test hours."
)


def _synth_load(wx: pl.DataFrame, res: int) -> pl.DataFrame:
    """Synthetic stationary demand: base + diurnal cycle + a per-cell noise gradient
    (west->east). Stationary day-to-day so split-conformal calibration is exchangeable
    and hits its nominal coverage; the west->east noise is what the error map surfaces."""
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
    ap = argparse.ArgumentParser(description="Render the SP5 verification error map.")
    ap.add_argument("--city", default="london", choices=sorted(PRESETS))
    ap.add_argument("--start", default="2020-01-15")
    ap.add_argument("--days", type=int, default=5)
    args = ap.parse_args()

    p = PRESETS[args.city]
    cells = cells_in_bbox(p["south"], p["west"], p["north"], p["east"], p["res"])
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=args.days - 1)

    reg = Registry()
    reg.register(OpenMeteoWeatherAdapter())
    print(f"fetching {len(cells)} cells × {args.days} d for {args.city} ...")
    wx = reg.get("weather.t2m", cells, start, end)

    supervised = build_supervised(_synth_load(wx, p["res"]), wx)
    results = verification_frame(GBMForecaster(), supervised, FEATURE_COLS, alpha=0.1)
    coverage = results["covered"].mean()
    err = as_layer(results, "abs_error")
    gmax = float(err["value"].max())

    times = err.select(pl.col("time").unique().sort()).to_series().to_list()
    frames = [
        {"label": t.strftime("%m-%d %H:%M"), "records": h3_layer_records(err, at=t, vmin=0.0, vmax=gmax)}
        for t in times
    ]

    html = to_self_contained_html(
        frames,
        lat=p["lat"],
        lon=p["lon"],
        zoom=p.get("zoom", 10.6),
        pitch=p.get("pitch", 50.0),
        elevation_scale=4.0 * h3.average_hexagon_edge_length(p["res"], unit="m"),
        title=f"{args.city.upper()} — load forecast |error|",
        subtitle=f"SP5 verify · GBM · split-conformal · coverage {coverage:.0%} · {len(times)} test h",
        about=_ABOUT,
        unit="load",
    )
    out = f"{args.city}_verify_3d.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} — {len(cells)} hexes × {len(times)} test h · empirical coverage {coverage:.1%}")


if __name__ == "__main__":
    main()

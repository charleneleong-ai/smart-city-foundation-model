"""Render a day of Open-Meteo temperature over a city/region's H3 grid as a 3D WebGL
map with a Play button + time slider.

Run: uv run python apps/demo_weather.py --city uk            (london, nyc, tokyo, uk)
     uv run python apps/demo_weather.py --city london --radius 40 --res 7
(hits the real Open-Meteo archive API; writes <city>_3d.html — open it in a browser)
"""

import argparse
import math
from datetime import datetime, timezone

import h3
import polars as pl

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry

from presets import PRESETS
from render_3d import to_self_contained_html

_ABOUT = (
    "Each hexagon is an H3 res-{res} cell (~{edge_km:.1f} km across). Colour and bar "
    "height encode 2 m air temperature (blue = cold → red = warm), normalised over the "
    "whole day so frames are comparable. Source: Open-Meteo reanalysis. Press Play (or "
    "drag the slider) to step through 24 h; Toggle 2D/3D to flatten the bars."
)
_MAX_CELLS = 400  # one Open-Meteo call per cell


def _bbox_around(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a day of Open-Meteo temperature as a 3D map.")
    ap.add_argument("--city", default="uk", choices=sorted(PRESETS), help="preset region")
    ap.add_argument("--date", default="2020-01-15", help="YYYY-MM-DD")
    ap.add_argument("--radius", type=float, default=None, help="km around the preset centre")
    ap.add_argument("--res", type=int, default=None, help="H3 resolution override (0..15)")
    args = ap.parse_args()

    p = PRESETS[args.city]
    res = args.res if args.res is not None else p["res"]
    if args.radius is not None:
        south, west, north, east = _bbox_around(p["lat"], p["lon"], args.radius)
        span = max(north - south, east - west)
        zoom = math.log2(360.0 / span) - 0.4
    else:
        south, west, north, east = p["south"], p["west"], p["north"], p["east"]
        zoom = p.get("zoom", 10.6)

    cells = cells_in_bbox(south, west, north, east, res)
    if not cells:
        raise SystemExit("no cells in that area — widen --radius or coarsen --res")
    if len(cells) > _MAX_CELLS:
        raise SystemExit(
            f"{len(cells)} cells > {_MAX_CELLS} (one API call each) — coarsen --res or shrink --radius"
        )

    day = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    reg = Registry()
    reg.register(OpenMeteoWeatherAdapter())
    print(f"fetching {len(cells)} cells for {args.city} {args.date} (res {res}) ...")
    frame = reg.get("weather.t2m", cells, day, day)  # 24 hourly rows per cell

    gmin, gmax = float(frame["value"].min()), float(frame["value"].max())
    hours = frame.select(pl.col("time").unique().sort()).to_series().to_list()
    frames = [
        {"label": h.strftime("%H:%M"), "records": h3_layer_records(frame, at=h, vmin=gmin, vmax=gmax)}
        for h in hours
    ]

    edge_m = h3.average_hexagon_edge_length(res, unit="m")
    html = to_self_contained_html(
        frames,
        lat=p["lat"],
        lon=p["lon"],
        zoom=zoom,
        pitch=p.get("pitch", 50.0),
        elevation_scale=4.0 * edge_m,  # extrusion scaled to hex size, so 3D is visible at any zoom
        title=f"{args.city.upper()} — 2 m air temperature",
        subtitle=f"Open-Meteo · {args.date} · H3 res {res} · 24 h",
        about=_ABOUT.format(res=res, edge_km=edge_m / 1000),
        unit="°C",
    )
    out = f"{args.city}_3d.html"
    with open(out, "w") as f:
        f.write(html)
    n = len(cells) * len(frames)
    print(f"wrote {out} — {len(cells)} hexes × {len(frames)} h = {n:,} temperature values")


if __name__ == "__main__":
    main()

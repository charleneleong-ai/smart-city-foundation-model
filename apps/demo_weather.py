"""Render a day of Open-Meteo temperature over a city/region's H3 grid as a 3D WebGL
map with a Play button + time slider.

Run: uv run python apps/demo_weather.py --city uk   (cities: london, nyc, tokyo; region: uk)
(hits the real Open-Meteo archive API; writes <city>_3d.html — open it in a browser)
"""

import argparse
from datetime import datetime, timezone

import polars as pl

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry

from presets import PRESETS
from render_3d import to_self_contained_html

_ABOUT = (
    "Each hexagon is an H3 res-{res} cell. Colour and bar height encode 2 m air "
    "temperature (blue = cold → red = warm), normalised over the whole day so frames are "
    "comparable. Source: Open-Meteo reanalysis. Press Play (or drag the slider) to step "
    "through 24 h; Toggle 2D/3D to flatten."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a day of Open-Meteo temperature as a 3D map.")
    ap.add_argument("--city", default="london", choices=sorted(PRESETS), help="preset region")
    ap.add_argument("--date", default="2020-01-15", help="YYYY-MM-DD")
    args = ap.parse_args()

    p = PRESETS[args.city]
    cells = cells_in_bbox(p["south"], p["west"], p["north"], p["east"], p["res"])
    if len(cells) > 400:
        raise SystemExit(f"{args.city}: {len(cells)} cells > 400 (one API call each) — coarsen res")

    day = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    reg = Registry()
    reg.register(OpenMeteoWeatherAdapter())
    print(f"fetching {len(cells)} cells for {args.city} {args.date} ...")
    frame = reg.get("weather.t2m", cells, day, day)  # 24 hourly rows per cell

    gmin, gmax = float(frame["value"].min()), float(frame["value"].max())
    hours = frame.select(pl.col("time").unique().sort()).to_series().to_list()
    frames = [
        {"label": h.strftime("%H:%M"), "records": h3_layer_records(frame, at=h, vmin=gmin, vmax=gmax)}
        for h in hours
    ]

    html = to_self_contained_html(
        frames,
        lat=p["lat"],
        lon=p["lon"],
        zoom=p.get("zoom", 10.6),
        pitch=p.get("pitch", 50.0),
        title=f"{args.city.upper()} — 2 m air temperature",
        subtitle=f"Open-Meteo · {args.date} · H3 res {p['res']} · 24 h",
        about=_ABOUT.format(res=p["res"]),
        unit="°C",
    )
    out = f"{args.city}_3d.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({len(cells)} hexes × {len(frames)} hours) — open it in a browser")


if __name__ == "__main__":
    main()

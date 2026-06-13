"""Render the city digital twin: switch between data domains (Weather, Energy) on one 3D map.

Run: uv run --extra forecast python apps/demo_twin.py --city uk
(writes <city>_twin_3d.html — open it; use the Domain dropdown to switch Weather <-> Energy,
the Layer dropdown to pick a field, the radius slider to filter, Play to step through time)
"""

import argparse

from presets import PRESETS
from render_3d import to_self_contained_html
from twin import energy_map, weather_map

_ABOUT = (
    "City digital twin. The Domain selector switches data layers — Weather (2 m temperature, "
    "heating degrees) and Energy (demand, forecast, |error|, delta, coverage from the SP5 "
    "split-conformal harness). Pick a layer within a domain; the radius slider filters "
    "preloaded cells (no re-fetch); Play steps through time. Energy demand is synthetic for now. "
    "More domains (infrastructure, aerial, construction) plug in as additional maps."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the city digital twin (multi-domain 3D map).")
    ap.add_argument("--city", default="uk", choices=sorted(PRESETS))
    ap.add_argument("--date", default="2020-01-15")
    ap.add_argument("--days", type=int, default=5, help="days of history for the energy forecast")
    ap.add_argument("--radius", type=float, default=None, help="km around the preset centre")
    ap.add_argument("--res", type=int, default=None, help="H3 resolution override")
    args = ap.parse_args()

    p = PRESETS[args.city]
    maps = [
        weather_map("Weather", p, args.date, radius=args.radius, res=args.res),
        energy_map("Energy", p, args.date, args.days, radius=args.radius, res=args.res),
    ]
    html = to_self_contained_html(maps, title=f"{args.city.upper()} — city digital twin", about=_ABOUT)
    out = f"{args.city}_twin_3d.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} — domains: " + ", ".join(m["name"] for m in maps))


if __name__ == "__main__":
    main()

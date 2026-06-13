"""Render a day of Open-Meteo weather over a city/region as a 3D map (Weather domain only).

Run: uv run python apps/demo_weather.py --city uk --radius 250 --res 4
(hits the real Open-Meteo archive API; writes <city>_3d.html — open it in a browser)
For the full multi-domain twin (Weather + Energy), use apps/demo_twin.py.
"""

import argparse

from presets import PRESETS
from render_3d import to_self_contained_html
from twin import weather_map

_ABOUT = (
    "Weather domain. Layers: 2 m air temperature and heating degrees (a heating-demand "
    "proxy). Colour and bar height encode the value, normalised over the day. The radius "
    "slider filters preloaded cells (no re-fetch); Play steps through 24 h; Toggle 2D/3D."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a day of Open-Meteo weather as a 3D map.")
    ap.add_argument("--city", default="uk", choices=sorted(PRESETS), help="preset region")
    ap.add_argument("--date", default="2020-01-15", help="YYYY-MM-DD")
    ap.add_argument("--radius", type=float, default=None, help="km around the preset centre")
    ap.add_argument("--res", type=int, default=None, help="H3 resolution override (0..15)")
    args = ap.parse_args()

    p = PRESETS[args.city]
    m = weather_map(f"{args.city.upper()} weather", p, args.date, radius=args.radius, res=args.res)
    html = to_self_contained_html(
        [m], title=f"{args.city.upper()} — 2 m air temperature", about=_ABOUT
    )
    out = f"{args.city}_3d.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} — {len(m['layers'])} layers ({', '.join(layer['name'] for layer in m['layers'])})")


if __name__ == "__main__":
    main()

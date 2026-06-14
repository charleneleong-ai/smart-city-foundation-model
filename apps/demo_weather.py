"""Render a day of Open-Meteo weather over a city/region as a 3D map (Weather domain only).

Run: uv run python apps/demo_weather.py --city uk --radius 250 --res 4
(hits the real Open-Meteo archive API; writes <city>_3d.html — open it in a browser)
For the full multi-domain twin (Weather + Energy), use apps/demo_twin.py.
"""

import os
from pathlib import Path
from typing import Annotated

import typer

from presets import PRESETS
from render_3d import to_self_contained_html
from twin import weather_map

_ABOUT = (
    "Weather domain. Layers: 2 m air temperature and heating degrees (a heating-demand "
    "proxy). Colour and bar height encode the value, normalised over the day. The radius "
    "slider filters preloaded cells (no re-fetch); Play steps through 24 h; Toggle 2D/3D."
)


def main(
    city: Annotated[str, typer.Option(help="preset region")] = "uk",
    date: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2020-01-15",
    radius: Annotated[float | None, typer.Option(help="km around the preset centre")] = None,
    res: Annotated[int | None, typer.Option(help="H3 resolution override (0..15)")] = None,
    source: Annotated[str, typer.Option(help="open-meteo, or era5 (gridded; needs CDS key)")] = "open-meteo",
) -> None:
    """Render a day of weather over a city/region as a 3D map (Weather domain only)."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    if source == "era5":
        os.environ["WEATHER_SOURCE"] = "era5"

    m = weather_map(f"{city.upper()} weather", PRESETS[city], date, radius=radius, res=res)
    html = to_self_contained_html([m], title=f"{city.upper()} — 2 m air temperature", about=_ABOUT)
    out = Path(f"{city}_3d.html")
    out.write_text(html)
    print(f"wrote {out} — {len(m['layers'])} layers ({', '.join(layer['name'] for layer in m['layers'])})")


if __name__ == "__main__":
    typer.run(main)

"""Static backtest plot: the macro fire CA's predicted front vs an observed burn perimeter.

The 2D companion to the deck.gl viewer (apps/demo_fire.py -> la_fire_3d.html). Rasterises the
perimeter to H3, runs the CA from the ignition seed, and draws predicted arrival (hot colormap)
over the observed burn + perimeter outline, annotated with IoU / recall.

Needs the viz extra:  uv sync --extra viz
Run: uv run --extra viz python apps/viz_fire.py --perimeter palisades.geojson --out fire_backtest.png
(fetch the Palisades perimeter as in apps/eval_fire.py; honest limits in src/sctwin/fire.py)

Note: on some macOS setups Pillow >=11 fails to load (libjpeg flat-namespace symbol); if you hit
`_jpeg_resync_to_restart`, append `--with 'pillow<11'` to the run command.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import h3
import typer

from demo_fire import spread_from_weather
from eval_fire import bbox

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import WEATHER_VARS, OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.geo import cell_of
from sctwin.verify.burn import cells_from_geojson, score

MAX_CELLS = 1500


def _hex(hx: str) -> list[tuple[float, float]]:
    """H3 cell boundary as [(lng, lat), ...] for a matplotlib polygon."""
    return [(lng, lat) for lat, lng in h3.cell_to_boundary(hx)]


def _plot(out: Path, *, cells, observed, arrival, gj, seed, meta, sc) -> None:
    import matplotlib  # heavy, optional (the `viz` extra) — imported only when rendering

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(PolyCollection([_hex(c.h3) for c in cells], facecolors="none", edgecolors="#e8e8e8", linewidths=0.2))
    ax.add_collection(PolyCollection([_hex(c) for c in observed], facecolors="#9ecae1", edgecolors="#6baed6", linewidths=0.2, alpha=0.55))
    mx = max(arrival.values()) or 1
    front = PolyCollection(
        [_hex(c) for c in arrival],
        array=[step / mx for step in arrival.values()],
        cmap="autumn_r", edgecolors="k", linewidths=0.15, alpha=0.7,
    )
    ax.add_collection(front)
    for feat in gj.get("features", [gj]):
        geom = feat.get("geometry", feat)
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for rings in polys:
            for ring in rings:
                ax.plot([p[0] for p in ring], [p[1] for p in ring], color="#d62728", linewidth=1.6)
    slat, slng = h3.cell_to_latlng(seed)
    ax.plot(slng, slat, "*", color="#1f77b4", markersize=20, markeredgecolor="k")

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(
        "macro CA front vs observed burn perimeter\n"
        f"wind from {meta['wind_from']:.0f}° @ {meta['wind_speed']:.0f} km/h · "
        f"predicted {sc['predicted']} · observed {sc['observed']} · overlap {sc['intersection']} · "
        f"IoU {sc['iou']:.2f} · recall {sc['recall']:.2f}",
        fontsize=11,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#9ecae1", edgecolor="#6baed6", label="observed burn"),
            plt.Line2D([0], [0], color="#d62728", lw=1.6, label="perimeter"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#1f77b4", markersize=14, label="ignition"),
        ],
        loc="lower left", fontsize=8,
    )
    cb = fig.colorbar(front, ax=ax, shrink=0.55)
    cb.set_label("predicted arrival (normalised CA step) — yellow=early, red=late")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main(
    perimeter: Annotated[Path, typer.Option(help="observed burn perimeter GeoJSON")],
    out: Annotated[Path, typer.Option(help="output image path")] = Path("fire_backtest.png"),
    date: Annotated[str, typer.Option(help="YYYY-MM-DD of the fire weather")] = "2025-01-07",
    res: Annotated[int, typer.Option(help="H3 resolution")] = 8,
    steps: Annotated[int, typer.Option(help="CA spread steps")] = 40,
    spread: Annotated[float, typer.Option(help="0..1 front threshold")] = 0.5,
    seed_lat: Annotated[float, typer.Option(help="ignition latitude")] = 34.0725,
    seed_lon: Annotated[float, typer.Option(help="ignition longitude")] = -118.5425,
    margin: Annotated[float, typer.Option(help="km padding around the perimeter bbox")] = 2.0,
) -> None:
    """Render the macro fire CA's predicted front against an observed burn perimeter."""
    gj = json.loads(perimeter.read_text())
    observed = cells_from_geojson(gj, res)
    south, west, north, east = bbox(gj)
    d = margin / 111.0
    cells = cells_in_bbox(south - d, west - d, north + d, east + d, res)
    if not 0 < len(cells) <= MAX_CELLS:
        raise SystemExit(f"{len(cells)} cells — keep 1..{MAX_CELLS}; raise --res or shrink --margin")

    cached = CachingAdapter(OpenMeteoWeatherAdapter(variables=WEATHER_VARS), ".cache/open-meteo-fire")
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    print(f"observed {len(observed)} cells · fetching {len(cells)} weather cells {date} ...")
    wx = cached.fetch(cells, day, day)
    seed = cell_of(seed_lat, seed_lon, res).h3
    arrival, meta = spread_from_weather(wx, seed, steps=steps, spread_fraction=spread)
    sc = score(set(arrival), observed)
    _plot(out, cells=cells, observed=observed, arrival=arrival, gj=gj, seed=seed, meta=meta, sc=sc)
    print(f"wrote {out} — IoU {sc['iou']:.2f} · recall {sc['recall']:.2f} · overlap {sc['intersection']}/{sc['observed']}")


if __name__ == "__main__":
    typer.run(main)

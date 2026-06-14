"""Gather + cache real total-load (Electricity Maps) for a set of zones — ground-truth demand
for modelling. The free endpoint returns only the last 24 h, so run this repeatedly (e.g. daily,
detached) to *accumulate* a growing per-zone series in .cache/electricitymaps/.

Needs the token:  export ELECTRICITYMAPS_TOKEN=<key>
Run:              uv run python apps/gather_demand.py --zones all
"""

from typing import Annotated

import typer

from sctwin.adapters.demand import _FAR_FUTURE, _FAR_PAST, ElectricityMapsAdapter
from sctwin.geo import cell_of

# a starter set of Electricity Maps zones (it covers ~200) + a representative city centre each
ZONES: dict[str, tuple[float, float]] = {
    "GB": (51.50, -0.12), "FR": (48.85, 2.35), "DE": (52.52, 13.40), "ES": (40.42, -3.70),
    "AU-NSW": (-33.87, 151.21), "US-CAL-CISO": (34.05, -118.24), "JP-TK": (35.68, 139.69),
    "IN-NO": (28.61, 77.21), "BR-CS": (-23.55, -46.63), "ZA": (-26.20, 28.04),
}


def main(
    zones: Annotated[str, typer.Option(help="comma-separated zones, or 'all'")] = "all",
    res: Annotated[int, typer.Option(help="H3 resolution for the zone cell")] = 4,
) -> None:
    """Poll Electricity Maps total-load for each zone and accumulate it in the cache."""
    chosen = ZONES if zones == "all" else {z: ZONES[z] for z in zones.split(",") if z in ZONES}
    print(f"gathering {len(chosen)} zones -> .cache/electricitymaps/ (accumulating; run repeatedly for history)")
    for zone, (lat, lon) in chosen.items():
        try:
            out = ElectricityMapsAdapter(zone=zone).fetch([cell_of(lat, lon, res)], _FAR_PAST, _FAR_FUTURE)
            latest = out.sort("time").tail(1)
            mw = latest["value"].item() if latest.height else None
            print(f"  {zone:14s} cached {out.height:>3d} h  latest {mw} MW")
        except Exception as exc:  # one zone failing (no access / unknown) shouldn't stop the gather
            print(f"  {zone:14s} {type(exc).__name__}: {str(exc)[:70]}")


if __name__ == "__main__":
    typer.run(main)

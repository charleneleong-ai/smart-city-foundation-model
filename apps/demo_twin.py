"""Render the city digital twin: switch between data domains (Weather, Energy) on one 3D map.

Run: uv run --extra forecast python apps/demo_twin.py --city uk
(writes <city>_twin_3d.html — open it; use the Domain dropdown to switch Weather <-> Energy,
the Layer dropdown to pick a field, the radius slider to filter, Play to step through time)
"""

import argparse
import calendar
import os

from presets import PRESETS
from render_3d import to_self_contained_html
from twin import twin_map


def _sample_dates(date: str, months: str | None, day: int) -> list[str]:
    """The single --date, or one YYYY-MM-DD per requested month ('all' or e.g. '1,4,7,10'),
    reusing the year from --date and clamping the day to each month's length."""
    if not months:
        return [date]
    year = int(date[:4])
    nums = range(1, 13) if months == "all" else [int(n) for n in months.split(",")]
    return [f"{year}-{m:02d}-{min(day, calendar.monthrange(year, m)[1]):02d}" for m in nums]

_ABOUT = (
    "City digital twin. The Layer dropdown groups every layer by domain — Weather (2 m "
    "temperature, heating degrees) and Energy (demand, forecast, |error|, delta, coverage "
    "from the SP4 GBM + SP5 split-conformal harness) — so all are visible at once. Hover a "
    "hex to compare a domain's layers (e.g. demand vs forecast). The radius slider filters "
    "preloaded cells (no re-fetch); Play steps through time. Energy demand is synthetic for "
    "now. More domains (infrastructure, aerial, construction) plug in as more layer groups."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the city digital twin (multi-domain 3D map).")
    ap.add_argument("--city", default="uk", choices=sorted(PRESETS))
    ap.add_argument("--date", default="2020-01-15")
    ap.add_argument("--months", default=None,
                    help="sample one map per month into a Month picker: 'all' or e.g. '1,4,7,10' (year from --date)")
    ap.add_argument("--day", type=int, default=15, help="day-of-month to sample when --months is set")
    ap.add_argument("--days", type=int, default=5, help="days of history for the energy forecast")
    ap.add_argument("--radius", type=float, default=None, help="km around the preset centre")
    ap.add_argument("--res", type=int, default=None, help="H3 resolution override")
    ap.add_argument("--source", default="open-meteo", choices=["open-meteo", "era5"],
                    help="weather source — era5 is gridded (dense, no rate limit; needs CDS key)")
    args = ap.parse_args()
    if args.source == "era5":
        os.environ["WEATHER_SOURCE"] = "era5"

    p = PRESETS[args.city]
    dates = _sample_dates(args.date, args.months, args.day)
    maps = [
        twin_map(
            f"{calendar.month_abbr[int(d[5:7])]} {d[:4]}" if args.months else f"{args.city.upper()} twin",
            p, d, args.days, radius=args.radius, res=args.res,
        )
        for d in dates
    ]
    html = to_self_contained_html(
        maps, title=f"{args.city.upper()} — city digital twin", about=_ABOUT,
        map_label="Month" if args.months else "Domain",
    )
    out = f"{args.city}_twin_3d.html"
    with open(out, "w") as f:
        f.write(html)
    layers = maps[0]["layers"]
    print(f"wrote {out} — {len(maps)} map(s) [{', '.join(dates)}], "
          f"{len(layers)} layers (" + ", ".join(layer["name"] for layer in layers) + ")")


if __name__ == "__main__":
    main()

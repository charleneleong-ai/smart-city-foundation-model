"""Render the city digital twin: sample each month, switch domains/layers on one 3D map.

Run: uv run --extra forecast python apps/demo_twin.py --city uk
(writes <city>_twin_3d.html — open it; the Month dropdown samples the 1st of each month,
the Layer dropdown picks a field, the radius slider filters, Play steps through the day.
--months none for a single --date build.)
"""

import argparse
import calendar
import json
import os
from pathlib import Path

from presets import PRESETS
from render_3d import to_lazy_html, to_self_contained_html
from twin import twin_map, unify_ranges


def _years(date: str, years: str | None) -> list[int]:
    """The year in --date by default, or an explicit set: '2018,2019' or an inclusive range '2018-2020'."""
    if not years:
        return [int(date[:4])]
    if "-" in years:
        lo, hi = (int(x) for x in years.split("-"))
        return list(range(lo, hi + 1))
    return [int(y) for y in years.split(",")]


def _sample_dates(date: str, months: str | None, day: int, years: list[int]) -> list[str]:
    """The single --date, or one YYYY-MM-DD per (year, month) — year-major — for the requested
    months ('all' or e.g. '1,4,7,10'), clamping the day to each month's length."""
    if not months or months.lower() == "none":
        return [date]
    nums = range(1, 13) if months == "all" else [int(n) for n in months.split(",")]
    return [f"{y}-{m:02d}-{min(day, calendar.monthrange(y, m)[1]):02d}" for y in years for m in nums]

_ABOUT = (
    "City digital twin. The Month dropdown samples the 1st of each month (loaded on demand); "
    "the Layer dropdown groups every layer by domain — Weather (2 m temperature, heating "
    "degrees) and Energy (demand, forecast, |error|, delta, coverage from the SP4 GBM + SP5 "
    "split-conformal harness). Hover a hex to compare a domain's layers (e.g. demand vs "
    "forecast). The radius slider filters preloaded cells (no re-fetch); Play steps through "
    "the day. Energy demand is synthetic for now. More domains (infrastructure, aerial, "
    "construction) plug in as more layer groups."
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the city digital twin (multi-domain 3D map).")
    ap.add_argument("--city", default="uk", choices=sorted(PRESETS))
    ap.add_argument("--date", default="2020-01-01")
    ap.add_argument("--months", default="all",
                    help="months sampled into the Month picker: 'all' (default), e.g. '1,4,7,10', or 'none' "
                         "for a single --date build (year from --date)")
    ap.add_argument("--day", type=int, default=1, help="day-of-month to sample (clamped per month)")
    ap.add_argument("--years", default=None,
                    help="sample across years into a Year picker: e.g. '2018,2019,2020' or '2018-2020' "
                         "(default: the year in --date)")
    ap.add_argument("--days", type=int, default=5, help="days of history for the energy forecast")
    ap.add_argument("--radius", type=float, default=None, help="km around the preset centre")
    ap.add_argument("--res", type=int, default=None, help="H3 resolution override")
    ap.add_argument("--source", default="open-meteo", choices=["open-meteo", "era5"],
                    help="weather source — era5 is gridded (dense, no rate limit; needs CDS key)")
    ap.add_argument("--inline", action="store_true",
                    help="force one self-contained file (no lazy month fetch); default lazy-loads months over http")
    args = ap.parse_args()
    if args.source == "era5":
        os.environ["WEATHER_SOURCE"] = "era5"

    p = PRESETS[args.city]
    years = _years(args.date, args.years)
    dates = _sample_dates(args.date, args.months, args.day, years)
    multi = len(dates) > 1
    multiyear = multi and len(years) > 1
    stride = len(dates) // len(years) if multi else 1  # months per year (year-major flat list)

    def _name(d: str) -> str:
        if not multi:
            return f"{args.city.upper()} twin"
        mon = calendar.month_abbr[int(d[5:7])]
        return mon if multiyear else f"{mon} {d[:4]}"  # year shown in its own picker when present

    maps = [twin_map(_name(d), p, d, args.days, radius=args.radius, res=args.res) for d in dates]
    unify_ranges(maps)  # absolute gradient: one colour/height scale per layer across all maps
    title = f"{args.city.upper()} — city digital twin"
    axes = {
        "map_label": "Month" if multi else "Domain",
        "group_options": [str(y) for y in years] if multiyear else None,
        "stride": stride,
    }
    out = Path(f"{args.city}_twin_3d.html")
    if multi and not args.inline:  # lazy: small shell + per-map JSON fetched on select (serve over http)
        data_dir = out.with_suffix(".data")
        html, payloads = to_lazy_html(maps, data_dir=data_dir.name, title=title, about=_ABOUT, **axes)
        data_dir.mkdir(exist_ok=True)
        for i, payload in enumerate(payloads):
            (data_dir / f"{i}.json").write_text(json.dumps(payload))
        served = (
            f"loads {data_dir.name}/<i>.json on demand — serve over http: "
            f"python -m http.server 8001 then open http://localhost:8001/{out.name}"
        )
    else:
        html = to_self_contained_html(maps, title=title, about=_ABOUT, **axes)
        served = "self-contained (open directly)"
    out.write_text(html)
    layers = maps[0]["layers"]
    print(f"wrote {out} — {len(maps)} map(s) [{', '.join(dates)}], {len(layers)} layers; {served}")


if __name__ == "__main__":
    main()

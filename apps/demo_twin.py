"""Render the city digital twin: sample each month, switch domains/layers on one 3D map.

Run: uv run --extra forecast python apps/demo_twin.py --city uk
(writes <city>_twin_3d.html — open it; the Month dropdown samples the 1st of each month,
the Layer dropdown picks a field, the radius slider filters, Play steps through the day.
--months none for a single --date build.)
"""

import calendar
import json
import os
from collections import Counter
from datetime import date as _date
from pathlib import Path
from typing import Annotated

import typer

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


def _sample_dates(
    date: str, months: str | None, day: int, years: list[int], *, today: str | None = None
) -> list[str]:
    """The single --date, or one YYYY-MM-DD per (year, month) — year-major — for the requested
    months ('all' or e.g. '1,4,7,10'), clamping the day to each month's length and dropping any
    sample later than `today` (so a 'last N years from now' span omits future, data-less months)."""
    if not months or months.lower() == "none":
        return [date]
    nums = range(1, 13) if months == "all" else [int(n) for n in months.split(",")]
    cutoff = today or _date.today().isoformat()
    sampled = [f"{y}-{m:02d}-{min(day, calendar.monthrange(y, m)[1]):02d}" for y in years for m in nums]
    return [d for d in sampled if d <= cutoff]


def _grid_stride(dates: list[str], years: list[int]) -> int | None:
    """Months-per-year if the year×month grid is rectangular (>1 year, equal months each) — drives the
    year sub-picker. None when single-year or ragged (e.g. a partial current year) → flat Month axis."""
    if len(years) <= 1:
        return None
    per_year = set(Counter(d[:4] for d in dates).values())
    return next(iter(per_year)) if len(per_year) == 1 else None

_ABOUT = (
    "City digital twin. The Month dropdown samples the 1st of each month (loaded on demand); "
    "the Layer dropdown groups every layer by domain — Weather (2 m temperature, heating "
    "degrees) and Energy (demand, forecast, |error|, delta, coverage from the SP4 GBM + SP5 "
    "split-conformal harness). Hover a hex to compare a domain's layers (e.g. demand vs "
    "forecast). The radius slider filters preloaded cells (no re-fetch); Play steps through "
    "the day. Energy demand is synthetic for now. More domains (infrastructure, aerial, "
    "construction) plug in as more layer groups."
)


def main(
    city: Annotated[str, typer.Option(help="preset region")] = "uk",
    date: Annotated[str, typer.Option(help="YYYY-MM-DD (year used when --months/--years set)")] = "2020-01-01",
    months: Annotated[str, typer.Option(
        help="Month picker: 'all', e.g. '1,4,7,10', or 'none' for a single --date build")] = "all",
    day: Annotated[int, typer.Option(help="day-of-month to sample (clamped per month)")] = 1,
    years: Annotated[str | None, typer.Option(
        help="Year picker: e.g. '2018,2019,2020' or '2018-2020' (default: the year in --date)")] = None,
    days: Annotated[int, typer.Option(help="days of history for the energy forecast")] = 5,
    radius: Annotated[float | None, typer.Option(help="km around the preset centre")] = None,
    res: Annotated[int | None, typer.Option(help="H3 resolution override")] = None,
    source: Annotated[str, typer.Option(
        help="open-meteo (archive), open-meteo-forecast (real NWP, near-now dates), or era5")] = "open-meteo",
    inline: Annotated[bool, typer.Option(
        help="force one self-contained file; default lazy-loads months over http")] = False,
) -> None:
    """Render the city digital twin (multi-domain, month/year-sampled 3D map)."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    if source not in ("open-meteo", "open-meteo-forecast", "era5"):
        raise typer.BadParameter("--source must be open-meteo, open-meteo-forecast, or era5")
    os.environ["WEATHER_SOURCE"] = source

    p = PRESETS[city]
    year_nums = _years(date, years)
    dates = _sample_dates(date, months, day, year_nums)
    multi = len(dates) > 1
    grid_stride = _grid_stride(dates, year_nums)  # months/year if rectangular, else None (flat axis)
    multiyear = multi and grid_stride is not None
    stride = grid_stride or 1  # year sub-picker stride; 1 = single flat Month axis

    def _name(d: str) -> str:
        if not multi:
            return f"{city.upper()} twin"
        mon = calendar.month_abbr[int(d[5:7])]
        return mon if multiyear else f"{mon} {d[:4]}"  # year shown in its own picker when present

    maps = [twin_map(_name(d), p, d, days, radius=radius, res=res) for d in dates]
    unify_ranges(maps)  # absolute gradient: one colour/height scale per layer across all maps
    if multi:
        for k, d in enumerate(dates):
            maps[k]["ym"] = d[:7]  # YYYY-MM drives the viewer's data-driven Year + Month pickers
    title = f"{city.upper()} — city digital twin"
    axes = {
        "map_label": "Month" if multi else "Domain",
        "group_options": [str(y) for y in year_nums] if multiyear else None,
        "stride": stride,
    }
    out = Path(f"{city}_twin_3d.html")
    if multi and not inline:  # lazy: small shell + per-map JSON fetched on select (serve over http)
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
    typer.run(main)

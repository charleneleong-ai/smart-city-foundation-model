"""Benchmark the weather forecast: score Open-Meteo NWP against the realized reanalysis (the
cached ground truth) vs persistence + climatology. The weather analog of the demand baseline
harness — it makes "is open-meteo-forecast a baseline?" a number, and an AI weather FM
(Aurora/GenCast) the thing that has to beat it.

Run on a RECENT window (the forecast endpoint covers ~last 90 days; the archive settles after a
few days):
    uv run --extra forecast python apps/eval_weather.py --city london
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import typer

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import OpenMeteoForecastAdapter, OpenMeteoWeatherAdapter
from sctwin.forecast.skill import benchmark

from presets import PRESETS
from twin import _resolve


def _day(value: str | None, default_days_ago: int) -> datetime:
    if value:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return datetime.combine(
        datetime.now(timezone.utc).date() - timedelta(days=default_days_ago), datetime.min.time(), timezone.utc
    )


def main(
    city: Annotated[str, typer.Option(help="preset region")] = "london",
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD (default: ~13 days ago)")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD (default: ~6 days ago, archive settled)")] = None,
    radius: Annotated[float, typer.Option(help="km around the preset centre")] = 25.0,
    res: Annotated[int, typer.Option(help="H3 resolution")] = 6,
) -> None:
    """Score Open-Meteo NWP vs persistence vs climatology against realized truth (2 m temp)."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    s, e = _day(start, 13), _day(end, 6)
    cells, _, _ = _resolve(PRESETS[city], radius, res)

    forecast = CachingAdapter(OpenMeteoForecastAdapter(), ".cache/open-meteo-forecast").fetch(cells, s, e)
    truth = CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e)  # realized ground truth

    print(f"\nweather-forecast skill — {len(cells)} cells, {s.date()}..{e.date()} (2 m temperature)")
    print(f"  {'baseline':26s} {'MAE °C':>7} {'RMSE °C':>8} {'bias':>6}")
    for name, m in benchmark(forecast, truth).items():
        print(f"  {name:26s} {m['mae']:7.2f} {m['rmse']:8.2f} {m['bias']:6.2f}")


if __name__ == "__main__":
    typer.run(main)

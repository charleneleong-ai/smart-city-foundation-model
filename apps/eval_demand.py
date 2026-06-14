"""GBM vs Chronos-2 on REAL hourly electricity load (Monash electricity_hourly, 321 meters) —
the honest version of the synthetic smoke test. These meters aren't geo-aligned to a weather
field, so the features are calendar + lags only; both forecasters get the same data.

Run: uv run --extra forecast --extra tsfm python apps/eval_demand.py
"""

from datetime import datetime
from typing import Annotated

import polars as pl
import typer

from sctwin.demand import ELECTRICITY_URL, electricity_to_long
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.chronos import ChronosForecaster
from sctwin.forecast.features import CALENDAR_COLS, build_calendar_supervised
from sctwin.verify.results import verification_frame


def main(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD window start")] = "2013-01-07",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD window end")] = "2013-01-21",
    meters: Annotated[int, typer.Option(help="number of real meters")] = 20,
    device: Annotated[str, typer.Option(help="chronos device (cpu/cuda)")] = "cpu",
) -> None:
    """Score GBM vs Chronos-2 on real electricity load (calendar + lags, no weather)."""
    s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    demand = electricity_to_long(pl.read_parquet(ELECTRICITY_URL), start=s, end=e, n_meters=meters)
    sup = build_calendar_supervised(demand)
    print(f"\nreal demand — {meters} meters, {s.date()}..{e.date()}, supervised rows={sup.height}")

    gbm = verification_frame(GBMForecaster(), sup, CALENDAR_COLS)
    print(f"  {'GBM (gradient-boosted trees)':30s} MAE {float(gbm['abs_error'].mean()):6.1f}  "
          f"coverage {float(gbm['covered'].mean()):.2f}")
    ch = ChronosForecaster(device=device).verify(sup, covariates=["hour", "dow", "month"])
    print(f"  {'Chronos-2 (TS foundation model)':30s} MAE {float(ch['abs_error'].mean()):6.1f}  "
          f"coverage {float(ch['covered'].mean()):.2f}")


if __name__ == "__main__":
    typer.run(main)

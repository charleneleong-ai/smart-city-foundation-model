"""Australia, weather-coupled: GBM vs Chronos-2 on REAL AEMO NSW regional demand (MW) overlaid
with REAL Sydney weather. Mirrors the UK benchmark on the other side of the planet — weather is
global, so only the demand source (AEMO) is new.

Run: uv run --extra forecast --extra tsfm python apps/eval_au_demand.py
"""

from datetime import datetime, timezone
from typing import Annotated

import typer

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.demand import AEMODemandAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.chronos import ChronosForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.geo import cell_of
from sctwin.verify.results import verification_frame

_COVARIATES = ["t2m", "hdd", "cdd", "hour", "dow", "month"]


def main(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD window start")] = "2023-01-07",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD window end")] = "2023-01-21",
    region: Annotated[str, typer.Option(help="AEMO region (NSW1, VIC1, QLD1, SA1, TAS1)")] = "NSW1",
    res: Annotated[int, typer.Option(help="H3 resolution")] = 7,
    device: Annotated[str, typer.Option(help="chronos device (cpu/cuda)")] = "cpu",
) -> None:
    """Score GBM vs Chronos-2 on real AEMO regional demand + real Sydney weather."""
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    cells = [cell_of(-33.87, 151.21, res)]  # one Sydney cell — AEMO demand is a single regional series
    demand = AEMODemandAdapter(region=region).fetch(cells, s, e)
    weather = CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e)
    sup = build_supervised(demand, weather)
    print(f"\nreal AEMO {region} demand + Sydney weather — {s.date()}..{e.date()}, rows={sup.height}")

    gbm = verification_frame(GBMForecaster(), sup, FEATURE_COLS)
    print(f"  {'GBM (+ weather, lags)':32s} MAE {float(gbm['abs_error'].mean()):7.1f}  "
          f"coverage {float(gbm['covered'].mean()):.2f}")
    ch = ChronosForecaster(device=device).verify(sup, covariates=_COVARIATES)
    print(f"  {'Chronos-2 (+ weather covariate)':32s} MAE {float(ch['abs_error'].mean()):7.1f}  "
          f"coverage {float(ch['covered'].mean()):.2f}")


if __name__ == "__main__":
    typer.run(main)

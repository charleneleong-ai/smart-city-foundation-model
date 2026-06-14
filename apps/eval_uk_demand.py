"""The coherent UK benchmark: GBM vs Chronos-2 on REAL London household load (Low Carbon
London smart meters) overlaid with REAL London weather — demand that is genuinely
weather-driven, with the weather as a covariate. The "for foundation" test: does a TS-FM (+
weather) beat a GBM on real, weather-coupled UK demand?

Run: uv run --extra forecast --extra tsfm python apps/eval_uk_demand.py
"""

from datetime import datetime, timezone
from typing import Annotated

import typer

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.demand import LondonSmartMeterAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.chronos import ChronosForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.verify.results import verification_frame

_COVARIATES = ["t2m", "hdd", "cdd", "hour", "dow", "month"]  # weather + calendar (no lags for the FM)


def main(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD window start")] = "2013-01-07",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD window end")] = "2013-01-21",
    meters: Annotated[int, typer.Option(help="number of London households")] = 20,
    res: Annotated[int, typer.Option(help="H3 resolution")] = 7,
    device: Annotated[str, typer.Option(help="chronos device (cpu/cuda)")] = "cpu",
) -> None:
    """Score GBM vs Chronos-2 on real London load + real London weather (weather-coupled)."""
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    cells = cells_in_bbox(51.40, -0.25, 51.60, 0.05, res)[:meters]
    demand = LondonSmartMeterAdapter().fetch(cells, s, e)
    weather = CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e)
    sup = build_supervised(demand, weather)
    print(f"\nreal London load + weather — {len(cells)} households, {s.date()}..{e.date()}, rows={sup.height}")

    gbm = verification_frame(GBMForecaster(), sup, FEATURE_COLS)
    print(f"  {'GBM (+ weather, lags)':32s} MAE {float(gbm['abs_error'].mean()):.3f}  "
          f"coverage {float(gbm['covered'].mean()):.2f}")
    ch = ChronosForecaster(device=device).verify(sup, covariates=_COVARIATES)
    print(f"  {'Chronos-2 (+ weather covariate)':32s} MAE {float(ch['abs_error'].mean()):.3f}  "
          f"coverage {float(ch['covered'].mean()):.2f}")


if __name__ == "__main__":
    typer.run(main)

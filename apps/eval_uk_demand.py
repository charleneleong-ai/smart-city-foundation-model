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
from sctwin.forecast.features import BASE_FEATURES, LAGS_BY_FREQ, build_supervised, feature_cols, resample
from sctwin.verify.results import verification_frame


def main(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD window start")] = "2013-01-07",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD window end")] = "2013-01-21",
    meters: Annotated[int, typer.Option(help="number of London households")] = 20,
    res: Annotated[int, typer.Option(help="H3 resolution")] = 7,
    freq: Annotated[str, typer.Option(help="forecast frequency: hour|day|week|month")] = "hour",
    device: Annotated[str, typer.Option(help="chronos device (cpu/cuda)")] = "cpu",
) -> None:
    """Score GBM vs Chronos-2 on real London load + real London weather (weather-coupled)."""
    if freq not in LAGS_BY_FREQ:
        raise typer.BadParameter(f"--freq must be one of {', '.join(LAGS_BY_FREQ)}")
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    cells = cells_in_bbox(51.40, -0.25, 51.60, 0.05, res)[:meters]
    demand = resample(LondonSmartMeterAdapter().fetch(cells, s, e), freq, agg="sum")
    weather = resample(CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e), freq)
    lags = LAGS_BY_FREQ[freq]
    sup = build_supervised(demand, weather, lags=lags)
    print(f"\nreal London load + weather — {len(cells)} households, {s.date()}..{e.date()}, "
          f"freq={freq}, lags={lags}, rows={sup.height}")

    gbm = verification_frame(GBMForecaster(), sup, feature_cols(lags))
    print(f"  {'GBM (+ weather, lags)':32s} MAE {float(gbm['abs_error'].mean()):.3f}  "
          f"coverage {float(gbm['covered'].mean()):.2f}")
    ch = ChronosForecaster(device=device).verify(sup, covariates=BASE_FEATURES)
    print(f"  {'Chronos-2 (+ weather covariate)':32s} MAE {float(ch['abs_error'].mean()):.3f}  "
          f"coverage {float(ch['covered'].mean()):.2f}")


if __name__ == "__main__":
    typer.run(main)

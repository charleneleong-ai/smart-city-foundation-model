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
from sctwin.forecast.features import BASE_FEATURES, LAGS_BY_FREQ, build_supervised, feature_cols, resample
from sctwin.geo import cell_of
from sctwin.verify.results import verification_frame


def main(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD window start")] = "2023-01-07",
    end: Annotated[str, typer.Option(help="YYYY-MM-DD window end")] = "2023-01-21",
    region: Annotated[str, typer.Option(help="AEMO region (NSW1, VIC1, QLD1, SA1, TAS1)")] = "NSW1",
    res: Annotated[int, typer.Option(help="H3 resolution")] = 7,
    freq: Annotated[str, typer.Option(help="forecast frequency: hour|day|week|month")] = "hour",
    device: Annotated[str, typer.Option(help="chronos device (cpu/cuda)")] = "cpu",
) -> None:
    """Score GBM vs Chronos-2 on real AEMO regional demand + real Sydney weather."""
    if freq not in LAGS_BY_FREQ:
        raise typer.BadParameter(f"--freq must be one of {', '.join(LAGS_BY_FREQ)}")
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    cells = [cell_of(-33.87, 151.21, res)]  # one Sydney cell — AEMO demand is a single regional series
    demand = resample(AEMODemandAdapter(region=region).fetch(cells, s, e), freq, agg="sum")
    weather = resample(CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e), freq)
    lags = LAGS_BY_FREQ[freq]
    sup = build_supervised(demand, weather, lags=lags)
    print(f"\nreal AEMO {region} demand + Sydney weather — {s.date()}..{e.date()}, "
          f"freq={freq}, lags={lags}, rows={sup.height}")

    gbm = verification_frame(GBMForecaster(), sup, feature_cols(lags))
    print(f"  {'GBM (+ weather, lags)':32s} MAE {float(gbm['abs_error'].mean()):7.1f}  "
          f"coverage {float(gbm['covered'].mean()):.2f}")
    ch = ChronosForecaster(device=device).verify(sup, covariates=BASE_FEATURES)
    print(f"  {'Chronos-2 (+ weather covariate)':32s} MAE {float(ch['abs_error'].mean()):7.1f}  "
          f"coverage {float(ch['covered'].mean()):.2f}")


if __name__ == "__main__":
    typer.run(main)

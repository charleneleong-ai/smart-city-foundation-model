"""Combined GBM-vs-Chronos-2 comparison across regions (UK London + AU NSW) and frequencies
(hour/day/week/month). Answers: does the foundation model's advantage grow at coarser
resolution, and does that hold across two regions on opposite sides of the planet? Chronos-2 is
loaded once and reused across the whole sweep.

Run: uv run --extra forecast --extra tsfm python apps/eval_compare.py --freqs day,week,month
"""

from datetime import datetime, timezone
from typing import Annotated

import polars as pl
import typer

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.demand import AEMODemandAdapter, LondonSmartMeterAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.chronos import ChronosForecaster
from sctwin.forecast.features import (
    BASE_FEATURES,
    LAGS_BY_FREQ,
    build_supervised,
    feature_cols,
    regularize,
    resample,
)
from sctwin.geo import cell_of
from sctwin.verify.results import verification_frame

_MIN_ROWS = 20  # below this the coarse-resolution sample is too thin to report


def _utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _london(meters: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    cells = cells_in_bbox(51.40, -0.25, 51.60, 0.05, 7)[:meters]
    s, e = _utc(2012, 10, 15), _utc(2014, 2, 15)  # the Low Carbon London coverage (~16 months)
    demand = LondonSmartMeterAdapter().fetch(cells, s, e)
    weather = CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e)
    return demand, weather


def _nsw() -> tuple[pl.DataFrame, pl.DataFrame]:
    cells = [cell_of(-33.87, 151.21, 7)]
    s, e = _utc(2021, 1, 1), _utc(2023, 12, 31)  # 3 years — enough for the weekly/monthly seasonal lags
    demand = AEMODemandAdapter(region="NSW1").fetch(cells, s, e)
    weather = CachingAdapter(OpenMeteoWeatherAdapter(), ".cache/open-meteo").fetch(cells, s, e)
    return demand, weather


def main(
    freqs: Annotated[str, typer.Option(help="comma list of hour|day|week|month")] = "day,week,month",
    meters: Annotated[int, typer.Option(help="London households")] = 8,
    device: Annotated[str, typer.Option(help="chronos device (cpu/cuda)")] = "cpu",
) -> None:
    """Sweep {UK London, AU NSW} × {freqs}, scoring GBM vs Chronos-2 (loaded once)."""
    chosen = freqs.split(",")
    if any(f not in LAGS_BY_FREQ for f in chosen):
        raise typer.BadParameter(f"--freqs must be from {', '.join(LAGS_BY_FREQ)}")
    pipe = ChronosForecaster(device=device).pipeline()  # load amazon/chronos-2 once, reuse across the sweep

    print(f"\n{'region':12s} {'freq':6s} {'rows':>5s}  {'GBM MAE':>11s}  {'Chronos MAE':>12s}  winner")
    for region, (demand, weather) in (("UK London", _london(meters)), ("AU NSW", _nsw())):
        for freq in chosen:
            lags = LAGS_BY_FREQ[freq]
            d = regularize(resample(demand, freq, agg="sum"), freq)  # gap-free grid so Chronos predict_df works
            sup = build_supervised(d, resample(weather, freq), lags=lags)
            if sup.height < _MIN_ROWS:
                print(f"{region:12s} {freq:6s} {sup.height:>5d}  (insufficient history for the seasonal lag)")
                continue
            gbm = float(verification_frame(GBMForecaster(), sup, feature_cols(lags))["abs_error"].mean())
            try:
                chronos = float(ChronosForecaster(pipeline=pipe).verify(sup, covariates=BASE_FEATURES)["abs_error"].mean())
            except Exception as exc:  # a gappy/short series can trip predict_df — don't kill the sweep
                print(f"{region:12s} {freq:6s} {sup.height:>5d}  {gbm:>11.1f}  {'chronos: ' + type(exc).__name__:>12s}")
                continue
            win = "Chronos" if chronos < gbm else "GBM"
            gap = 100 * (1 - min(gbm, chronos) / max(gbm, chronos))
            print(f"{region:12s} {freq:6s} {sup.height:>5d}  {gbm:>11.1f}  {chronos:>12.1f}  {win} ({gap:.0f}%)")


if __name__ == "__main__":
    typer.run(main)

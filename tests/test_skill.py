import math
from datetime import datetime, timedelta, timezone

import polars as pl

from sctwin.forecast.skill import benchmark, climatology, persistence, skill


def _truth(days: int = 3, cells: tuple[str, ...] = ("a", "b"), trend: float = 0.0) -> pl.DataFrame:
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    rows = [
        {"cell": c, "time": t0 + timedelta(hours=h), "layer": "t2m",
         # diurnal cycle + an optional day-to-day warming trend (trend=0 -> perfectly periodic)
         "value": 10.0 + 6.0 * math.sin(2 * math.pi * h / 24) + trend * (h // 24)}
        for c in cells for h in range(24 * days)
    ]
    return pl.DataFrame(rows)


def test_skill_metrics_against_a_known_offset():
    truth = _truth(days=1)
    fc = truth.with_columns(pl.col("value") + 2.0)  # forecast biased +2°C everywhere
    s = skill(fc, truth)
    assert s["mae"] == 2.0 and s["bias"] == 2.0 and s["rmse"] == 2.0
    assert s["n"] == 48.0  # 2 cells x 24 h


def test_a_good_forecast_beats_persistence_and_climatology():
    truth = _truth(days=3, trend=2.0)  # warming trend -> naive baselines have real error
    fc = truth.with_columns(pl.col("value") + 0.3)  # near-perfect NWP (tiny error)
    table = benchmark(fc, truth)
    assert table["nwp forecast"]["mae"] < table["persistence (24 h)"]["mae"]
    assert table["nwp forecast"]["mae"] < table["climatology (hourly mean)"]["mae"]


def test_persistence_is_exact_on_a_daily_periodic_signal():
    truth = _truth(days=3)  # signal repeats every 24 h, so yesterday == today
    assert skill(persistence(truth), truth)["mae"] < 1e-9


def test_climatology_recovers_the_diurnal_mean_shape():
    truth = _truth(days=3)
    clim = climatology(truth)
    assert clim.height == truth.height
    assert skill(clim, truth)["mae"] < 1e-9  # constant-per-day signal -> hourly mean is exact

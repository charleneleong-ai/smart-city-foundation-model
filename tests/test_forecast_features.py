from datetime import datetime, timedelta, timezone

import polars as pl

from sctwin.forecast.features import FEATURE_COLS, build_supervised, to_xy


def _series(cell: str, layer: str, values: list[float]) -> pl.DataFrame:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "cell": [cell] * len(values),
            "time": [t0 + timedelta(hours=i) for i in range(len(values))],
            "layer": [layer] * len(values),
            "value": values,
        }
    )


def test_build_joins_and_adds_features_without_nulls():
    n = 48
    load = _series("a", "load", [float(i % 24) for i in range(n)])
    weather = _series("a", "t2m", [float(5 + (i % 10)) for i in range(n)])
    sup = build_supervised(load, weather)
    assert sup.height == n - 24  # 24h max lag drops the first 24 rows
    for col in FEATURE_COLS + ["y"]:
        assert col in sup.columns
        assert sup[col].null_count() == 0


def test_degree_days_are_one_sided():
    load = _series("a", "load", [1.0] * 30)
    cold = _series("a", "t2m", [0.0] * 30)  # below 18 -> HDD>0, CDD=0
    sup = build_supervised(load, cold)
    assert (sup["hdd"] > 0).all()
    assert (sup["cdd"] == 0).all()


def test_to_xy_shapes_match_feature_cols():
    load = _series("a", "load", [float(i % 24) for i in range(48)])
    weather = _series("a", "t2m", [float(5 + (i % 10)) for i in range(48)])
    sup = build_supervised(load, weather)
    x, y = to_xy(sup, FEATURE_COLS)
    assert x.shape == (sup.height, len(FEATURE_COLS))
    assert y.shape == (sup.height,)

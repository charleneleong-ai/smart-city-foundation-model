# SP4 (PR 1) — Energy Forecasting Baseline Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A leakage-safe forecasting harness on the canonical `(cell, time, layer, value)` schema, with weather→load features and the baselines a world model must beat — degree-day regression, gradient-boosted trees, and naive persistence.

**Architecture:** Build supervised features by joining a load target frame with a weather feature frame on `(cell, time)`, adding calendar + lag + degree-day features. Forecasters are pure `(X, y)` numpy estimators behind a `Forecaster` protocol, so the *caller* picks the feature subset (degree-day gets `[hdd, cdd]`; GBM gets all). A temporal backtest splits strictly by time (no leakage) and scores MAE/RMSE/MAPE against persistence.

**Tech Stack:** `polars` (features), `numpy` + `scikit-learn` (estimators), `pytest`. Why baselines first: per the design spec, a spatiotemporal world model earns its place only by beating these on public benchmarks (BDG2 / ASHRAE GEPIII). TS foundation models (TimesFM / Chronos) are the *next* SP4 PR — they need torch/GPU.

**Depends on:** SP1 (canonical schema). Stacked on `feat/sp1-ingestion-schema`.

---

## File Structure

- `pyproject.toml` — add `forecast` extra (numpy, scikit-learn)
- `src/sctwin/forecast/__init__.py`
- `src/sctwin/forecast/features.py` — `build_supervised`, `to_xy`, `FEATURE_COLS`
- `src/sctwin/forecast/baselines.py` — `Forecaster` protocol, `DegreeDayRegressor`, `GBMForecaster`, `PersistenceForecaster`
- `src/sctwin/forecast/backtest.py` — `temporal_split`, `metrics`, `backtest`, `BacktestResult`
- `tests/test_forecast_features.py`, `tests/test_forecast_baselines.py`, `tests/test_forecast_backtest.py`

---

### Task 1: `forecast` dependency group

- [ ] **Step 1: add the extra to `pyproject.toml`**

```toml
forecast = ["numpy>=1.26", "scikit-learn>=1.4"]
```

- [ ] **Step 2:** `uv sync --extra dev --extra forecast`  → installs numpy + sklearn.
- [ ] **Step 3:** commit `chore: add forecast dependency group (numpy, scikit-learn)`.

---

### Task 2: features (`features.py`)

Join target (`layer="load"`) with weather (`layer="t2m"`) on `(cell, time)`; add `hdd = max(0, 18 - t2m)`, `cdd = max(0, t2m - 18)`, calendar (`hour`, `dow`, `month`), and lags of `y` (`y_lag_1`, `y_lag_24`). Drop lag-boundary rows so there are no null features.

- [ ] **Step 1: failing test** — `tests/test_forecast_features.py`:

```python
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
    # 24h max lag drops the first 24 rows
    assert sup.height == n - 24
    for col in FEATURE_COLS + ["y"]:
        assert col in sup.columns
        assert sup[col].null_count() == 0


def test_degree_days_are_one_sided():
    load = _series("a", "load", [1.0] * 30)
    cold = _series("a", "t2m", [0.0] * 30)  # below 18 -> HDD>0, CDD=0
    sup = build_supervised(cold_target := load, cold)
    assert (sup["hdd"] > 0).all()
    assert (sup["cdd"] == 0).all()


def test_to_xy_shapes_match_feature_cols():
    load = _series("a", "load", [float(i % 24) for i in range(48)])
    weather = _series("a", "t2m", [float(5 + (i % 10)) for i in range(48)])
    sup = build_supervised(load, weather)
    X, y = to_xy(sup, FEATURE_COLS)
    assert X.shape == (sup.height, len(FEATURE_COLS))
    assert y.shape == (sup.height,)
```

- [ ] **Step 2:** run → FAIL (no module).
- [ ] **Step 3: implement** `src/sctwin/forecast/features.py`:

```python
from datetime import datetime  # noqa: F401  (kept for type clarity in callers)

import numpy as np
import polars as pl

BASE_TEMP_C = 18.0
LAGS = [1, 24]
FEATURE_COLS = ["t2m", "hdd", "cdd", "hour", "dow", "month", "y_lag_1", "y_lag_24"]


def build_supervised(target: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    tgt = target.select("cell", "time", pl.col("value").alias("y"))
    wx = weather.select("cell", "time", pl.col("value").alias("t2m"))
    df = tgt.join(wx, on=["cell", "time"], how="inner").sort("cell", "time")
    df = df.with_columns(
        (pl.max_horizontal(BASE_TEMP_C - pl.col("t2m"), 0.0)).alias("hdd"),
        (pl.max_horizontal(pl.col("t2m") - BASE_TEMP_C, 0.0)).alias("cdd"),
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.weekday().alias("dow"),
        pl.col("time").dt.month().alias("month"),
    )
    df = df.with_columns(
        *[pl.col("y").shift(lag).over("cell").alias(f"y_lag_{lag}") for lag in LAGS]
    )
    return df.drop_nulls(subset=[f"y_lag_{lag}" for lag in LAGS])


def to_xy(frame: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = frame.select(feature_cols).to_numpy()
    y = frame["y"].to_numpy()
    return x, y
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat: weather->load supervised features (calendar, lags, degree-days)`.

---

### Task 3: baselines (`baselines.py`)

- [ ] **Step 1: failing test** — `tests/test_forecast_baselines.py`:

```python
import numpy as np

from sctwin.forecast.baselines import (
    DegreeDayRegressor,
    GBMForecaster,
    PersistenceForecaster,
)


def _mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))


def test_degree_day_recovers_linear_signal():
    rng = np.random.default_rng(0)
    hdd = rng.uniform(0, 20, size=500)
    cdd = rng.uniform(0, 10, size=500)
    y = 3.0 * hdd + 1.5 * cdd + rng.normal(0, 0.1, size=500)
    X = np.column_stack([hdd, cdd])
    m = DegreeDayRegressor().fit(X, y)
    pred = m.predict(X)
    assert _mae(pred, y) < 0.5
    assert _mae(pred, y) < _mae(np.full_like(y, y.mean()), y)  # beats mean


def test_gbm_beats_persistence_on_nonlinear():
    rng = np.random.default_rng(1)
    n = 800
    lag = rng.uniform(0, 10, size=n)
    feat = rng.uniform(0, 10, size=n)
    y = np.sin(feat) * 5 + lag * 0.2 + rng.normal(0, 0.1, size=n)
    X = np.column_stack([feat, lag])
    cut = 600
    gbm = GBMForecaster().fit(X[:cut], y[:cut])
    gbm_mae = _mae(gbm.predict(X[cut:]), y[cut:])
    pers = PersistenceForecaster().fit(X[:cut, 1:2], y[:cut])  # lag column
    pers_mae = _mae(pers.predict(X[cut:, 1:2]), y[cut:])
    assert gbm_mae < pers_mae


def test_persistence_returns_lag_column():
    X = np.array([[2.0], [5.0], [9.0]])
    pred = PersistenceForecaster().fit(X, np.zeros(3)).predict(X)
    assert np.allclose(pred, [2.0, 5.0, 9.0])
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** `src/sctwin/forecast/baselines.py`:

```python
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression


@runtime_checkable
class Forecaster(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Forecaster": ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class DegreeDayRegressor:
    def __init__(self) -> None:
        self._lr = LinearRegression()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DegreeDayRegressor":
        self._lr.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._lr.predict(x)


class GBMForecaster:
    def __init__(self, **kw) -> None:
        self._m = HistGradientBoostingRegressor(**kw)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GBMForecaster":
        self._m.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._m.predict(x)


class PersistenceForecaster:
    """Predict the lag column passed as the single feature (naive baseline)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PersistenceForecaster":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0]
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat: forecasting baselines (degree-day, GBM, persistence)`.

---

### Task 4: backtest (`backtest.py`)

- [ ] **Step 1: failing test** — `tests/test_forecast_backtest.py`:

```python
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from sctwin.forecast.backtest import backtest, metrics, temporal_split
from sctwin.forecast.baselines import DegreeDayRegressor


def _frame(n: int) -> pl.DataFrame:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    hdd = rng.uniform(0, 20, size=n)
    return pl.DataFrame(
        {
            "cell": ["a"] * n,
            "time": [t0 + timedelta(hours=i) for i in range(n)],
            "y": 3.0 * hdd + rng.normal(0, 0.1, size=n),
            "hdd": hdd,
            "cdd": np.zeros(n),
        }
    )


def test_temporal_split_has_no_leakage():
    train, test = temporal_split(_frame(100), test_frac=0.25)
    assert train.height == 75 and test.height == 25
    assert train["time"].max() < test["time"].min()


def test_metrics_known_values():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["mae"] == 2.0 / 3.0
    assert m["rmse"] == (4.0 / 3.0) ** 0.5


def test_backtest_returns_metrics_and_predictions():
    res = backtest(DegreeDayRegressor(), _frame(200), ["hdd", "cdd"], test_frac=0.25)
    assert res.metrics["mae"] < 0.5
    assert len(res.predictions) == 50
    assert res.split_time is not None
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** `src/sctwin/forecast/backtest.py`:

```python
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from sctwin.forecast.baselines import Forecaster
from sctwin.forecast.features import to_xy


@dataclass(frozen=True)
class BacktestResult:
    metrics: dict[str, float]
    predictions: np.ndarray
    split_time: datetime


def temporal_split(frame: pl.DataFrame, test_frac: float = 0.25) -> tuple[pl.DataFrame, pl.DataFrame]:
    ordered = frame.sort("time")
    cut = int(round(ordered.height * (1 - test_frac)))
    return ordered[:cut], ordered[cut:]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    out = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2)))}
    nz = y_true != 0
    out["mape"] = float(np.mean(np.abs(err[nz] / y_true[nz]))) if nz.any() else float("nan")
    return out


def backtest(
    model: Forecaster, frame: pl.DataFrame, feature_cols: list[str], test_frac: float = 0.25
) -> BacktestResult:
    train, test = temporal_split(frame, test_frac)
    xtr, ytr = to_xy(train, feature_cols)
    xte, yte = to_xy(test, feature_cols)
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    return BacktestResult(metrics(yte, pred), pred, test["time"].min())
```

(`to_xy` reads `y` from the frame, so backtest frames must carry a `y` column — the
feature frames from Task 2 do.)

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat: leakage-safe temporal backtest + MAE/RMSE/MAPE`.

---

### Task 5: exports + full gate

- [ ] **Step 1:** `src/sctwin/forecast/__init__.py` exports `build_supervised`, `to_xy`, `FEATURE_COLS`, the three forecasters, `backtest`, `temporal_split`, `metrics`, `BacktestResult`.
- [ ] **Step 2:** `uv run pytest -q && uv run ruff check src tests && uv run mypy src` → all clean.
- [ ] **Step 3:** commit `feat: sctwin.forecast public exports`.

---

## Self-Review notes

- **Spec coverage (SP4 baselines rule):** degree-day ✅, GBM ✅, persistence ✅, leakage-safe backtest ✅. Public benchmark *loaders* (BDG2 / GEPIII) and TS-FMs are the **next** SP4 PR — flagged out-of-scope here.
- **No leakage:** `temporal_split` sorts by time and cuts by index, so every train timestamp precedes every test timestamp; `y_lag_*` features use only past values via `shift`.
- **Real energy data:** this PR proves the harness on synthetic + the SP1 weather adapter; a real BDG2/NREL load adapter lands with the benchmark PR (same `LayerAdapter` protocol, `layer="load"`).

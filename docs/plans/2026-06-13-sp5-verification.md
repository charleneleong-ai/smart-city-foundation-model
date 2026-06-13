# SP5 (PR 1) — Verification & Calibration Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Checkbox (`- [ ]`) steps.

**Goal:** Turn SP4 forecasts into *trustable, renderable* outputs: split-conformal prediction intervals with guaranteed coverage, a per-`(cell, time)` verification frame (predicted-vs-actual, error, interval, covered), and drift detection.

**Architecture:** Split-conformal calibration on a temporal holdout gives marginal-coverage intervals independent of model family. `verification_frame()` does a 3-way temporal split (train → calibrate → test), fits the model, calibrates conformal on the calib residuals, and emits a wide results frame keyed on `(cell, time)`. `as_layer()` projects any field to the canonical `(cell, time, layer, value)` schema so the existing H3 renderer (SP8) draws error/coverage/drift maps for free. Drift = empirical coverage falling below target over time buckets.

**Tech Stack:** `numpy` (moved to core), `polars`. Reuses SP4 `temporal_split` + the `Forecaster` protocol.

**Depends on:** SP1 (schema), SP4 (forecaster + split). Branch `feat/sp5-verification` off `main`.

---

## File Structure

- `pyproject.toml` — move `numpy` to core deps (verify + forecast both need it)
- `src/sctwin/verify/__init__.py`
- `src/sctwin/verify/conformal.py` — `ConformalCalibrator`, `coverage`
- `src/sctwin/verify/results.py` — `verification_frame`, `as_layer`, `RESULT_FIELDS`
- `src/sctwin/verify/drift.py` — `coverage_over_time`, `drift_flags`
- `tests/test_conformal.py`, `tests/test_verify_results.py`, `tests/test_drift.py`

---

### Task 1: move numpy to core

- [ ] **Step 1:** in `pyproject.toml`, add `"numpy>=1.26"` to `[project].dependencies`; set `forecast = ["scikit-learn>=1.4"]`.
- [ ] **Step 2:** `uv sync --extra dev --extra forecast --extra app` → resolves.
- [ ] **Step 3:** commit `chore: move numpy to core deps (verify + forecast need it)`.

---

### Task 2: split-conformal (`conformal.py`)

Split-conformal: on a calibration set, the `(1-alpha)` quantile (with finite-sample
correction `ceil((n+1)(1-alpha))/n`) of absolute residuals is the interval half-width;
applied to new predictions it gives marginal coverage ≥ `1-alpha`.

- [ ] **Step 1: failing test** — `tests/test_conformal.py`:

```python
import numpy as np

from sctwin.verify.conformal import ConformalCalibrator, coverage


def test_intervals_achieve_target_coverage_on_fresh_data():
    rng = np.random.default_rng(0)
    cal_resid = rng.normal(0, 1, 2000)
    cal_true = rng.normal(0, 1, 2000)
    cal_pred = cal_true + cal_resid  # residual = pred - true symmetric
    cab = ConformalCalibrator.fit(cal_true, cal_pred, alpha=0.1)
    # fresh data, same noise
    true = rng.normal(0, 1, 5000)
    pred = true + rng.normal(0, 1, 5000)
    lo, hi = cab.interval(pred)
    cov = coverage(true, lo, hi)
    assert 0.86 <= cov <= 0.94  # ~90% +/- sampling


def test_tighter_alpha_widens_interval():
    rng = np.random.default_rng(1)
    t = rng.normal(0, 1, 1000)
    p = t + rng.normal(0, 1, 1000)
    w90 = ConformalCalibrator.fit(t, p, alpha=0.1).quantile
    w99 = ConformalCalibrator.fit(t, p, alpha=0.01).quantile
    assert w99 > w90


def test_coverage_counts_points_inside():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 2.5, 2.0])  # 2.0 is below its lo
    hi = np.array([2.0, 3.5, 4.0])
    assert coverage(y, lo, hi) == 2.0 / 3.0
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** `src/sctwin/verify/conformal.py`:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalCalibrator:
    quantile: float

    @classmethod
    def fit(cls, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> "ConformalCalibrator":
        resid = np.abs(np.asarray(y_pred) - np.asarray(y_true))
        n = len(resid)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # finite-sample correction
        return cls(quantile=float(np.quantile(resid, level, method="higher")))

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y_pred)
        return y - self.quantile, y + self.quantile


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y_true)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat: split-conformal calibrator + coverage`.

---

### Task 3: verification frame (`results.py`)

3-way temporal split, fit, calibrate, evaluate; emit a wide `(cell, time)` frame.
`as_layer` projects a field to the canonical schema for the SP8 renderer.

- [ ] **Step 1: failing test** — `tests/test_verify_results.py`:

```python
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from sctwin.forecast.baselines import DegreeDayRegressor
from sctwin.verify.results import RESULT_FIELDS, as_layer, verification_frame


def _frame(n: int) -> pl.DataFrame:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    hdd = rng.uniform(0, 20, n)
    return pl.DataFrame(
        {
            "cell": ["a"] * n,
            "time": [t0 + timedelta(hours=i) for i in range(n)],
            "y": 3.0 * hdd + rng.normal(0, 0.3, n),
            "hdd": hdd,
            "cdd": np.zeros(n),
        }
    )


def test_results_frame_has_expected_columns_and_covered_logic():
    res = verification_frame(DegreeDayRegressor(), _frame(400), ["hdd", "cdd"], alpha=0.1)
    for col in ["cell", "time", "y_true", "y_pred", "abs_error", "lo", "hi", "covered"]:
        assert col in res.columns
    # covered iff y_true within [lo, hi]
    chk = res.with_columns(
        ((pl.col("y_true") >= pl.col("lo")) & (pl.col("y_true") <= pl.col("hi"))).alias("exp")
    )
    assert (chk["covered"] == chk["exp"]).all()


def test_empirical_coverage_near_target():
    res = verification_frame(DegreeDayRegressor(), _frame(600), ["hdd", "cdd"], alpha=0.1)
    assert 0.8 <= res["covered"].mean() <= 1.0


def test_as_layer_projects_to_canonical_schema():
    res = verification_frame(DegreeDayRegressor(), _frame(400), ["hdd", "cdd"])
    layer = as_layer(res, "abs_error")
    assert layer.columns == ["cell", "time", "layer", "value"]
    assert layer["layer"].unique().to_list() == ["abs_error"]


def test_as_layer_rejects_unknown_field():
    res = verification_frame(DegreeDayRegressor(), _frame(400), ["hdd", "cdd"])
    try:
        as_layer(res, "nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** `src/sctwin/verify/results.py`:

```python
import numpy as np
import polars as pl

from sctwin.forecast.backtest import temporal_split
from sctwin.forecast.baselines import Forecaster
from sctwin.forecast.features import to_xy
from sctwin.verify.conformal import ConformalCalibrator

RESULT_FIELDS = ["y_true", "y_pred", "abs_error", "lo", "hi", "covered"]


def verification_frame(
    model: Forecaster,
    frame: pl.DataFrame,
    feature_cols: list[str],
    *,
    calib_frac: float = 0.3,
    test_frac: float = 0.25,
    alpha: float = 0.1,
) -> pl.DataFrame:
    fit_part, test = temporal_split(frame, test_frac)
    train, calib = temporal_split(fit_part, calib_frac)

    xtr, ytr = to_xy(train, feature_cols)
    model.fit(xtr, ytr)

    xca, yca = to_xy(calib, feature_cols)
    cal = ConformalCalibrator.fit(yca, model.predict(xca), alpha=alpha)

    xte, yte = to_xy(test, feature_cols)
    pred = model.predict(xte)
    lo, hi = cal.interval(pred)
    return test.select("cell", "time").with_columns(
        pl.Series("y_true", yte),
        pl.Series("y_pred", pred),
        pl.Series("abs_error", np.abs(pred - yte)),
        pl.Series("lo", lo),
        pl.Series("hi", hi),
        pl.Series("covered", (yte >= lo) & (yte <= hi)),
    )


def as_layer(results: pl.DataFrame, field: str) -> pl.DataFrame:
    if field not in RESULT_FIELDS:
        raise ValueError(f"unknown field {field!r}; expected one of {RESULT_FIELDS}")
    return results.select(
        "cell", "time", pl.lit(field).alias("layer"), pl.col(field).alias("value")
    )
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat: per-(cell,time) verification frame + canonical as_layer projection`.

---

### Task 4: drift detection (`drift.py`)

Empirical coverage per time bucket; flag buckets where coverage falls below
`target_coverage - tol` (the twin has drifted from reality there).

- [ ] **Step 1: failing test** — `tests/test_drift.py`:

```python
from datetime import datetime, timedelta, timezone

import polars as pl

from sctwin.verify.drift import coverage_over_time, drift_flags


def _results(covered_pattern: list[bool]) -> pl.DataFrame:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "cell": ["a"] * len(covered_pattern),
            "time": [t0 + timedelta(hours=i) for i in range(len(covered_pattern))],
            "covered": covered_pattern,
        }
    )


def test_coverage_over_time_buckets():
    res = _results([True] * 6 + [False] * 6)
    cov = coverage_over_time(res, every="3h")
    assert cov.height == 4  # 12h / 3h
    assert cov["coverage"].to_list() == [1.0, 1.0, 0.0, 0.0]


def test_drift_flags_low_coverage_windows():
    res = _results([True] * 6 + [False] * 6)
    flagged = drift_flags(res, target_coverage=0.9, tol=0.1, every="3h")
    # first two windows ok (1.0), last two drifted (0.0 < 0.8)
    assert flagged["drift"].to_list() == [False, False, True, True]
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** `src/sctwin/verify/drift.py`:

```python
import polars as pl


def coverage_over_time(results: pl.DataFrame, *, every: str = "1d") -> pl.DataFrame:
    return (
        results.sort("time")
        .group_by_dynamic("time", every=every)
        .agg(pl.col("covered").mean().alias("coverage"))
    )


def drift_flags(
    results: pl.DataFrame, *, target_coverage: float, tol: float = 0.1, every: str = "1d"
) -> pl.DataFrame:
    cov = coverage_over_time(results, every=every)
    return cov.with_columns((pl.col("coverage") < target_coverage - tol).alias("drift"))
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat: drift detection over time buckets`.

---

### Task 5: exports + gate

- [ ] **Step 1:** `src/sctwin/verify/__init__.py` exports `ConformalCalibrator`, `coverage`, `verification_frame`, `as_layer`, `RESULT_FIELDS`, `coverage_over_time`, `drift_flags`.
- [ ] **Step 2:** `uv run pytest -q && uv run ruff check src tests && uv run mypy src` → clean.
- [ ] **Step 3:** commit `feat: sctwin.verify public exports`.

---

## Self-Review notes

- **Spec coverage (SP5):** conformal calibration ✅, predicted-vs-actual results frame ✅, coverage metric ✅, drift ✅. RLVR verifiable-reward training and physics-simulator oracle are the *next* SP5 work — out of scope here (this PR is the calibration + results-data spine).
- **Renderable by construction:** `as_layer` yields the exact `(cell, time, layer, value)` schema SP8's `h3_layer_records` consumes — the accuracy/verify map is then a no-new-viz reuse.
- **Conformal caveat:** temporal split breaks strict exchangeability; split-conformal on a holdout is the standard pragmatic choice and documented as such.

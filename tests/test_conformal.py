import numpy as np

from sctwin.verify.conformal import ConformalCalibrator, coverage


def test_intervals_achieve_target_coverage_on_fresh_data():
    rng = np.random.default_rng(0)
    cal_true = rng.normal(0, 1, 2000)
    cal_pred = cal_true + rng.normal(0, 1, 2000)
    cab = ConformalCalibrator.fit(cal_true, cal_pred, alpha=0.1)
    true = rng.normal(0, 1, 5000)
    pred = true + rng.normal(0, 1, 5000)
    lo, hi = cab.interval(pred)
    assert 0.86 <= coverage(true, lo, hi) <= 0.94  # ~90% +/- sampling


def test_tighter_alpha_widens_interval():
    rng = np.random.default_rng(1)
    t = rng.normal(0, 1, 1000)
    p = t + rng.normal(0, 1, 1000)
    w90 = ConformalCalibrator.fit(t, p, alpha=0.1).quantile
    w99 = ConformalCalibrator.fit(t, p, alpha=0.01).quantile
    assert w99 > w90


def test_coverage_counts_points_inside():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 2.5, 2.0])  # middle point's true (2.0) is below its lo (2.5)
    hi = np.array([2.0, 3.5, 4.0])
    assert coverage(y, lo, hi) == 2.0 / 3.0

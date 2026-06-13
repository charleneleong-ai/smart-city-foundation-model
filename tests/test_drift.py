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
    cov = coverage_over_time(_results([True] * 6 + [False] * 6), every="3h")
    assert cov.height == 4  # 12h / 3h
    assert cov["coverage"].to_list() == [1.0, 1.0, 0.0, 0.0]


def test_drift_flags_low_coverage_windows():
    flagged = drift_flags(
        _results([True] * 6 + [False] * 6), target_coverage=0.9, tol=0.1, every="3h"
    )
    # first two windows ok (1.0); last two drifted (0.0 < 0.8)
    assert flagged["drift"].to_list() == [False, False, True, True]

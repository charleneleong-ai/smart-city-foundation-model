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
    """Flag time buckets where empirical coverage falls below target_coverage - tol:
    the twin's calibrated intervals no longer hold there, so it has drifted from reality."""
    cov = coverage_over_time(results, every=every)
    return cov.with_columns((pl.col("coverage") < target_coverage - tol).alias("drift"))

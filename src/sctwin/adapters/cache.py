from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.geo import Cell
from sctwin.schema import empty_frame, validate_frame


class CachingAdapter:
    """Wrap a LayerAdapter with an on-disk parquet cache keyed on (cell, day).

    Only cells not already cached for the requested days are fetched from the inner
    adapter; everything else is served from disk. Lets repeated runs (and progressive
    panning around a centre) reuse downloads instead of re-hitting the API.
    """

    def __init__(self, inner: LayerAdapter, cache_dir: str | Path) -> None:
        self.name = inner.name
        self._inner = inner
        self._path = Path(cache_dir) / f"{inner.name}.parquet"

    def _load(self) -> pl.DataFrame:
        return pl.read_parquet(self._path) if self._path.exists() else empty_frame()

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        cached = self._load()
        days = [start.date() + timedelta(days=i) for i in range((end.date() - start.date()).days + 1)]
        have: set[tuple[str, object]] = set()
        if cached.height:
            hv = cached.select("cell", pl.col("time").dt.date().alias("d")).unique()
            have = set(zip(hv["cell"].to_list(), hv["d"].to_list(), strict=True))

        missing = [c for c in cells if any((c.h3, d) not in have for d in days)]
        if missing:
            cached = pl.concat([cached, self._inner.fetch(missing, start, end)]).unique(
                ["cell", "time", "layer"], keep="last"  # multi-layer adapters: 1 row per (cell,time,layer)
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            cached.write_parquet(self._path)

        want = [c.h3 for c in cells]
        out = cached.filter(pl.col("cell").is_in(want) & pl.col("time").dt.date().is_in(days))
        return validate_frame(out)

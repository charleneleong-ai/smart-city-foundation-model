from datetime import datetime
from typing import Protocol, runtime_checkable

import polars as pl

from sctwin.geo import Cell


@runtime_checkable
class LayerAdapter(Protocol):
    name: str

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        """Return a canonical frame: columns cell, time, layer, value."""
        ...

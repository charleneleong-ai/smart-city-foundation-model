from datetime import datetime

import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.geo import Cell


class Registry:
    def __init__(self) -> None:
        self._adapters: dict[str, LayerAdapter] = {}

    def register(self, adapter: LayerAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"adapter {adapter.name!r} already registered")
        self._adapters[adapter.name] = adapter

    def get(self, layer: str, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        if layer not in self._adapters:
            raise KeyError(f"no adapter for layer {layer!r}")
        return self._adapters[layer].fetch(cells, start, end)

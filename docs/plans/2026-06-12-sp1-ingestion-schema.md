# SP1 — Ingestion & Canonical Spatiotemporal Schema — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A typed, tested foundation that pulls open urban data through pluggable adapters and normalizes it to one canonical schema keyed on `(H3 cell, UTC time, layer)`.

**Architecture:** Everything in the twin joins on `(cell, time)`. SP1 provides: (1) H3 spatial keying helpers, (2) a pydantic-validated canonical record + a typed DataFrame contract, (3) a `LayerAdapter` protocol, (4) one real no-auth adapter (Open-Meteo weather), and (5) a `Registry` that routes `get(layer, cells, time_range)` to the right adapter. Region-specific datasets register as additional adapters later without touching the core.

**Tech Stack:** Python 3.11, `uv`, `h3` (spatial indexing), `pydantic` v2 (schema), `polars` (canonical frames), `httpx` (HTTP), `pytest` + `respx` (mock HTTP), `ruff` + `mypy`.

**Roadmap context:** This is the first of three PRs for the energy vertical — SP1 (this) → SP4 (energy world model) → SP5 (verification harness). SP4/SP5 get their own plans once SP1 lands.

---

## File Structure

- `pyproject.toml` — package metadata, deps, ruff/mypy/pytest config
- `src/sctwin/__init__.py` — public exports
- `src/sctwin/geo.py` — H3 keying: `cell_of`, `center_of`, `Cell`
- `src/sctwin/schema.py` — `LayerRecord` (pydantic), canonical polars schema + validation
- `src/sctwin/adapters/base.py` — `LayerAdapter` protocol
- `src/sctwin/adapters/open_meteo.py` — `OpenMeteoWeatherAdapter`
- `src/sctwin/registry.py` — `Registry.register` / `Registry.get`
- `tests/test_geo.py`, `tests/test_schema.py`, `tests/test_open_meteo.py`, `tests/test_registry.py`

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/sctwin/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "sctwin"
version = "0.0.1"
description = "Smart-city digital twin — ingestion & canonical spatiotemporal schema"
requires-python = ">=3.11"
dependencies = [
    "h3>=4.1",
    "pydantic>=2.7",
    "polars>=1.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "respx>=0.21", "ruff>=0.5", "mypy>=1.10"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sctwin"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
```

- [ ] **Step 2: Create empty package init**

```python
# src/sctwin/__init__.py
```

- [ ] **Step 3: Sync and verify install**

Run: `uv sync --extra dev`
Expected: resolves and installs without error; `.venv/` created.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/sctwin/__init__.py
git commit -m "chore: sctwin project scaffold"
```

---

### Task 2: H3 spatial keying (`geo.py`)

**Files:**
- Create: `src/sctwin/geo.py`
- Test: `tests/test_geo.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_geo.py
import pytest
from sctwin.geo import Cell, cell_of, center_of

def test_cell_of_is_stable():
    # Central London, resolution 9
    c1 = cell_of(51.5074, -0.1278, res=9)
    c2 = cell_of(51.5074, -0.1278, res=9)
    assert c1 == c2
    assert isinstance(c1.h3, str)
    assert c1.res == 9

def test_center_round_trips_within_cell():
    c = cell_of(51.5074, -0.1278, res=9)
    lat, lon = center_of(c)
    # center of the same cell must map back to the same cell
    assert cell_of(lat, lon, res=9) == c

def test_resolution_changes_cell():
    assert cell_of(51.5074, -0.1278, res=7) != cell_of(51.5074, -0.1278, res=9)

@pytest.mark.parametrize("res", [-1, 16])
def test_invalid_resolution_rejected(res):
    with pytest.raises(ValueError):
        cell_of(51.5074, -0.1278, res=res)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_geo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.geo'`

- [ ] **Step 3: Implement `geo.py`**

```python
# src/sctwin/geo.py
from dataclasses import dataclass

import h3


@dataclass(frozen=True)
class Cell:
    h3: str
    res: int


def cell_of(lat: float, lon: float, res: int) -> Cell:
    if not 0 <= res <= 15:
        raise ValueError(f"H3 resolution must be 0..15, got {res}")
    return Cell(h3=h3.latlng_to_cell(lat, lon, res), res=res)


def center_of(cell: Cell) -> tuple[float, float]:
    lat, lon = h3.cell_to_latlng(cell.h3)
    return lat, lon
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_geo.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/geo.py tests/test_geo.py
git commit -m "feat: H3 spatial keying (cell_of/center_of)"
```

---

### Task 3: Canonical schema (`schema.py`)

**Files:**
- Create: `src/sctwin/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schema.py
from datetime import datetime, timezone

import polars as pl
import pytest
from pydantic import ValidationError

from sctwin.schema import CANONICAL_COLUMNS, LayerRecord, validate_frame


def test_record_coerces_time_to_utc():
    rec = LayerRecord(
        cell="891f1d4894bffff", time=datetime(2020, 1, 1, 12), layer="t2m", value=4.5
    )
    assert rec.time.tzinfo == timezone.utc


def test_record_rejects_missing_value():
    with pytest.raises(ValidationError):
        LayerRecord(cell="891f1d4894bffff", time=datetime(2020, 1, 1), layer="t2m")


def test_validate_frame_accepts_canonical_columns():
    df = pl.DataFrame(
        {
            "cell": ["891f1d4894bffff"],
            "time": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
            "layer": ["t2m"],
            "value": [4.5],
        }
    )
    out = validate_frame(df)
    assert out.columns == CANONICAL_COLUMNS


def test_validate_frame_rejects_missing_column():
    df = pl.DataFrame({"cell": ["x"], "time": [datetime(2020, 1, 1)], "value": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(df)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.schema'`

- [ ] **Step 3: Implement `schema.py`**

```python
# src/sctwin/schema.py
from datetime import datetime, timezone

import polars as pl
from pydantic import BaseModel, field_validator

CANONICAL_COLUMNS = ["cell", "time", "layer", "value"]


class LayerRecord(BaseModel):
    cell: str
    time: datetime
    layer: str
    value: float

    @field_validator("time")
    @classmethod
    def _to_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def validate_frame(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df.select(CANONICAL_COLUMNS)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/schema.py tests/test_schema.py
git commit -m "feat: canonical (cell,time,layer,value) schema + frame validation"
```

---

### Task 4: Adapter protocol (`adapters/base.py`)

**Files:**
- Create: `src/sctwin/adapters/__init__.py`
- Create: `src/sctwin/adapters/base.py`
- Test: `tests/test_registry.py` (covers the protocol via a fake adapter in Task 6)

- [ ] **Step 1: Implement the protocol (no separate test — exercised in Task 6)**

```python
# src/sctwin/adapters/__init__.py
```

```python
# src/sctwin/adapters/base.py
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
```

- [ ] **Step 2: Commit**

```bash
git add src/sctwin/adapters/
git commit -m "feat: LayerAdapter protocol"
```

---

### Task 5: Open-Meteo weather adapter (`adapters/open_meteo.py`)

**Files:**
- Create: `src/sctwin/adapters/open_meteo.py`
- Test: `tests/test_open_meteo.py`

Open-Meteo archive API is free and keyless. Endpoint:
`https://archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&hourly=temperature_2m`
Response shape: `{"hourly": {"time": ["2020-01-01T00:00", ...], "temperature_2m": [4.5, ...]}}`.

- [ ] **Step 1: Write the failing test (mocked HTTP — no network)**

```python
# tests/test_open_meteo.py
from datetime import datetime, timezone

import httpx
import respx

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.geo import cell_of

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@respx.mock
def test_fetch_maps_response_to_canonical_frame():
    respx.get(ARCHIVE).mock(
        return_value=httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2020-01-01T00:00", "2020-01-01T01:00"],
                    "temperature_2m": [4.5, 4.1],
                }
            },
        )
    )
    cell = cell_of(51.5074, -0.1278, res=7)
    df = OpenMeteoWeatherAdapter().fetch(
        [cell], datetime(2020, 1, 1), datetime(2020, 1, 1)
    )
    assert df.columns == ["cell", "time", "layer", "value"]
    assert df.height == 2
    assert df["layer"].unique().to_list() == ["t2m"]
    assert df["cell"].unique().to_list() == [cell.h3]
    assert df["value"].to_list() == [4.5, 4.1]
    assert df["time"][0] == datetime(2020, 1, 1, 0, tzinfo=timezone.utc)


@respx.mock
def test_fetch_multiple_cells_concats():
    respx.get(ARCHIVE).mock(
        return_value=httpx.Response(
            200,
            json={"hourly": {"time": ["2020-01-01T00:00"], "temperature_2m": [3.0]}},
        )
    )
    cells = [cell_of(51.5, -0.1, res=7), cell_of(48.85, 2.35, res=7)]
    df = OpenMeteoWeatherAdapter().fetch(cells, datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert df.height == 2
    assert set(df["cell"].to_list()) == {c.h3 for c in cells}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_open_meteo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.adapters.open_meteo'`

- [ ] **Step 3: Implement the adapter**

```python
# src/sctwin/adapters/open_meteo.py
from datetime import datetime

import httpx
import polars as pl

from sctwin.geo import Cell, center_of
from sctwin.schema import validate_frame

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoWeatherAdapter:
    name = "weather.t2m"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0)

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        frames = [self._fetch_one(cell, start, end) for cell in cells]
        return validate_frame(pl.concat(frames)) if frames else _empty()

    def _fetch_one(self, cell: Cell, start: datetime, end: datetime) -> pl.DataFrame:
        lat, lon = center_of(cell)
        resp = self._client.get(
            ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "hourly": "temperature_2m",
            },
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        return pl.DataFrame(
            {
                "cell": cell.h3,
                "time": pl.Series(hourly["time"]).str.to_datetime(time_zone="UTC"),
                "layer": "t2m",
                "value": hourly["temperature_2m"],
            }
        )


def _empty() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"cell": pl.String, "time": pl.Datetime("us", "UTC"), "layer": pl.String, "value": pl.Float64}
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_open_meteo.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/adapters/open_meteo.py tests/test_open_meteo.py
git commit -m "feat: Open-Meteo weather adapter (t2m) -> canonical frame"
```

---

### Task 6: Registry (`registry.py`)

**Files:**
- Create: `src/sctwin/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests (with a fake adapter that asserts the protocol)**

```python
# tests/test_registry.py
from datetime import datetime, timezone

import polars as pl
import pytest

from sctwin.adapters.base import LayerAdapter
from sctwin.geo import cell_of
from sctwin.registry import Registry


class _FakeAdapter:
    name = "weather.t2m"

    def fetch(self, cells, start, end) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "cell": [c.h3 for c in cells],
                "time": [datetime(2020, 1, 1, tzinfo=timezone.utc)] * len(cells),
                "layer": ["t2m"] * len(cells),
                "value": [1.0] * len(cells),
            }
        )


def test_fake_adapter_satisfies_protocol():
    assert isinstance(_FakeAdapter(), LayerAdapter)


def test_get_routes_to_registered_adapter():
    reg = Registry()
    reg.register(_FakeAdapter())
    cell = cell_of(51.5, -0.1, res=7)
    df = reg.get("weather.t2m", [cell], datetime(2020, 1, 1), datetime(2020, 1, 1))
    assert df["cell"].to_list() == [cell.h3]


def test_get_unknown_layer_raises():
    reg = Registry()
    with pytest.raises(KeyError, match="no adapter"):
        reg.get("missing", [], datetime(2020, 1, 1), datetime(2020, 1, 1))


def test_register_duplicate_raises():
    reg = Registry()
    reg.register(_FakeAdapter())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_FakeAdapter())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.registry'`

- [ ] **Step 3: Implement `registry.py`**

```python
# src/sctwin/registry.py
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/registry.py tests/test_registry.py
git commit -m "feat: adapter Registry (register/get routing)"
```

---

### Task 7: Public exports + full test run

**Files:**
- Modify: `src/sctwin/__init__.py`

- [ ] **Step 1: Add public exports**

```python
# src/sctwin/__init__.py
from sctwin.geo import Cell, cell_of, center_of
from sctwin.registry import Registry
from sctwin.schema import CANONICAL_COLUMNS, LayerRecord, validate_frame

__all__ = [
    "Cell",
    "cell_of",
    "center_of",
    "Registry",
    "CANONICAL_COLUMNS",
    "LayerRecord",
    "validate_frame",
]
```

- [ ] **Step 2: Run the full suite + lint + types**

Run: `uv run pytest -q && uv run ruff check src tests && uv run mypy src`
Expected: all tests pass, ruff clean, mypy clean.

- [ ] **Step 3: Commit**

```bash
git add src/sctwin/__init__.py
git commit -m "feat: public sctwin exports"
```

---

## Self-Review notes

- **Spec coverage (SP1 row):** canonical spatiotemporal schema ✅ (Task 3, H3 cell+UTC), pluggable adapters ✅ (Task 4 protocol + Task 6 registry), open no-auth data source ✅ (Task 5 Open-Meteo). Sentinel/ERA5/OSM adapters are deliberately deferred to follow-up adapters — the protocol makes them additive.
- **Provenance/uncertainty** (a cross-cutting spec concern) is *not* in SP1's minimal schema — tracked as a follow-up: add `source` + `confidence` columns once a second adapter exists to disambiguate. Noted in the PR's out-of-scope.
- **Privacy/k-anonymity** applies at *release* time, not raw ingestion — belongs to SP5, not SP1.

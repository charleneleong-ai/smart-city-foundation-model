# SP8 — Serving & Visualization App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the twin over HTTP and render it on an H3 hex map in the browser — including the verification view (predicted-vs-actual, drift, calibration) and the reasoning trace.

**Architecture:** A FastAPI service exposes the canonical `(cell, time, layer, value)` frames and twin queries; a deck.gl `H3HexagonLayer` over a MapLibre basemap renders them. Because the whole twin is H3-keyed, rendering needs **no geometry conversion** — hex ids go straight to the GPU. The app is built in two phases: **Phase A** (buildable now on SP1 data — render the Open-Meteo weather layer) and **Phase B** (full twin serving, depends on SP4 world model + SP5 verification).

**Tech Stack:** `fastapi` + `uvicorn` (serve), `polars` → Arrow/Parquet (transport), `pydeck` (Python-side render for Phase A), `deck.gl` `H3HexagonLayer` + `MapLibre GL` (web), Hugging Face Spaces or Streamlit/Gradio (demo hosting). Tests: `pytest` + `fastapi.testclient`.

**Dependency note:** Phase A depends only on **SP1** (merged/PR #1). Phase B tasks are outlined but blocked on SP4/SP5 and get their own detailed steps once those land.

---

## File Structure

- `pyproject.toml` — add `app` optional-dependency group (fastapi, uvicorn, pydeck)
- `src/sctwin/app/__init__.py`
- `src/sctwin/app/service.py` — FastAPI app + routes
- `src/sctwin/app/render.py` — frame → deck.gl H3 layer spec (pure, testable)
- `src/sctwin/app/cells.py` — helpers: bbox → covering H3 cells (for a city viewport)
- `tests/test_app_render.py`, `tests/test_app_service.py`
- `apps/web/` — (Phase B) deck.gl + MapLibre frontend (React/Vite), out of scope for Phase A

---

## Phase A — Render SP1 data (buildable now)

### Task 1: App dependency group

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add an `app` optional-dependency group**

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "respx>=0.21", "ruff>=0.5", "mypy>=1.10"]
app = ["fastapi>=0.115", "uvicorn>=0.32", "pydeck>=0.9"]
```

- [ ] **Step 2: Sync**

Run: `uv sync --extra dev --extra app`
Expected: resolves and installs fastapi/uvicorn/pydeck.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add app dependency group (fastapi, uvicorn, pydeck)"
```

---

### Task 2: bbox → covering H3 cells (`app/cells.py`)

**Files:**
- Create: `src/sctwin/app/__init__.py`
- Create: `src/sctwin/app/cells.py`
- Test: `tests/test_app_cells.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app_cells.py
from sctwin.app.cells import cells_in_bbox
from sctwin.geo import center_of


def test_cells_cover_bbox_and_lie_inside():
    # small bbox around central London
    cells = cells_in_bbox(south=51.50, west=-0.13, north=51.52, east=-0.10, res=8)
    assert len(cells) > 0
    # every returned cell's center is within the bbox (with a small margin)
    for c in cells:
        lat, lon = center_of(c)
        assert 51.49 <= lat <= 51.53
        assert -0.14 <= lon <= -0.09


def test_finer_resolution_returns_more_cells():
    box = dict(south=51.50, west=-0.13, north=51.52, east=-0.10)
    assert len(cells_in_bbox(**box, res=9)) > len(cells_in_bbox(**box, res=7))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_app_cells.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.app.cells'`

- [ ] **Step 3: Implement `cells.py`**

```python
# src/sctwin/app/cells.py
import h3

from sctwin.geo import Cell


def cells_in_bbox(south: float, west: float, north: float, east: float, res: int) -> list[Cell]:
    if not 0 <= res <= 15:
        raise ValueError(f"H3 resolution must be 0..15, got {res}")
    poly = h3.LatLngPoly(
        [(south, west), (south, east), (north, east), (north, west)]
    )
    return [Cell(h3=cell, res=res) for cell in h3.polygon_to_cells(poly, res)]
```

```python
# src/sctwin/app/__init__.py
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_app_cells.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/app/__init__.py src/sctwin/app/cells.py tests/test_app_cells.py
git commit -m "feat: bbox -> covering H3 cells for map viewports"
```

---

### Task 3: frame → deck.gl H3 layer spec (`app/render.py`)

**Files:**
- Create: `src/sctwin/app/render.py`
- Test: `tests/test_app_render.py`

Renders a canonical frame at a single timestamp into a deck.gl-ready record list:
`[{"cell": "...", "value": 4.5, "color": [r,g,b,a]}, ...]`. Color = linear ramp over
the value range; alpha is a fixed default (uncertainty wiring is Phase B).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app_render.py
from datetime import datetime, timezone

import polars as pl

from sctwin.app.render import h3_layer_records


def _frame() -> pl.DataFrame:
    t = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "cell": ["a", "b", "c"],
            "time": [t, t, t],
            "layer": ["t2m"] * 3,
            "value": [0.0, 5.0, 10.0],
        }
    )


def test_records_one_per_cell_with_color():
    recs = h3_layer_records(_frame(), at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert [r["cell"] for r in recs] == ["a", "b", "c"]
    assert all(len(r["color"]) == 4 for r in recs)
    # min value -> low end of ramp, max value -> high end
    assert recs[0]["color"] != recs[2]["color"]


def test_filters_to_requested_timestamp():
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2020, 1, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {"cell": ["a", "a"], "time": [t0, t1], "layer": ["t2m", "t2m"], "value": [1.0, 9.0]}
    )
    recs = h3_layer_records(df, at=t1)
    assert len(recs) == 1
    assert recs[0]["value"] == 9.0


def test_constant_field_does_not_divide_by_zero():
    t = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame({"cell": ["a", "b"], "time": [t, t], "layer": ["t2m"] * 2, "value": [3.0, 3.0]})
    recs = h3_layer_records(df, at=t)
    assert len(recs) == 2  # no crash on zero range
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_app_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.app.render'`

- [ ] **Step 3: Implement `render.py`**

```python
# src/sctwin/app/render.py
from datetime import datetime

import polars as pl

RGBA = tuple[int, int, int, int]


def _ramp(t: float) -> RGBA:
    # simple blue -> red linear ramp, fixed alpha
    t = max(0.0, min(1.0, t))
    return (int(255 * t), 40, int(255 * (1 - t)), 160)


def h3_layer_records(frame: pl.DataFrame, at: datetime) -> list[dict]:
    snap = frame.filter(pl.col("time") == at)
    if snap.height == 0:
        return []
    vmin = snap["value"].min()
    vmax = snap["value"].max()
    span = (vmax - vmin) or 1.0
    return [
        {"cell": row["cell"], "value": row["value"], "color": list(_ramp((row["value"] - vmin) / span))}
        for row in snap.iter_rows(named=True)
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_app_render.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/app/render.py tests/test_app_render.py
git commit -m "feat: canonical frame -> deck.gl H3 layer records (value color ramp)"
```

---

### Task 4: FastAPI service (`app/service.py`)

**Files:**
- Create: `src/sctwin/app/service.py`
- Test: `tests/test_app_service.py`

Exposes a `/layer` endpoint that takes a bbox + time + layer, fetches via the SP1
`Registry`, and returns deck.gl-ready records. The registry is injected so tests can
use a fake adapter (no network).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app_service.py
from datetime import datetime, timezone

import polars as pl
from fastapi.testclient import TestClient

from sctwin.app.service import build_app
from sctwin.registry import Registry


class _FakeWeather:
    name = "weather.t2m"

    def fetch(self, cells, start, end) -> pl.DataFrame:
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        return pl.DataFrame(
            {
                "cell": [c.h3 for c in cells],
                "time": [t] * len(cells),
                "layer": ["t2m"] * len(cells),
                "value": [float(i) for i in range(len(cells))],
            }
        )


def _client() -> TestClient:
    reg = Registry()
    reg.register(_FakeWeather())
    return TestClient(build_app(reg))


def test_layer_endpoint_returns_colored_records():
    resp = _client().get(
        "/layer",
        params={
            "layer": "weather.t2m",
            "south": 51.50, "west": -0.13, "north": 51.52, "east": -0.10,
            "res": 8, "date": "2020-01-01",
        },
    )
    assert resp.status_code == 200
    recs = resp.json()
    assert len(recs) > 0
    assert {"cell", "value", "color"} <= set(recs[0])


def test_unknown_layer_returns_404():
    resp = _client().get(
        "/layer",
        params={"layer": "nope", "south": 51.5, "west": -0.13, "north": 51.52,
                "east": -0.10, "res": 8, "date": "2020-01-01"},
    )
    assert resp.status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_app_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sctwin.app.service'`

- [ ] **Step 3: Implement `service.py`**

```python
# src/sctwin/app/service.py
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry


def build_app(registry: Registry) -> FastAPI:
    app = FastAPI(title="sctwin")

    @app.get("/layer")
    def layer(
        layer: str, south: float, west: float, north: float, east: float, res: int, date: str
    ) -> list[dict]:
        cells = cells_in_bbox(south=south, west=west, north=north, east=east, res=res)
        at = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        try:
            frame = registry.get(layer, cells, at, at)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return h3_layer_records(frame, at=at)

    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_app_service.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sctwin/app/service.py tests/test_app_service.py
git commit -m "feat: FastAPI /layer endpoint -> deck.gl H3 records (registry-injected)"
```

---

### Task 5: Runnable demo (pydeck notebook/script + uvicorn entry)

**Files:**
- Create: `apps/demo_weather.py`
- Modify: `README.md` (run instructions)

- [ ] **Step 1: Write a pydeck demo that renders real Open-Meteo data**

```python
# apps/demo_weather.py
"""Render one day of Open-Meteo t2m over a city's H3 grid. Run: uv run python apps/demo_weather.py"""
from datetime import datetime, timezone

import pydeck as pdk

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry

reg = Registry()
reg.register(OpenMeteoWeatherAdapter())

cells = cells_in_bbox(south=51.46, west=-0.20, north=51.55, east=-0.05, res=7)
day = datetime(2020, 1, 15, tzinfo=timezone.utc)
frame = reg.get("weather.t2m", cells, day, day)
records = h3_layer_records(frame, at=datetime(2020, 1, 15, 12, tzinfo=timezone.utc))

layer = pdk.Layer(
    "H3HexagonLayer",
    records,
    get_hexagon="cell",
    get_fill_color="color",
    pickable=True,
    extruded=False,
)
view = pdk.ViewState(latitude=51.5, longitude=-0.12, zoom=10)
pdk.Deck(layers=[layer], initial_view_state=view, map_style="light").to_html("weather_map.html")
print("wrote weather_map.html")
```

- [ ] **Step 2: Run the demo (real network call to Open-Meteo)**

Run: `uv run --extra app python apps/demo_weather.py`
Expected: prints `wrote weather_map.html`; opening it shows a colored H3 hex grid over London.

- [ ] **Step 3: Add README run section**

```markdown
## Run the map demo

\`\`\`bash
uv sync --extra app
uv run python apps/demo_weather.py   # -> weather_map.html (H3 weather map)
uv run uvicorn --factory sctwin.app.service:build_app  # if wiring a live registry
\`\`\`
```

- [ ] **Step 4: Commit**

```bash
git add apps/demo_weather.py README.md
git commit -m "feat: pydeck weather map demo over real Open-Meteo H3 grid"
```

---

## Phase B — Full twin serving (blocked on SP4 + SP5)

Detailed steps are written once SP4/SP5 land; the routes and panels are fixed now so
the frontend can be built against a stable contract.

- **`POST /scenario`** — body `{intervention, horizon}` → calls L4 `rollout`, returns
  trajectory + conformal intervals. *Blocked on SP4.*
- **`GET /backtest`** — predicted-vs-actual per cell for the **verification view**
  (drift cells flagged, interval width → hex opacity). *Blocked on SP5.*
- **`GET /reasoning/{id}`** — the grounded reasoning trace (each step's cells +
  simulator-checked verdict) for the **reasoning panel**. *Blocked on SP7 reasoning model.*
- **`apps/web/`** — deck.gl `H3HexagonLayer` + MapLibre basemap, time slider bound to
  the canonical `time` axis, scenario + reasoning + verify panels. Hosted on a
  Hugging Face Space.

---

## Self-Review notes

- **Phase A is fully buildable on merged SP1** — `/layer` only needs the `Registry` +
  `OpenMeteoWeatherAdapter`, both of which exist.
- **Uncertainty → opacity** is stubbed (fixed alpha) in Phase A because conformal
  intervals come from SP5; the `color` RGBA already carries an alpha slot so wiring it
  later is a one-line change in `_ramp`.
- **No geometry conversion anywhere** — the H3-native deck.gl layer is the entire
  reason the canonical schema keys on hex ids; this plan validates that bet end-to-end.
- **Tests use an injected fake registry** (no network) for the service; the pydeck demo
  is the one place that hits the real Open-Meteo API, kept out of the test suite.

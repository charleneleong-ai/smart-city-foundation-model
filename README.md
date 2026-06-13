# Smart City Foundation Model — Urban Digital Twin

A **verifiable urban digital twin** built from open data and open foundation models
(LLMs, VLMs / geospatial FMs, world models, RLVR) for smart-city planning predictions
— energy first.

- **Modular, not monolithic** — a foundation model per concern, each independently
  verifiable.
- **Verification is a first-class spine** — RLVR rewards, backtesting, conformal
  calibration, and drift-triggered re-grounding against real measurements.
- **Portable** — instantiable for any city from global open layers; region data is a
  pluggable adapter.

See the design: [`docs/specs/2026-06-12-urban-digital-twin-design.md`](docs/specs/2026-06-12-urban-digital-twin-design.md).

## Quickstart

```bash
uv sync --extra dev --extra forecast --extra app
uv run pytest -q                     # full test suite
```

## Map demo (3D H3 + deck.gl + MapLibre)

Render a day of Open-Meteo temperature over London's H3 grid as a self-contained 3D
WebGL map (extruded hexes, dark basemap, no token — opens straight from `file://`):

```bash
mise run map                                       # London -> london_3d.html
uv run python apps/demo_weather.py --city uk       # region gradient -> uk_3d.html
uv run python apps/demo_weather.py --city london --radius 40 --res 7   # control extent + detail
open london_3d.html                                # Play button + time slider + 2D/3D toggle
```

Presets: `london`, `nyc`, `tokyo` (city, H3 res 8), `uk` (region, res 4 — north-south gradient).
`--radius <km>` sets the area around the preset centre; `--res <0..15>` sets H3 detail
(coarser = fewer cells = fewer API calls; the demo caps at 400 cells). Extrusion auto-scales
to hex size so 3D is visible at any zoom.

Or serve the twin and query deck.gl-ready records:

```bash
uv run --extra app uvicorn --factory sctwin.app.service:build_app   # needs a wired Registry
# GET /layer?layer=weather.t2m&south=51.46&west=-0.20&north=51.55&east=-0.05&res=7&date=2020-01-15
```

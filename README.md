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

## Twin viewer (3D H3 + deck.gl + MapLibre)

Self-contained 3D WebGL map (extruded hexes, dark basemap, no token — opens from `file://`)
with three nested selectors — **Domain** (Weather, Energy, …), **Layer**, and a **time**
slider — plus an in-map **radius** slider (filters preloaded cells, no re-fetch), Play, and
2D/3D toggle. Hovering a hex shows *all* layers at once (e.g. demand vs forecast vs error).

```bash
mise run twin                                      # full UK twin -> uk_twin_3d.html
uv run --extra forecast python apps/demo_twin.py --city london --radius 30 --res 8
open uk_twin_3d.html
```

- **Weather domain:** 2 m temperature, heating degrees.
- **Energy domain:** demand (`y_true`), forecast (`y_pred`), |error|, delta (diverging), coverage —
  from the SP4 GBM + SP5 split-conformal harness; subtitle reports MAE / RMSE / coverage.

Weather-only map: `mise run map` (`apps/demo_weather.py`). Presets: `london`, `nyc`, `tokyo`
(city, H3 res 8), `uk` (region, res 4). `--radius <km>` + `--res <0..15>` control extent/detail
(capped at 400 cells = 400 API calls); extrusion auto-scales to hex size.

Or serve the twin and query deck.gl-ready records:

```bash
uv run --extra app uvicorn --factory sctwin.app.service:build_app   # needs a wired Registry
# GET /layer?layer=weather.t2m&south=51.46&west=-0.20&north=51.55&east=-0.05&res=7&date=2020-01-15
```

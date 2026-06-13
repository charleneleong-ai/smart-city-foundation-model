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

## Map demo (H3 + deck.gl)

Render a day of Open-Meteo temperature over London's H3 grid:

```bash
uv run --extra app --extra forecast python apps/demo_weather.py   # -> weather_map.html
```

Or serve the twin and query deck.gl-ready records:

```bash
uv run --extra app uvicorn --factory sctwin.app.service:build_app   # needs a wired Registry
# GET /layer?layer=weather.t2m&south=51.46&west=-0.20&north=51.55&east=-0.05&res=7&date=2020-01-15
```

# Smart City Foundation Model — Urban Digital Twin

A **verifiable urban digital twin** built from open data and open foundation models
(time-series FMs, geospatial FMs, world models, RLVR) — the **calibrated predictive
world-model that Physical (Embodied) AI plans against**. Robots, drones, AVs and smart
infrastructure don't act on weather; they act on *where and when demand, load, and risk will
be* — predicted per H3 cell, at macro and micro scale, with the uncertainty needed to act safely.

- **Multi-domain, one substrate** — weather, energy, **EV charging**, … are channels of one
  forecaster ([Chronos-2](https://hf.co/amazon/chronos-2)); weather conditions energy via covariates.
- **Verification is a first-class spine** — RLVR rewards, backtesting, conformal calibration, and
  drift-triggered re-grounding, so an embodied agent knows *when not to trust* a prediction.
- **Portable, macro + micro** — any city from global open layers; H3 res 4 (region) to res 8
  (neighbourhood / charging cluster).

See: [`docs/physical-ai-world-model.md`](docs/physical-ai-world-model.md) ·
[design spec](docs/specs/2026-06-12-urban-digital-twin-design.md).

![Palisades fire backtest: the macro cellular-automaton front advancing over the real burn perimeter, with a personalised firefighter deployment, on satellite terrain](https://github.com/charleneleong-ai/smart-city-foundation-model/blob/feat/weather-fire-variables/docs/media/palisades_fire.gif?raw=true)

*Macro fire-spread **backtest** on the 2025 LA Palisades fire: the wind- and terrain-driven
cellular-automaton front (yellow→red = **hit**, magenta = **over-reach**) walking over the real
NIFC burn perimeter (blue = **missed**), water-masked to land, with a **personalised
exposure→health firefighter deployment** advancing with the front. IoU 0.35 · recall 0.40.
Built by [`apps/viz_fire_3d.py`](apps/viz_fire_3d.py) → self-contained `la_fire_3d.html`.*

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
  from the SP4 GBM + SP5 split-conformal harness.
- **EV charging (physical-AI consumable):** evening-peaked, cold-amplified charging-demand surface
  ([`sctwin.demand`](src/sctwin/demand.py)) → forecast + interval a fleet/depot operator acts on.

Weather-only map: `mise run map` (`apps/demo_weather.py`). Presets: `london`, `nyc`, `tokyo`
(city, H3 res 8), `uk` (region, res 4). `--radius <km>` + `--res <0..15>` control extent/detail
(capped at 400 cells = 400 API calls); extrusion auto-scales to hex size.

Or serve the twin and query deck.gl-ready records:

```bash
uv run --extra app uvicorn --factory sctwin.app.service:build_app   # needs a wired Registry
# GET /layer?layer=weather.t2m&south=51.46&west=-0.20&north=51.55&east=-0.05&res=7&date=2020-01-15
```

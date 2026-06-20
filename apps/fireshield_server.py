"""Live HTTP feed server: serves the smart-city fire model's per-step EnvData to the Fire-Shield
app so the dashboard polls model predictions instead of a regenerated file. Same `build_feed`
the CLI exporter uses; CORS-open so the app (localhost:3000) can fetch cross-origin.

  GET /feed                  -> the cached feed (computed once at startup)
  GET /feed?spread=0.35      -> recompute with model overrides (spread / seed_lat / seed_lon / date),
                                so changing the *model* live changes what the app shows

Run: uv run --extra app python apps/fireshield_server.py --perimeter palisades.geojson --port 8787
"""

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from export_fireshield_feed import build_feed

# One-click scenarios for the control panel — each maps to build_feed() overrides.
PRESETS: dict[str, dict] = {
    "default": {},
    "wider front": {"spread": 0.25},
    "contained": {"spread": 0.7},
    "north ridge": {"seed_lat": 34.09, "seed_lon": -118.55},
    "coastal canyon": {"seed_lat": 34.05, "seed_lon": -118.55},
}


def create_app(perimeter: Path) -> FastAPI:
    app = FastAPI(title="fireshield-feed")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
    )
    cache: dict[str, dict] = {}
    state = {"config": {}}  # the scenario the app pulls when it polls /feed with no params

    def feed_for(config: dict) -> dict:
        key = repr(sorted(config.items()))
        if key not in cache:  # compute once per distinct model config, then serve from memory
            try:
                cache[key] = build_feed(perimeter, **config)
            except (ValueError, FileNotFoundError) as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        return cache[key]

    @app.get("/feed")
    def feed(
        spread: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        seed_lat: float | None = None,
        seed_lon: float | None = None,
        date: str | None = None,
    ) -> dict:
        # explicit params = one-off override; otherwise serve the panel-selected scenario
        overrides = {k: v for k, v in
                     {"spread": spread, "seed_lat": seed_lat, "seed_lon": seed_lon, "date": date}.items()
                     if v is not None}
        return feed_for(overrides or state["config"])

    @app.post("/select")
    def select(preset: Annotated[str, Query()] = "default") -> dict:
        """Set the scenario the app pulls. The control panel calls this; the app picks it up on its
        next poll / Refresh."""
        if preset not in PRESETS:
            raise HTTPException(status_code=400, detail=f"unknown preset {preset!r}")
        state["config"] = PRESETS[preset]
        return {"preset": preset, **feed_for(state["config"])}

    @app.get("/", response_class=HTMLResponse)
    def panel() -> str:
        return _CONTROL_PANEL

    return app


_CONTROL_PANEL = """<!doctype html><meta charset=utf-8><title>Fire model → app</title>
<style>
 body{background:#0f172a;color:#e2e8f0;font:14px/1.5 system-ui;margin:0;padding:2rem;max-width:640px}
 h1{font-size:1.1rem;color:#fb923c}.sub{color:#94a3b8;font-size:.85rem;margin-bottom:1.5rem}
 button{background:#1e293b;border:1px solid #fb923c33;color:#e2e8f0;padding:.6rem 1rem;border-radius:.6rem;
   font-weight:700;cursor:pointer;margin:.25rem}button:hover{background:#fb923c;color:#0f172a}
 pre{background:#020617;border-radius:.6rem;padding:1rem;font-size:.8rem;overflow:auto;margin-top:1.2rem}
 .live{color:#34d399}
</style>
<h1>🔥 Smart-City Fire Model → Fire-Shield app</h1>
<div class=sub>Pick a scenario to drive the app's environment. The dashboard pulls it on its next
poll (~5s) or when you hit <b>Refresh from model</b>.</div>
<div id=btns></div>
<pre id=out>select a scenario…</pre>
<script>
 const PRESETS=["default","wider front","contained","north ridge","coastal canyon"];
 const b=document.getElementById("btns"),o=document.getElementById("out");
 PRESETS.forEach(p=>{const x=document.createElement("button");x.textContent=p;
   x.onclick=async()=>{o.textContent="running model…";
     const r=await fetch("/select?preset="+encodeURIComponent(p),{method:"POST"});
     const f=await r.json();const peak=Math.max(...f.frames.map(x=>x.env.temperature));
     o.innerHTML='<span class=live>● selected: '+f.preset+'</span>\\n'+
       'cell '+f.cell+' · '+f.steps+' steps · wind '+f.windFrom+' @ '+f.windKph+'km/h\\n'+
       'peak temp '+peak+'°C · the app will show this on its next poll / Refresh';};
   b.appendChild(x);});
</script>
"""


def main(
    perimeter: Annotated[Path, typer.Option(help="observed burn perimeter GeoJSON")] = Path("palisades.geojson"),
    host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port")] = 8787,
) -> None:
    """Serve the live Fire-Shield model feed."""
    uvicorn.run(create_app(perimeter), host=host, port=port)


if __name__ == "__main__":
    typer.run(main)

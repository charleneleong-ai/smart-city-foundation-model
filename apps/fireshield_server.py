"""Live HTTP feed server: serves the smart-city fire model's per-step EnvData to the Fire-Shield
app so the dashboard polls model predictions instead of a regenerated file. Runs the deployment
engine once and places each deployed firefighter at a cell along the front, so the app can monitor
whichever member is selected on the operator map. CORS-open so the app (localhost:3000) can fetch.

  GET  /          -> crew control panel (pick a deployed firefighter)
  GET  /crew      -> the deployed roster (id, role, deploy risk, cell, front-arrival)
  POST /select?member=FF-03  -> set which member the app monitors
  GET  /feed      -> the selected member's EnvData feed (+ member info); the app polls this

Run: uv run --extra app python apps/fireshield_server.py --perimeter palisades.geojson --port 8787
"""

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from export_fireshield_feed import build_deployment, feed_at_cell


def create_app(perimeter: Path) -> FastAPI:
    app = FastAPI(title="fireshield-feed")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
    )
    dep: dict = {}            # {arrival, meta, wx, members} — model + deployment, computed once
    state = {"member": None}  # id of the member the app currently monitors

    def deployment() -> dict:
        if not dep:
            arrival, meta, wx, members = build_deployment(perimeter)
            dep.update(arrival=arrival, meta=meta, wx=wx, members=members)
            mid = (max(arrival.values()) if arrival else 0) / 2.0  # default to a mid-front member
            state["member"] = min(members, key=lambda m: abs((m["arrival"] or 0) - mid))["id"]
        return dep

    def member_feed(m: dict) -> dict:
        feed = feed_at_cell(dep["arrival"], dep["meta"], dep["wx"], m["cell"])
        feed["member"] = {k: m[k] for k in ("id", "role", "ppe", "deployRisk", "arrival")}
        return feed

    @app.get("/crew")
    def crew() -> dict:
        d = deployment()
        keys = ("id", "role", "ppe", "deployRisk", "cell", "arrival")
        return {"selected": state["member"], "members": [{k: m[k] for k in keys} for m in d["members"]]}

    @app.post("/select")
    def select(member: Annotated[str, Query()]) -> dict:
        d = deployment()
        m = next((x for x in d["members"] if x["id"] == member), None)
        if m is None:
            raise HTTPException(status_code=404, detail=f"no deployed member {member!r}")
        state["member"] = member
        return member_feed(m)

    @app.get("/feed")
    def feed() -> dict:
        d = deployment()
        m = next((x for x in d["members"] if x["id"] == state["member"]), d["members"][0])
        return member_feed(m)

    @app.get("/", response_class=HTMLResponse)
    def panel() -> str:
        return _CONTROL_PANEL

    return app


def main(
    perimeter: Annotated[Path, typer.Option(help="observed burn perimeter GeoJSON")] = Path("palisades.geojson"),
    host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port")] = 8787,
) -> None:
    """Serve the live Fire-Shield deployment feed."""
    uvicorn.run(create_app(perimeter), host=host, port=port)


if __name__ == "__main__":
    typer.run(main)


_CONTROL_PANEL = """<!doctype html><meta charset=utf-8><title>Deployment → app</title>
<style>
 body{background:#0f172a;color:#e2e8f0;font:14px/1.5 system-ui;margin:0;padding:2rem;max-width:680px}
 h1{font-size:1.1rem;color:#fb923c}.sub{color:#94a3b8;font-size:.85rem;margin-bottom:1.2rem}
 button{display:block;width:100%;text-align:left;background:#1e293b;border:1px solid #fb923c33;color:#e2e8f0;
   padding:.7rem 1rem;border-radius:.6rem;font-weight:700;cursor:pointer;margin:.4rem 0}
 button:hover{background:#fb923c;color:#0f172a}button small{font-weight:400;opacity:.8}
 pre{background:#020617;border-radius:.6rem;padding:1rem;font-size:.8rem;white-space:pre-wrap;margin-top:1.2rem}
 .live{color:#34d399}
</style>
<h1>🚒 Deployed crew → Fire-Shield app</h1>
<div class=sub>Pick a deployed firefighter (these match the labelled markers on the operator map).
The app switches to monitor that member's environment on its next poll (~5s) or on <b>Refresh from model</b>.</div>
<div id=btns>loading deployed crew…</div>
<pre id=out>select a firefighter…</pre>
<script>
 const b=document.getElementById("btns"),o=document.getElementById("out");
 fetch("/crew").then(r=>r.json()).then(d=>{b.innerHTML="";
   d.members.forEach(m=>{const x=document.createElement("button");
     x.innerHTML=m.id+' <small>· '+m.role+' · deploy risk '+m.deployRisk+
       ' · front-arrival step '+(m.arrival==null?'—':m.arrival)+'</small>';
     x.onclick=async()=>{o.textContent="selecting "+m.id+"…";
       const r=await fetch("/select?member="+encodeURIComponent(m.id),{method:"POST"});const f=await r.json();
       const peak=Math.max(...f.frames.map(x=>x.env.temperature));
       o.innerHTML='<span class=live>● app now monitoring '+f.member.id+' ('+f.member.role+')</span>\\n'+
         'cell '+f.cell+' · '+f.steps+' steps · peak '+peak+'°C · deploy risk '+f.member.deployRisk+'\\n'+
         'the app switches to this member on its next poll (~5s) or hit Refresh from model';};
     b.appendChild(x);});});
</script>
"""

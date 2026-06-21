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

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from export_fireshield_feed import build_deployment, feed_at_cell

_TICK_SECONDS = 1.5  # operator-clock cadence: one CA step per tick


def create_app(perimeter: Path) -> FastAPI:
    app = FastAPI(title="fireshield-feed")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
    )
    dep: dict = {}  # {arrival, meta, wx, members} — model + deployment, computed once
    # The operator side is the master clock: it sets member + step here, and the app follows via /state.
    state = {"member": None, "step": 0, "playing": False}
    clock: dict = {"task": None}

    def deployment() -> dict:
        if not dep:
            arrival, meta, wx, members = build_deployment(perimeter)
            dep.update(arrival=arrival, meta=meta, wx=wx, members=members)
            state["member"] = members[0]["id"]  # auto-select the first deployed firefighter
        return dep

    def maxstep() -> int:
        d = deployment()
        return max(d["arrival"].values()) if d["arrival"] else 0

    def member_feed(m: dict) -> dict:
        feed = feed_at_cell(dep["arrival"], dep["meta"], dep["wx"], m["cell"])
        feed["member"] = {k: m[k] for k in
                          ("id", "role", "ppe", "deployRisk", "riskLow", "riskHigh", "drivers", "arrival")}
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

    # ── Operator-driven playback clock — the app follows /state in lockstep ───────────────────
    @app.get("/state")
    def get_state() -> dict:
        deployment()
        return {"member": state["member"], "step": state["step"], "maxstep": maxstep(), "playing": state["playing"]}

    @app.post("/step")
    def set_step(step: Annotated[int, Query()] = 0) -> dict:
        state["step"] = max(0, min(step, maxstep()))
        return {"step": state["step"]}

    @app.post("/play")
    async def play() -> dict:
        deployment()
        if clock["task"] and not clock["task"].done():
            return {"playing": True, "step": state["step"]}
        if state["step"] >= maxstep():
            state["step"] = 0  # restart from the top
        state["playing"] = True

        async def run() -> None:
            while state["playing"] and state["step"] < maxstep():
                await asyncio.sleep(_TICK_SECONDS)
                if state["playing"]:
                    state["step"] = min(state["step"] + 1, maxstep())
            state["playing"] = False

        clock["task"] = asyncio.create_task(run())
        return {"playing": True, "step": state["step"]}

    @app.post("/pause")
    def pause() -> dict:
        state["playing"] = False
        return {"playing": False, "step": state["step"]}

    @app.post("/reset")
    def reset() -> dict:
        state["playing"] = False
        state["step"] = 0
        return {"step": 0}

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


_CONTROL_PANEL = """<!doctype html><meta charset=utf-8><title>Operator → app</title>
<style>
 body{background:#0f172a;color:#e2e8f0;font:14px/1.5 system-ui;margin:0;padding:2rem;max-width:680px}
 h1{font-size:1.1rem;color:#fb923c}.sub{color:#94a3b8;font-size:.85rem;margin-bottom:1rem}
 button{display:block;width:100%;text-align:left;background:#1e293b;border:1px solid #fb923c33;color:#e2e8f0;
   padding:.7rem 1rem;border-radius:.6rem;font-weight:700;cursor:pointer;margin:.4rem 0}
 button:hover{background:#fb923c;color:#0f172a}button small{font-weight:400;opacity:.8}
 .ctrl{display:flex;align-items:center;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}
 .ctrl button{display:inline-block;width:auto;margin:0;padding:.6rem 1.1rem}
 #clock{font:700 15px ui-monospace,monospace;color:#fb923c;margin-left:auto}
 pre{background:#020617;border-radius:.6rem;padding:1rem;font-size:.8rem;white-space:pre-wrap;margin-top:1.2rem}
 .live{color:#34d399}
</style>
<h1>🚒 Operator simulation → Fire-Shield app</h1>
<div class=sub>Run the simulation here — the app follows the operator clock in lockstep. Pick a
firefighter to choose whose environment the app monitors; rows match the labelled markers on the map.</div>
<div class=ctrl>
  <button id=play>▶ Run simulation</button>
  <button id=pause>⏸ Pause</button>
  <button id=reset>⏮ Reset</button>
  <span id=clock>step –/–</span>
</div>
<div id=btns>loading deployed crew…</div>
<pre id=out>first firefighter auto-selected — press ▶ Run, or pick another below…</pre>
<script>
 const b=document.getElementById("btns"),o=document.getElementById("out"),clk=document.getElementById("clock");
 const post=p=>fetch(p,{method:"POST"});
 document.getElementById("play").onclick=()=>post("/play");
 document.getElementById("pause").onclick=()=>post("/pause");
 document.getElementById("reset").onclick=()=>post("/reset");
 setInterval(async()=>{try{const s=await(await fetch("/state")).json();
   clk.textContent='step '+s.step+'/'+s.maxstep+(s.playing?' \\u25b6 running':' \\u23f8');}catch(e){}},600);
 fetch("/crew").then(r=>r.json()).then(d=>{b.innerHTML="";
   d.members.forEach(m=>{const x=document.createElement("button");
     x.innerHTML=m.id+' <small>· '+m.role+' · deploy risk '+m.deployRisk+
       ' · front-arrival step '+(m.arrival==null?'—':m.arrival)+'</small>';
     x.onclick=async()=>{o.textContent="selecting "+m.id+"…";
       const r=await fetch("/select?member="+encodeURIComponent(m.id),{method:"POST"});const f=await r.json();
       const peak=Math.max(...f.frames.map(x=>x.env.temperature));
       o.innerHTML='<span class=live>● app now monitoring '+f.member.id+' ('+f.member.role+')</span>\\n'+
         'cell '+f.cell+' · '+f.steps+' steps · peak '+peak+'°C · deploy risk '+f.member.deployRisk+'\\n'+
         'press \\u25b6 Run — the app follows the operator clock';};
     b.appendChild(x);});});
</script>
"""

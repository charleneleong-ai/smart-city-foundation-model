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

import h3
import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from export_fireshield_feed import build_deployment, feed_at_cell

_TICK_SECONDS = 0.5  # operator-clock cadence — ~2 sim-min/sec so each minute is readable
_ACRES_PER_KM2 = 247.105
_ACRES_PER_ENGINE = 50.0  # illustrative span-of-control: one engine company per ~50 burned acres
_FF_PER_ENGINE = 4
_MIN_PER_TICK = 1    # advance one sim-minute per tick — true minute-by-minute progression
_DAY_START_MIN, _DAY_END_MIN = 6 * 60, 9 * 60  # fire's active window (06:00–09:00) — keeps per-minute change visible
_DAY_SPAN = _DAY_END_MIN - _DAY_START_MIN
_DEPLOY_AFTER_MIN = 15  # assess the fire first; commit the crew ~15 min into the incident


def create_app(perimeter: Path) -> FastAPI:
    app = FastAPI(title="fireshield-feed")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
    )
    dep: dict = {}  # {arrival, meta, wx, members} — model + deployment, computed once
    # The operator side is the master clock — a wall-clock minute over the operational day. The app
    # follows /state in lockstep and interpolates the environment between CA steps for continuity.
    state = {"member": None, "minute": float(_DAY_START_MIN), "playing": False}
    clock: dict = {"task": None}

    def deployment() -> dict:
        if not dep:
            arrival, meta, wx, members = build_deployment(perimeter)
            dep.update(arrival=arrival, meta=meta, wx=wx, members=members,
                       total=wx["cell"].n_unique(), hex_acres=h3.average_hexagon_area(8, unit="km^2") * _ACRES_PER_KM2)
            state["member"] = members[0]["id"]  # auto-select the first deployed firefighter
        return dep

    def maxstep() -> int:
        d = deployment()
        return max(d["arrival"].values()) if d["arrival"] else 0

    def assess(minute: float) -> dict:
        """Stage 1 (damage) + Stage 2 (manpower) at the current clock — what has burned and what the
        fire would demand. Manpower is an illustrative engine-company span-of-control on burned area."""
        d = deployment()
        s = round(step_for(minute))
        burned = sum(1 for v in d["arrival"].values() if v <= s)
        front = sum(1 for v in d["arrival"].values() if v == s)
        burned_acres = round(burned * d["hex_acres"])
        engines = max(1, round(burned_acres / _ACRES_PER_ENGINE)) if burned else 0
        deployed = len(d["members"]) if minute >= _DAY_START_MIN + _DEPLOY_AFTER_MIN else 0
        return {"burnedCells": burned, "frontCells": front, "totalCells": d["total"],
                "burnedAcres": burned_acres, "engines": engines, "firefighters": engines * _FF_PER_ENGINE,
                "deployed": deployed}

    def step_for(minute: float) -> float:  # fractional CA step the app interpolates the env at
        return (minute - _DAY_START_MIN) / _DAY_SPAN * maxstep()

    def mmclock(minute: float) -> str:
        m = int(round(minute))
        return f"{m // 60:02d}:{m % 60:02d}"

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
        deployed_yet = state["minute"] >= _DAY_START_MIN + _DEPLOY_AFTER_MIN
        return {"member": state["member"], "minute": round(state["minute"]), "clock": mmclock(state["minute"]),
                "step": round(step_for(state["minute"]), 3), "maxstep": maxstep(),
                "dayStart": mmclock(_DAY_START_MIN), "dayEnd": mmclock(_DAY_END_MIN), "playing": state["playing"],
                "deployedYet": deployed_yet, "deployClock": mmclock(_DAY_START_MIN + _DEPLOY_AFTER_MIN),
                "assessment": assess(state["minute"])}

    @app.post("/seek")
    def seek(minute: Annotated[int, Query()] = _DAY_START_MIN) -> dict:
        state["minute"] = float(max(_DAY_START_MIN, min(minute, _DAY_END_MIN)))
        return {"clock": mmclock(state["minute"])}

    async def _run() -> None:  # advance wall-clock minutes, looping the day until paused
        while state["playing"]:
            await asyncio.sleep(_TICK_SECONDS)
            if state["playing"]:
                nxt = state["minute"] + _MIN_PER_TICK
                state["minute"] = float(_DAY_START_MIN) if nxt >= _DAY_END_MIN else nxt

    def _start_clock() -> None:
        if not (clock["task"] and not clock["task"].done()):
            clock["task"] = asyncio.create_task(_run())

    @app.post("/play")
    async def play() -> dict:
        deployment()
        state["playing"] = True
        _start_clock()
        return {"playing": True, "clock": mmclock(state["minute"])}

    @app.on_event("startup")
    async def _autostart() -> None:  # boot straight into a running sim so the app shows it driving
        deployment()
        state["playing"] = True
        _start_clock()

    @app.post("/pause")
    def pause() -> dict:
        state["playing"] = False
        return {"playing": False, "clock": mmclock(state["minute"])}

    @app.post("/reset")
    def reset() -> dict:
        state["playing"] = False
        state["minute"] = float(_DAY_START_MIN)
        return {"clock": mmclock(state["minute"])}

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


_CONTROL_PANEL = """<!doctype html><meta charset=utf-8><title>Incident command → app</title>
<style>
 body{background:#0f172a;color:#e2e8f0;font:14px/1.5 system-ui;margin:0;padding:2rem;max-width:700px}
 h1{font-size:1.1rem;color:#fb923c;margin-bottom:.2rem}.sub{color:#94a3b8;font-size:.85rem;margin-bottom:1rem}
 .ctrl{display:flex;align-items:center;gap:.5rem;margin-bottom:1.2rem;flex-wrap:wrap}
 .ctrl button{background:#1e293b;border:1px solid #fb923c33;color:#e2e8f0;padding:.6rem 1.1rem;border-radius:.6rem;font-weight:700;cursor:pointer}
 .ctrl button:hover{background:#fb923c;color:#0f172a}
 #clock{font:700 16px ui-monospace,monospace;color:#fb923c;margin-left:auto}
 .stage{background:#111c33;border:1px solid #ffffff14;border-radius:.7rem;padding:.8rem 1rem;margin:.6rem 0}
 .stagehead{font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;color:#fb923c;font-weight:800;margin-bottom:.4rem}
 .stat{font:600 14px system-ui}.stat b{color:#fff;font-family:ui-monospace,monospace}
 .crew{display:block;width:100%;text-align:left;background:#1e293b;border:1px solid #fb923c22;color:#e2e8f0;padding:.55rem .9rem;border-radius:.5rem;font-weight:700;cursor:pointer;margin:.3rem 0}
 .crew:hover{background:#fb923c;color:#0f172a}.crew small{font-weight:400;opacity:.8}
 #out{color:#94a3b8;font-size:.8rem;margin-top:.6rem}.live{color:#34d399}
</style>
<h1>🚒 Incident command — fire → manpower → deployment</h1>
<div class=sub>Run the fire on the operator clock; assess the damage, size the manpower, then follow a
firefighter. The map and the Fire-Shield app follow in lockstep.</div>
<div class=ctrl>
  <button id=play>▶ Run</button><button id=pause>⏸ Pause</button><button id=reset>⏮ Reset</button>
  <span id=clock>06:00</span>
</div>
<div class=stage><div class=stagehead>1 · Fire &amp; damage assessment</div><div id=damage class=stat>press ▶ Run…</div></div>
<div class=stage><div class=stagehead>2 · Manpower required</div><div id=manpower class=stat>—</div></div>
<div class=stage><div class=stagehead>3 · Deploy &amp; follow a firefighter</div>
  <div id=deploywait style="color:#fb923c"></div>
  <div id=btns>loading deployed crew…</div><div id=out>first firefighter auto-selected.</div></div>
<script>
 const b=document.getElementById("btns"),o=document.getElementById("out"),clk=document.getElementById("clock");
 const dmg=document.getElementById("damage"),mp=document.getElementById("manpower");
 const post=p=>fetch(p,{method:"POST"});
 document.getElementById("play").onclick=()=>post("/play");
 document.getElementById("pause").onclick=()=>post("/pause");
 document.getElementById("reset").onclick=()=>post("/reset");
 const dw=document.getElementById("deploywait");
 setInterval(async()=>{try{const s=await(await fetch("/state")).json();
   clk.textContent=s.clock+(s.playing?' \\u25b6':' \\u23f8');
   const a=s.assessment||{};
   dmg.innerHTML='<b>'+(a.burnedAcres||0).toLocaleString()+'</b> acres burned \\u00b7 <b>'+(a.burnedCells||0)+'/'+(a.totalCells||0)+
     '</b> cells \\u00b7 active front <b>'+(a.frontCells||0)+'</b> \\u00b7 '+s.clock;
   mp.innerHTML='\\u2248 <b>'+(a.firefighters||0).toLocaleString()+'</b> firefighters / <b>'+(a.engines||0)+
     '</b> engine companies for the current fire \\u00b7 tracking sector: <b>'+(a.deployed||0)+'</b> deployed';
   if(s.deployedYet){b.style.display="block";dw.style.display="none";}
   else{b.style.display="none";dw.style.display="block";dw.textContent='\\u23f3 assessing the fire \\u2014 crew deploys at '+s.deployClock;}
 }catch(e){}},250);
 fetch("/crew").then(r=>r.json()).then(d=>{b.innerHTML="";
   d.members.forEach(m=>{const x=document.createElement("button");x.className="crew";
     x.innerHTML=m.id+' <small>\\u00b7 '+m.role+' \\u00b7 deploy risk '+m.deployRisk+'</small>';
     x.onclick=async()=>{o.textContent="selecting "+m.id+"\\u2026";
       const r=await fetch("/select?member="+encodeURIComponent(m.id),{method:"POST"});const f=await r.json();
       o.innerHTML='<span class=live>\\u25cf app now following '+f.member.id+' ('+f.member.role+
         ') \\u00b7 deploy risk '+f.member.deployRisk+'</span>';};
     b.appendChild(x);});});
</script>
"""

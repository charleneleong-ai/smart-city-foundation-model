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

from export_fireshield_feed import build_feed


def create_app(perimeter: Path) -> FastAPI:
    app = FastAPI(title="fireshield-feed")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
    )
    cache: dict[str, dict] = {}

    @app.get("/feed")
    def feed(
        spread: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        seed_lat: float | None = None,
        seed_lon: float | None = None,
        date: str | None = None,
    ) -> dict:
        key = f"{spread}|{seed_lat}|{seed_lon}|{date}"
        if key not in cache:  # compute once per distinct model config, then serve from memory
            overrides = {k: v for k, v in
                         {"spread": spread, "seed_lat": seed_lat, "seed_lon": seed_lon, "date": date}.items()
                         if v is not None}
            try:
                cache[key] = build_feed(perimeter, **overrides)
            except (ValueError, FileNotFoundError) as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        return cache[key]

    return app


def main(
    perimeter: Annotated[Path, typer.Option(help="observed burn perimeter GeoJSON")] = Path("palisades.geojson"),
    host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port")] = 8787,
) -> None:
    """Serve the live Fire-Shield model feed."""
    uvicorn.run(create_app(perimeter), host=host, port=port)


if __name__ == "__main__":
    typer.run(main)

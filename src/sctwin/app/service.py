from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry


def build_app(registry: Registry) -> FastAPI:
    app = FastAPI(title="sctwin")

    @app.get("/layer")
    def get_layer(
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

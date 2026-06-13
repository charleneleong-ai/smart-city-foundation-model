from datetime import datetime, timezone

import polars as pl
from fastapi.testclient import TestClient

from sctwin.app.service import build_app
from sctwin.registry import Registry

BOX = {"south": 51.50, "west": -0.13, "north": 51.52, "east": -0.10, "res": 8, "date": "2020-01-01"}


class _FakeWeather:
    name = "weather.t2m"

    def fetch(self, cells, start, end) -> pl.DataFrame:
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        return pl.DataFrame(
            {
                "cell": [c.h3 for c in cells],
                "time": [t] * len(cells),
                "layer": ["t2m"] * len(cells),
                "value": [float(i) for i in range(len(cells))],
            }
        )


def _client() -> TestClient:
    reg = Registry()
    reg.register(_FakeWeather())
    return TestClient(build_app(reg))


def test_layer_endpoint_returns_colored_records():
    resp = _client().get("/layer", params={"layer": "weather.t2m", **BOX})
    assert resp.status_code == 200
    recs = resp.json()
    assert len(recs) > 0
    assert {"cell", "value", "color"} <= set(recs[0])


def test_unknown_layer_returns_404():
    resp = _client().get("/layer", params={"layer": "nope", **BOX})
    assert resp.status_code == 404

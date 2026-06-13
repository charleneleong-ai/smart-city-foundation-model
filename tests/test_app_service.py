from datetime import timedelta

import polars as pl
from fastapi.testclient import TestClient

from sctwin.app.service import build_app
from sctwin.registry import Registry

BOX = {"south": 51.50, "west": -0.13, "north": 51.52, "east": -0.10, "res": 8, "date": "2020-01-01"}
TILES = {"layer": "weather.t2m", "lat": 51.505, "lon": -0.12, "radius_km": 5, "res": 8, "date": "2020-01-15"}


class _FakeWeather:
    name = "weather.t2m"

    def fetch(self, cells, start, end) -> pl.DataFrame:
        rows = [(c.h3, start + timedelta(hours=h), "t2m", float(h)) for c in cells for h in range(24)]
        return pl.DataFrame(
            {
                "cell": [r[0] for r in rows],
                "time": [r[1] for r in rows],
                "layer": [r[2] for r in rows],
                "value": [r[3] for r in rows],
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


def test_tiles_endpoint_returns_deckgl_hexes():
    resp = _client().get("/tiles", params=TILES)
    assert resp.status_code == 200
    recs = resp.json()
    assert len(recs) > 0
    assert {"cell", "value", "color", "polygon"} <= set(recs[0])
    assert recs[0]["value"] == 12.0  # default hour=12 -> fake's value == hour


def test_index_serves_the_frontend():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "deck.gl" in resp.text and "/tiles" in resp.text  # the live-tiles page

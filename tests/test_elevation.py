import httpx
import respx

from sctwin.adapters.elevation import ELEVATION_URL, fetch_elevation
from sctwin.geo import cell_of


@respx.mock
def test_fetch_elevation_maps_cells_to_metres():
    respx.get(ELEVATION_URL).mock(return_value=httpx.Response(200, json={"elevation": [120.0, 80.0]}))
    cells = [cell_of(34.07, -118.54, 8), cell_of(34.05, -118.52, 8)]
    out = fetch_elevation(cells)
    assert out == {cells[0].h3: 120.0, cells[1].h3: 80.0}

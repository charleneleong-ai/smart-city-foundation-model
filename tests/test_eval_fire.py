import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from eval_fire import bbox  # noqa: E402


def test_bbox_reads_lng_lat_order_through_nested_geometry():
    # MultiPolygon nesting; GeoJSON coords are [lng, lat] -> expect (south, west, north, east)
    gj = {
        "type": "MultiPolygon",
        "coordinates": [[[[-118.6, 34.0], [-118.4, 34.0], [-118.4, 34.1], [-118.6, 34.1], [-118.6, 34.0]]]],
    }
    assert bbox(gj) == (34.0, -118.6, 34.1, -118.4)

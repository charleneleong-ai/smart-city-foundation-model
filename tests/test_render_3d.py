import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

import h3  # noqa: E402

from render_3d import to_self_contained_html  # noqa: E402

_HOT = h3.latlng_to_cell(51.51, -0.12, 8)
_COLD = h3.latlng_to_cell(51.50, -0.10, 8)
RECORDS = [
    {"cell": _HOT, "value": 4.5, "color": [200, 40, 55, 160], "height": 1.0},
    {"cell": _COLD, "value": 1.0, "color": [0, 40, 255, 160], "height": 0.0},
]


def test_html_is_self_contained_with_inlined_polygons():
    html = to_self_contained_html(RECORDS, lat=51.5, lon=-0.12, title="T", unit="°C")
    assert "<!DOCTYPE html>" in html
    assert "PolygonLayer" in html  # core deck layer (avoids the broken bundled h3-js)
    assert "cartocdn.com" in html  # dark basemap, no token
    assert '"polygon"' in html  # python-computed hex boundary inlined (opens via file://)
    assert "__DATA__" not in html and "__LAT__" not in html  # placeholders filled


def test_legend_uses_value_range():
    html = to_self_contained_html(RECORDS, lat=51.5, lon=-0.12, unit="°C")
    assert "1.0" in html and "4.5" in html  # vmin / vmax in the legend

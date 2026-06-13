import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from render_3d import to_self_contained_html  # noqa: E402

RECORDS = [
    {"cell": "871f1d489ffffff", "value": 4.5, "color": [200, 40, 55, 160], "height": 1.0},
    {"cell": "871f1d48bffffff", "value": 1.0, "color": [0, 40, 255, 160], "height": 0.0},
]


def test_html_is_self_contained_with_inlined_data():
    html = to_self_contained_html(RECORDS, lat=51.5, lon=-0.12, title="T", unit="°C")
    assert "<!DOCTYPE html>" in html
    assert "H3HexagonLayer" in html  # the WebGL layer
    assert "cartocdn.com" in html  # dark basemap, no token
    assert "871f1d489ffffff" in html  # data inlined (opens via file://, no server)
    assert "__DATA__" not in html and "__LAT__" not in html  # all placeholders filled


def test_legend_uses_value_range():
    html = to_self_contained_html(RECORDS, lat=51.5, lon=-0.12, unit="°C")
    assert "1.0" in html and "4.5" in html  # vmin / vmax in the legend

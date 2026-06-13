import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

import h3  # noqa: E402

from render_3d import to_self_contained_html  # noqa: E402

_HOT = h3.latlng_to_cell(51.51, -0.12, 8)
_COLD = h3.latlng_to_cell(51.50, -0.10, 8)


def _frame(label: str, hot_val: float) -> dict:
    return {
        "label": label,
        "records": [
            {"cell": _HOT, "value": hot_val, "color": [200, 40, 55, 160], "height": 1.0},
            {"cell": _COLD, "value": 1.0, "color": [0, 40, 255, 160], "height": 0.0},
        ],
    }


def _layer(name: str, unit: str, frames: list[dict]) -> dict:
    return {"name": name, "unit": unit, "frames": frames}


def _css_rule(html: str, selector: str) -> str:
    return html.split(selector + " {")[1].split("}")[0]


def test_html_is_self_contained_with_inlined_polygons():
    html = to_self_contained_html([_layer("t2m", "°C", [_frame("12:00", 4.5)])], lat=51.5, lon=-0.12)
    assert "<!DOCTYPE html>" in html
    assert "PolygonLayer" in html  # core deck layer (avoids the broken bundled h3-js)
    assert "cartocdn.com" in html  # dark basemap, no token
    assert '"polygon"' in html  # python-computed hex boundary inlined (opens via file://)
    assert "__CELLS__" not in html and "__LAYERS__" not in html  # placeholders filled


def test_single_layer_hides_dropdown():
    html = to_self_contained_html([_layer("t2m", "°C", [_frame("12:00", 4.5)])], lat=51.5, lon=-0.12)
    assert "display: none" in _css_rule(html, "#layerwrap")
    assert html.count("<option") == 1


def test_multiple_layers_show_dropdown_with_options():
    layers = [_layer("error", "x", [_frame("00:00", 4.5)]), _layer("coverage", "in", [_frame("00:00", 1.0)])]
    html = to_self_contained_html(layers, lat=51.5, lon=-0.12)
    assert "display: block" in _css_rule(html, "#layerwrap")
    assert '<option value="0">error</option>' in html
    assert '<option value="1">coverage</option>' in html


def test_time_controls_hidden_for_single_frame():
    html = to_self_contained_html([_layer("t2m", "°C", [_frame("12:00", 4.5)])], lat=51.5, lon=-0.12)
    assert "display: none" in _css_rule(html, "#controls")


def test_time_controls_and_play_shown_for_multi_frame():
    frames = [_frame("00:00", 4.5), _frame("06:00", 9.0), _frame("12:00", 2.0)]
    html = to_self_contained_html([_layer("err", "x", frames)], lat=51.5, lon=-0.12)
    assert "display: block" in _css_rule(html, "#controls")
    assert 'id="play"' in html  # play button present for animation
    assert "00:00" in html and "12:00" in html  # frame labels embedded
    assert '"vmin": 1.0' in html and '"vmax": 9.0' in html  # layer spans its global value range

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


def test_html_is_self_contained_with_inlined_polygons():
    html = to_self_contained_html([_frame("12:00", 4.5)], lat=51.5, lon=-0.12, unit="°C")
    assert "<!DOCTYPE html>" in html
    assert "PolygonLayer" in html  # core deck layer (avoids the broken bundled h3-js)
    assert "cartocdn.com" in html  # dark basemap, no token
    assert '"polygon"' in html  # python-computed hex boundary inlined (opens via file://)
    assert "__CELLS__" not in html and "__FRAMES__" not in html  # placeholders filled


def test_single_frame_hides_time_controls():
    html = to_self_contained_html([_frame("12:00", 4.5)], lat=51.5, lon=-0.12)
    assert "display: none;" in html  # controls hidden for a single frame


def test_multiple_frames_show_slider_and_span_global_range():
    frames = [_frame("00:00", 4.5), _frame("06:00", 9.0), _frame("12:00", 2.0)]
    html = to_self_contained_html(frames, lat=51.5, lon=-0.12, unit="°C")
    assert "display: block;" in html  # slider shown
    assert 'id="play"' in html  # play button present for animation
    assert 'max="2"' in html  # three frames -> indices 0..2
    assert "00:00" in html and "12:00" in html  # frame labels embedded
    assert "9.0 °C" in html and "1.0 °C" in html  # legend spans the global value range

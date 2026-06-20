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


def _layer(name: str, frames: list[dict]) -> dict:
    return {"name": name, "unit": "u", "frames": frames}


def _map(name: str, layers: list[dict]) -> dict:
    return {"name": name, "subtitle": name, "lat": 51.5, "lon": -0.11, "zoom": 10.0, "layers": layers}


def _css_rule(html: str, selector: str) -> str:
    return html.split(selector + " {")[1].split("}")[0]


def test_html_is_self_contained_with_inlined_polygons():
    html = to_self_contained_html([_map("weather", [_layer("t2m", [_frame("12:00", 4.5)])])])
    assert "<!DOCTYPE html>" in html
    assert "PolygonLayer" in html  # core deck layer (avoids the broken bundled h3-js)
    assert "cartocdn.com" in html  # dark basemap, no token
    assert '"polygon"' in html  # python-computed hex boundary inlined (opens via file://)
    assert "__MAPS__" not in html and "__MAP_OPTIONS__" not in html  # placeholders filled


def test_single_map_hides_domain_dropdown():
    html = to_self_contained_html([_map("weather", [_layer("t2m", [_frame("12:00", 4.5)])])])
    assert "display: none" in _css_rule(html, "#mapwrap")
    assert '<select id="layer"></select>' in html  # layer options are built client-side, not server-side


def test_multiple_maps_show_domain_dropdown_with_options():
    maps = [
        _map("Weather", [_layer("t2m", [_frame("00:00", 4.5)])]),
        _map("Energy", [_layer("demand", [_frame("00:00", 9.0)]), _layer("forecast", [_frame("00:00", 8.0)])]),
    ]
    html = to_self_contained_html(maps)
    assert "display: block" in _css_rule(html, "#mapwrap")
    assert '<option value="0">Weather</option>' in html
    assert '<option value="1">Energy</option>' in html
    # every layer is embedded so the tooltip can show all of them at once
    assert '"name": "demand"' in html and '"name": "forecast"' in html


def test_radius_filter_and_movable_centre_wired():
    records = [
        {"cell": _HOT, "value": 1.0, "color": [0, 0, 0, 1], "height": 0.0},
        {"cell": _COLD, "value": 2.0, "color": [0, 0, 0, 1], "height": 1.0},
    ]
    html = to_self_contained_html([_map("m", [_layer("t", [{"label": "a", "records": records}])])])
    assert 'id="radius"' in html  # in-map radius slider present
    assert '"cen":' in html  # per-cell centroid embedded -> client-side distance from a movable centre
    assert "maplibregl.Marker" in html and "function setCenter" in html  # click-to-recentre wired


def test_layer_spans_its_global_value_range():
    frames = [_frame("00:00", 4.5), _frame("06:00", 9.0), _frame("12:00", 2.0)]
    html = to_self_contained_html([_map("m", [_layer("err", frames)])])
    assert "00:00" in html and "12:00" in html  # frame labels embedded
    assert '"vmin": 1.0' in html and '"vmax": 9.0' in html  # layer spans its global value range
    assert 'id="play"' in html  # play button present for animation


def test_basemap_satellite_swaps_in_esri_raster():
    m = _map("m", [_layer("t", [_frame("12:00", 4.5)])])
    sat = to_self_contained_html([m], basemap="satellite")
    assert "World_Imagery" in sat and '"type": "raster"' in sat  # esri imagery raster style
    assert "cartocdn.com" not in sat  # dark vector style replaced
    assert "World_Imagery" not in to_self_contained_html([m])  # default stays dark (cartocdn)


def test_categorical_legend_embeds_swatches():
    m = _map("m", [_layer("t", [_frame("12:00", 4.5)])])
    m["legend"] = [{"color": [255, 0, 0], "label": "hot"}, {"color": [0, 0, 255], "label": "cold"}]
    html = to_self_contained_html([m])
    assert '"label": "hot"' in html  # legend data embedded
    assert "m.legend" in html and 'class="sw"' in html  # swatch render wired in the viewer
    assert '"legend": []' in to_self_contained_html([_map("m2", [_layer("t", [_frame("00:00", 1.0)])])])  # default empty

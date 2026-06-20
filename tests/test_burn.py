import h3

from sctwin.verify.burn import cells_from_geojson, score

RES = 8
# a ~0.1 deg square over Pacific Palisades (lng, lat order, GeoJSON convention)
_SQUARE = [[[-118.58, 34.02], [-118.48, 34.02], [-118.48, 34.10], [-118.58, 34.10], [-118.58, 34.02]]]


def test_cells_from_geojson_rasterises_polygon_and_multipolygon():
    poly = {"type": "Polygon", "coordinates": _SQUARE}
    cells = cells_from_geojson(poly, RES)
    assert len(cells) > 0
    assert all(h3.get_resolution(c) == RES for c in cells)
    # a Feature wrapping the same geometry yields the same cells
    feat = {"type": "Feature", "geometry": poly, "properties": {}}
    assert cells_from_geojson({"type": "FeatureCollection", "features": [feat]}, RES) == cells
    # MultiPolygon of two disjoint copies is a superset (union) of the single polygon
    far = [[[lng + 0.5, lat] for lng, lat in _SQUARE[0]]]
    multi = {"type": "MultiPolygon", "coordinates": [_SQUARE, far]}
    assert cells_from_geojson(multi, RES) > cells


def test_score_iou_precision_recall():
    obs = {"a", "b", "c", "d"}
    pred = {"b", "c", "d", "e"}  # 3 hit, 1 false alarm (e), 1 missed (a)
    s = score(pred, obs)
    assert s["intersection"] == 3
    assert s["iou"] == 3 / 5  # union {a,b,c,d,e}
    assert s["precision"] == 3 / 4 and s["recall"] == 3 / 4
    assert abs(s["f1"] - 0.75) < 1e-9


def test_score_handles_empty_and_perfect():
    assert score(set(), {"a"})["iou"] == 0.0  # nothing predicted
    perfect = score({"a", "b"}, {"a", "b"})
    assert perfect["iou"] == 1.0 and perfect["f1"] == 1.0

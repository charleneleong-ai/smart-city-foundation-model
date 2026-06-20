"""Burned-area verification: rasterise an observed fire perimeter (GeoJSON) to H3 cells and
score a predicted burn set against it (IoU / precision / recall / F1) — the verification-spine
join that turns the macro fire CA from illustrative into measurable.

Note the honest limit: the CA's burned *extent* is a tuned free parameter (steps x spread
threshold), so recall/IoU partly reflect that tuning, not predictive size skill. The signal
that is *not* circular is directional — whether the front overlaps where the fire actually ran.
"""

import h3


def _polygon_cells(rings: list, res: int) -> set[str]:
    """H3 cells covering one GeoJSON polygon: rings[0] outer, rings[1:] holes. GeoJSON is
    [lng, lat]; H3 LatLngPoly wants (lat, lng)."""
    latlng = [[(lat, lng) for lng, lat in ring] for ring in rings]
    return set(h3.polygon_to_cells(h3.LatLngPoly(latlng[0], *latlng[1:]), res))


def cells_from_geojson(gj: dict, res: int) -> set[str]:
    """Union of H3 cells covering every Polygon / MultiPolygon in a GeoJSON FeatureCollection,
    Feature, or bare geometry."""
    features = gj.get("features", [gj])
    cells: set[str] = set()
    for feat in features:
        geom = feat.get("geometry", feat)
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for rings in polys:
            cells |= _polygon_cells(rings, res)
    return cells


def score(predicted: set[str], observed: set[str]) -> dict:
    """Overlap of a predicted burn set against the observed one. IoU = intersection/union;
    precision = how much of the prediction burned; recall = how much of the burn was caught."""
    tp = len(predicted & observed)
    fp = len(predicted - observed)
    fn = len(observed - predicted)
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": tp / union if union else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted": len(predicted),
        "observed": len(observed),
        "intersection": tp,
    }

"""Self-contained deck.gl + MapLibre 3D renderer for H3 layer records.

Produces a single HTML file (data inlined, libs via CDN) — opens straight from file://,
no server, no map token. WebGL 3D-extruded hexes on a dark vector basemap, pitched,
lit, with hover tooltips.

Hex boundaries are computed in Python (h3.cell_to_boundary) and drawn with deck.gl's
core PolygonLayer — deck's own H3HexagonLayer relies on a CDN-bundled h3-js that is
broken in the standalone build, so we avoid it entirely.
"""

import json

import h3

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>__TITLE__</title>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
<script src="https://unpkg.com/deck.gl@9.0.38/dist.min.js"></script>
<style>
  html, body, #map { margin: 0; height: 100%; width: 100%; background: #0b0b10; }
  #panel { position: absolute; top: 16px; left: 16px; padding: 12px 14px; z-index: 2;
    background: rgba(16,18,28,.82); color: #e8eaf2; border-radius: 10px;
    font: 13px/1.4 -apple-system, system-ui, sans-serif; box-shadow: 0 6px 24px rgba(0,0,0,.4); }
  #panel b { font-size: 14px; }
  #bar { height: 8px; width: 180px; margin: 8px 0 4px;
    background: linear-gradient(90deg, rgb(0,40,255), rgb(255,40,0)); border-radius: 4px; }
  #scale { display: flex; justify-content: space-between; font-size: 11px; opacity: .8; }
  .maplibregl-popup-content { background: #11141c; color: #e8eaf2; border-radius: 8px;
    font: 12px system-ui; padding: 6px 9px; }
  .maplibregl-popup-tip { display: none; }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <b>__TITLE__</b><br/>__SUBTITLE__
  <div id="bar"></div>
  <div id="scale"><span>__VMIN__</span><span>__VMAX__</span></div>
</div>
<script>
  const DATA = __DATA__;
  const map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [__LON__, __LAT__], zoom: __ZOOM__, pitch: __PITCH__, bearing: __BEARING__,
    antialias: true,
  });
  map.addControl(new maplibregl.NavigationControl());
  const layer = new deck.PolygonLayer({
    id: 'hex', data: DATA, extruded: true, filled: true, wireframe: false,
    getPolygon: d => d.polygon, getFillColor: d => d.color,
    getElevation: d => d.height, elevationScale: __ELEV__,
    opacity: 0.86, pickable: true,
    material: { ambient: 0.55, diffuse: 0.65, shininess: 28, specularColor: [60, 64, 90] },
  });
  const overlay = new deck.MapboxOverlay({
    layers: [layer],
    getTooltip: ({ object }) => object && {
      html: `<b>${object.value.toFixed(1)} __UNIT__</b>`,
      style: { background: '#11141c', color: '#e8eaf2', fontSize: '12px', borderRadius: '6px' },
    },
  });
  map.addControl(overlay);
</script>
</body>
</html>
"""


def _hex_ring(cell: str) -> list[list[float]]:
    """H3 cell -> closed [lng, lat] ring for deck.gl PolygonLayer."""
    ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(cell)]
    ring.append(ring[0])
    return ring


def to_self_contained_html(
    records: list[dict],
    *,
    lat: float,
    lon: float,
    title: str = "sctwin",
    subtitle: str = "",
    unit: str = "",
    zoom: float = 10.6,
    pitch: float = 52.0,
    bearing: float = 18.0,
    elevation_scale: float = 3200.0,
) -> str:
    drawable = [{**r, "polygon": _hex_ring(r["cell"])} for r in records]
    vals = [r["value"] for r in records] or [0.0]
    repl = {  # repr() on floats yields valid JS number literals; json.dumps for the data array
        "__DATA__": json.dumps(drawable),
        "__LON__": repr(lon),
        "__LAT__": repr(lat),
        "__ZOOM__": repr(zoom),
        "__PITCH__": repr(pitch),
        "__BEARING__": repr(bearing),
        "__ELEV__": repr(elevation_scale),
        "__TITLE__": title,
        "__SUBTITLE__": subtitle,
        "__UNIT__": unit,
        "__VMIN__": f"{min(vals):.1f}",
        "__VMAX__": f"{max(vals):.1f}",
    }
    html = _TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, v)
    return html

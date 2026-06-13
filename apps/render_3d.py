"""Self-contained deck.gl + MapLibre 3D renderer for multi-layer, time-series H3 data.

Produces a single HTML file (data inlined, libs via CDN) — opens straight from file://,
no server, no map token. WebGL hexes on a dark vector basemap, pitched, lit, with a
**layer dropdown**, a time slider, a 2D/3D toggle, a Play button, a value legend, an
"about" panel, and hover tooltips.

Hex boundaries are computed in Python (h3.cell_to_boundary) and drawn with deck.gl's
core PolygonLayer — deck's own H3HexagonLayer relies on a CDN-bundled h3-js that is
broken in the standalone build. Geometry is embedded once; each layer carries per-frame
value/color/height arrays.
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
  #panel { position: absolute; top: 16px; left: 16px; width: 300px; padding: 14px 16px; z-index: 2;
    background: rgba(16,18,28,.86); color: #e8eaf2; border-radius: 12px;
    font: 13px/1.45 -apple-system, system-ui, sans-serif; box-shadow: 0 8px 28px rgba(0,0,0,.45); }
  #panel h1 { font-size: 15px; margin: 0 0 2px; }
  #panel .sub { opacity: .72; font-size: 12px; }
  #layerwrap { margin: 11px 0 2px; display: __LAYER_DISPLAY__; }
  #layer { width: 100%; padding: 5px; color: #e8eaf2; background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.16); border-radius: 7px; font: 12px system-ui; }
  #bar { height: 9px; margin: 11px 0 4px;
    background: linear-gradient(90deg, rgb(0,40,255), rgb(140,40,160), rgb(255,40,0)); border-radius: 5px; }
  #scale { display: flex; justify-content: space-between; font-size: 11px; opacity: .82; }
  #about { font-size: 12px; opacity: .82; margin: 11px 0 0; }
  #controls { margin-top: 12px; border-top: 1px solid rgba(255,255,255,.1); padding-top: 11px;
    display: __CTRL_DISPLAY__; }
  #trow { display: flex; align-items: center; gap: 9px; }
  #time { flex: 1; accent-color: #ff5a3c; }
  #tlabel { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 78px; }
  #btns { display: flex; gap: 8px; margin-top: 9px; }
  .btn { flex: 1; padding: 6px; cursor: pointer; color: #e8eaf2;
    background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.16); border-radius: 7px;
    font: 12px system-ui; }
  .btn:hover { background: rgba(255,255,255,.13); }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div id="layerwrap"><select id="layer">__OPTIONS__</select></div>
  <div id="bar"></div>
  <div id="scale"><span id="vmin"></span><span id="vmax"></span></div>
  <p id="about">__ABOUT__</p>
  <div id="controls">
    <div id="trow">
      <input id="time" type="range" min="0" value="0" step="1" />
      <span id="tlabel"></span>
    </div>
    <div id="btns">
      <button id="play" class="btn">&#9654; Play</button>
      <button id="toggle" class="btn">2D / 3D</button>
    </div>
  </div>
</div>
<script>
  const CELLS = __CELLS__;     // [{polygon:[[lng,lat]...]}]  — static geometry
  const LAYERS = __LAYERS__;   // [{name, unit, vmin, vmax, frames:[{label,v,c,h}]}]
  let layerIdx = 0, frame = 0, extruded = true;
  const map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [__LON__, __LAT__], zoom: __ZOOM__, pitch: __PITCH__, bearing: __BEARING__,
    antialias: true,
  });
  map.addControl(new maplibregl.NavigationControl());
  const overlay = new deck.MapboxOverlay({
    layers: [],
    getTooltip: ({ object }) => object && {
      html: `<b>${object.value.toFixed(2)} ${LAYERS[layerIdx].unit}</b>`,
      style: { background: '#11141c', color: '#e8eaf2', fontSize: '12px', borderRadius: '6px' },
    },
  });
  map.addControl(overlay);

  const slider = document.getElementById('time'), playBtn = document.getElementById('play');
  let timer = null;

  function render() {
    const L = LAYERS[layerIdx], F = L.frames[frame];
    const data = CELLS.map((c, i) => ({ polygon: c.polygon, value: F.v[i], color: F.c[i], height: F.h[i] }));
    overlay.setProps({ layers: [new deck.PolygonLayer({
      id: 'hex', data, extruded, filled: true, wireframe: false,
      getPolygon: d => d.polygon, getFillColor: d => d.color,
      getElevation: d => d.height, elevationScale: extruded ? __ELEV__ : 0,
      opacity: extruded ? 0.86 : 0.7, pickable: true,
      material: { ambient: 0.55, diffuse: 0.65, shininess: 28, specularColor: [60, 64, 90] },
      updateTriggers: { getFillColor: [layerIdx, frame], getElevation: [layerIdx, frame, extruded] },
    })] });
    document.getElementById('tlabel').textContent = F.label;
    document.getElementById('vmin').textContent = L.vmin.toFixed(1) + ' ' + L.unit;
    document.getElementById('vmax').textContent = L.vmax.toFixed(1) + ' ' + L.unit;
  }
  function setFrame(i) { frame = i; slider.value = i; render(); }

  document.getElementById('layer').addEventListener('change', e => {
    layerIdx = +e.target.value;
    slider.max = LAYERS[layerIdx].frames.length - 1;
    setFrame(0);
  });
  slider.addEventListener('input', e => setFrame(+e.target.value));
  document.getElementById('toggle').addEventListener('click', () => { extruded = !extruded; render(); });
  playBtn.addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; playBtn.innerHTML = '&#9654; Play'; }
    else { timer = setInterval(() => setFrame((frame + 1) % LAYERS[layerIdx].frames.length), 420);
           playBtn.innerHTML = '&#9208; Pause'; }
  });
  slider.max = LAYERS[0].frames.length - 1;
  render();
</script>
</body>
</html>
"""


def _hex_ring(cell: str) -> list[list[float]]:
    """H3 cell -> closed [lng, lat] ring for deck.gl PolygonLayer."""
    ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(cell)]
    ring.append(ring[0])
    return ring


def _js_layer(layer: dict) -> dict:
    frames = [
        {
            "label": f["label"],
            "v": [r["value"] for r in f["records"]],
            "c": [r["color"] for r in f["records"]],
            "h": [r["height"] for r in f["records"]],
        }
        for f in layer["frames"]
    ]
    vals = [v for f in frames for v in f["v"]] or [0.0]
    return {
        "name": layer["name"],
        "unit": layer.get("unit", ""),
        "vmin": min(vals),
        "vmax": max(vals),
        "frames": frames,
    }


def to_self_contained_html(
    layers: list[dict],
    *,
    lat: float,
    lon: float,
    title: str = "sctwin",
    subtitle: str = "",
    about: str = "",
    zoom: float = 10.6,
    pitch: float = 50.0,
    bearing: float = 18.0,
    elevation_scale: float = 900.0,
) -> str:
    """Render named layers (each with time frames) as a self-contained 3D map.

    layers: [{"name": str, "unit": str, "frames": [{"label": str, "records": [...]}]}].
    All frames across all layers must share the same cells in the same order
    (geometry is taken from the first frame of the first layer).
    """
    cells = [{"polygon": _hex_ring(r["cell"])} for r in layers[0]["frames"][0]["records"]]
    js_layers = [_js_layer(layer) for layer in layers]
    n_frames = len(js_layers[0]["frames"])
    options = "".join(f'<option value="{i}">{layer["name"]}</option>' for i, layer in enumerate(layers))
    repl = {  # repr() on floats yields valid JS number literals; json.dumps for arrays
        "__CELLS__": json.dumps(cells),
        "__LAYERS__": json.dumps(js_layers),
        "__OPTIONS__": options,
        "__LON__": repr(lon),
        "__LAT__": repr(lat),
        "__ZOOM__": repr(zoom),
        "__PITCH__": repr(pitch),
        "__BEARING__": repr(bearing),
        "__ELEV__": repr(elevation_scale),
        "__LAYER_DISPLAY__": "block" if len(layers) > 1 else "none",
        "__CTRL_DISPLAY__": "block" if n_frames > 1 else "none",
        "__TITLE__": title,
        "__SUBTITLE__": subtitle,
        "__ABOUT__": about,
    }
    html = _TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, v)
    return html

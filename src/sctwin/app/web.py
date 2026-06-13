"""Frontend for the served dynamic-load twin: a deck.gl + MapLibre page that fetches tiles
around a movable centre from the /tiles endpoint (so panning loads on demand, via cache)."""

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>sctwin — live tiles</title>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
<script src="https://unpkg.com/deck.gl@9.0.38/dist.min.js"></script>
<style>
  html, body, #map { margin: 0; height: 100%; width: 100%; background: #0b0b10; }
  #panel { position: absolute; top: 16px; left: 16px; width: 260px; padding: 14px 16px; z-index: 2;
    background: rgba(16,18,28,.86); color: #e8eaf2; border-radius: 12px;
    font: 13px/1.45 -apple-system, system-ui, sans-serif; box-shadow: 0 8px 28px rgba(0,0,0,.45); }
  #panel h1 { font-size: 15px; margin: 0 0 2px; }
  #panel .sub { opacity: .7; font-size: 12px; margin-bottom: 10px; }
  label { display: block; font-size: 10px; text-transform: uppercase; letter-spacing: .04em;
    opacity: .55; margin: 9px 0 3px; }
  input[type=range] { width: 100%; accent-color: #ff5a3c; }
  select, input[type=date] { width: 100%; padding: 5px; color: #e8eaf2; background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.16); border-radius: 7px; font: 12px system-ui; }
  #status { margin-top: 11px; font-size: 12px; opacity: .8; font-variant-numeric: tabular-nums; }
  #hint { margin-top: 8px; font-size: 10.5px; opacity: .5; }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>Live tiles</h1>
  <div class="sub">2 m temperature · loads around the centre on demand</div>
  <label>Radius <span id="rval"></span></label>
  <input id="radius" type="range" min="2" max="60" value="12" step="1" />
  <label>Resolution (H3)</label>
  <select id="res"><option>6</option><option selected>7</option><option>8</option></select>
  <label>Date</label>
  <input id="date" type="date" value="2020-01-15" />
  <div id="status">—</div>
  <div id="hint">Click the map to move the centre &#8853;</div>
</div>
<script>
  const P = { lat: 51.505, lon: -0.12, radius: 12, res: 7, date: '2020-01-15', layer: 'weather.t2m' };
  let data = [], busy = false;
  const map = new maplibregl.Map({
    container: 'map', style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [P.lon, P.lat], zoom: 8.5, pitch: 45, antialias: true,
  });
  map.addControl(new maplibregl.NavigationControl());
  const marker = new maplibregl.Marker({ color: '#ff5a3c' }).setLngLat([P.lon, P.lat]).addTo(map);
  const overlay = new deck.MapboxOverlay({
    layers: [],
    getTooltip: ({ object }) => object && {
      html: `<b>${object.value.toFixed(1)} °C</b>`,
      style: { background: '#11141c', color: '#e8eaf2', fontSize: '12px', borderRadius: '6px' },
    },
  });
  map.addControl(overlay);

  function render() {
    const elev = 2000 * Math.pow(2.6, 8 - P.res);  // scale extrusion to hex size
    overlay.setProps({ layers: [new deck.PolygonLayer({
      id: 'hex', data, extruded: true, filled: true, getPolygon: d => d.polygon,
      getFillColor: d => d.color, getElevation: d => d.height, elevationScale: elev,
      opacity: 0.86, pickable: true,
      material: { ambient: 0.55, diffuse: 0.65, shininess: 28, specularColor: [60, 64, 90] },
    })] });
  }
  async function load() {
    if (busy) return;
    busy = true; status('loading…');
    try {
      const u = new URLSearchParams({ layer: P.layer, lat: P.lat, lon: P.lon,
        radius_km: P.radius, res: P.res, date: P.date });
      data = await (await fetch('/tiles?' + u)).json();
      render(); status(data.length + ' tiles · ' + P.radius + ' km · res ' + P.res);
    } catch (e) { status('error: ' + e); }
    busy = false;
  }
  function status(t) { document.getElementById('status').textContent = t; }

  map.on('click', e => {
    P.lat = e.lngLat.lat; P.lon = e.lngLat.lng; marker.setLngLat(e.lngLat); load();
  });
  const radius = document.getElementById('radius'), rval = document.getElementById('rval');
  function setR() { P.radius = +radius.value; rval.textContent = P.radius + ' km'; }
  radius.addEventListener('change', () => { setR(); load(); });
  radius.addEventListener('input', setR);
  document.getElementById('res').addEventListener('change', e => { P.res = +e.target.value; load(); });
  document.getElementById('date').addEventListener('change', e => { P.date = e.target.value; load(); });
  setR();
  map.on('load', load);
</script>
</body>
</html>
"""

"""Self-contained deck.gl + MapLibre 3D renderer for multi-map, multi-layer, time-series
H3 data.

Produces a single HTML file (data inlined, libs via CDN) — opens straight from file://,
no server, no map token. Three nested selectors: a **map** dropdown (e.g. weather inputs
vs verification, or different cities — each with its own geometry/centre), a **layer**
dropdown within the active map, and a **time** slider over frames. Plus an in-map radius
slider (filters preloaded cells by distance — no re-fetch), a 2D/3D toggle, Play, a value
legend, an "about" panel, and hover tooltips.

Hex boundaries are computed in Python (h3.cell_to_boundary) and drawn with deck.gl's core
PolygonLayer — deck's own H3HexagonLayer relies on a CDN-bundled h3-js that is broken in
the standalone build.
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
  .maplibregl-marker { z-index: 5; }  /* keep the radius-centre pin above the deck.gl overlay */
  #panel { position: absolute; top: 16px; left: 16px; width: 300px; padding: 14px 16px; z-index: 2;
    background: rgba(16,18,28,.86); color: #e8eaf2; border-radius: 12px;
    font: 13px/1.45 -apple-system, system-ui, sans-serif; box-shadow: 0 8px 28px rgba(0,0,0,.45); }
  #panel h1 { font-size: 15px; margin: 0 0 2px; }
  #panel .sub { opacity: .72; font-size: 12px; }
  #yearwrap { margin: 11px 0 0; }
  #mapwrap { margin: 11px 0 0; display: __MAP_DISPLAY__; }
  #layerwrap { margin: 8px 0 2px; }
  .srow { display: flex; gap: 6px; }
  #panel label { display: block; font-size: 10px; letter-spacing: .04em; text-transform: uppercase;
    opacity: .55; margin: 0 0 3px; }
  select { width: 100%; padding: 5px; color: #e8eaf2; background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.16); border-radius: 7px; font: 12px system-ui; }
  #bar { height: 9px; margin: 11px 0 4px;
    background: linear-gradient(90deg, rgb(0,40,255), rgb(140,40,160), rgb(255,40,0)); border-radius: 5px; }
  #scale { display: flex; justify-content: space-between; font-size: 11px; opacity: .82; }
  #scalehint { font-size: 10px; opacity: .6; margin: 2px 0 0; }
  #ldesc { font-size: 11px; opacity: .7; margin: 4px 0 0; line-height: 1.35; }
  #legend { margin: 9px 0 2px; display: none; }
  #legend .lrow { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; opacity: .92; }
  #legend .sw { width: 14px; height: 14px; border-radius: 3px; flex: 0 0 14px; border: 1px solid rgba(255,255,255,.3); }
  #about { font-size: 12px; opacity: .82; margin: 11px 0 0; }
  #controls { margin-top: 12px; border-top: 1px solid rgba(255,255,255,.1); padding-top: 11px; }
  .ctl { margin-bottom: 9px; }
  .trow { display: flex; align-items: center; gap: 9px; }
  #time, #radius { flex: 1; accent-color: #ff5a3c; }
  #tlabel, #dlabel, #rlabel { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 92px; font-size: 11px; }
  #btns { display: flex; gap: 8px; margin-top: 3px; }
  .btn, .mini { padding: 6px; cursor: pointer; color: #e8eaf2;
    background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.16); border-radius: 7px;
    font: 12px system-ui; }
  .btn { flex: 1; }
  .mini { flex: 0 0 30px; }
  .btn:hover, .mini:hover { background: rgba(255,255,255,.13); }
  #keys { position: absolute; top: 16px; right: 16px; z-index: 2; padding: 11px 13px;
    background: rgba(16,18,28,.86); color: #e8eaf2; border-radius: 12px;
    font: 12px/1.5 -apple-system, system-ui, sans-serif; box-shadow: 0 8px 28px rgba(0,0,0,.45); }
  #keys .kh { font-size: 10px; letter-spacing: .04em; text-transform: uppercase; opacity: .55; margin-bottom: 6px; }
  #keys .kr { display: flex; align-items: center; gap: 9px; opacity: .85; margin: 3px 0; }
  #keys .kr span { display: inline-flex; gap: 3px; flex: 0 0 52px; }
  #keys kbd { background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.22);
    border-bottom-width: 2px; border-radius: 5px; padding: 1px 6px; min-width: 9px; text-align: center;
    font: 11px ui-monospace, "SF Mono", monospace; }
</style>
</head>
<body>
<div id="map"></div>
<div id="keys">
  <div class="kh">Shortcuts</div>
  <div class="kr"><span><kbd>[</kbd><kbd>]</kbd></span> layer</div>
  <div class="kr"><span><kbd>,</kbd><kbd>.</kbd></span> month</div>
  <div class="kr" style="display: __YEAR_DISPLAY__"><span><kbd>-</kbd><kbd>=</kbd></span> year</div>
  <div class="kr"><span><kbd>;</kbd><kbd>&#39;</kbd></span> time</div>
  <div class="kr"><span><kbd>click</kbd></span> move radius centre</div>
</div>
<div id="roster" style="position:absolute;right:10px;top:160px;max-width:320px;background:rgba(20,20,20,.86);
     color:#eee;font:12px/1.4 system-ui;padding:8px 10px;border-radius:8px;display:none"></div>
<div id="info" style="position:absolute;right:175px;top:16px;max-width:280px;background:rgba(20,20,20,.88);
     color:#eee;font:12px/1.45 system-ui;padding:10px 12px;border-radius:8px;display:none"></div>
<div id="panel">
  <h1>__TITLE__</h1>
  <div class="sub" id="subtitle"></div>
  <div id="yearwrap" style="display: __YEAR_DISPLAY__"><label>__YEAR_LABEL__</label>
    <div class="srow"><button id="yprev" class="mini">&#10094;</button><select id="yearsel">__YEAR_OPTIONS__</select><button id="ynext" class="mini">&#10095;</button></div>
  </div>
  <div id="mapwrap"><label>__MAP_LABEL__</label>
    <div class="srow"><button id="mprev" class="mini">&#10094;</button><select id="mapsel">__MAP_OPTIONS__</select><button id="mnext" class="mini">&#10095;</button></div>
  </div>
  <div id="layerwrap"><label>Layer</label>
    <div class="srow"><button id="lprev" class="mini">&#10094;</button><select id="layer"></select><button id="lnext" class="mini">&#10095;</button></div>
    <div id="ldesc"></div>
  </div>
  <div id="bar"></div>
  <div id="scale"><span id="vmin"></span><span id="vmax"></span></div>
  <div id="scalehint">colour &amp; 3-D height both encode value &mdash; taller &amp; brighter = higher</div>
  <div id="legend"></div>
  <div id="controls">
    <div class="ctl"><label>&#9678; Radius</label>
      <div class="trow"><input id="radius" type="range" min="1" step="1" /><span id="rlabel"></span></div>
    </div>
    <div class="ctl" id="daywrap"><label>&#128197; Day</label>
      <div class="trow"><input id="day" type="range" min="0" value="0" step="1" /><span id="dlabel"></span></div>
    </div>
    <div class="ctl"><label>&#9201; Time of day</label>
      <div class="trow"><input id="time" type="range" min="0" max="23" value="0" step="1" /><span id="tlabel"></span></div>
    </div>
    <div id="btns">
      <button id="play" class="btn">&#9654; Play</button>
      <button id="playday" class="btn">&#9654; Day</button>
      <button id="toggle" class="btn">2D / 3D</button>
    </div>
  </div>
  <p id="about">__ABOUT__</p>
</div>
<script>
  const MAPS = __MAPS__;  // [{name, subtitle, lat, lon, zoom, pitch, elev, cells, layers}]
  const DATA_DIR = "__DATA_DIR__";  // '' = fully inline; else fetch DATA_DIR/<i>.json on first select (lazy)
  const STRIDE = __STRIDE__;  // maps per outer axis (months per year); MAPS is year-major flat
  const NYEARS = MAPS.length / STRIDE;
  const monthOf = i => i % STRIDE, yearOf = i => Math.floor(i / STRIDE);
  // data-driven Year+Month picker: "YYYY" -> [{mon, idx}] from each map's ym ("YYYY-MM"). Handles a
  // ragged grid (e.g. a partial current year) the flat STRIDE index can't. Empty for undated maps.
  const _MON = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const byYear = {};
  MAPS.forEach((m, i) => { if (m.ym) (byYear[m.ym.slice(0, 4)] ||= []).push({ mon: +m.ym.slice(5, 7), idx: i }); });
  const YEARS = Object.keys(byYear).sort();
  const DATED = YEARS.length > 1;  // >1 dated year -> Year dropdown filters the Month dropdown
  const fillMonths = y => { document.getElementById('mapsel').innerHTML =
    byYear[y].map(e => `<option value="${e.idx}">${_MON[e.mon]}</option>`).join(''); };
  const monthPos = i => byYear[MAPS[i].ym.slice(0, 4)].findIndex(e => e.idx === i);
  let mapIdx = 0, layerIdx = 0, frame = 0, extruded = true;
  const map = new maplibregl.Map({
    container: 'map',
    style: __BASEMAP_STYLE__,
    center: [MAPS[0].lon, MAPS[0].lat], zoom: MAPS[0].zoom, pitch: MAPS[0].pitch,
    bearing: __BEARING__, antialias: true,
  });
  map.addControl(new maplibregl.NavigationControl());
  const overlay = new deck.MapboxOverlay({
    layers: [],
    getTooltip: ({ object }) => object && (object.ff_id ? {
      html: `<b>${object.ff_id}</b> — age ${object.age}`
        + (object.cardiovascular ? ' · CV' : '') + (object.respiratory ? ' · resp' : '')
        + `<br>${object.role.toUpperCase()} · ppe ${object.ppe} · rotate@${object.rotation}min`
        + `<br>risk <b>${object.risk}</b> [${object.low}, ${object.high}]`
        + `<br>acute ${object.drivers.acute} · incident ${object.drivers.incident} · career ${object.drivers.career}`,
      style: { background: '#181818', color: '#eee', fontSize: '12px', padding: '6px' },
    } : {  // hex tooltip: show same-group layers for the hovered cell (active bold)
      html: MAPS[mapIdx].layers.map((L, j) => ({ L, j }))
        .filter(({ L }) => L.group === MAPS[mapIdx].layers[layerIdx].group)
        .map(({ L, j }) => {
          const line = `${L.name}: ${L.frames[frame].v[object.idx].toFixed(1)} ${L.unit}`;
          return j === layerIdx ? `<b>${line}</b>` : line;
        }).join('<br>'),
      style: { background: '#11141c', color: '#e8eaf2', fontSize: '12px', borderRadius: '6px' },
    }),
  });
  map.addControl(overlay);

  const slider = document.getElementById('time'), radius = document.getElementById('radius');
  const daySlider = document.getElementById('day');
  function frameDims() {  // split the active layer's hourly frames into (days, hoursPerDay)
    const n = M().layers[layerIdx].frames.length;
    return n > 24 ? { days: Math.ceil(n / 24), hours: 24 } : { days: 1, hours: n };
  }
  function syncSliders() {  // set Day/Time ranges + hide the Day row on single-day layers
    const { days, hours } = frameDims();
    daySlider.max = days - 1; slider.max = hours - 1;
    document.getElementById('daywrap').style.display = days > 1 ? '' : 'none';
  }
  function fromSliders() {  // clamp guards a partial last day (frame count not a multiple of 24)
    const { hours } = frameDims(), n = M().layers[layerIdx].frames.length;
    setFrame(Math.min((+daySlider.value) * hours + (+slider.value), n - 1));
  }
  const playBtn = document.getElementById('play');
  const dayBtn = document.getElementById('playday');  // loops the current day (24 hourly frames); hidden on single-day layers
  function syncDayBtn() {
    const multi = M().layers[layerIdx].frames.length > 24;
    dayBtn.style.display = multi ? '' : 'none';
    if (!multi && playMode === 'day') setPlay('day');  // pause a day-loop that just lost its button
  }
  let visRadius = 0, timer = null, playMode = null;  // km radius around the (movable) centre; set once distances are computed
  let center = [MAPS[0].lon, MAPS[0].lat], cellDist = [];
  const M = () => MAPS[mapIdx];
  const marker = new maplibregl.Marker({ color: '#ff5a3c', scale: 1.25 }).setLngLat(center).addTo(map);
  marker.getElement().style.zIndex = '5';  // float the centre pin above the deck.gl hex overlay canvas

  function hav(aLng, aLat, bLng, bLat) {  // km between two [lng,lat]
    const r = Math.PI / 180, R = 6371;
    const dp = (bLat - aLat) * r, dl = (bLng - aLng) * r;
    const x = Math.sin(dp / 2) ** 2 + Math.cos(aLat * r) * Math.cos(bLat * r) * Math.sin(dl / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(x));
  }
  function recompute() {  // distances from the (movable) centre to each cell
    cellDist = M().cells.map(c => hav(center[0], center[1], c.cen[0], c.cen[1]));
    radius.max = Math.ceil(Math.max(...cellDist, 1));
  }
  function setCenter(lngLat) {  // click a hex -> move the radius centre there (the circle follows it)
    center = [lngLat[0], lngLat[1]]; marker.setLngLat(center);
    recompute(); visRadius = Math.min(visRadius, +radius.max); radius.value = visRadius; render();
  }

  function render() {
    const m = M(), L = m.layers[layerIdx], F = L.frames[frame];
    const data = [];
    m.cells.forEach((c, i) => {  // cells within visRadius km of the (movable) centre — radius follows the click
      if (cellDist[i] <= visRadius) data.push({ idx: i, polygon: c.polygon, value: F.v[i], color: F.c[i], height: F.h[i] });
    });
    overlay.setProps({ layers: [new deck.PolygonLayer({
      id: 'hex', data, extruded, filled: true, wireframe: false,
      getPolygon: d => d.polygon, getFillColor: d => d.color,
      getElevation: d => d.height, elevationScale: extruded ? m.elev : 0,
      opacity: extruded ? 0.86 : 0.7, pickable: true,
      onClick: info => { if (info.coordinate) setCenter(info.coordinate); },
      material: { ambient: 0.55, diffuse: 0.65, shininess: 28, specularColor: [60, 64, 90] },
      updateTriggers: { getFillColor: [mapIdx, layerIdx, frame], getElevation: [mapIdx, layerIdx, frame, extruded] },
    }), new deck.ScatterplotLayer({
      // per-frame crew (advance with the front) if provided, else the static plan
      id: 'crew', data: ((M().plan_frames && M().plan_frames[Math.min(frame, M().plan_frames.length - 1)]) || M().plan || []),
      getPosition: d => [d.lon, d.lat], getFillColor: d => d.color,
      getRadius: d => 12 + 60 * d.risk, radiusUnits: 'meters', radiusMinPixels: 5,
      stroked: true, getLineColor: [10, 10, 10], lineWidthMinPixels: 1, pickable: true,
    })] });
    const parts = F.label.split(' \\u00b7 ');  // "YYYY · Mon DD · HH:MM" -> day label vs time label
    document.getElementById('dlabel').textContent = parts.length > 2 ? parts.slice(0, 2).join(' \\u00b7 ') : '';
    document.getElementById('tlabel').textContent = parts.length > 2 ? parts[2] : F.label;
    document.getElementById('rlabel').textContent = '\\u2264 ' + visRadius + ' km (' + data.length + ' tiles)';
    document.getElementById('vmin').textContent = L.vmin.toFixed(1) + ' ' + L.unit;
    document.getElementById('vmax').textContent = L.vmax.toFixed(1) + ' ' + L.unit;
    document.getElementById('ldesc').textContent = L.desc || '';
    const info = document.getElementById('info');  // top-right "what am I looking at" box (twin maps; deploy uses #roster)
    if (M().plan && M().plan.length) info.style.display = 'none';
    else {
      info.style.display = 'block';
      info.innerHTML = `<div style="font-size:10px;letter-spacing:.05em;text-transform:uppercase;opacity:.6">${L.group || 'Layer'}</div>`
        + `<div style="font-weight:600;font-size:14px;margin:1px 0 3px">${L.name}</div>`
        + (L.desc ? `<div style="opacity:.85">${L.desc}</div>` : '')
        + `<div style="opacity:.6;margin-top:6px">range ${L.vmin.toFixed(1)}–${L.vmax.toFixed(1)} ${L.unit} &middot; ${F.label}</div>`;
    }
  }
  function setFrame(i) {  // i is the combined day*24+hour index; split it across the two sliders
    const { hours } = frameDims();
    frame = i; daySlider.value = Math.floor(i / hours); slider.value = i % hours; render();
  }

  function renderRoster() {
    const plan = M().plan || [];
    const el = document.getElementById('roster');
    if (!plan.length) { el.style.display = 'none'; return; }
    const maxRisk = Math.max(...plan.map(p => p.risk));
    const rows = [...plan].sort((a, b) => b.risk - a.risk).map(d => {
      const c = d.color, bar = Math.round(100 * d.risk / maxRisk);
      const flags = (d.cardiovascular ? 'CV ' : '') + (d.respiratory ? 'R' : '') || '–';
      return `<div style="margin:3px 0"><b>${d.ff_id}</b> ${d.age} ${flags}
        <span style="float:right">${d.role.toUpperCase()} @${d.rotation}m</span><br>
        <span style="display:inline-block;height:7px;width:${bar}%;background:rgb(${c[0]},${c[1]},${c[2]})"></span>
        risk ${d.risk} [${d.low}, ${d.high}]</div>`;
    }).join('');
    el.innerHTML = `<div style="font-weight:600;margin-bottom:4px">Deployment — ${M().name}</div>${rows}`;
    el.style.display = 'block';
  }

  async function selectMap(i) {
    mapIdx = i; layerIdx = 0; frame = 0;
    const m = M();
    if (!m.layers) {  // lazy: fetch this map's cells+layers once, then cache on the object
      document.getElementById('subtitle').textContent = 'loading ' + m.name + ' \\u2026';
      const d = await (await fetch(DATA_DIR + '/' + i + '.json')).json();
      m.cells = d.cells; m.layers = d.layers;
    }
    map.jumpTo({ center: [m.lon, m.lat], zoom: m.zoom, pitch: m.pitch });
    document.getElementById('subtitle').textContent = m.subtitle;
    if (DATED) {  // sync Year (rebuild its Month list when the year changes) + Month from this map's ym
      const y = m.ym.slice(0, 4), ysel = document.getElementById('yearsel');
      if (ysel.value !== y) { ysel.value = y; fillMonths(y); }
      document.getElementById('mapsel').value = i;
    } else {
      document.getElementById('mapsel').value = monthOf(i);  // legacy flat index (rectangular / single-year)
      const ys = document.getElementById('yearsel'); if (ys) ys.value = yearOf(i);
    }
    const groups = {};
    m.layers.forEach((L, j) => { (groups[L.group || ''] ||= []).push(`<option value="${j}">${L.name}</option>`); });
    document.getElementById('layer').innerHTML = Object.entries(groups)
      .map(([g, opts]) => g ? `<optgroup label="${g}">${opts.join('')}</optgroup>` : opts.join('')).join('');
    center = [m.lon, m.lat]; marker.setLngLat(center); recompute();
    visRadius = +radius.max; radius.value = visRadius;
    syncSliders(); slider.value = 0; daySlider.value = 0; syncDayBtn();
    const lg = document.getElementById('legend'), catLegend = m.legend && m.legend.length;  // categorical swatches vs gradient
    lg.style.display = catLegend ? 'block' : 'none';
    document.getElementById('bar').style.display = catLegend ? 'none' : '';
    document.getElementById('scale').style.display = catLegend ? 'none' : '';
    document.getElementById('scalehint').style.display = catLegend ? 'none' : '';
    if (catLegend) lg.innerHTML = m.legend.map(e =>
      `<div class="lrow"><span class="sw" style="background:rgb(${e.color.slice(0, 3).join(',')})"></span>${e.label}</div>`).join('');
    render();
    renderRoster();
  }

  function selectLayer(i) {
    layerIdx = (i + M().layers.length) % M().layers.length;
    document.getElementById('layer').value = layerIdx;
    syncSliders(); setFrame(0); syncDayBtn();
  }
  function stepTime(d) {
    const n = M().layers[layerIdx].frames.length; setFrame((frame + d + n) % n);
  }

  function stepMonth(d) {
    if (DATED) { const ms = byYear[MAPS[mapIdx].ym.slice(0, 4)]; selectMap(ms[(monthPos(mapIdx) + d + ms.length) % ms.length].idx); }
    else selectMap(yearOf(mapIdx) * STRIDE + (monthOf(mapIdx) + d + STRIDE) % STRIDE);
  }
  function stepYear(d) {
    if (DATED) {  // keep the month position, clamped if the target year has fewer months (ragged)
      const yi = YEARS.indexOf(MAPS[mapIdx].ym.slice(0, 4)), ms = byYear[YEARS[(yi + d + YEARS.length) % YEARS.length]];
      selectMap(ms[Math.min(monthPos(mapIdx), ms.length - 1)].idx);
    } else selectMap(((yearOf(mapIdx) + d + NYEARS) % NYEARS) * STRIDE + monthOf(mapIdx));
  }
  document.getElementById('mapsel').addEventListener('change', e => selectMap(DATED ? +e.target.value : yearOf(mapIdx) * STRIDE + (+e.target.value)));
  document.getElementById('mprev').addEventListener('click', () => stepMonth(-1));
  document.getElementById('mnext').addEventListener('click', () => stepMonth(1));
  const yearsel = document.getElementById('yearsel');
  if (yearsel) {
    yearsel.addEventListener('change', e => { if (DATED) { fillMonths(e.target.value); selectMap(byYear[e.target.value][0].idx); } else selectMap((+e.target.value) * STRIDE + monthOf(mapIdx)); });
    document.getElementById('yprev').addEventListener('click', () => stepYear(-1));
    document.getElementById('ynext').addEventListener('click', () => stepYear(1));
  }
  document.getElementById('layer').addEventListener('change', e => selectLayer(+e.target.value));
  document.getElementById('lprev').addEventListener('click', () => selectLayer(layerIdx - 1));
  document.getElementById('lnext').addEventListener('click', () => selectLayer(layerIdx + 1));
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;  // don't double-handle
    if (e.key === '[') selectLayer(layerIdx - 1);
    else if (e.key === ']') selectLayer(layerIdx + 1);
    else if (e.key === ',') stepMonth(-1);
    else if (e.key === '.') stepMonth(1);
    else if (e.key === '-' && yearsel) stepYear(-1);
    else if (e.key === '=' && yearsel) stepYear(1);
    else if (e.key === ';') stepTime(-1);  // arrows are left to MapLibre for panning
    else if (e.key === "'") stepTime(1);
  });
  slider.addEventListener('input', fromSliders);
  daySlider.addEventListener('input', fromSliders);
  radius.addEventListener('input', e => { visRadius = +e.target.value; render(); });
  document.getElementById('toggle').addEventListener('click', () => { extruded = !extruded; render(); });
  function tick() {  // 'day' loops the current day's 24 hourly frames; 'all' runs the whole window continuously
    const n = M().layers[layerIdx].frames.length;
    if (playMode === 'day') {
      const start = Math.floor(frame / 24) * 24, end = Math.min(start + 24, n);
      setFrame(frame + 1 >= end ? start : frame + 1);
    } else { setFrame((frame + 1) % n); }
  }
  function setPlay(mode) {
    if (timer) { clearInterval(timer); timer = null; }
    playMode = playMode === mode ? null : mode;  // clicking the active mode pauses
    if (playMode) timer = setInterval(tick, 420);
    playBtn.innerHTML = playMode === 'all' ? '&#9208; Pause' : '&#9654; Play';
    dayBtn.innerHTML = playMode === 'day' ? '&#9208; Pause' : '&#9654; Day';
  }
  playBtn.addEventListener('click', () => setPlay('all'));
  dayBtn.addEventListener('click', () => setPlay('day'));
  if (DATED) {  // populate + reveal the Year picker (static placeholders only cover rectangular grids)
    document.getElementById('yearsel').innerHTML = YEARS.map(y => `<option value="${y}">${y}</option>`).join('');
    fillMonths(YEARS[0]);
    document.getElementById('yearwrap').style.display = '';
  }
  selectMap(MAPS.length - 1);  // open on the latest snapshot (most recent month) by default
</script>
</body>
</html>
"""


# MapLibre basemap styles. "dark" = Carto vector style URL; "satellite" = Esri World Imagery
# raster style object (xyz tiles use {z}/{y}/{x} — Esri's order, not the {z}/{x}/{y} default).
# Both go through json.dumps in _basemap_style so the template's `style: __BASEMAP_STYLE__`
# receives a valid JS literal (a quoted URL or an inline object).
_DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
_SATELLITE_STYLE = {
    "version": 8,
    "sources": {
        "esri": {
            "type": "raster",
            "tiles": ["https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            "tileSize": 256,
            "attribution": "Esri, Maxar, Earthstar Geographics",
        }
    },
    "layers": [{"id": "esri", "type": "raster", "source": "esri"}],
}


def _basemap_style(basemap: str) -> str:
    return json.dumps(_SATELLITE_STYLE if basemap == "satellite" else _DARK_STYLE)


def _hex_ring(cell: str) -> list[list[float]]:
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
    return {"name": layer["name"], "unit": layer.get("unit", ""), "group": layer.get("group", ""),
            "desc": layer.get("desc", ""),
            "vmin": layer.get("vmin", min(vals)), "vmax": layer.get("vmax", max(vals)), "frames": frames}


def _js_map(m: dict) -> dict:
    cells = []
    for r in m["layers"][0]["frames"][0]["records"]:
        clat, clon = h3.cell_to_latlng(r["cell"])
        cells.append({"polygon": _hex_ring(r["cell"]), "cen": [round(clon, 5), round(clat, 5)]})
    return {  # distance from the (movable) centre is computed client-side from each cell's "cen"
        "name": m["name"], "subtitle": m.get("subtitle", ""), "ym": m.get("ym", ""),
        "lat": m["lat"], "lon": m["lon"], "zoom": m["zoom"], "pitch": m.get("pitch", 50.0),
        "elev": m.get("elevation_scale", 900.0), "legend": m.get("legend", []),
        "cells": cells, "layers": [_js_layer(L) for L in m["layers"]],
        "plan": m.get("plan", []),  # per-firefighter deployment markers (empty for non-fire maps)
        "plan_frames": m.get("plan_frames"),  # optional per-frame crew positions (advance with the front)
    }


_META_KEYS = ("name", "subtitle", "ym", "lat", "lon", "zoom", "pitch", "elev")


def _fill(
    maps_json: list[dict], *, title: str, about: str, bearing: float, map_label: str, data_dir: str,
    group_label: str, group_options: list[str] | None, stride: int | None, basemap: str = "dark",
) -> str:
    stride = stride or len(maps_json)  # maps per outer (year) group; default = single group
    years = group_options or []
    month_opts = "".join(f'<option value="{i}">{m["name"]}</option>' for i, m in enumerate(maps_json[:stride]))
    year_opts = "".join(f'<option value="{i}">{y}</option>' for i, y in enumerate(years))
    repl = {  # repr() on floats -> valid JS number literals; json.dumps for arrays
        "__MAPS__": json.dumps(maps_json),
        "__MAP_OPTIONS__": month_opts,
        "__MAP_LABEL__": map_label,
        "__YEAR_OPTIONS__": year_opts,
        "__YEAR_LABEL__": group_label,
        "__YEAR_DISPLAY__": "block" if len(years) > 1 else "none",
        "__STRIDE__": str(stride),
        "__BEARING__": repr(bearing),
        "__MAP_DISPLAY__": "block" if len(maps_json) > 1 else "none",
        "__TITLE__": title,
        "__ABOUT__": about,
        "__DATA_DIR__": data_dir,
        "__BASEMAP_STYLE__": _basemap_style(basemap),
    }
    html = _TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, v)
    return html


def to_self_contained_html(
    maps: list[dict], *, title: str = "sctwin", about: str = "", bearing: float = 18.0, map_label: str = "Domain",
    group_label: str = "Year", group_options: list[str] | None = None, stride: int | None = None,
    basemap: str = "dark",
) -> str:
    """Render named maps as a self-contained 3D viewer with map/layer/time selectors.

    maps: [{"name", "subtitle", "lat", "lon", "zoom", "pitch", "elevation_scale",
            "layers": [{"name", "unit", "frames": [{"label", "records": [...]}]}]}].
    Each map is self-contained (own geometry + centre); switching maps recentres the view.
    `map_label` titles the inner selector (e.g. "Month"). For a 2-axis grid, pass `stride`
    (inner maps per outer group, year-major) and `group_options` (e.g. the year labels).
    """
    return _fill([_js_map(m) for m in maps], title=title, about=about, bearing=bearing, map_label=map_label,
                 data_dir="", group_label=group_label, group_options=group_options, stride=stride, basemap=basemap)


def to_lazy_html(
    maps: list[dict], *, data_dir: str, title: str = "sctwin", about: str = "", bearing: float = 18.0,
    map_label: str = "Domain", group_label: str = "Year", group_options: list[str] | None = None,
    stride: int | None = None,
) -> tuple[str, list[dict]]:
    """Like `to_self_contained_html` but embeds only per-map metadata; each map's heavy
    cells+layers payload is returned separately to be written as `<data_dir>/<i>.json` and
    fetched on demand when that map is first selected. Keeps the initial page small (one map's
    worth) instead of all N — needed once N months blow past a single-file budget. The viewer
    must be served over http (fetch can't read file://). Returns (html, payloads)."""
    payloads = [_js_map(m) for m in maps]
    meta = [{k: p[k] for k in _META_KEYS} for p in payloads]
    html = _fill(meta, title=title, about=about, bearing=bearing, map_label=map_label, data_dir=data_dir,
                 group_label=group_label, group_options=group_options, stride=stride)
    return html, payloads

/* greenpy viz — map front end. Plain JS on MapLibre GL; all data comes from the local server. */
"use strict";

// ColorBrewer ramps (7 classes) and the categorical colours for rule flags
const RAMPS = {
  high: ["#ffffcc", "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#238443", "#005a32"], // YlGn
  neutral: ["#edf8fb", "#bfd3e6", "#9ebcda", "#8c96c6", "#8c6bb1", "#88419d", "#6e016b"], // BuPu
  diverging: ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"], // BrBG, fail → pass
};
const PASS = "#1a9850", FAIL = "#d73027", NODATA = "#c9c4b8", FILTERED = "#e4dfd3";
const INK = "#1d1d1b", PAPER = "#f4f1ea";
const MODULE_ORDER = ["Rule", "T3", "T30", "T30_buildings", "T300", "Visibility", "Tree_count", "Merge", "Spectral", "Other"];

const BASEMAPS = {
  paper: { label: "None", style: null },
  positron: { label: "OpenFreeMap Positron", style: "https://tiles.openfreemap.org/styles/positron" },
  carto: {
    label: "CARTO Light",
    raster: ["a", "b", "c", "d"].map(s => `https://${s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png`),
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
    maxzoom: 20,
  },
  osm: {
    label: "OpenStreetMap",
    raster: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxzoom: 19, saturation: -0.5,
  },
  eox: {
    label: "Sentinel-2 imagery",
    note: "non-commercial",
    raster: ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"],
    attribution: '<a href="https://s2maps.eu">Sentinel-2 cloudless</a> by EOX IT Services GmbH (Copernicus Sentinel data 2020), CC BY-NC-SA 4.0',
    maxzoom: 15, saturation: -0.3,
  },
};

const state = {
  catalog: null,
  basemap: "paper",
  buildings: { visible: true, metric: null, mode: "quantile", stats: null, range: null, hidden: new Set() },
  units: { visible: false, layer: null, style: "outline", metric: null, mode: "quantile", stats: null, range: null, hidden: new Set() },
  trees: { visible: false },
  selected: null, // {layer, id}
};

const $ = id => document.getElementById(id);
const metricInfo = (layer, name) => state.catalog.layers[layer].metrics.find(m => m.name === name);
const tileUrl = (layer, metric) => `${location.origin}/tiles/${encodeURIComponent(layer)}/${encodeURIComponent(metric || "_")}/{z}/{x}/{y}.pbf`;
const getJSON = url => fetch(url).then(r => (r.ok ? r.json() : null));

/* ---------- formatting ---------- */

function fmt(value, info) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value !== "number") return String(value);
  const kind = info ? info.kind : "value";
  if (kind === "count") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(1);
  if (kind === "percent") return value.toFixed(1) + " %";
  if (kind === "distance") return Math.round(value).toLocaleString() + " m";
  const a = Math.abs(value);
  return a >= 1000 ? Math.round(value).toLocaleString() : a >= 1 ? value.toFixed(2) : value.toPrecision(3);
}

const tick = (v, info) => fmt(v, info).replace(" %", "%").replace(" m", "");

/* ---------- classification ---------- */

function sample(ramp, n) {
  if (n >= ramp.length) return ramp.slice();
  if (n <= 1) return [ramp[Math.floor(ramp.length / 2)]];
  return Array.from({ length: n }, (_, i) => ramp[Math.round((i * (ramp.length - 1)) / (n - 1))]);
}

/** Class breaks and colours for a numeric metric: {breaks: [b1..bk-1], colors: [c0..ck-1]} */
function classify(info, stats, mode) {
  if (!stats.domain) return { breaks: [], colors: [NODATA] };
  const [lo, hi] = stats.domain;
  const integer = !!stats.integer;
  const clean = bs => [...new Set(bs.map(b => (integer ? Math.round(b) : +b.toPrecision(6))))].filter(b => b > lo && b <= hi).sort((a, b) => a - b);

  if (mode === "rule" && info.threshold !== null) {
    const t = info.threshold;
    // three classes either side of the rule threshold, which is always a break
    const below = [1 / 3, 2 / 3].map(f => lo + (t - lo) * f).filter(b => b < t);
    const above = [1 / 3, 2 / 3].map(f => t + (hi - t) * f).filter(b => b > t);
    // "meets" means >= t for counts/percent and <= t for distances, so the break sits on the passing side
    const tBreak = info.better === "low" ? (integer ? t + 1 : t + 1e-9) : t;
    const breaks = [...new Set([...below.map(b => (integer ? Math.ceil(b) : b)), tBreak, ...above.map(b => (integer ? Math.ceil(b) : b))])].sort((a, b) => a - b);
    const nBelow = breaks.filter(b => b <= tBreak).length;
    const fail = RAMPS.diverging.slice(0, 3), pass = RAMPS.diverging.slice(3);
    let colors;
    if (info.better === "low") colors = [...pass.slice().reverse().slice(0, nBelow), ...fail.slice().reverse().slice(0, breaks.length + 1 - nBelow)];
    else colors = [...fail.slice(3 - nBelow), ...pass.slice(0, breaks.length + 1 - nBelow)];
    return { breaks, colors, threshold: t };
  }
  const breaks = clean(mode === "equal" ? stats.equal : stats.quantile);
  const base = info.better === "high" ? RAMPS.high : info.better === "low" ? RAMPS.high.slice().reverse() : RAMPS.neutral;
  return { breaks, colors: sample(base, breaks.length + 1) };
}

function colorForValue(v, cls) {
  let i = 0;
  while (i < cls.breaks.length && v >= cls.breaks[i]) i++;
  return cls.colors[Math.min(i, cls.colors.length - 1)];
}

/** MapLibre fill-color expression for a layer state (handles nulls, booleans and the brush). */
function colorExpression(info, stats, s) {
  const v = ["get", info.name];
  if (info.kind === "boolean") {
    const col = (val, c) => (s.hidden.has(String(val)) ? FILTERED : c);
    return ["case", ["==", v, true], col(true, PASS), ["==", v, false], col(false, FAIL), col(null, NODATA)];
  }
  const cls = classify(info, stats, s.mode);
  const step = cls.breaks.length ? ["step", ["to-number", v], cls.colors[0], ...cls.breaks.flatMap((b, i) => [b, cls.colors[i + 1]])] : cls.colors[0];
  let expr = step;
  if (s.range) {
    const [a, b] = s.range;
    const inRange = ["all", [">=", ["to-number", v], a], ["<=", ["to-number", v], b]];
    expr = ["case", inRange, step, FILTERED];
  }
  return ["case", ["==", v, null], s.range ? FILTERED : NODATA, expr];
}

/* ---------- map ---------- */

function baseStyle(key) {
  const b = BASEMAPS[key];
  const style = {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {},
    layers: [{ id: "paper", type: "background", paint: { "background-color": PAPER } }],
  };
  if (b.raster) {
    style.sources.basemap = { type: "raster", tiles: b.raster, tileSize: 256, attribution: b.attribution, maxzoom: b.maxzoom };
    style.layers.push({ id: "basemap", type: "raster", source: "basemap", paint: { "raster-saturation": b.saturation ?? -0.2, "raster-opacity": 0.9 } });
  }
  return b.style || style;
}

function dataSources() {
  const c = state.catalog;
  const sources = {
    buildings: { type: "vector", tiles: [tileUrl("buildings", state.buildings.metric)], minzoom: c.min_zoom.buildings, maxzoom: 16, promoteId: "id" },
  };
  if (state.units.layer) {
    sources.units = { type: "vector", tiles: [tileUrl(state.units.layer, state.units.metric)], minzoom: 0, maxzoom: 14, promoteId: "id" };
  }
  if (c.trees) sources.trees = { type: "vector", tiles: [tileUrl("trees", null)], minzoom: c.min_zoom.trees, maxzoom: 16 };
  return sources;
}

function treeRadius() {
  const lat = (state.catalog.bounds[1] + state.catalog.bounds[3]) / 2;
  // metres → pixels on MapLibre's 512 px world: px = m · 2^z / (78271.517 · cos φ)
  const k = z => 2 ** z / (78271.517 * Math.cos((lat * Math.PI) / 180));
  if (!state.catalog.trees.sized) return ["interpolate", ["linear"], ["zoom"], 14, 1.5, 18, 3];
  const r = ["coalesce", ["get", "radius_m"], 1.5];
  return ["interpolate", ["exponential", 2], ["zoom"], 14, ["max", 1, ["*", r, k(14)]], 22, ["*", r, k(22)]];
}

function dataLayers() {
  const vis = on => (on ? "visible" : "none");
  const u = state.units, fill = u.style === "fill";
  const layers = [];
  if (state.units.layer) {
    layers.push({
      id: "gp-units-fill", type: "fill", source: "units", "source-layer": "features",
      layout: { visibility: vis(u.visible) },
      paint: {
        "fill-color": fill && u.stats ? colorExpression(metricInfo(u.layer, u.metric), u.stats, u) : PAPER,
        // outline mode keeps an invisible fill so units stay clickable
        "fill-opacity": fill ? ["interpolate", ["linear"], ["zoom"], 12, 0.85, 15, 0.3] : 0,
      },
    });
  }
  layers.push(
    {
      id: "gp-buildings", type: "fill", source: "buildings", "source-layer": "features",
      layout: { visibility: vis(state.buildings.visible) },
      paint: { "fill-color": state.buildings.stats ? colorExpression(metricInfo("buildings", state.buildings.metric), state.buildings.stats, state.buildings) : NODATA },
    },
    {
      id: "gp-buildings-line", type: "line", source: "buildings", "source-layer": "features", minzoom: 16,
      layout: { visibility: vis(state.buildings.visible) },
      paint: { "line-color": INK, "line-opacity": 0.35, "line-width": 0.5 },
    },
  );
  if (state.units.layer) {
    layers.push({
      id: "gp-units-line", type: "line", source: "units", "source-layer": "features",
      layout: { visibility: vis(u.visible) },
      paint: fill
        ? { "line-color": PAPER, "line-width": 0.8 }
        : { "line-color": INK, "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.8, 16, 1.6], "line-opacity": 0.85 },
    });
  }
  if (state.catalog.trees) {
    const t = state.catalog.trees;
    layers.push({
      id: "gp-trees", type: "circle", source: "trees", "source-layer": "features",
      layout: { visibility: vis(state.trees.visible) },
      paint: {
        "circle-radius": treeRadius(),
        "circle-color": t.has_height ? ["interpolate", ["linear"], ["coalesce", ["get", "height"], 8], 3, "#a1d76a", 12, "#4d9221", 25, "#1b4314"] : "#4d9221",
        "circle-opacity": 0.55,
        "circle-stroke-color": "#1b4314",
        "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 14, 0, 17, 0.6],
        "circle-stroke-opacity": 0.7,
      },
    });
  }
  const sel = state.selected;
  for (const [id, src] of [["gp-selected-building", "buildings"], ["gp-selected-unit", "units"]]) {
    if (src === "units" && !state.units.layer) continue;
    const match = sel && (sel.layer === "buildings") === (src === "buildings") ? sel.id : "";
    layers.push({
      id, type: "line", source: src, "source-layer": "features",
      filter: ["==", ["to-string", ["coalesce", ["id"], ["get", "id"]]], String(match)],
      paint: { "line-color": INK, "line-width": 2.5 },
    });
  }
  return layers;
}

let map;

function applyStyle() {
  const transformStyle = (_prev, next) => ({
    ...next,
    sources: { ...next.sources, ...dataSources() },
    layers: [...next.layers, ...dataLayers()],
  });
  map.setStyle(baseStyle(state.basemap), { transformStyle, diff: false });
}

function refreshLayer(which) {
  // cheap paint/visibility updates without reloading the style
  if (!map.isStyleLoaded()) return map.once("idle", () => refreshLayer(which));
  for (const layer of dataLayers()) {
    if (!map.getLayer(layer.id)) continue;
    if (which && !layer.id.includes(which)) continue;
    for (const [k, v] of Object.entries(layer.paint || {})) map.setPaintProperty(layer.id, k, v);
    for (const [k, v] of Object.entries(layer.layout || {})) map.setLayoutProperty(layer.id, k, v);
    if (layer.filter) map.setFilter(layer.id, layer.filter);
  }
}

/* ---------- metric switching ---------- */

async function setMetric(target, metric) {
  const s = state[target];
  const layer = target === "buildings" ? "buildings" : s.layer;
  s.metric = metric;
  s.range = null;
  s.hidden = new Set();
  const info = metricInfo(layer, metric);
  if (s.mode === "rule" && info.threshold === null) s.mode = "quantile";
  s.stats = await getJSON(`/api/stats/${encodeURIComponent(layer)}/${encodeURIComponent(metric)}`);
  const src = map.getSource(target);
  if (src) src.setTiles([tileUrl(layer, metric)]);
  refreshLayer(target === "buildings" ? "buildings" : "units");
  renderLegend();
}

function fillMetricSelect(select, layer, current) {
  const groups = {};
  for (const m of state.catalog.layers[layer].metrics) (groups[m.module] ||= []).push(m);
  select.innerHTML = "";
  const order = Object.keys(groups).sort((a, b) => (MODULE_ORDER.indexOf(a) + 99) % 99 - (MODULE_ORDER.indexOf(b) + 99) % 99);
  for (const g of order) {
    const og = document.createElement("optgroup");
    og.label = g === "Merge" ? `Merge (${state.catalog.layers[layer].label})` : g;
    for (const m of groups[g]) og.append(new Option(m.label, m.name, false, m.name === current));
    select.append(og);
  }
}

function defaultMetric(layer) {
  const ms = state.catalog.layers[layer].metrics.map(m => m.name);
  const prefs = ["meets_3_30_300", "pct_meets_3_30_300"];
  return prefs.find(p => ms.includes(p)) || ms.find(m => /tree_count_\d+m$/.test(m)) || ms.find(m => m === "canopy_cover") || ms[0] || null;
}

/* ---------- legend ---------- */

function renderLegend() {
  const legend = $("legend");
  legend.innerHTML = "";
  if (state.buildings.visible && state.buildings.metric) legend.append(legendBlock("buildings"));
  if (state.units.visible && state.units.style === "fill" && state.units.metric) legend.append(legendBlock("units"));
}

function legendBlock(target) {
  const s = state[target];
  const layer = target === "buildings" ? "buildings" : s.layer;
  const info = metricInfo(layer, s.metric);
  const total = state.catalog.layers[layer];
  const block = document.createElement("div");
  block.className = "legend-block";
  block.innerHTML = `<h3></h3><div class="sub"></div>`;
  block.querySelector("h3").textContent = info.label;
  block.querySelector(".sub").textContent = target === "buildings" ? "per building" : `per unit · ${total.label}`;
  if (!s.stats) return block;

  if (info.kind === "boolean") {
    const cats = document.createElement("div");
    cats.className = "cats";
    for (const [key, label, color, n] of [["true", "Meets", PASS, s.stats.true], ["false", "Does not meet", FAIL, s.stats.false], ["null", "No data", NODATA, s.stats.null]]) {
      if (!n && key === "null") continue;
      const row = document.createElement("div");
      row.className = "cat" + (s.hidden.has(key) ? " off" : "");
      row.title = "Click to fade / restore";
      row.innerHTML = `<span class="swatch" style="background:${color}"></span><span>${label}</span><span>${n.toLocaleString()}</span>`;
      row.onclick = () => {
        s.hidden.has(key) ? s.hidden.delete(key) : s.hidden.add(key);
        refreshLayer(target === "buildings" ? "buildings" : "units");
        renderLegend();
      };
      cats.append(row);
    }
    block.append(cats);
  } else {
    block.append(modeControl(target, info), histogram(target, info));
    const foot = document.createElement("div");
    foot.className = "legend-foot";
    const shown = s.range ? countInRange(s) : null;
    const left = document.createElement("span");
    left.textContent = shown === null
      ? `${(s.stats.n - s.stats.null).toLocaleString()} ${target === "buildings" ? "buildings" : "units"}` + (s.stats.null ? ` · ${s.stats.null.toLocaleString()} no data` : "")
      : `${shown.toLocaleString()} of ${(s.stats.n - s.stats.null).toLocaleString()} shown`;
    foot.append(left);
    if (s.range) {
      const clear = document.createElement("a");
      clear.textContent = "clear";
      clear.onclick = () => { s.range = null; refreshLayer(target === "buildings" ? "buildings" : "units"); renderLegend(); };
      foot.append(clear);
    } else {
      const hint = document.createElement("span");
      hint.textContent = "drag to filter";
      foot.append(hint);
    }
    block.append(foot);
  }
  if (target === "buildings" && map.getZoom() < state.catalog.min_zoom.buildings) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = `Buildings appear from zoom ${state.catalog.min_zoom.buildings} — zoom in, or show units.`;
    block.append(note);
  }
  return block;
}

function modeControl(target, info) {
  const s = state[target];
  const seg = document.createElement("div");
  seg.className = "seg";
  for (const [mode, label] of [["quantile", "Quantile"], ["equal", "Equal"], ["rule", info.threshold !== null ? `Rule ${tick(info.threshold, info)}` : "Rule"]]) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.className = s.mode === mode ? "on" : "";
    b.disabled = mode === "rule" && info.threshold === null;
    b.title = mode === "rule" ? "Diverging colours centred on the 3-30-300 threshold" : "";
    b.onclick = () => { s.mode = mode; refreshLayer(target === "buildings" ? "buildings" : "units"); renderLegend(); };
    seg.append(b);
  }
  return seg;
}

function binEdges(stats) {
  const [lo, hi] = stats.domain, n = stats.hist.length, w = (hi - lo) / n;
  return { lo, hi, n, w };
}

function countInRange(s) {
  const { lo, w } = binEdges(s.stats);
  const [a, b] = s.range;
  return s.stats.hist.reduce((acc, c, i) => {
    const mid = lo + (i + 0.5) * w;
    return acc + (mid >= a && mid <= b ? c : 0);
  }, 0);
}

function histogram(target, info) {
  const s = state[target];
  const NS = "http://www.w3.org/2000/svg";
  const W = 240, H = 64, top = 4, base = 44;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "hist");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  if (!s.stats.domain) return svg;
  const { lo, hi, n, w } = binEdges(s.stats);
  const max = Math.max(...s.stats.hist, 1);
  const cls = classify(info, s.stats, s.mode);
  const bw = W / n;
  const x = v => ((v - lo) / (hi - lo)) * W;
  const el = (tag, attrs) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    svg.append(e);
    return e;
  };

  s.stats.hist.forEach((c, i) => {
    const mid = lo + (i + 0.5) * w;
    const h = c ? Math.max(1, (Math.sqrt(c) / Math.sqrt(max)) * (base - top)) : 0;
    const inRange = !s.range || (mid >= s.range[0] && mid <= s.range[1]);
    el("rect", { x: i * bw + 0.5, y: base - h, width: bw - 1, height: h, fill: inRange ? colorForValue(mid, cls) : FILTERED, stroke: inRange ? "rgba(29,29,27,.35)" : "none", "stroke-width": 0.5 });
  });
  el("line", { class: "axis", x1: 0, x2: W, y1: base + 0.5, y2: base + 0.5 });
  for (const b of cls.breaks) if (b > lo && b < hi) el("line", { class: "axis", x1: x(b), x2: x(b), y1: base, y2: base + 4 });
  if (info.threshold !== null && info.threshold > lo && info.threshold < hi) {
    el("line", { class: "threshold", x1: x(info.threshold), x2: x(info.threshold), y1: top - 4, y2: base + 4 });
  }
  const label = (v, anchor, xx) => { const t = el("text", { x: xx, y: H - 4, "text-anchor": anchor }); t.textContent = tick(v, info); };
  label(lo, "start", 0);
  label(s.stats.integer ? hi - w : hi, "end", W);
  if (info.threshold !== null && info.threshold > lo && info.threshold < hi) {
    const tx = x(info.threshold);
    if (tx > 30 && tx < W - 30) label(info.threshold, "middle", tx);
  }

  // brush: drag across bins to keep only that value range
  let start = null;
  const binAt = ev => {
    const r = svg.getBoundingClientRect();
    return Math.max(0, Math.min(n - 1, Math.floor(((ev.clientX - r.left) / r.width) * n)));
  };
  const rangeOf = (a, b) => {
    const [i, j] = [Math.min(a, b), Math.max(a, b)];
    // the end bins also hold the outliers beyond the 0.5-99.5 % domain
    return [i === 0 ? -Infinity : lo + i * w, j === n - 1 ? Infinity : lo + (j + 1) * w];
  };
  svg.addEventListener("pointerdown", ev => { start = binAt(ev); svg.setPointerCapture(ev.pointerId); });
  svg.addEventListener("pointerup", ev => {
    if (start === null) return;
    const end = binAt(ev);
    s.range = rangeOf(start, end);
    if (s.range[0] === -Infinity && s.range[1] === Infinity) s.range = null;
    start = null;
    refreshLayer(target === "buildings" ? "buildings" : "units");
    renderLegend();
  });
  return svg;
}

/* ---------- details ---------- */

async function showDetails(feature) {
  const layerId = feature.layer.id;
  const body = $("details-body");
  if (layerId === "gp-trees") {
    const p = feature.properties;
    state.selected = null;
    body.innerHTML = `<h3>Tree</h3><div class="sub">from the tree layer</div><table></table>`;
    const rows = [["Crown radius", p.radius_m != null ? `${(+p.radius_m).toFixed(1)} m` : "—"], ["Height", p.height != null ? `${(+p.height).toFixed(1)} m` : "—"]];
    body.querySelector("table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  } else {
    const isBuilding = layerId.startsWith("gp-buildings");
    const layer = isBuilding ? "buildings" : state.units.layer;
    const id = String(feature.id ?? feature.properties.id);
    state.selected = { layer, id };
    const data = await getJSON(`/api/feature/${encodeURIComponent(layer)}/${encodeURIComponent(id)}`);
    if (!data) return;
    const metrics = state.catalog.layers[layer].metrics;
    const current = isBuilding ? state.buildings.metric : state.units.metric;
    const metricNames = new Set(metrics.map(m => m.name));
    const codes = Object.keys(data).filter(k => k !== "id" && !metricNames.has(k) && data[k] !== null);
    body.innerHTML = "<h3></h3><div class='sub'></div><table></table>";
    body.querySelector("h3").textContent = isBuilding ? `Building ${id}` : `${state.catalog.layers[layer].label} ${id}`;
    body.querySelector(".sub").textContent = codes.map(k => `${k} ${data[k]}`).join(" · ") || (isBuilding ? "building" : "unit");
    const table = body.querySelector("table");
    for (const m of metrics) {
      const tr = document.createElement("tr");
      if (m.name === current) tr.className = "current";
      const [a, b] = [document.createElement("td"), document.createElement("td")];
      a.textContent = m.label;
      b.textContent = fmt(data[m.name], m);
      tr.append(a, b);
      table.append(tr);
    }
  }
  $("details").hidden = false;
  refreshLayer("selected");
}

function hideDetails() {
  $("details").hidden = true;
  state.selected = null;
  refreshLayer("selected");
}

/* ---------- controls ---------- */

function buildControls() {
  const c = state.catalog;
  $("area-name").textContent = c.study_area_name;
  const counts = [];
  if (state.buildings.stats) counts.push(`${state.buildings.stats.n.toLocaleString()} buildings`);
  if (c.trees) counts.push(`${c.trees.count.toLocaleString()} trees`);
  $("counts").textContent = counts.join(" · ");

  // basemaps
  const bm = $("basemaps");
  for (const [key, b] of Object.entries(BASEMAPS)) {
    const row = document.createElement("label");
    row.className = "row";
    row.innerHTML = `<input type="radio" name="basemap" value="${key}"${key === state.basemap ? " checked" : ""}> <span></span>`;
    row.querySelector("span").textContent = b.label + (b.note ? ` (${b.note})` : "");
    row.querySelector("input").onchange = () => { state.basemap = key; saveView(); applyStyle(); };
    bm.append(row);
  }

  // buildings
  const bsel = $("building-metric");
  if (c.layers.buildings.metrics.length) {
    fillMetricSelect(bsel, "buildings", state.buildings.metric);
    bsel.onchange = () => { setMetric("buildings", bsel.value); saveView(); };
  } else {
    bsel.disabled = true;
    bsel.append(new Option("no building metrics yet", ""));
  }
  $("show-buildings").onchange = e => { state.buildings.visible = e.target.checked; refreshLayer("buildings"); renderLegend(); };

  // units
  const unitLayers = Object.keys(c.layers).filter(k => k !== "buildings");
  if (!unitLayers.length) $("units-controls").hidden = true;
  const lsel = $("unit-layer");
  for (const k of unitLayers) lsel.append(new Option(`${c.layers[k].label} · ${c.layers[k].metrics.length} metrics`, k, false, k === state.units.layer));
  lsel.onchange = async () => {
    state.units.layer = lsel.value;
    state.units.metric = defaultMetric(lsel.value);
    fillMetricSelect($("unit-metric"), lsel.value, state.units.metric);
    if (state.units.metric) state.units.stats = await getJSON(`/api/stats/${encodeURIComponent(lsel.value)}/${encodeURIComponent(state.units.metric)}`);
    state.units.range = null;
    applyStyle();
    renderLegend();
    saveView();
  };
  $("show-units").onchange = e => { state.units.visible = e.target.checked; refreshLayer("units"); renderLegend(); saveView(); };
  for (const b of $("unit-style").querySelectorAll("button")) {
    b.onclick = () => {
      state.units.style = b.dataset.value;
      for (const o of $("unit-style").querySelectorAll("button")) o.classList.toggle("on", o === b);
      $("unit-metric").hidden = state.units.style !== "fill";
      if (state.units.style === "fill" && !state.units.visible) { state.units.visible = true; $("show-units").checked = true; }
      refreshLayer("units");
      renderLegend();
      saveView();
    };
  }
  const usel = $("unit-metric");
  if (state.units.layer) fillMetricSelect(usel, state.units.layer, state.units.metric);
  usel.onchange = () => { setMetric("units", usel.value); saveView(); };

  // trees
  const tbox = $("show-trees");
  if (!c.trees) {
    tbox.disabled = true;
    $("trees-row").classList.add("disabled");
    $("trees-row").title = "No tree polygons configured (data.trees_dir), or the map was started with --no-trees";
    $("trees-note").textContent = "not available";
  } else {
    $("trees-note").textContent = c.trees.sized ? "· to scale" : "· as dots";
    tbox.onchange = e => { state.trees.visible = e.target.checked; refreshLayer("trees"); saveView(); };
  }
  $("details-close").onclick = hideDetails;
  document.addEventListener("keydown", e => { if (e.key === "Escape") hideDetails(); });
}

/* ---------- persistence (per browser; best effort) ---------- */

const VIEW_KEY = "greenpy-viz-view";
function saveView() {
  try {
    localStorage.setItem(VIEW_KEY, JSON.stringify({
      area: state.catalog.study_area_name,
      basemap: state.basemap,
      buildingMetric: state.buildings.metric,
      unitLayer: state.units.layer,
      unitMetric: state.units.metric,
      unitStyle: state.units.style,
      units: state.units.visible,
      trees: state.trees.visible,
    }));
  } catch (_) { /* storage unavailable */ }
}

function loadView() {
  try {
    const v = JSON.parse(localStorage.getItem(VIEW_KEY) || "null");
    return v && v.area === state.catalog.study_area_name ? v : null;
  } catch (_) {
    return null;
  }
}

/* ---------- boot ---------- */

async function main() {
  const catalog = await getJSON("/api/catalog");
  state.catalog = catalog;
  document.title = `${catalog.study_area_name} · greenpy`;

  const has = (layer, m) => layer && catalog.layers[layer] && catalog.layers[layer].metrics.some(x => x.name === m);
  const saved = loadView() || {};
  const unitLayers = Object.keys(catalog.layers).filter(k => k !== "buildings");
  state.basemap = BASEMAPS[saved.basemap] ? saved.basemap : "paper";
  state.buildings.metric = has("buildings", saved.buildingMetric) ? saved.buildingMetric : defaultMetric("buildings");
  state.units.layer = unitLayers.includes(saved.unitLayer) ? saved.unitLayer : unitLayers[unitLayers.length - 1] || null;
  state.units.metric = has(state.units.layer, saved.unitMetric) ? saved.unitMetric : state.units.layer ? defaultMetric(state.units.layer) : null;
  state.units.style = saved.unitStyle === "fill" ? "fill" : "outline";
  state.units.visible = !!saved.units;
  state.trees.visible = !!saved.trees && !!catalog.trees;
  $("show-units").checked = state.units.visible;
  $("show-trees").checked = state.trees.visible;
  for (const o of $("unit-style").querySelectorAll("button")) o.classList.toggle("on", o.dataset.value === state.units.style);
  $("unit-metric").hidden = state.units.style !== "fill";

  const statsFor = (layer, m) => (m ? getJSON(`/api/stats/${encodeURIComponent(layer)}/${encodeURIComponent(m)}`) : null);
  [state.buildings.stats, state.units.stats] = await Promise.all([
    statsFor("buildings", state.buildings.metric),
    state.units.layer ? statsFor(state.units.layer, state.units.metric) : null,
  ]);

  map = new maplibregl.Map({
    container: "map",
    style: { version: 8, sources: {}, layers: [] },
    bounds: catalog.bounds,
    fitBoundsOptions: { padding: 40 },
    hash: true,
    attributionControl: { compact: true },
    maxPitch: 0,
    dragRotate: false,
  });
  map.touchZoomRotate.disableRotation();
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
  buildControls();
  applyStyle();
  renderLegend();

  map.on("zoomend", renderLegend);
  map.on("click", ev => {
    const order = ["gp-trees", "gp-buildings", "gp-units-fill"];
    const layers = order.filter(id => map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== "none");
    const hits = map.queryRenderedFeatures(ev.point, { layers });
    if (!hits.length) return hideDetails();
    hits.sort((a, b) => order.indexOf(a.layer.id) - order.indexOf(b.layer.id));
    showDetails(hits[0]);
  });
  for (const id of ["gp-trees", "gp-buildings", "gp-units-fill"]) {
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  }
}

main().catch(err => {
  document.body.insertAdjacentHTML("beforeend", `<section class="panel" style="top:40%;left:50%;transform:translateX(-50%)"><h1>Could not start the map</h1><p></p></section>`);
  document.querySelector("body > section:last-child p").textContent = String(err);
  console.error(err);
});

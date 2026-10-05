/* greenpy viz — map front end. Plain JS on MapLibre GL; all data comes from the local server. */
"use strict";

// Sequential palettes run low → high. Greens are kept for the tree layer and purples for
// parks, so metric colours never read as canopy or green space.
const SEQUENTIAL = {
  Blues: ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
  YlGnBu: ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"],
  YlOrBr: ["#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"],
  OrRd: ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"],
  Viridis: ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"],
  Magma: ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf"],
  Cividis: ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b"],
  Greys: ["#f7f7f7", "#e5e5e5", "#cccccc", "#b0b0b0", "#969696", "#737373", "#525252", "#333333", "#141414"],
};
// Diverging palettes run fail → pass; the rule view splits them at the 3-30-300 threshold
const DIVERGING = {
  RdBu: ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"],
  PuOr: ["#b35806", "#f1a340", "#fee0b6", "#d8daeb", "#998ec3", "#542788"],
  BrBG: ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"],
  RdYlBu: ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4"],
};
const METHODS = {
  quantile: { label: "Quantile", note: "Each class holds the same number of features." },
  equal: { label: "Equal interval", note: "Classes split the value range evenly." },
  jenks: { label: "Natural breaks", note: "Breaks fall in the gaps of the distribution (Jenks)." },
  log: { label: "Logarithmic", note: "Equal steps on a log scale — spreads out skewed data." },
  rank: { label: "Continuous (rank)", note: "Colour follows each value's rank, so every part of the distribution gets contrast." },
  rule: { label: "Rule threshold", note: "Diverging colours split at the 3-30-300 threshold." },
};
const NODATA = "#c9c4b8", FILTERED = "#e4dfd3";
const PARK = "#7b3294", PARK_LINE = "#542788", GRID_LINE = "#3f4a63";
const GRID_RE = /^(h3|s2|geohash|a5|rhealpix)_\d+$/;
const INK = "#1d1d1b", PAPER = "#f4f1ea";
const MODULE_ORDER = ["Rule", "T3", "T30", "T30_buildings", "T300", "Visibility", "Tree_count", "Merge", "Spectral", "Other"];

const BASEMAPS = {
  paper: { label: "None (paper)", style: null },
  positron: { label: "OpenFreeMap Positron", style: "https://tiles.openfreemap.org/styles/positron" },
  liberty: { label: "OpenFreeMap Liberty", style: "https://tiles.openfreemap.org/styles/liberty" },
  osm: {
    label: "OpenStreetMap",
    raster: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxzoom: 19, saturation: -0.5,
  },
  eox: {
    label: "Sentinel-2 imagery (non-commercial)",
    raster: ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"],
    attribution: '<a href="https://s2maps.eu">Sentinel-2 cloudless</a> by EOX IT Services GmbH (Copernicus Sentinel data 2020), CC BY-NC-SA 4.0',
    maxzoom: 15, saturation: -0.3,
  },
};

const target = () => ({
  visible: true, metric: null, stats: null, flagStats: null, range: null, hidden: new Set(),
  // view: a rule flag is drawn as a "gradient" of the value it tests, or as plain pass/fail ("flag")
  view: "gradient",
  method: "quantile", k: 7, palette: null, diverging: "RdBu", reverse: false,
});
const state = {
  catalog: null,
  basemap: "paper",
  buildings: target(),
  units: { ...target(), visible: false, layer: null, style: "outline" },
  trees: { visible: false },
  parks: { visible: false },
  outlines: new Set(), // unit layers drawn as boundary lines, independent of the fill layer
  selected: null, // {layer, id}
};

const $ = id => document.getElementById(id);
const metricInfo = (layer, name) => state.catalog.layers[layer].metrics.find(m => m.name === name);
const tileUrl = (layer, metric) => `${location.origin}/tiles/${encodeURIComponent(layer)}/${encodeURIComponent(metric || "_")}/{z}/{x}/{y}.pbf`;
const getJSON = url => fetch(url).then(r => (r.ok ? r.json() : null));
const statsFor = (layer, m) => (m ? getJSON(`/api/stats/${encodeURIComponent(layer)}/${encodeURIComponent(m)}`) : Promise.resolve(null));
const unitLayerNames = () => Object.keys(state.catalog.layers).filter(k => k !== "buildings");
const layerKey = t => (t === "buildings" ? "buildings" : "units");

/** What a target draws: a rule flag in gradient view shows the metric it tests. */
function drawn(t) {
  const s = state[t];
  const layer = t === "buildings" ? "buildings" : s.layer;
  const base = layer && s.metric ? metricInfo(layer, s.metric) : null;
  const flag = base && base.kind === "boolean" && base.gradient && s.view === "gradient" ? base : null;
  return { layer, base, flag, info: flag ? metricInfo(layer, base.gradient) : base };
}

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

/* ---------- colour ---------- */

function sample(ramp, n) {
  if (n >= ramp.length) return ramp.slice();
  if (n <= 1) return [ramp[Math.floor(ramp.length / 2)]];
  return Array.from({ length: n }, (_, i) => ramp[Math.round((i * (ramp.length - 1)) / (n - 1))]);
}

function hexMix(a, b, f) {
  const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * f).toString(16).padStart(2, "0")).join("");
}

/** Colour at fraction f (0-1) along a ramp, interpolated. */
function rampAt(ramp, f) {
  const x = Math.max(0, Math.min(1, f)) * (ramp.length - 1);
  const i = Math.min(Math.floor(x), ramp.length - 2);
  return hexMix(ramp[i], ramp[i + 1], x - i);
}

/** The target's sequential ramp, oriented so the "good" end is darkest. */
function sequentialRamp(s, info) {
  const name = s.palette || (info.better ? "Blues" : "YlOrBr");
  let ramp = SEQUENTIAL[name] || SEQUENTIAL.Blues;
  if (info.better === "low") ramp = ramp.slice().reverse();
  return s.reverse ? ramp.slice().reverse() : ramp;
}

function divergingRamp(s) {
  const ramp = DIVERGING[s.diverging] || DIVERGING.RdBu;
  return s.reverse ? ramp.slice().reverse() : ramp;
}

const passColor = s => divergingRamp(s)[5];
const failColor = s => divergingRamp(s)[0];

/* ---------- class breaks ---------- */

/** Values at the server's 101 percentiles (older stores lack them: fall back to their quantiles). */
const pcts = stats => stats.pcts || [stats.domain[0], ...stats.quantile, stats.domain[1]];

function cleanBreaks(bs, stats) {
  const [lo, hi] = stats.domain;
  const r = b => (stats.integer ? Math.round(b) : +b.toPrecision(6));
  return [...new Set(bs.map(r))].filter(b => b > lo && b <= hi).sort((a, b) => a - b);
}

function quantileBreaks(stats, k) {
  const p = pcts(stats);
  const at = q => p[Math.round(q * (p.length - 1))];
  let breaks = cleanBreaks(Array.from({ length: k - 1 }, (_, i) => at((i + 1) / k)), stats);
  if (breaks.length < Math.min(3, k - 1)) {
    // one value (often 0) dominates: it keeps its own class and the rest split the values above it
    const above = p.filter(v => v > p[0]);
    if (above.length) {
      const q = f => above[Math.round(f * (above.length - 1))];
      breaks = cleanBreaks([above[0], ...Array.from({ length: k - 2 }, (_, i) => q((i + 1) / (k - 1)))], stats);
    }
  }
  return breaks;
}

function logBreaks(stats, k) {
  const [lo, hi] = stats.domain;
  const top = Math.log1p(hi - lo);
  return cleanBreaks(Array.from({ length: k - 1 }, (_, i) => lo + Math.expm1((top * (i + 1)) / k)), stats);
}

/** Fisher-Jenks natural breaks over the percentile values (each stands for 1 % of the features). */
function jenksBreaks(stats, k) {
  const v = pcts(stats).slice().sort((a, b) => a - b);
  const n = v.length;
  if (n <= k) return cleanBreaks(v.slice(1), stats);
  const lower = Array.from({ length: n + 1 }, () => new Array(k + 1).fill(0));
  const cost = Array.from({ length: n + 1 }, () => new Array(k + 1).fill(Infinity));
  for (let j = 1; j <= k; j++) { lower[1][j] = 1; cost[1][j] = 0; }
  for (let l = 2; l <= n; l++) {
    let s1 = 0, s2 = 0, w = 0, var_ = 0;
    for (let m = 1; m <= l; m++) {
      const i3 = l - m + 1, val = v[i3 - 1];
      s2 += val * val; s1 += val; w++;
      var_ = s2 - (s1 * s1) / w;
      if (i3 > 1) {
        for (let j = 2; j <= k; j++) {
          if (cost[l][j] >= var_ + cost[i3 - 1][j - 1]) { lower[l][j] = i3; cost[l][j] = var_ + cost[i3 - 1][j - 1]; }
        }
      }
    }
    lower[l][1] = 1; cost[l][1] = var_;
  }
  const breaks = [];
  let idx = n;
  for (let j = k; j >= 2; j--) { idx = lower[idx][j] - 1; breaks.unshift(v[idx]); }
  return cleanBreaks(breaks, stats);
}

/**
 * How a numeric metric is coloured: stepped classes {breaks, colors} or, for the rank method,
 * continuous {stops, colors} interpolated between percentile values.
 */
function classify(info, stats, s) {
  if (!stats || !stats.domain) return { breaks: [], colors: [NODATA] };
  const [lo, hi] = stats.domain;
  const integer = !!stats.integer;

  if (s.method === "rule" && info.threshold !== null) {
    const t = info.threshold;
    // three classes either side of the rule threshold, which is always a break
    const below = [1 / 3, 2 / 3].map(f => lo + (t - lo) * f).filter(b => b < t);
    const above = [1 / 3, 2 / 3].map(f => t + (hi - t) * f).filter(b => b > t);
    // "meets" means >= t for counts/percent and <= t for distances, so the break sits on the passing side
    const tBreak = info.better === "low" ? (integer ? t + 1 : t + 1e-9) : t;
    const breaks = [...new Set([...below.map(b => (integer ? Math.ceil(b) : b)), tBreak, ...above.map(b => (integer ? Math.ceil(b) : b))])].sort((a, b) => a - b);
    const nBelow = breaks.filter(b => b <= tBreak).length;
    const ramp = divergingRamp(s);
    const fail = ramp.slice(0, 3), pass = ramp.slice(3);
    let colors;
    if (info.better === "low") colors = [...pass.slice().reverse().slice(0, nBelow), ...fail.slice().reverse().slice(0, breaks.length + 1 - nBelow)];
    else colors = [...fail.slice(3 - nBelow), ...pass.slice(0, breaks.length + 1 - nBelow)];
    return { breaks, colors, threshold: t };
  }

  const ramp = sequentialRamp(s, info);
  if (s.method === "rank") {
    // stops every 5th percentile; colour position = rank, so skewed data still spans the ramp
    const p = pcts(stats);
    const stops = [], colors = [];
    for (let i = 0; i < p.length; i += Math.max(1, Math.round((p.length - 1) / 20))) {
      if (stops.length && p[i] <= stops[stops.length - 1]) continue;
      stops.push(p[i]);
      colors.push(rampAt(ramp, i / (p.length - 1)));
    }
    return stops.length > 1 ? { stops, colors, continuous: true } : { breaks: [], colors: [ramp[ramp.length - 1]] };
  }
  const k = Math.max(2, Math.min(9, s.k));
  const breaks = s.method === "equal" ? cleanBreaks(Array.from({ length: k - 1 }, (_, i) => lo + ((hi - lo) * (i + 1)) / k), stats)
    : s.method === "log" ? logBreaks(stats, k)
    : s.method === "jenks" ? jenksBreaks(stats, k)
    : quantileBreaks(stats, k);
  return { breaks, colors: sample(ramp, breaks.length + 1) };
}

function colorForValue(v, cls) {
  if (cls.continuous) {
    const { stops, colors } = cls;
    if (v <= stops[0]) return colors[0];
    for (let i = 1; i < stops.length; i++) {
      if (v <= stops[i]) return hexMix(colors[i - 1], colors[i], (v - stops[i - 1]) / (stops[i] - stops[i - 1]));
    }
    return colors[colors.length - 1];
  }
  let i = 0;
  while (i < cls.breaks.length && v >= cls.breaks[i]) i++;
  return cls.colors[Math.min(i, cls.colors.length - 1)];
}

/** MapLibre fill-color expression for a target (handles nulls, booleans and the brush). */
function colorExpression(info, stats, s) {
  const v = ["get", info.name];
  if (info.kind === "boolean") {
    const col = (val, c) => (s.hidden.has(String(val)) ? FILTERED : c);
    return ["case", ["==", v, true], col(true, passColor(s)), ["==", v, false], col(false, failColor(s)), col(null, NODATA)];
  }
  const cls = classify(info, stats, s);
  const num = ["to-number", v];
  let expr;
  if (cls.continuous) expr = ["interpolate", ["linear"], num, ...cls.stops.flatMap((b, i) => [b, cls.colors[i]])];
  else expr = cls.breaks.length ? ["step", num, cls.colors[0], ...cls.breaks.flatMap((b, i) => [b, cls.colors[i + 1]])] : cls.colors[0];
  if (s.range) {
    const [a, b] = s.range;
    expr = ["case", ["all", [">=", num, a], ["<=", num, b]], expr, FILTERED];
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
  const b = drawn("buildings");
  const sources = {
    buildings: { type: "vector", tiles: [tileUrl("buildings", b.info && b.info.name)], minzoom: c.min_zoom.buildings, maxzoom: 16, promoteId: "id" },
  };
  if (state.units.layer) {
    const u = drawn("units");
    sources.units = { type: "vector", tiles: [tileUrl(state.units.layer, u.info && u.info.name)], minzoom: 0, maxzoom: 14, promoteId: "id" };
  }
  for (const layer of state.outlines) {
    if (c.layers[layer]) sources[`outline-${layer}`] = { type: "vector", tiles: [tileUrl(layer, null)], minzoom: 0, maxzoom: 14 };
  }
  if (c.parks) sources.parks = { type: "vector", tiles: [tileUrl("parks", null)], minzoom: 0, maxzoom: 14, promoteId: "id" };
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
    const d = drawn("units");
    layers.push({
      id: "gp-units-fill", type: "fill", source: "units", "source-layer": "features",
      layout: { visibility: vis(u.visible) },
      paint: {
        "fill-color": fill && u.stats && d.info ? colorExpression(d.info, u.stats, u) : PAPER,
        // outline mode keeps an invisible fill so units stay clickable
        "fill-opacity": fill ? ["interpolate", ["linear"], ["zoom"], 12, 0.85, 15, 0.3] : 0,
      },
    });
  }
  if (state.catalog.parks) {
    const pv = { visibility: vis(state.parks.visible) };
    const used = ["==", ["get", "used"], true];
    layers.push(
      { id: "gp-parks", type: "fill", source: "parks", "source-layer": "features", layout: pv,
        paint: { "fill-color": PARK, "fill-opacity": ["case", used, 0.35, 0.08] } },
      { id: "gp-parks-line", type: "line", source: "parks", "source-layer": "features", layout: pv, filter: used,
        paint: { "line-color": PARK_LINE, "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.8, 16, 2] } },
      { id: "gp-parks-small", type: "line", source: "parks", "source-layer": "features", layout: pv, filter: ["!", used],
        paint: { "line-color": PARK, "line-width": 1, "line-dasharray": [2, 1.5], "line-opacity": 0.8 } },
    );
  }
  const b = drawn("buildings");
  layers.push(
    {
      id: "gp-buildings", type: "fill", source: "buildings", "source-layer": "features",
      layout: { visibility: vis(state.buildings.visible) },
      paint: { "fill-color": state.buildings.stats && b.info ? colorExpression(b.info, state.buildings.stats, state.buildings) : NODATA },
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
  // boundary lines: census levels solid (coarser = heavier), grids thin and dashed
  const census = unitLayerNames().filter(k => !GRID_RE.test(k));
  for (const layer of state.outlines) {
    if (!state.catalog.layers[layer]) continue;
    const grid = GRID_RE.test(layer);
    const rank = census.indexOf(layer);
    const width = grid ? 0.7 : Math.max(0.8, 2.6 - 0.7 * rank);
    layers.push({
      id: `gp-outline-${layer}`, type: "line", source: `outline-${layer}`, "source-layer": "features",
      paint: grid
        ? { "line-color": GRID_LINE, "line-width": width, "line-dasharray": [3, 2], "line-opacity": 0.75 }
        : { "line-color": INK, "line-width": ["interpolate", ["linear"], ["zoom"], 10, width * 0.7, 16, width * 1.3], "line-opacity": 0.85 },
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

/** Repaint a target's layer and redraw its sidebar controls and legend. */
function restyle(t) {
  refreshLayer(layerKey(t));
  renderPanel(t);
  saveView();
}

/* ---------- metric switching ---------- */

async function setMetric(t, metric) {
  const s = state[t];
  s.metric = metric;
  s.range = null;
  s.hidden = new Set();
  const d = drawn(t);
  // a flag's gradient centres on its threshold, like the park distance in rule mode
  if (d.flag) s.method = "rule";
  if (s.method === "rule" && (!d.info || d.info.threshold === null)) s.method = "quantile";
  // switch tiles before awaiting the stats: tiles requested meanwhile (e.g. while panning)
  // would otherwise come from the old URL and lack the new metric
  const src = map.getSource(t);
  if (src && d.info) src.setTiles([tileUrl(d.layer, d.info.name)]);
  [s.stats, s.flagStats] = await Promise.all([statsFor(d.layer, d.info && d.info.name), d.flag ? statsFor(d.layer, d.flag.name) : null]);
  restyle(t);
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

/* ---------- sidebar panels: colour controls + legend per target ---------- */

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e[k] = v;
    else e.setAttribute(k, v);
  }
  if (text !== undefined) e.textContent = text;
  return e;
}

function selectOf(options, value, onchange, attrs = {}) {
  const sel = el("select", attrs);
  for (const [v, label, group] of options) {
    let parent = sel;
    if (group) parent = sel.querySelector(`optgroup[label="${group}"]`) || sel.appendChild(el("optgroup", { label: group }));
    parent.append(new Option(label, v, false, v === value));
  }
  sel.onchange = () => onchange(sel.value);
  return sel;
}

function rampPreview(colors) {
  const bar = el("div", { class: "ramp-preview" });
  for (const c of colors) { const sw = el("span"); sw.style.background = c; bar.append(sw); }
  return bar;
}

function renderPanel(t) {
  const panel = $(`${t}-panel`);
  panel.innerHTML = "";
  const s = state[t];
  const { base, flag, info } = drawn(t);
  if (!base) return;

  if (base.kind === "boolean" && base.gradient) {
    const seg = el("div", { class: "seg" });
    for (const [view, label] of [["gradient", "Gradient"], ["flag", "Pass / fail"]]) {
      seg.append(el("button", { type: "button", class: s.view === view ? "on" : "", onclick: () => { s.view = view; setMetric(t, s.metric); } }, label));
    }
    panel.append(seg);
  }
  if (flag && s.flagStats) panel.append(passFail(s.flagStats, s));
  panel.append(styleControls(t, info));
  const legend = el("div", { class: "legend" });
  if (s.stats) legend.append(info.kind === "boolean" ? categories(t) : histogramBlock(t, info));
  if (t === "buildings" && map && map.getZoom() < state.catalog.min_zoom.buildings) {
    legend.append(el("p", { class: "note" }, `Buildings appear from zoom ${state.catalog.min_zoom.buildings} — zoom in, or fill the areas below.`));
  }
  panel.append(legend);
}

function styleControls(t, info) {
  const s = state[t];
  const grid = el("div", { class: "style-grid" });
  const boolean = info.kind === "boolean";
  const rule = boolean || s.method === "rule";

  grid.append(el("span", { class: "label" }, "Colours"));
  const palettes = rule ? Object.keys(DIVERGING) : Object.keys(SEQUENTIAL);
  const current = rule ? s.diverging : s.palette || (info.better ? "Blues" : "YlOrBr");
  const pal = selectOf(palettes.map(p => [p, p]), current, v => { rule ? (s.diverging = v) : (s.palette = v); restyle(t); }, { "aria-label": "Palette" });
  const flip = el("button", { type: "button", class: "flip" + (s.reverse ? " on" : ""), title: "Reverse the colour ramp", onclick: () => { s.reverse = !s.reverse; restyle(t); } }, "Reverse");
  grid.append(pal, flip);
  const preview = rule ? divergingRamp(s) : sequentialRamp(s, info);
  const pv = rampPreview(boolean ? [failColor(s), passColor(s)] : preview);
  pv.style.gridColumn = "1 / -1";
  grid.append(pv);

  if (!boolean) {
    grid.append(el("span", { class: "label" }, "Classes"));
    const methods = Object.entries(METHODS).filter(([m]) => m !== "rule" || info.threshold !== null);
    grid.append(selectOf(methods.map(([m, d]) => [m, d.label]), s.method, v => { s.method = v; s.range = null; restyle(t); }, { "aria-label": "Classification" }));
    const ks = [3, 4, 5, 6, 7, 8, 9].map(k => [String(k), `${k}`]);
    const ksel = selectOf(ks, String(s.k), v => { s.k = +v; restyle(t); }, { class: "k", "aria-label": "Number of classes", title: "Number of classes" });
    ksel.disabled = s.method === "rule" || s.method === "rank";
    grid.append(ksel);
    grid.append(el("p", { class: "method-note" }, s.method === "rule" ? `Split at ${tick(info.threshold, info)} — ${info.better === "low" ? "at or below" : "at or above"} meets the rule.` : METHODS[s.method].note));
  }
  return grid;
}

function passFail(stats, s) {
  const row = el("div", { class: "passfail" });
  const n = stats.true + stats.false;
  const pct = n ? Math.round((100 * stats.true) / n) : 0;
  row.innerHTML = `<span class="swatch" style="background:${passColor(s)}"></span><span>${stats.true.toLocaleString()} meet (${pct}%)</span>
    <span class="swatch" style="background:${failColor(s)}"></span><span>${stats.false.toLocaleString()} don't</span>`;
  return row;
}

function categories(t) {
  const s = state[t];
  const cats = el("div", { class: "cats" });
  for (const [key, label, color, n] of [["true", "Meets", passColor(s), s.stats.true], ["false", "Does not meet", failColor(s), s.stats.false], ["null", "No data", NODATA, s.stats.null]]) {
    if (!n && key === "null") continue;
    const row = el("div", { class: "cat" + (s.hidden.has(key) ? " off" : ""), title: "Click to fade / restore" });
    row.innerHTML = `<span class="swatch" style="background:${color}"></span><span>${label}</span><span>${n.toLocaleString()}</span>`;
    row.onclick = () => { s.hidden.has(key) ? s.hidden.delete(key) : s.hidden.add(key); restyle(t); };
    cats.append(row);
  }
  return cats;
}

/**
 * Histogram bins in value space. Linear scales use the server's exact counts; the log
 * method redraws them on a log axis from the percentiles, so counts there are estimates.
 */
function bins(stats, method) {
  const [lo, hi] = stats.domain;
  const n = stats.n - stats.null;
  if (method !== "log" || !stats.pcts) {
    const w = (hi - lo) / stats.hist.length;
    return { exact: true, scale: v => (v - lo) / (hi - lo), bins: stats.hist.map((c, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, count: c })) };
  }
  const top = Math.log1p(hi - lo), N = 30;
  const p = stats.pcts;
  const cdf = x => {
    if (x <= p[0]) return 0;
    if (x >= p[p.length - 1]) return 1;
    let i = 1;
    while (p[i] < x) i++;
    const span = p[i] - p[i - 1];
    return (i - 1 + (span ? (x - p[i - 1]) / span : 1)) / (p.length - 1);
  };
  const edge = i => lo + Math.expm1((top * i) / N);
  return {
    exact: false,
    scale: v => Math.log1p(Math.max(0, v - lo)) / top,
    bins: Array.from({ length: N }, (_, i) => ({ x0: edge(i), x1: edge(i + 1), count: Math.round(n * (cdf(edge(i + 1)) - cdf(edge(i)))) })),
  };
}

function histogramBlock(t, info) {
  const s = state[t];
  const wrap = el("div");
  if (!s.stats.domain) { wrap.append(el("p", { class: "note" }, "No values to show.")); return wrap; }
  const B = bins(s.stats, s.method);
  const cls = classify(info, s.stats, s);
  const NS = "http://www.w3.org/2000/svg";
  const W = 280, H = 64, top = 4, base = 44;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "hist");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const add = (tag, attrs) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    svg.append(e);
    return e;
  };
  const x = v => B.scale(v) * W;
  const [lo, hi] = s.stats.domain;
  const n = B.bins.length;
  const bw = W / n;
  const max = Math.max(...B.bins.map(b => b.count), 1);
  const inRange = b => !s.range || ((b.x0 + b.x1) / 2 >= s.range[0] && (b.x0 + b.x1) / 2 <= s.range[1]);
  B.bins.forEach((b, i) => {
    const h = b.count ? Math.max(1, (Math.sqrt(b.count) / Math.sqrt(max)) * (base - top)) : 0;
    const on = inRange(b);
    add("rect", { x: i * bw + 0.5, y: base - h, width: Math.max(bw - 1, 0.5), height: h, fill: on ? colorForValue((b.x0 + b.x1) / 2, cls) : FILTERED, stroke: on ? "rgba(29,29,27,.35)" : "none", "stroke-width": 0.5 });
  });
  add("line", { class: "axis", x1: 0, x2: W, y1: base + 0.5, y2: base + 0.5 });
  for (const b of cls.breaks || []) if (b > lo && b < hi) add("line", { class: "axis", x1: x(b), x2: x(b), y1: base, y2: base + 4 });
  const showT = info.threshold !== null && info.threshold > lo && info.threshold < hi;
  if (showT) add("line", { class: "threshold", x1: x(info.threshold), x2: x(info.threshold), y1: top - 4, y2: base + 4 });
  const label = (v, anchor, xx) => { add("text", { x: xx, y: H - 4, "text-anchor": anchor }).textContent = tick(v, info); };
  label(lo, "start", 0);
  label(s.stats.integer && B.exact ? hi - (hi - lo) / n : hi, "end", W);
  if (showT && x(info.threshold) > 30 && x(info.threshold) < W - 30) label(info.threshold, "middle", x(info.threshold));

  // brush: drag across bins to keep only that value range
  let start = null;
  const binAt = ev => {
    const r = svg.getBoundingClientRect();
    return Math.max(0, Math.min(n - 1, Math.floor(((ev.clientX - r.left) / r.width) * n)));
  };
  svg.addEventListener("pointerdown", ev => { start = binAt(ev); svg.setPointerCapture(ev.pointerId); });
  svg.addEventListener("pointerup", ev => {
    if (start === null) return;
    const [i, j] = [Math.min(start, binAt(ev)), Math.max(start, binAt(ev))];
    start = null;
    // the end bins also hold the outliers beyond the 0.5-99.5 % domain
    s.range = i === 0 && j === n - 1 ? null : [i === 0 ? -Infinity : B.bins[i].x0, j === n - 1 ? Infinity : B.bins[j].x1];
    restyle(t);
  });
  wrap.append(svg);

  const foot = el("div", { class: "legend-foot" });
  const valid = s.stats.n - s.stats.null;
  const noun = t === "buildings" ? "buildings" : "areas";
  if (s.range) {
    const shown = B.bins.filter(inRange).reduce((a, b) => a + b.count, 0);
    foot.append(el("span", {}, `${B.exact ? "" : "≈ "}${shown.toLocaleString()} of ${valid.toLocaleString()} shown`));
    foot.append(el("a", { onclick: () => { s.range = null; restyle(t); } }, "clear"));
  } else {
    foot.append(el("span", {}, `${valid.toLocaleString()} ${noun}` + (s.stats.null ? ` · ${s.stats.null.toLocaleString()} no data` : "")));
    foot.append(el("span", {}, "drag to filter"));
  }
  wrap.append(foot);
  return wrap;
}

function renderParksLegend() {
  const box = $("parks-legend");
  box.innerHTML = "";
  const p = state.catalog.parks;
  if (!p || !state.parks.visible) return;
  const min = p.min_area_ha ? `≥ ${p.min_area_ha} ha` : "all sizes";
  box.innerHTML = `<div class="cats">
    <div class="cat"><span class="swatch" style="background:${PARK};opacity:.55;border:1.5px solid ${PARK_LINE}"></span><span>Counted for 300 (${min})</span><span>${p.used.toLocaleString()}</span></div>
    ${p.count > p.used ? `<div class="cat"><span class="swatch dashed" style="border-color:${PARK}"></span><span>Smaller, ignored</span><span>${(p.count - p.used).toLocaleString()}</span></div>` : ""}
  </div>`;
}

/* ---------- study-area summary ---------- */

function renderSummary() {
  const sm = state.catalog.summary;
  const box = $("summary");
  if (!sm) return;
  box.hidden = false;
  // on a phone the card would cover most of the map, so it starts folded
  const pref = loadPref("summaryCollapsed");
  const collapsed = pref === undefined ? window.innerWidth <= 720 : pref === true;
  box.classList.toggle("collapsed", collapsed);
  const figures = [["Buildings", sm.buildings.toLocaleString()]];
  if (sm.trees !== null) figures.push(["Trees", sm.trees.toLocaleString()]);
  if (sm.parks) figures.push(["Parks counted for 300", `${sm.parks.used.toLocaleString()} of ${sm.parks.count.toLocaleString()}`]);
  if (sm.canopy_cover) {
    const cc = sm.canopy_cover;
    figures.push(["Canopy cover", fmt(cc.value, { kind: "percent" }),
      `Total canopy ÷ total measured area of the ${cc.units.toLocaleString()} ${cc.layer} units T30 covered (nodata pixels excluded) — not an average of unit percentages`]);
  }
  for (const m of sm.medians) figures.push([`Median ${m.label[0].toLowerCase()}${m.label.slice(1)}`, fmt(m.value, m)]);

  const share = st => (st && st.true + st.false ? (100 * st.true) / (st.true + st.false) : null);
  const all = share(sm.rule.meets_3_30_300);
  const crit = [["3", "meets_3", "trees"], ["30", "meets_30", "canopy"], ["300", "meets_300", "park"]].filter(([, f]) => sm.rule[f]);

  box.innerHTML = `<h2><span>Study area</span><button type="button" class="toggle" id="summary-toggle" aria-expanded="${!collapsed}"
    title="${collapsed ? "Expand" : "Collapse"} the summary" aria-label="${collapsed ? "Expand" : "Collapse"} the summary">${collapsed ? "+" : "−"}</button></h2>
    <div class="summary-body"><div class="sub"></div><dl class="figures"></dl>${crit.length ? `<div class="rule"></div>` : ""}</div>`;
  box.querySelector(".sub").textContent = state.catalog.study_area_name;
  const dl = box.querySelector(".figures");
  for (const [k, v, how] of figures) {
    const dt = el("dt", {}, k);
    if (how) { dt.title = how; dt.className = "explained"; }
    dl.append(dt, el("dd", {}, v));
  }
  if (sm.canopy_cover) {
    box.querySelector(".summary-body").insertBefore(
      el("p", { class: "method" }, `Canopy: total canopy ÷ total area over all ${sm.canopy_cover.units.toLocaleString()} ${sm.canopy_cover.layer} units.`),
      box.querySelector(".rule"));
  }
  if (crit.length) {
    const r = box.querySelector(".rule");
    const head = el("div", { class: "rule-head" });
    head.append(el("span", {}, "Buildings meeting 3-30-300"), el("strong", {}, all === null ? "—" : `${all.toFixed(1)}%`));
    r.append(head);
    const bars = el("div", { class: "bars" });
    for (const [num, f, what] of crit) {
      const pct = share(sm.rule[f]);
      const bar = el("div", { class: "bar", title: `${sm.rule[f].true.toLocaleString()} of ${(sm.rule[f].true + sm.rule[f].false).toLocaleString()} evaluated buildings` });
      const fillEl = el("i");
      fillEl.style.width = `${pct ?? 0}%`;
      bar.append(fillEl);
      bars.append(el("span", {}, `${num} · ${what}`), bar, el("span", { class: "pct" }, pct === null ? "—" : `${pct.toFixed(1)}%`));
    }
    r.append(bars);
  }
  $("summary-toggle").onclick = () => { savePref("summaryCollapsed", !collapsed); renderSummary(); };
}

/* ---------- collapsible sidebar ---------- */

function setSidebar(collapsed, save = true) {
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  const b = $("sidebar-toggle");
  const what = collapsed ? "Show the control panel" : "Hide the control panel";
  b.textContent = collapsed ? "›" : "‹";
  b.title = what;
  b.setAttribute("aria-label", what);
  b.setAttribute("aria-expanded", String(!collapsed));
  if (save) savePref("sidebarCollapsed", collapsed);
  // the map container changed size
  if (map) map.resize();
}

/* ---------- details ---------- */

async function showDetails(feature) {
  const layerId = feature.layer.id;
  const body = $("details-body");
  if (layerId === "gp-parks") {
    const p = feature.properties;
    state.selected = null;
    const min = state.catalog.parks.min_area_ha;
    body.innerHTML = `<h3></h3><div class="sub">park</div><table></table>`;
    body.querySelector("h3").textContent = p.name || "Unnamed park";
    const rows = [["Area", p.area_ha != null ? `${(+p.area_ha).toFixed(2)} ha` : "—"],
      ["Counted for 300", p.used === true || p.used === "true" ? "yes" : `no (under ${min} ha)`]];
    body.querySelector("table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  } else if (layerId === "gp-trees") {
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
    const label = k => (state.catalog.layers[k] ? state.catalog.layers[k].label : k);
    // unit codes, shown with their level label and name (columns.geo_level_labels / geo_level_names)
    const codes = Object.keys(data).filter(k => k !== "id" && k !== "name" && !k.startsWith("name:") && !metricNames.has(k) && data[k] !== null);
    body.innerHTML = "<h3></h3><div class='sub'></div><table></table>";
    body.querySelector("h3").textContent = isBuilding ? `Building ${id}` : `${label(layer)} ${data.name || id}`;
    body.querySelector(".sub").textContent = codes.map(k => `${label(k)} ${data["name:" + k] ?? data[k]}`).join(" · ") || (isBuilding ? "building" : "area");
    const table = body.querySelector("table");
    for (const m of metrics) {
      const tr = el("tr", { class: m.name === current ? "current" : "" });
      tr.append(el("td", {}, m.label), el("td", {}, fmt(data[m.name], m)));
      table.append(tr);
    }
  }
  const details = $("details");
  details.hidden = false;
  // keep the popup clear of the summary card above it
  const summary = $("summary");
  details.style.maxHeight = !summary.hidden && window.innerWidth > 720
    ? `${Math.max(160, window.innerHeight - summary.getBoundingClientRect().bottom - 24)}px`
    : "";
  refreshLayer("selected");
}

function hideDetails() {
  $("details").hidden = true;
  state.selected = null;
  refreshLayer("selected");
}

/* ---------- controls ---------- */

function setSectionVisible(id, on) {
  $(id).classList.toggle("off", !on);
}

function buildControls() {
  const c = state.catalog;
  $("area-name").textContent = c.study_area_name;

  // basemap
  const bm = $("basemap");
  for (const [key, b] of Object.entries(BASEMAPS)) bm.append(new Option(b.label, key, false, key === state.basemap));
  bm.onchange = () => { state.basemap = bm.value; saveView(); applyStyle(); };

  // buildings
  const bsel = $("building-metric");
  if (c.layers.buildings.metrics.length) {
    fillMetricSelect(bsel, "buildings", state.buildings.metric);
    bsel.onchange = () => setMetric("buildings", bsel.value);
  } else {
    bsel.disabled = true;
    bsel.append(new Option("no building metrics yet", ""));
  }
  $("show-buildings").checked = state.buildings.visible;
  $("show-buildings").onchange = e => { state.buildings.visible = e.target.checked; setSectionVisible("sec-buildings", e.target.checked); restyle("buildings"); };
  setSectionVisible("sec-buildings", state.buildings.visible);

  // areas
  const unitLayers = unitLayerNames();
  if (!unitLayers.length) $("sec-units").hidden = true;
  const lsel = $("unit-layer");
  for (const k of unitLayers) lsel.append(new Option(`${c.layers[k].label} · ${c.layers[k].metrics.length} metrics`, k, false, k === state.units.layer));
  lsel.onchange = async () => {
    const u = state.units;
    u.layer = lsel.value;
    u.metric = defaultMetric(lsel.value);
    u.range = null;
    fillMetricSelect($("unit-metric"), lsel.value, u.metric);
    const d = drawn("units");
    u.stats = await statsFor(u.layer, d.info && d.info.name);
    applyStyle();
    renderPanel("units");
    saveView();
  };
  $("show-units").checked = state.units.visible;
  $("show-units").onchange = e => { state.units.visible = e.target.checked; setSectionVisible("sec-units", e.target.checked); restyle("units"); };
  setSectionVisible("sec-units", state.units.visible);
  const setStyle = style => {
    state.units.style = style;
    for (const o of $("unit-style").querySelectorAll("button")) o.classList.toggle("on", o.dataset.value === style);
    $("unit-fill").hidden = style !== "fill";
  };
  setStyle(state.units.style);
  for (const b of $("unit-style").querySelectorAll("button")) {
    b.onclick = () => {
      setStyle(b.dataset.value);
      if (state.units.style === "fill" && !state.units.visible) { state.units.visible = true; $("show-units").checked = true; setSectionVisible("sec-units", true); }
      restyle("units");
    };
  }
  const usel = $("unit-metric");
  if (state.units.layer) fillMetricSelect(usel, state.units.layer, state.units.metric);
  usel.onchange = () => setMetric("units", usel.value);

  // boundary outlines, any number of unit layers at once
  const ol = $("outlines");
  for (const k of unitLayers) {
    const row = el("label", { class: "row" });
    const box = el("input", { type: "checkbox" });
    box.checked = state.outlines.has(k);
    box.onchange = e => { e.target.checked ? state.outlines.add(k) : state.outlines.delete(k); applyStyle(); saveView(); };
    row.append(box, el("span", {}, c.layers[k].label));
    ol.append(row);
  }
  if (!unitLayers.length) $("outlines-controls").hidden = true;

  // parks
  const pbox = $("show-parks");
  pbox.checked = state.parks.visible;
  if (!c.parks) {
    pbox.disabled = true;
    $("parks-row").classList.add("disabled");
    $("parks-note").textContent = "not available";
  } else {
    $("parks-note").textContent = `· ${c.parks.used.toLocaleString()} counted`;
    pbox.onchange = e => { state.parks.visible = e.target.checked; refreshLayer("parks"); renderParksLegend(); saveView(); };
  }

  // trees
  const tbox = $("show-trees");
  tbox.checked = state.trees.visible;
  if (!c.trees) {
    tbox.disabled = true;
    $("trees-row").classList.add("disabled");
    $("trees-row").title = "No tree polygons configured (data.trees_dir), or the map was started with --no-trees";
    $("trees-note").textContent = "not available";
  } else {
    $("trees-note").textContent = c.trees.sized ? "· to scale" : "· as dots";
    tbox.onchange = e => { state.trees.visible = e.target.checked; refreshLayer("trees"); saveView(); };
  }
  $("sidebar-toggle").onclick = () => setSidebar(!document.body.classList.contains("sidebar-collapsed"));
  $("details-close").onclick = hideDetails;
  document.addEventListener("keydown", e => { if (e.key === "Escape") hideDetails(); });
}

/* ---------- persistence (per browser; best effort) ---------- */

const VIEW_KEY = "greenpy-viz-view";
const STYLE_KEYS = ["method", "k", "palette", "diverging", "reverse", "view"];

function readStore() {
  try { return JSON.parse(localStorage.getItem(VIEW_KEY) || "null") || {}; } catch (_) { return {}; }
}
function writeStore(obj) {
  try { localStorage.setItem(VIEW_KEY, JSON.stringify(obj)); } catch (_) { /* storage unavailable */ }
}
function loadPref(key) { return readStore()[key]; }
function savePref(key, value) { writeStore({ ...readStore(), [key]: value }); }

function saveView() {
  const pick = s => Object.fromEntries(STYLE_KEYS.map(k => [k, s[k]]));
  writeStore({
    ...readStore(),
    area: state.catalog.study_area_name,
    basemap: state.basemap,
    buildingMetric: state.buildings.metric,
    buildingStyle: pick(state.buildings),
    buildings: state.buildings.visible,
    unitLayer: state.units.layer,
    unitMetric: state.units.metric,
    unitStyle: state.units.style,
    unitColours: pick(state.units),
    units: state.units.visible,
    trees: state.trees.visible,
    parks: state.parks.visible,
    outlines: [...state.outlines],
  });
}

function loadView() {
  const v = readStore();
  return v.area === state.catalog.study_area_name ? v : {};
}

/* ---------- boot ---------- */

async function main() {
  const catalog = await getJSON("/api/catalog");
  state.catalog = catalog;
  document.title = `${catalog.study_area_name} · greenpy`;

  const has = (layer, m) => layer && catalog.layers[layer] && catalog.layers[layer].metrics.some(x => x.name === m);
  const saved = loadView();
  const unitLayers = unitLayerNames();
  const restoreStyle = (s, st) => {
    if (!st) return;
    if (METHODS[st.method]) s.method = st.method;
    if (st.k >= 3 && st.k <= 9) s.k = st.k;
    if (SEQUENTIAL[st.palette]) s.palette = st.palette;
    if (DIVERGING[st.diverging]) s.diverging = st.diverging;
    s.reverse = !!st.reverse;
    s.view = st.view === "flag" ? "flag" : "gradient";
  };
  state.basemap = BASEMAPS[saved.basemap] ? saved.basemap : "paper";
  state.parks.visible = !!saved.parks && !!catalog.parks;
  state.outlines = new Set((saved.outlines || []).filter(k => unitLayers.includes(k)));
  state.buildings.visible = saved.buildings !== false;
  state.buildings.metric = has("buildings", saved.buildingMetric) ? saved.buildingMetric : defaultMetric("buildings");
  restoreStyle(state.buildings, saved.buildingStyle);
  state.units.layer = unitLayers.includes(saved.unitLayer) ? saved.unitLayer : unitLayers[unitLayers.length - 1] || null;
  state.units.metric = has(state.units.layer, saved.unitMetric) ? saved.unitMetric : state.units.layer ? defaultMetric(state.units.layer) : null;
  state.units.style = saved.unitStyle === "fill" ? "fill" : "outline";
  state.units.visible = !!saved.units;
  restoreStyle(state.units, saved.unitColours);
  state.trees.visible = !!saved.trees && !!catalog.trees;

  const b = drawn("buildings"), u = drawn("units");
  if (b.flag && !saved.buildingStyle) state.buildings.method = "rule";
  [state.buildings.stats, state.buildings.flagStats, state.units.stats] = await Promise.all([
    b.info ? statsFor("buildings", b.info.name) : null,
    b.flag ? statsFor("buildings", b.flag.name) : null,
    u.info ? statsFor(state.units.layer, u.info.name) : null,
  ]);

  setSidebar(loadPref("sidebarCollapsed") === true, false);
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
  renderPanel("buildings");
  renderPanel("units");
  renderParksLegend();
  renderSummary();

  let lastZoomOk = map.getZoom() >= catalog.min_zoom.buildings;
  map.on("zoomend", () => {
    const ok = map.getZoom() >= catalog.min_zoom.buildings;
    if (ok !== lastZoomOk) { lastZoomOk = ok; renderPanel("buildings"); }
  });
  map.on("click", ev => {
    const order = ["gp-trees", "gp-buildings", "gp-parks", "gp-units-fill"];
    const layers = order.filter(id => map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== "none");
    const hits = map.queryRenderedFeatures(ev.point, { layers });
    if (!hits.length) return hideDetails();
    hits.sort((a, b) => order.indexOf(a.layer.id) - order.indexOf(b.layer.id));
    showDetails(hits[0]);
  });
  for (const id of ["gp-trees", "gp-buildings", "gp-parks", "gp-units-fill"]) {
    map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
    map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
  }
}

main().catch(err => {
  document.body.insertAdjacentHTML("beforeend", `<section class="card" style="top:40%;left:50%;transform:translateX(-50%)"><h1>Could not start the map</h1><p></p></section>`);
  document.querySelector("body > section:last-child p").textContent = String(err);
  console.error(err);
});

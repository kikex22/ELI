'use strict';
/* Graficas SVG sin dependencias. Reglas: colores de serie en orden fijo (nunca ciclados),
   leyenda siempre con 2+ series, marcas finas, rejilla recesiva, texto en tintas de texto (no en el color de la serie)
   y tooltip al pasar el cursor. Cada grafica entrega tambien su tabla de datos. */

const NS = 'http://www.w3.org/2000/svg';
const seriesColor = (i) => `var(--series-${(i % 8) + 1})`;

function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  kids.flat().forEach((kid) => kid && el.append(kid));
  return el;
}
function stext(x, y, text, attrs) {
  const t = svg('text', { x, y, ...attrs });
  t.textContent = text;
  return t;
}
const shorten = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

/* names: textos o {name, i}. `i` es el indice de color estable del modelo (no su posicion en la grafica). */
function legend(names) {
  if (names.length < 2) return null;
  return h('div', { class: 'legend' }, names.map((n, k) => {
    const item = typeof n === 'string' ? { name: n, i: k } : n;
    return h('span', {}, h('i', { style: `background:${seriesColor(item.i)}` }), item.name);
  }));
}

function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function topRounded(x, y, w, hgt, r) {
  r = Math.min(r, w / 2, hgt);
  return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`;
}

/* Barras agrupadas: groups = ['Logica', ...]; series = [{name, values:[0..max|null], tip:[texto|null]}] */
function barChart({ groups, series, yMax = 100, unit = '%', label }) {
  const W = 760, H = 306, pad = { l: 44, r: 10, t: 26, b: 52 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const Y = (v) => pad.t + ph - (Math.min(v, yMax) / yMax) * ph;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': label || 'Grafico de barras' });
  for (let i = 0; i <= 4; i++) {
    const v = (yMax / 4) * i, y = Math.round(Y(v)) + 0.5;
    root.append(svg('line', { x1: pad.l, x2: pad.l + pw, y1: y, y2: y, class: i === 0 ? 'axis' : 'grid' }), stext(pad.l - 6, y + 4, `${fmt(v, 0)}${i === 0 ? '' : unit === '%' ? '' : ''}`, { 'text-anchor': 'end' }));
  }
  root.append(stext(4, 14, unit === '%' ? 'Aciertos (%)' : unit, { 'text-anchor': 'start' }));
  const gw = pw / Math.max(groups.length, 1);
  const bw = Math.max(4, Math.min(24, (gw * 0.8) / Math.max(series.length, 1) - 2));
  groups.forEach((g, gi) => {
    const x0 = pad.l + gi * gw + (gw - (bw + 2) * series.length) / 2;
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (v == null) return;
      const x = x0 + si * (bw + 2), y = Y(v), hgt = Math.max(1, pad.t + ph - y);
      const bar = svg('path', { d: topRounded(x, y, bw, hgt, 4), style: `fill:${seriesColor(s.i ?? si)}`, tabindex: 0 });
      const text = s.tip?.[gi] || `${s.name} · ${g}: ${fmt(v, 1)}${unit}`;
      tipOn(bar, text);
      root.append(bar);
    });
    const lab = stext(pad.l + gi * gw + gw / 2, H - pad.b + 16, shorten(g, 18), { 'text-anchor': 'middle' });
    const full = svg('title', {});
    full.textContent = g;
    lab.append(full);
    root.append(lab);
  });
  return root;
}

/* Dispersion: points = [{name, x, y, i}] con i = indice de serie (color estable por modelo). */
function scatterChart({ points, xLabel, yLabel, xFmt, yFmt, yMax = 100, label }) {
  const W = 760, H = 332, pad = { l: 52, r: 24, t: 28, b: 44 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const xs = points.map((p) => p.x);
  const xMax = niceMax(Math.max(...xs, 0.0001) * 1.08), xMin = 0;
  const X = (v) => pad.l + ((v - xMin) / (xMax - xMin)) * pw;
  const Y = (v) => pad.t + ph - (Math.min(v, yMax) / yMax) * ph;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': label || 'Grafico de dispersion' });
  for (let i = 0; i <= 4; i++) {
    const y = Math.round(Y((yMax / 4) * i)) + 0.5, x = Math.round(X((xMax / 4) * i)) + 0.5;
    root.append(svg('line', { x1: pad.l, x2: pad.l + pw, y1: y, y2: y, class: i === 0 ? 'axis' : 'grid' }),
      stext(pad.l - 6, y + 4, yFmt((yMax / 4) * i), { 'text-anchor': 'end' }),
      svg('line', { x1: x, x2: x, y1: pad.t, y2: pad.t + ph, class: i === 0 ? 'axis' : 'grid' }),
      stext(x, pad.t + ph + 16, xFmt((xMax / 4) * i), { 'text-anchor': 'middle' }));
  }
  root.append(stext(pad.l + pw / 2, H - 6, xLabel, { 'text-anchor': 'middle' }), stext(4, 14, yLabel, { 'text-anchor': 'start' }));
  points.forEach((p) => {
    const cx = X(p.x), cy = Y(p.y), flip = cx > pad.l + pw - 120;
    const dot = svg('circle', { cx, cy, r: 6, style: `fill:${seriesColor(p.i)};stroke:var(--surface-1);stroke-width:2`, tabindex: 0 });
    tipOn(dot, `${p.name}: ${xLabel} ${xFmt(p.x)} · ${yLabel} ${yFmt(p.y)}`);
    root.append(dot, stext(cx + (flip ? -10 : 10), cy + 4, shorten(p.name, 22), { 'text-anchor': flip ? 'end' : 'start', class: 'label' }));
  });
  return root;
}

/* Lineas con eje x categorico: xs = ['2048', ...]; series = [{name, values:[n|null]}] */
function lineChart({ xs, series, yFmt, yLabel, yMax, label }) {
  const W = 760, H = 292, pad = { l: 52, r: 16, t: 28, b: 40 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const top = yMax || niceMax(Math.max(...series.flatMap((s) => s.values.filter((v) => v != null)), 0.0001) * 1.1);
  const X = (i) => pad.l + (xs.length === 1 ? pw / 2 : (i / (xs.length - 1)) * pw);
  const Y = (v) => pad.t + ph - (Math.min(v, top) / top) * ph;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': label || 'Grafico de lineas' });
  for (let i = 0; i <= 4; i++) {
    const y = Math.round(Y((top / 4) * i)) + 0.5;
    root.append(svg('line', { x1: pad.l, x2: pad.l + pw, y1: y, y2: y, class: i === 0 ? 'axis' : 'grid' }), stext(pad.l - 6, y + 4, yFmt((top / 4) * i), { 'text-anchor': 'end' }));
  }
  xs.forEach((x, i) => root.append(stext(X(i), pad.t + ph + 16, x, { 'text-anchor': 'middle' })));
  root.append(stext(4, 14, yLabel, { 'text-anchor': 'start' }));
  series.forEach((s, si) => {
    const ci = s.i ?? si;
    const pts = s.values.map((v, i) => (v == null ? null : [X(i), Y(v), v, i])).filter(Boolean);
    if (pts.length > 1) root.append(svg('polyline', { points: pts.map((p) => `${p[0]},${p[1]}`).join(' '), fill: 'none', style: `stroke:${seriesColor(ci)}`, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    pts.forEach((p) => {
      const dot = svg('circle', { cx: p[0], cy: p[1], r: 4.5, style: `fill:${seriesColor(ci)};stroke:var(--surface-1);stroke-width:2`, tabindex: 0 });
      tipOn(dot, `${s.name} · ${xs[p[3]]}: ${yFmt(p[2])}`);
      root.append(dot);
    });
  });
  return root;
}

/* Contenedor estandar: titulo, leyenda, grafica y tabla de datos plegable. */
function chartBox({ title, note, chart, names, columns, rows }) {
  const table = h('table', { class: 'data' }, h('thead', {}, h('tr', {}, columns.map((c) => h('th', {}, c)))),
    h('tbody', {}, rows.map((r) => h('tr', {}, r.map((c, i) => h('td', { class: i ? 'num' : '' }, c ?? '—'))))));
  return h('div', { class: 'chartbox' }, h('h3', {}, title), note ? h('p', { class: 'hint' }, note) : null, legend(names), chart,
    h('details', {}, h('summary', { class: 'hint' }, 'Ver como tabla'), h('div', { class: 'scroll' }, table)));
}

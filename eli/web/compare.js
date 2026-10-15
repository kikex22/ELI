'use strict';
/* Compare: tablas, clasificaciones, graficas y filtros por tamano. Solo se compara lo que todos hicieron en las mismas condiciones. */

let cmpModels = [];
let cmpBuckets = [];
let cmpData = null;
let cmpSort = { key: 'quality', dir: -1 };
const MAX_COMPARE = 8;

const RANK_TITLE = {
  reasoning: 'Mejor en razonamiento', code: 'Mejor en programación', instructions: 'Mejor siguiendo instrucciones',
  fastest: 'Más rápido', lowest_vram: 'Menor consumo de VRAM', efficiency: 'Mejor eficiencia general',
};
const RANK_VALUE = {
  reasoning: (v) => pct(v), code: (v) => pct(v), instructions: (v) => pct(v), fastest: (v) => `${fmt(v, 1)} tok/s`,
  lowest_vram: (v) => `${fmt(v, 2)} GiB`, efficiency: (v) => `${v} frente(s) de Pareto`,
};

async function loadCompareSetup() {
  const r = await rawApi('/api/compare/models');
  if (!r.ok) { banner(r.data.error); return; }
  cmpModels = r.data.models.filter((m) => m.has_results);
  cmpBuckets = r.data.size_buckets;
  const sel = $('cmp-size');
  if (sel.options.length === 1) cmpBuckets.forEach((b) => sel.append(h('option', { value: b.id }, b.id)));
  renderModelChecks();
  const ref = $('cmp-similar'), keep = ref.value;
  clear(ref).append(h('option', { value: '' }, '—'), ...cmpModels.filter((m) => m.meta.parameters_b).map((m) => h('option', { value: m.name }, `${m.name} (${m.meta.parameter_size})`)));
  ref.value = cmpModels.some((m) => m.name === keep) ? keep : '';
  $('cmp-sub').textContent = cmpModels.length ? `${cmpModels.length} modelos con resultados` : '';
  if (!cmpModels.length) clear($('cmp-out')).append(h('div', { class: 'card' }, h('p', { class: 'hint' }, 'Todavía no hay modelos evaluados. Ejecuta una evaluación desde Benchmark.')));
}

function bucketRange() {
  const b = cmpBuckets.find((x) => x.id === $('cmp-size').value);
  return b ? { min: b.min, max: b.max } : null;
}

function renderModelChecks() {
  const keep = new Set(checked($('cmp-models'))), range = bucketRange(), box = clear($('cmp-models'));
  for (const m of cmpModels) {
    const p = m.meta.parameters_b;
    if (range && (p == null || p < range.min || p >= range.max)) continue;
    const input = h('input', { type: 'checkbox', value: m.name, checked: keep.has(m.name) });
    input.addEventListener('change', () => {
      if (checked(box).length > MAX_COMPARE) { input.checked = false; banner(`Como máximo ${MAX_COMPARE} modelos.`); }
    });
    box.append(h('label', {}, input, h('span', {}, m.name, h('small', {}, [m.meta.parameter_size, m.meta.quantization, `evaluado ${fmtDate(m.created)}`].filter(Boolean).join(' · ')))));
  }
  if (!box.children.length) box.append(h('p', { class: 'hint' }, range ? 'Ningún modelo evaluado tiene ese tamaño (o no se conoce su número de parámetros).' : 'Sin modelos.'));
}

function selectSimilar() {
  const ref = cmpModels.find((m) => m.name === $('cmp-similar').value);
  if (!ref) return;
  const lo = ref.meta.parameters_b * 0.7, hi = ref.meta.parameters_b * 1.3;
  $('cmp-models').querySelectorAll('input').forEach((i) => {
    const m = cmpModels.find((x) => x.name === i.value);
    i.checked = m.meta.parameters_b != null && m.meta.parameters_b >= lo && m.meta.parameters_b <= hi;
  });
}

async function runCompare() {
  const names = checked($('cmp-models'));
  if (!names.length) { banner('Elige al menos un modelo.'); return; }
  banner('');
  const q = new URLSearchParams({ models: names.join(',') });
  const range = bucketRange();
  if (range) { q.set('param_min', range.min); q.set('param_max', range.max); }
  const r = await rawApi('/api/compare?' + q);
  if (!r.ok) { banner(r.data.error || 'No se pudo comparar'); return; }
  cmpData = r.data;
  const key = $('cmp-sort').value;
  cmpSort = { key, dir: key === 'vram' ? 1 : -1 };
  renderCompare();
}

const sortValue = {
  name: (m) => m.name, params: (m) => m.meta.parameters_b, quality: (m) => m.indicators.quality, speed: (m) => m.indicators.speed_tok_s,
  ttft: (m) => m.performance.ttft_s, vram: (m) => m.indicators.vram_gib, disk: (m) => m.indicators.size_gib, energy: (m) => m.indicators.wh_per_token,
  qvram: (m) => m.indicators.quality_per_gib_vram, efficiency: (m) => m.pareto_fronts,
};

function sortedModels() {
  const f = sortValue[cmpSort.key] || sortValue.quality;
  return [...cmpData.models].sort((a, b) => {
    const x = f(a), y = f(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1; // sin dato siempre al final
    if (y == null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * cmpSort.dir;
  });
}

function renderCompare() {
  const d = cmpData, out = clear($('cmp-out'));
  if (!d.models.length) { out.append(h('div', { class: 'card' }, d.warnings.map((w) => h('p', { class: 'hint' }, w)))); return; }
  const color = new Map(d.models.map((m, i) => [m.name, i])); // el color sigue al modelo, no a su posicion en el orden
  out.append(h('div', { class: 'card' }, h('h2', {}, 'Clasificación', h('span', { class: 'sub' }, `${d.common.conditions} pruebas comparables`)),
    d.warnings.map((w) => h('p', { class: 'hint' }, '⚠ ', w)),
    h('div', { class: 'cards' }, Object.entries(RANK_TITLE).map(([k, title]) => rankCard(k, title, d.rankings[k]))),
    h('p', { class: 'hint' }, `No hay un indicador único. Se necesitan ≥ ${d.thresholds.min_graded_category} pruebas calificadas por categoría, ≥ ${d.thresholds.min_speed_samples} mediciones de velocidad y ≥ 2 modelos con datos para clasificar.`)));
  out.append(h('div', { class: 'card' }, h('h2', {}, 'Resultados por modelo'), h('div', { class: 'scroll' }, mainTable(color)),
    h('p', { class: 'hint' }, 'Las pruebas «sin calificar» (respuesta cortada, ambigua, sin sandbox…) no cuentan como acierto ni como fallo. Calidad = superadas / calificadas. La VRAM es lo usado sobre lo que ya había antes de cargar el modelo. Wh y tok/s son del modelo caliente; la energía es una estimación.')));
  out.append(h('div', { class: 'card' }, h('h2', {}, 'Por categoría'), categoryTable(), ...charts(color)));
  if (d.quant_groups.length) {
    out.append(h('div', { class: 'card' }, h('h2', {}, 'Misma familia, distinta cuantización'),
      d.quant_groups.map((g) => h('p', {}, `${g.family} ${g.parameter_size}: `, g.models.map((m) => `${m.name} (${m.quantization || 'cuantización desconocida'})`).join(' · ')))));
  }
}

function rankCard(key, title, r) {
  if (!r || r.status !== 'ok') return h('div', { class: 'rank insufficient' }, h('div', { class: 't' }, title), h('div', { class: 'w' }, 'Datos insuficientes'), h('div', { class: 'b' }, r ? r.reason : ''));
  const v = r.values[r.winners[0]];
  const card = h('div', { class: 'rank' }, h('div', { class: 't' }, title), h('div', { class: 'w' }, r.winners.join(' · ')), h('div', { class: 'b' }, `${RANK_VALUE[key](v)}${r.winners.length > 1 ? ' (empate)' : ''}`), h('div', { class: 'b' }, r.basis));
  tipOn(card, () => r.order.map((n, i) => `${i + 1}. ${n}: ${RANK_VALUE[key](r.values[n])}`).join('\n'));
  return card;
}

function mainTable(color) {
  const cols = [['name', 'Modelo'], ['params', 'Params'], [null, 'Cuantización'], [null, 'Pruebas'], ['quality', 'Calidad'], ['speed', 'tok/s'], ['ttft', 'TTFT s'], ['vram', 'VRAM GiB'],
    ['disk', 'Disco GiB'], ['energy', 'mWh/token*'], ['qvram', 'Calidad/GiB'], [null, 'Contexto comprobado'], ['efficiency', 'Frentes Pareto']];
  const head = h('tr', {}, cols.map(([k, label]) => h('th', { class: k ? 'sortable' : '', title: k ? 'Ordenar' : '', onclick: k ? () => { cmpSort = { key: k, dir: cmpSort.key === k ? -cmpSort.dir : (k === 'vram' || k === 'name' || k === 'ttft' || k === 'energy' || k === 'disk' ? 1 : -1) }; renderCompare(); } : null },
    label + (cmpSort.key === k ? (cmpSort.dir > 0 ? ' ▲' : ' ▼') : ''))));
  const rows = sortedModels().map((m) => {
    const o = m.overall, i = m.indicators, c = m.context;
    return h('tr', {},
      h('td', { class: 'name' }, h('i', { style: `display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:${seriesColor(color.get(m.name))}` }),
        h('a', { href: '#history', class: 'link', onclick: (e) => { e.preventDefault(); window.pendingModel = m.name; openRun(m.run_id); } }, m.name)),
      h('td', { class: 'num' }, unk(m.meta.parameter_size)), h('td', {}, unk(m.meta.quantization)),
      h('td', {}, h('div', { class: 'chips' }, countsChips(o, ['passed', 'failed', 'ungraded', 'error', 'timeout']))),
      h('td', { class: 'num', title: `${o.graded} pruebas calificadas` }, i.quality != null ? pct(i.quality) : h('span', { class: 'unknown', title: `Solo ${o.graded} pruebas calificadas (mínimo ${cmpData.thresholds.min_graded_overall})` }, 'n/d')),
      h('td', { class: 'num', title: `${m.performance.decode_samples} mediciones; variación ${m.performance.decode_cv ?? 'n/d'}` }, i.speed_tok_s != null ? fmt(i.speed_tok_s, 1) : h('span', { class: 'unknown' }, 'n/d')),
      h('td', { class: 'num' }, m.performance.ttft_s != null ? fmt(m.performance.ttft_s, 2) : '—'),
      h('td', { class: 'num', title: `Pico de GPU ${fmtMiB(m.vram.gpu_peak_mib) || '—'}; antes de cargar ${fmtMiB(m.vram.idle_mib) || '—'}` }, i.vram_gib != null ? fmt(i.vram_gib, 2) : '—'),
      h('td', { class: 'num' }, i.size_gib != null ? fmt(i.size_gib, 1) : '—'),
      h('td', { class: 'num', title: m.energy.valid ? 'Estimación' : `Solo ${m.energy.samples} mediciones (mínimo ${cmpData.thresholds.min_energy_samples})` }, m.energy.valid && m.energy.wh_per_token != null ? fmt(m.energy.wh_per_token * 1000, 2) : h('span', { class: 'unknown' }, 'n/d')),
      h('td', { class: 'num' }, i.quality_per_gib_vram != null ? fmt(i.quality_per_gib_vram, 2) : '—'),
      h('td', { class: 'num', title: 'Mayor prompt (tokens procesados) con el que recuperó el dato; no es el contexto configurado.' }, c.verified_tokens != null ? `${c.verified_tokens.toLocaleString('es')} tok` : (c.max_requested ? h('span', { class: 'unknown' }, 'ninguno superado') : h('span', { class: 'muted' }, 'sin prueba'))),
      h('td', { class: 'num' }, m.pareto_fronts));
  });
  return h('table', { class: 'data' }, h('thead', {}, head), h('tbody', {}, rows));
}

function categoryTable() {
  const cats = [...new Set(cmpData.models.flatMap((m) => Object.keys(m.categories)))];
  const names = suiteInfo?.categories_names || {};
  return h('div', { class: 'scroll' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, h('th', {}, 'Categoría'), cmpData.models.map((m) => h('th', { class: 'num' }, shortName(m.name))))),
    h('tbody', {}, cats.map((c) => h('tr', {}, h('td', {}, names[c] || c), cmpData.models.map((m) => {
      const t = m.categories[c];
      if (!t) return h('td', { class: 'num' }, '—');
      return h('td', { class: 'num', title: `${t.passed} superadas, ${t.failed} falladas, ${t.ungraded} sin calificar, ${t.error + t.timeout} errores` },
        t.graded ? `${t.passed}/${t.graded}` : 'n/d', t.ungraded ? h('span', { class: 'muted' }, ` (+${t.ungraded}?)`) : null);
    }))))));
}

const shortName = (n) => (n.length > 26 ? `…${n.slice(-25)}` : n);

function charts(color) {
  const d = cmpData, boxes = [];
  const legendItems = d.models.map((m) => ({ name: m.name, i: color.get(m.name) }));
  const cats = [...new Set(d.models.flatMap((m) => Object.keys(m.categories)))];
  const names = suiteInfo?.categories_names || {};
  const groups = cats.map((c) => names[c] || c);
  const series = d.models.map((m) => ({
    name: m.name, i: color.get(m.name), values: cats.map((c) => (m.categories[c]?.graded >= d.thresholds.min_graded_category ? m.categories[c].score * 100 : null)),
    tip: cats.map((c) => { const t = m.categories[c]; return t ? `${m.name} · ${names[c] || c}: ${t.passed}/${t.graded} superadas${t.ungraded ? `, ${t.ungraded} sin calificar` : ''}` : null; }),
  }));
  // las barras usan el color del modelo (no su posicion): se reordena la paleta con el indice estable
  const svgBars = barChart({ groups, series, label: 'Calidad por categoría' });
  boxes.push(chartBox({ title: 'Calidad por categoría', note: 'Porcentaje de aciertos sobre pruebas calificadas. Una categoría sin barra no tiene suficientes pruebas calificadas.',
    chart: svgBars, names: legendItems, columns: ['Modelo', ...groups], rows: series.map((s) => [s.name, ...s.values.map((v) => (v == null ? null : `${fmt(v, 0)} %`))]) }));

  const scat = [
    ['calidad_vs_vram', 'Calidad frente a VRAM', 'vram_gib', 'VRAM (GiB)', (v) => fmt(v, 1)],
    ['calidad_vs_velocidad', 'Calidad frente a velocidad', 'speed_tok_s', 'tok/s', (v) => fmt(v, 0)],
    ['calidad_vs_tamano', 'Calidad frente a tamaño del modelo', 'size_gib', 'Disco (GiB)', (v) => fmt(v, 0)],
    ['calidad_vs_energia', 'Calidad frente a consumo energético (estimado)', 'wh_per_token', 'mWh por token', (v) => fmt(v * 1000, 1)],
  ];
  for (const [front, title, key, xLabel, xFmt] of scat) {
    const pts = d.models.filter((m) => m.indicators[key] != null && m.indicators.quality != null).map((m) => ({ name: m.name, x: m.indicators[key], y: m.indicators.quality * 100, i: color.get(m.name) }));
    if (pts.length < 2) continue;
    const front_names = d.pareto[front] || [];
    boxes.push(chartBox({ title, note: `Frente de Pareto (no dominados): ${front_names.join(', ') || '—'}. Un modelo está en el frente si ningún otro es a la vez mejor en calidad y en este recurso.`,
      chart: scatterChart({ points: pts, xLabel, yLabel: 'Calidad %', xFmt, yFmt: (v) => fmt(v, 0), label: title }), names: legendItems,
      columns: ['Modelo', xLabel, 'Calidad %', 'En el frente'], rows: pts.map((p) => [p.name, xFmt(p.x), fmt(p.y, 0), front_names.includes(p.name) ? 'sí' : 'no']) }));
  }

  const withCtx = d.models.filter((m) => m.context.curve.length);
  if (withCtx.length) {
    const sizes = [...new Set(withCtx.flatMap((m) => m.context.curve.map((r) => r.num_ctx_requested)))].sort((a, b) => a - b);
    const at = (m, s, f) => { const r = m.context.curve.find((x) => x.num_ctx_requested === s); return r ? f(r) : null; };
    boxes.push(chartBox({ title: 'Contexto largo: tiempo hasta el primer token', note: 'Tamaño de contexto pedido (num_ctx). El prompt ocupa ≈ 60 % del contexto; los tokens realmente procesados están en la tabla.',
      chart: lineChart({ xs: sizes.map(String), series: withCtx.map((m) => ({ name: m.name, i: color.get(m.name), values: sizes.map((s) => at(m, s, (r) => r.ttft_s)) })), yFmt: (v) => `${fmt(v, 1)} s`, yLabel: 'TTFT', label: 'TTFT frente al contexto' }),
      names: withCtx.map((m) => ({ name: m.name, i: color.get(m.name) })), columns: ['Modelo', ...sizes.map((s) => `${s} (tokens procesados · TTFT)`)],
      rows: withCtx.map((m) => [m.name, ...sizes.map((s) => at(m, s, (r) => `${r.prompt_tokens_processed ?? '?'} · ${r.ttft_s != null ? fmt(r.ttft_s, 2) : '?'} s`))]) }));
    boxes.push(chartBox({ title: 'Contexto largo: acierto al recuperar el dato', note: 'Porcentaje de aciertos sobre pruebas calificadas en cada tamaño. Un hueco significa sin pruebas calificadas.',
      chart: lineChart({ xs: sizes.map(String), series: withCtx.map((m) => ({ name: m.name, i: color.get(m.name), values: sizes.map((s) => at(m, s, (r) => (r.graded ? r.score * 100 : null))) })), yFmt: (v) => `${fmt(v, 0)} %`, yLabel: 'Acierto', yMax: 100, label: 'Acierto frente al contexto' }),
      names: withCtx.map((m) => ({ name: m.name, i: color.get(m.name) })), columns: ['Modelo', ...sizes.map(String)],
      rows: withCtx.map((m) => [m.name, ...sizes.map((s) => at(m, s, (r) => (r.graded ? `${r.passed}/${r.graded}` : 'n/d')))]) }));
  }
  return boxes;
}

$('cmp-run').addEventListener('click', runCompare);
$('cmp-size').addEventListener('change', renderModelChecks);
$('cmp-similar').addEventListener('change', selectSimilar);
onView('compare', async () => {
  if (!suiteInfo) { const s = await rawApi('/api/bench/suite'); if (s.ok) suiteInfo = s.data; }
  await loadCompareSetup();
});

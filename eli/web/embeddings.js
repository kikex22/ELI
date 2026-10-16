'use strict';
/* Embeddings: evaluacion de modelos de embeddings con un panel oscuro de tiles y paneles.
   `tile` y `panel` estan pensados para llevarse despues al resto de secciones. */

let embRun = null; // {run, status, events} de la ejecucion mostrada
let embTimer = null;
let embWasActive = false;

/* Umbrales de los tiles: provisionales, para ajustar con datos reales. good/warn = limites inferiores. */
const EMB_LIMITS = {
  recall1: { good: 0.7, warn: 0.4 }, recall5: { good: 0.9, warn: 0.7 }, mrr: { good: 0.8, warn: 0.5 },
  sim: { good: 0.9, warn: 0.7 }, stab: { good: 0.9999, warn: 0.999 },
};
const embLevel = (key, v) => (v == null || !EMB_LIMITS[key] ? 'none' : v >= EMB_LIMITS[key].good ? 'good' : v >= EMB_LIMITS[key].warn ? 'warn' : 'bad');
const embPct = (v, d = 1) => (v == null ? null : `${fmt(v * 100, d)} %`);

/* Tile de un numero: etiqueta, valor, unidad, nivel de color y una linea de contexto. */
function tile({ label, value, unit, level = 'none', hint }) {
  return h('div', { class: `stat ${level}` }, h('div', { class: 'stat-label' }, label),
    h('div', { class: 'stat-value' }, value ?? '—', value != null && unit ? h('small', {}, ` ${unit}`) : null),
    hint ? h('div', { class: 'stat-hint' }, hint) : null);
}

/* Panel con titulo y ancho en columnas (de 12). */
function panel(title, body, { span = 6, sub } = {}) {
  return h('div', { class: `panel span-${span}` }, h('h3', {}, title, sub ? h('span', { class: 'sub' }, sub) : null), body);
}

const embSteps = (events, model) => Object.fromEntries(events.filter((e) => e.type === 'step' && e.model === model && e.status === 'ok').map((e) => [e.step, e]));
const embErrors = (events, model) => events.filter((e) => e.type === 'step' && e.model === model && e.status === 'error');

/* ---------- configuracion ---------- */
async function loadEmbSetup() {
  const [suite, models] = await Promise.all([rawApi('/api/emb/suite'), rawApi('/api/emb/models')]);
  if (suite.ok) $('emb-suite').textContent = `suite ${suite.data.id} ${suite.data.version} · ${suite.data.documents} documentos · ${suite.data.queries} consultas · ${suite.data.triplets} tripletes`;
  const keep = new Set(checked($('emb-models')));
  const box = clear($('emb-models'));
  const list = models.ok ? models.data.models : [];
  const llmSel = $('emb-llm'), chosenLlm = llmSel.value;
  clear(llmSel);
  put(llmSel, h('option', { value: '' }, 'Ninguno (solo las escritas a mano)'));
  for (const m of (models.ok ? models.data.llms || [] : [])) put(llmSel, h('option', { value: m.name }, m.name));
  llmSel.value = chosenLlm;
  if (!list.length) put(box, h('p', { class: 'hint' }, 'No hay modelos de embeddings instalados en Ollama (por ejemplo: ollama pull nomic-embed-text).'));
  for (const m of list) {
    const input = h('input', { type: 'checkbox', value: m.name, checked: keep.has(m.name) || list.length === 1 });
    input.addEventListener('change', updateEmbSummary);
    box.append(h('label', {}, input, h('span', {}, m.name, h('small', {}, [m.architecture, m.embedding_length ? `${m.embedding_length} dim.` : null, fmtBytes(m.disk_bytes)].filter(Boolean).join(' · ') || 'sin datos'))));
  }
  updateEmbSummary();
}

function updateEmbSummary() {
  const n = checked($('emb-models')).length;
  $('emb-summary').textContent = n ? `${n} modelo${n > 1 ? 's' : ''}, uno tras otro` : 'Elige al menos un modelo.';
  $('emb-start').disabled = !n;
}

const showEmbError = (t) => { $('emb-error').hidden = !t; $('emb-error').textContent = t || ''; };

function embBody() {
  const folder = $('emb-folder').value.trim(), llm = $('emb-llm').value;
  return { models: checked($('emb-models')), prefixes: $('emb-prefixes').value, corpus_mode: $('emb-corpus-mode').value, repetitions: Number($('emb-reps').value) || 5,
    unload_between: $('emb-unload').checked, ...(folder ? { corpus_folder: folder } : {}),
    ...(llm ? { subquery_llm: llm, subquery_lang: $('emb-llm-lang').value, subquery_count: Number($('emb-llm-count').value) || 6,
      iter_rounds: Number($('emb-iter').value) || 0 } : {}) };
}

async function previewEmbFolder() {
  const out = clear($('emb-preview-out')), folder = $('emb-folder').value.trim();
  if (!folder) { out.textContent = 'Escribe la ruta de una carpeta.'; return; }
  const r = await postJson('/api/emb/corpus-preview', { folder, mode: $('emb-corpus-mode').value });
  if (!r.ok) { put(out, h('span', { class: 'error' }, r.data.error)); return; }
  const s = r.data.stats;
  const kinds = Object.entries(s.queries_by_kind || {}).map(([k, n]) => `${n} ${k}`).join(', ');
  const langs = Object.entries(s.code_by_language || {}).map(([k, n]) => `${n} ${k}`).join(', ');
  put(out, `${(s.files || 0) + (s.code_files || 0)} archivos · ${s.documents} fragmentos${langs ? ` (código: ${langs})` : ''} · ${s.queries} consultas (${kinds}). Ejemplos: ${r.data.sample_queries.slice(0, 3).join(' · ')}`,
    r.data.warnings.length ? h('div', {}, r.data.warnings.join(' ')) : null);
}

async function startEmb() {
  showEmbError('');
  const body = embBody();
  let r = await postJson('/api/emb/start', body);
  if (r.status === 409 && r.data.code === 'models_loaded') {
    const ok = await askDialog({
      title: 'Descargar otros modelos de la VRAM',
      body: h('div', {}, h('p', {}, 'Para medir sin interferencias hay que liberar la VRAM. Se descargarán de memoria (no se borran del disco):'),
        h('ul', {}, r.data.loaded.map((n) => h('li', {}, n)))),
      okLabel: 'Descargar y empezar',
    });
    if (!ok) return;
    r = await postJson('/api/emb/start', { ...body, unload_others: true });
  }
  if (!r.ok) { showEmbError(r.data.error || `Error ${r.status}`); return; }
  window.pendingEmbRun = r.data.run_id;
  pollEmb();
}

/* ---------- progreso ---------- */
async function pollEmb() {
  clearTimeout(embTimer);
  const r = await rawApi('/api/emb/status');
  if (!r.ok) return;
  const d = r.data, box = $('emb-progress');
  box.hidden = !d.run_id;
  $('emb-cancel').hidden = !d.active;
  $('emb-start').disabled = d.active || !checked($('emb-models')).length;
  $('emb-dot').hidden = !d.active;
  if (d.run_id) {
    $('emb-pfill').style.width = `${d.total ? Math.round((d.done / d.total) * 100) : 0}%`;
    $('emb-pstate').textContent = STATE_LABEL[d.state] || d.state;
    $('emb-pinfo').textContent = d.active ? `${d.current_model || '…'} · paso: ${d.current_step || '…'} · ${d.done}/${d.total} pasos · ${fmtDur(d.elapsed_s)}`
      : `${d.done}/${d.total} pasos · ${d.counts.error} con error · ${d.counts.skipped} omitidos${d.message ? ` · ${d.message}` : ''}`;
  }
  if (d.active) embTimer = setTimeout(pollEmb, 1000);
  if (embWasActive && !d.active) { window.pendingEmbRun = d.run_id; await loadEmbRuns(); }
  embWasActive = d.active;
}

/* ---------- ejecuciones y panel ---------- */
async function loadEmbRuns() {
  const r = await rawApi('/api/emb/history');
  const sel = clear($('emb-run')), runs = r.ok ? r.data.runs : [];
  if (!runs.length) {
    embRun = null;
    put(clear($('emb-dash')), h('div', { class: 'panel span-12' }, h('p', { class: 'hint' }, 'Todavía no hay evaluaciones de embeddings. Elige un modelo y pulsa «Iniciar evaluación».')));
    return;
  }
  runs.forEach((x) => sel.append(h('option', { value: x.run_id }, `${fmtDate(x.created)} · ${(x.models || []).join(', ')} · ${STATE_LABEL[x.state] || x.state}`)));
  const want = window.pendingEmbRun && runs.some((x) => x.run_id === window.pendingEmbRun) ? window.pendingEmbRun : runs[0].run_id;
  window.pendingEmbRun = null;
  sel.value = want;
  await showEmbRun(want);
}

async function showEmbRun(runId) {
  const r = await rawApi(`/api/emb/history/${encodeURIComponent(runId)}`);
  if (!r.ok) { banner(r.data.error); return; }
  embRun = r.data;
  const sel = clear($('emb-model'));
  (embRun.run.models || []).forEach((m) => sel.append(h('option', { value: m.name }, m.name)));
  const link = $('emb-export');
  link.hidden = false;
  link.href = `/api/emb/history/${encodeURIComponent(runId)}/export?format=json`;
  renderEmbDash();
}

function embGpuFacts(steps, stats) {
  const g = steps.performance?.gpu;
  const emb = steps.performance?.embeddings_done;
  const perK = g?.energy_net_wh != null && emb ? (g.energy_net_wh / emb) * 1000 : null;
  return { g, perK, vram: stats?.vram_used_mib };
}

function renderEmbDash() {
  const dash = clear($('emb-dash'));
  if (!embRun) return;
  const model = $('emb-model').value || embRun.run.models[0]?.name;
  const events = embRun.events, steps = embSteps(events, model), errs = embErrors(events, model);
  const stats = (embRun.status.models || {})[model] || {};
  const start = events.find((e) => e.type === 'model_start' && e.model === model);
  const ret = steps.retrieval, sim = steps.similarity, stab = steps.stability, perf = steps.performance, user = steps.retrieval_user, warm = steps.warmup;
  const { g, perK, vram } = embGpuFacts(steps, stats);
  const px = start?.prefixes;
  $('emb-meta').textContent = `suite ${embRun.run.suite.id} ${embRun.run.suite.version} · prefijos: ${px ? (px.mode === 'task' ? (px.rule || 'sin regla para este modelo') : 'ninguno') : '—'} · ${embRun.status.state === 'running' ? 'en curso' : STATE_LABEL[embRun.status.state] || ''}`;

  const m = ret?.metrics || {};
  const tiles = h('div', { class: 'stats' },
    tile({ label: 'Recall@1', value: embPct(m['recall@1']), level: embLevel('recall1', m['recall@1']), hint: ret ? `${m.n} consultas` : 'sin datos' }),
    tile({ label: 'Recall@5', value: embPct(m['recall@5']), level: embLevel('recall5', m['recall@5']) }),
    tile({ label: 'MRR', value: m.mrr != null ? fmt(m.mrr, 3) : null, level: embLevel('mrr', m.mrr) }),
    tile({ label: 'Similitud', value: embPct(sim?.accuracy), level: embLevel('sim', sim?.accuracy), hint: sim ? `${sim.triplets.length} tripletes` : null }),
    tile({ label: 'Estabilidad', value: stab?.min_cos_repeat != null ? fmt(Math.min(stab.min_cos_repeat, stab.min_cos_batch ?? 1), 5) : null, level: embLevel('stab', stab ? Math.min(stab.min_cos_repeat, stab.min_cos_batch ?? 1) : null), hint: 'coseno mínimo' }),
    tile({ label: 'Cobertura de archivos @10', value: user?.coverage ? embPct(user.coverage['cov@10']) : null, hint: user?.coverage ? `${user.coverage.n} preguntas de varios archivos` : 'necesita tu carpeta' }),
    tile({ label: 'Cobertura por archivo @5', value: user?.file_level?.multi ? embPct(user.file_level.multi['cov@5']) : null, hint: user?.file_level?.multi ? `top 5 archivos · ${user.file_level.multi.n} preguntas` : 'necesita tu carpeta' }),
    tile({ label: 'Con subpreguntas (2 archivos c/u)', value: user?.subquery ? embPct(user.subquery.by_m['2'].coverage) : null,
      hint: user?.subquery ? `pregunta sola, mismo nº de archivos: ${embPct(user.subquery.by_m['2'].baseline, 0)}` : 'necesita subpreguntas' }),
    tile({ label: 'Latencia (80 pal.)', value: perf ? fmt(perf.latency.medium.median_ms, 0) : null, unit: 'ms', hint: perf ? `${perf.latency.medium.tokens ?? '?'} tokens` : null }),
    tile({ label: 'Velocidad (lote 32)', value: perf?.throughput['32'] ? fmt(perf.throughput['32'].emb_per_s, 0) : null, unit: 'emb/s' }),
    tile({ label: 'VRAM del modelo', value: vram != null ? fmt(vram, 0) : (stats.model_vram_bytes ? fmt(stats.model_vram_bytes / 2 ** 20, 0) : null), unit: 'MiB', hint: vram != null ? 'medida en la GPU' : stats.model_vram_bytes ? 'según Ollama' : null }),
    tile({ label: 'Energía / 1000 emb.', value: perK != null ? fmt(perK, 3) : null, unit: 'Wh', hint: g?.power_mean_w != null ? `${fmt(g.power_mean_w, 0)} W de media` : null }));
  dash.append(tiles);

  if (errs.length) dash.append(h('div', { class: 'panel span-12 warnbox' }, h('h3', {}, 'Pasos con error'), h('ul', {}, errs.map((e) => h('li', {}, `${e.step}: ${e.error}`)))));

  const grid = h('div', { class: 'pgrid' });
  if (ret) {
    const series = [{ name: 'Mismo idioma', values: ['recall@1', 'recall@3', 'recall@5', 'mrr'].map((k) => (ret.by_kind.mono[k] ?? null) == null ? null : ret.by_kind.mono[k] * 100), i: 0 },
      { name: 'Entre idiomas', values: ['recall@1', 'recall@3', 'recall@5', 'mrr'].map((k) => (ret.by_kind.cross[k] ?? null) == null ? null : ret.by_kind.cross[k] * 100), i: 1 }];
    if (user) {
      const kinds = Object.keys(user.by_kind || {});
      (kinds.length ? kinds : [null]).forEach((k, j) => series.push({ name: k ? `Tuyo: ${k}` : 'Tus documentos', values: ['recall@1', 'recall@3', 'recall@5', 'mrr'].map((m) => { const v = (k ? user.by_kind[k] : user.metrics)[m]; return v == null ? null : v * 100; }), i: 2 + j }));
    }
    put(grid, panel('Recuperación', chartBox({ title: 'Recall y MRR (%)', chart: barChart({ groups: ['Recall@1', 'Recall@3', 'Recall@5', 'MRR'], series, label: 'Recuperación' }), names: series.map((s) => ({ name: s.name, i: s.i })),
      columns: ['Métrica', ...series.map((s) => s.name)], rows: ['Recall@1', 'Recall@3', 'Recall@5', 'MRR'].map((g, i) => [g, ...series.map((s) => (s.values[i] == null ? null : fmt(s.values[i], 1)))]) }), { span: 6, sub: 'mayor es mejor' }));
    put(grid, panel('Consultas falladas', failedTable(ret, user), { span: 6, sub: 'posición > 1' }));
  }
  if (user?.coverage) put(grid, panel('Cobertura de archivos', coveragePanel(user), { span: 12, sub: 'preguntas de «cómo funciona» con varios archivos' }));
  if (user?.subquery) put(grid, panel('Descomposición en subpreguntas', subqueryPanel(user), { span: 12, sub: 'cobertura de archivos con la unión de las subpreguntas' }));
  if (steps.iterative?.metrics) put(grid, panel('Búsqueda iterativa', iterPanel(steps.iterative), { span: 12, sub: 'el LLM decide qué buscar en cada ronda, viendo lo ya encontrado' }));
  if (user?.subquery_llm) put(grid, panel('Subpreguntas del LLM', llmPanel(user, events.filter((e) => e.type === 'subquery_gen' && e.model === embRun.run.config?.subquery_llm)), { span: 12, sub: 'mismas preguntas, sin ver los archivos' }));
  if (user?.file_level) put(grid, panel('Recuperación agrupada por archivo', fileLevelPanel(user), { span: 12, sub: 'cada archivo vale lo que su mejor fragmento' }));
  if (sim) {
    const kinds = ['es', 'en', 'cross'], names = { es: 'Español', en: 'Inglés', cross: 'Entre idiomas' };
    put(grid, panel('Similitud por tripletes', chartBox({ title: 'Acierto: paráfrasis más cerca que el negativo (%)',
      chart: barChart({ groups: kinds.map((k) => names[k]), series: [{ name: 'Acierto', values: kinds.map((k) => (sim.by_kind[k] == null ? null : sim.by_kind[k] * 100)), i: 0 }], label: 'Similitud' }),
      names: ['Acierto'], columns: ['Tipo', 'Acierto'], rows: kinds.map((k) => [names[k], embPct(sim.by_kind[k])]) }), { span: 6 }));
  }
  if (perf) {
    const lat = perf.latency, labs = [['short', 'Corto'], ['medium', 'Medio'], ['long', 'Largo']];
    put(grid, panel('Latencia por tamaño de texto', chartBox({ title: 'Mediana (ms)', chart: barChart({ groups: labs.map(([, n]) => n), unit: 'ms', yMax: niceMax(Math.max(...labs.map(([k]) => lat[k].median_ms)) * 1.1), label: 'Latencia',
      series: [{ name: 'Mediana', values: labs.map(([k]) => lat[k].median_ms), i: 0, tip: labs.map(([k]) => `${lat[k].words} palabras (${lat[k].tokens ?? '?'} tokens): ${fmt(lat[k].median_ms, 1)} ms · mín ${fmt(lat[k].min_ms, 1)} · máx ${fmt(lat[k].max_ms, 1)}`) }] }),
      names: ['Mediana'], columns: ['Texto', 'Palabras', 'Tokens', 'Mediana ms', 'Mín', 'Máx'], rows: labs.map(([k, n]) => [n, lat[k].words, lat[k].tokens, fmt(lat[k].median_ms, 1), fmt(lat[k].min_ms, 1), fmt(lat[k].max_ms, 1)]) }), { span: 6, sub: `n=${lat.short.n}` }));
    const sizes = Object.keys(perf.throughput);
    put(grid, panel('Rendimiento por tamaño de lote', chartBox({ title: 'Embeddings por segundo', chart: lineChart({ xs: sizes, yFmt: (v) => fmt(v, 0), yLabel: 'emb/s', label: 'Rendimiento',
      series: [{ name: 'emb/s', values: sizes.map((s) => perf.throughput[s].emb_per_s), i: 0 }] }), names: ['emb/s'],
      columns: ['Lote', 'emb/s', 'Mediana s'], rows: sizes.map((s) => [s, fmt(perf.throughput[s].emb_per_s, 1), fmt(perf.throughput[s].median_s, 4)]) }), { span: 6 }));
  }
  put(grid, panel('Recursos y condiciones', resourcesList({ g, perK, vram, stats, warm, start, stab, ret, user }), { span: 6 }));
  dash.append(grid);
  if ((embRun.run.models || []).length > 1) dash.append(embCompareTable());
}

function failedTable(ret, user) {
  const rows = ret.queries.filter((q) => q.rank !== 1).map((q) => ({ ...q, set: 'banco' }))
    .concat((user?.queries || []).filter((q) => q.rank !== 1).map((q) => ({ ...q, set: 'tuyo' })));
  if (!rows.length) return h('p', { class: 'hint' }, 'Todas las consultas acertaron en la primera posición.');
  return h('div', { class: 'scroll tall' }, h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['Consulta', 'Tipo', 'Posición'].map((c) => h('th', {}, c)))),
    h('tbody', {}, rows.slice(0, 40).map((q) => h('tr', {}, h('td', { title: q.source || '' }, q.query), h('td', {}, q.set === 'tuyo' ? 'tuyo' : q.kind === 'cross' ? 'entre idiomas' : 'mismo idioma'), h('td', { class: 'num' }, q.rank ?? '—'))))),
    rows.length > 40 ? h('p', { class: 'hint' }, `Se muestran 40 de ${rows.length}.`) : null);
}

/* Para explicar «como funciona X» hacen falta varios archivos: que fraccion de ellos aparece entre los k mejores fragmentos. */
function coveragePanel(user) {
  const kinds = Object.keys(user.coverage_by_kind || {});
  const groups = ['Cobertura @5', 'Cobertura @10', 'Completas @10'];
  const val = (c, k) => (c[k] == null ? null : c[k] * 100);
  const series = (kinds.length ? kinds : ['todas']).map((k, j) => {
    const c = kinds.length ? user.coverage_by_kind[k] : user.coverage;
    return { name: `${k} (n=${c.n})`, values: [val(c, 'cov@5'), val(c, 'cov@10'), val(c, 'full@10')], i: j };
  });
  const rows = user.queries.filter((q) => q.coverage).sort((a, b) => a.coverage['10'] - b.coverage['10'] || a.coverage['5'] - b.coverage['5']);
  const table = h('div', { class: 'scroll tall' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['Pregunta', 'Archivos', 'Frag. @10', 'Arch. @5', 'Arch. @8'].map((c) => h('th', {}, c)))),
    h('tbody', {}, rows.map((q) => h('tr', {}, h('td', { title: q.source || '' }, q.query), h('td', { class: 'num' }, q.files), h('td', { class: 'num' }, embPct(q.coverage['10'], 0)), h('td', { class: 'num' }, q.file_level ? embPct(q.file_level.coverage['5'], 0) : '—'), h('td', { class: 'num' }, q.file_level ? embPct(q.file_level.coverage['8'], 0) : '—'))))));
  return h('div', {}, chartBox({ title: 'Cobertura (%)', chart: barChart({ groups, series, label: 'Cobertura de archivos' }), names: series.map((s) => ({ name: s.name, i: s.i })),
    columns: ['Medida', ...series.map((s) => s.name)], rows: groups.map((g, i) => [g, ...series.map((s) => (s.values[i] == null ? null : fmt(s.values[i], 1)))]) }),
  h('p', { class: 'hint' }, 'Cobertura: fracción de los archivos necesarios que aparece entre los k mejores fragmentos. «Completas»: preguntas con todos sus archivos en los 10 primeros. Ordenadas de peor a mejor.'), table);
}

/* Descomponer «como funciona X» en preguntas atomicas: se unen los m mejores archivos de cada una. La comparacion justa es contra
   la pregunta original con el MISMO numero de archivos (si no, mirar mas archivos ya mejoraria la cobertura por si solo). */
function subqueryPanel(user) {
  const sq = user.subquery, ms = Object.keys(sq.by_m);
  const groups = ms.map((m) => `${m} archivo${m === '1' ? '' : 's'} por subpregunta`);
  const series = [{ name: 'Con subpreguntas', values: ms.map((m) => sq.by_m[m].coverage * 100), i: 0 },
    { name: 'Pregunta sola (mismo nº de archivos)', values: ms.map((m) => sq.by_m[m].baseline * 100), i: 1 }];
  const rows = user.queries.filter((q) => q.subquery).sort((a, b) => b.subquery.by_m['2'].coverage - a.subquery.by_m['2'].coverage);
  const table = h('div', { class: 'scroll tall' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['Pregunta', 'Subpreg.', 'Archivos', 'Unión (m=2)', 'Con subpreg.', 'Sola'].map((c) => h('th', {}, c)))),
    h('tbody', {}, rows.map((q) => h('tr', {}, h('td', { title: (q.subqueries || []).join(' · ') }, q.query), h('td', { class: 'num' }, q.subquery.subqueries), h('td', { class: 'num' }, q.files),
      h('td', { class: 'num' }, q.subquery.by_m['2'].union), h('td', { class: 'num' }, embPct(q.subquery.by_m['2'].coverage, 0)), h('td', { class: 'num' }, embPct(q.subquery.by_m['2'].baseline, 0)))))));
  const full = ms.map((m) => `m=${m}: ${embPct(sq.by_m[m].full, 0)} con subpreguntas contra ${embPct(sq.by_m[m].full_baseline, 0)} sola`).join(' · ');
  return h('div', {}, chartBox({ title: 'Cobertura de archivos (%)', chart: barChart({ groups, series, label: 'Cobertura con subpreguntas' }), names: series.map((s) => ({ name: s.name, i: s.i })),
    columns: ['m', 'Archivos en la unión', 'Con subpreguntas', 'Pregunta sola', 'Ganancia'], rows: ms.map((m) => [m, fmt(sq.by_m[m].union_mean, 1), embPct(sq.by_m[m].coverage), embPct(sq.by_m[m].baseline), `${sq.by_m[m].gain >= 0 ? '+' : ''}${fmt(sq.by_m[m].gain * 100, 1)} pts`]) }),
  h('p', { class: 'hint' }, `${sq.n} preguntas, ${fmt(sq.subqueries_mean, 1)} subpreguntas y ${fmt(sq.files_mean, 1)} archivos de media. Preguntas con todos sus archivos: ${full}.`), table);
}

/* Busqueda iterativa: cobertura de la union de todas las busquedas vs la pregunta sola con el mismo numero de archivos. */
function iterPanel(it) {
  const ms = Object.keys(it.metrics.by_m);
  const series = [
    { name: 'Iterativa', i: 0, values: ms.map((m) => it.metrics.by_m[m].coverage * 100) },
    { name: 'Pregunta sola (mismo nº)', i: 1, values: ms.map((m) => it.metrics.by_m[m].baseline * 100) },
  ];
  const groups = ms.map((m) => `${m} archivo${m === '1' ? '' : 's'} por búsqueda`);
  const qs = it.queries || [], asked = qs.filter((q) => q.phrases?.length);
  const examples = asked.slice(0, 3).map((q) => h('li', {}, h('strong', {}, q.id), ': ', (q.phrases || []).join(' · ')));
  return h('div', {}, chartBox({ title: 'Cobertura de archivos (%)', chart: barChart({ groups, series, label: 'Búsqueda iterativa' }), names: series.map((s) => ({ name: s.name, i: s.i })),
    columns: ['m', ...series.map((s) => s.name), 'Ganancia'], rows: ms.map((m, j) => [m, ...series.map((s) => fmt(s.values[j], 1)), `${it.metrics.by_m[m].gain >= 0 ? '+' : ''}${fmt(it.metrics.by_m[m].gain * 100, 1)} pts`]) }),
  h('p', { class: 'hint' }, `${it.metrics.n} preguntas de varios archivos, ${fmt(it.rounds_mean ?? 0, 1)} rondas de media. Preguntas con todos sus archivos: ${ms.map((m) => `m=${m}: ${embPct(it.metrics.by_m[m].full, 0)} frente a ${embPct(it.metrics.by_m[m].full_baseline, 0)}`).join(' · ')}.`),
  examples.length ? h('details', {}, h('summary', {}, 'Frases que pidió el LLM'), h('ul', {}, examples)) : null);
}

/* Subpreguntas a mano vs generadas por el LLM, solo en las preguntas que tienen ambas (mismo conjunto, comparacion justa). */
function llmPanel(user, gen) {
  const ms = Object.keys(user.subquery_llm.by_m);
  const both = user.queries.filter((q) => q.subquery && q.subquery_llm);
  const avg = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const llm = user.subquery_llm, g = gen[0];
  const series = [
    { name: 'Escritas a mano', i: 0, values: ms.map((m) => (both.length ? avg(both.map((q) => q.subquery.by_m[m].coverage)) : null)) },
    { name: 'Generadas por LLM', i: 1, values: ms.map((m) => (both.length ? avg(both.map((q) => q.subquery_llm.by_m[m].coverage)) : llm.by_m[m].coverage)) },
    { name: 'Pregunta sola (mismo nº)', i: 2, values: ms.map((m) => (both.length ? avg(both.map((q) => q.subquery_llm.by_m[m].baseline)) : llm.by_m[m].baseline)) },
  ].map((s) => ({ ...s, values: s.values.map((v) => (v == null ? null : v * 100)) }));
  const groups = ms.map((m) => `${m} archivo${m === '1' ? '' : 's'} por subpregunta`);
  const examples = (g?.examples || []).map((e) => h('li', {}, h('strong', {}, e.query), h('ul', {}, e.subqueries.map((s) => h('li', {}, s)))));
  return h('div', {}, chartBox({ title: 'Cobertura de archivos (%)', chart: barChart({ groups, series, label: 'Subpreguntas a mano vs LLM' }), names: series.map((s) => ({ name: s.name, i: s.i })),
    columns: ['m', ...series.map((s) => s.name)], rows: ms.map((m, j) => [m, ...series.map((s) => (s.values[j] == null ? null : fmt(s.values[j], 1)))]) }),
  h('p', { class: 'hint' }, `${llm.n} preguntas con subpreguntas del LLM${both.length ? `; ${both.length} también con las escritas a mano (la tabla compara solo esas)` : ''}. ` +
    (g ? `Generadas: ${g.generated} de ${g.queries} preguntas${g.cached ? `, ${g.cached} desde la caché` : ''}${g.failed ? `, ${g.failed} sin respuesta usable` : ''}, en ${fmt(g.duration_s, 0)} s.` : '')),
  examples.length ? h('details', {}, h('summary', {}, 'Ejemplos de subpreguntas generadas'), h('ul', {}, examples)) : null);
}

/* Los archivos se ordenan por su mejor fragmento: un archivo grande (muchas ventanas) ya no llena el ranking. */
function fileLevelPanel(user) {
  const kinds = Object.keys(user.file_level_by_kind || {});
  const groups = ['Acierto @1', 'Acierto @3', 'Acierto @5', 'Cobertura @5', 'Completas @8'];
  const v = (x) => (x == null ? null : x * 100);
  const cells = (f) => [v(f['hit@1']), v(f['hit@3']), v(f['hit@5']), f.multi ? v(f.multi['cov@5']) : null, f.multi ? v(f.multi['full@8']) : null];
  const series = (kinds.length ? kinds : ['todas']).map((k, j) => {
    const f = kinds.length ? user.file_level_by_kind[k] : user.file_level;
    return { name: `${k} (n=${f.n})`, values: cells(f), i: j };
  });
  return h('div', {}, chartBox({ title: 'Recuperación por archivo (%)', chart: barChart({ groups, series, label: 'Recuperación por archivo' }), names: series.map((s) => ({ name: s.name, i: s.i })),
    columns: ['Medida', ...series.map((s) => s.name)], rows: groups.map((g, i) => [g, ...series.map((s) => (s.values[i] == null ? null : fmt(s.values[i], 1)))]) }),
  h('p', { class: 'hint' }, 'Acierto @k: el mejor archivo correcto está entre los k primeros archivos. Cobertura @5 y «Completas @8» solo cuentan preguntas con varios archivos: fracción de sus archivos entre los 5 primeros, y preguntas con todos sus archivos entre los 8 primeros.'));
}

function resourcesList({ g, perK, vram, stats, warm, start, stab, ret, user }) {
  const facts = [
    ['Dimensión', warm?.dims], ['Norma del vector', warm?.norm != null ? `${fmt(warm.norm, 3)}${Math.abs(warm.norm - 1) < 0.01 ? ' (normalizado)' : ' (sin normalizar)'}` : null],
    ['Carga del modelo', warm?.load_s != null ? `${fmt(warm.load_s, 2)} s` : null],
    ['VRAM del modelo (Ollama)', fmtBytes(stats.model_vram_bytes)], ['VRAM usada (medida)', fmtMiB(vram)], ['Pico de VRAM de la GPU', fmtMiB(stats.vram_gpu_peak_mib)],
    ['Potencia media', g?.power_mean_w != null ? `${fmt(g.power_mean_w, 0)} W` : null], ['Pico de potencia', g?.power_peak_w != null ? `${fmt(g.power_peak_w, 0)} W` : null],
    ['Energía neta del paso', g?.energy_net_wh != null ? `${fmt(g.energy_net_wh, 4)} Wh` : null], ['Energía por 1000 embeddings', perK != null ? `${fmt(perK, 3)} Wh` : null],
    ['Temperatura máx.', g?.temp_max_c != null ? `${g.temp_max_c} °C` : null], ['Frenado de la GPU', g ? (g.throttle?.length ? g.throttle.join(', ') : 'no') : null],
    ['Referencia TensorRT', stab?.reference ? `${stab.reference.vectors.length} vectores guardados (${stab.reference.fingerprint})` : null],
    ['Corpus propio', user ? `${user.timing.documents} fragmentos · ${user.corpus_fingerprint.slice(0, 12)}` : null],
  ];
  const dl = h('dl', { class: 'facts' });
  facts.forEach(([k, v]) => dl.append(h('dt', {}, k), h('dd', {}, v ?? h('span', { class: 'unknown' }, 'sin dato'))));
  return dl;
}

function embCompareTable() {
  const rows = embRun.run.models.map((mm) => {
    const s = embSteps(embRun.events, mm.name), st = (embRun.status.models || {})[mm.name] || {};
    return [mm.name, embPct(s.retrieval?.metrics['recall@1']), embPct(s.retrieval?.metrics['recall@5']), s.retrieval?.metrics.mrr != null ? fmt(s.retrieval.metrics.mrr, 3) : null,
      embPct(s.retrieval?.by_kind.cross['recall@1']), s.retrieval_user?.coverage ? embPct(s.retrieval_user.coverage['cov@10']) : null, s.retrieval_user?.file_level?.multi ? embPct(s.retrieval_user.file_level.multi['cov@5']) : null, embPct(s.similarity?.accuracy), s.performance ? fmt(s.performance.latency.medium.median_ms, 0) : null,
      s.performance?.throughput['32'] ? fmt(s.performance.throughput['32'].emb_per_s, 0) : null, fmtMiB(st.vram_used_mib)];
  });
  return panel('Comparación de modelos de esta ejecución', h('div', { class: 'scroll' }, h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['Modelo', 'R@1', 'R@5', 'MRR', 'R@1 entre idiomas', 'Cobertura @10', 'Archivos @5', 'Similitud', 'ms (80 pal.)', 'emb/s (lote 32)', 'VRAM'].map((c) => h('th', {}, c)))),
    h('tbody', {}, rows.map((r) => h('tr', {}, r.map((c, i) => h('td', { class: i ? 'num' : 'name' }, c ?? '—')))))), ),
  { span: 12, sub: 'solo si las condiciones son iguales' });
}

$('emb-start').addEventListener('click', startEmb);
$('emb-cancel').addEventListener('click', async () => { $('emb-cancel').disabled = true; await postJson('/api/emb/cancel', {}); $('emb-cancel').disabled = false; pollEmb(); });
$('emb-preview').addEventListener('click', previewEmbFolder);
$('emb-run').addEventListener('change', () => showEmbRun($('emb-run').value));
$('emb-model').addEventListener('change', renderEmbDash);

onView('embeddings', async () => { await loadEmbSetup(); await loadEmbRuns(); pollEmb(); });
setInterval(async () => {
  if (currentView === 'embeddings') return;
  const r = await rawApi('/api/status').catch(() => null);
  if (r?.ok) $('emb-dot').hidden = !r.data.emb_running;
}, 5000);

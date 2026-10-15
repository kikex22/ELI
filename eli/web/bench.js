'use strict';
/* Benchmark: seleccion de modelos, configuracion, inicio, progreso y cancelacion. */

let suiteInfo = null;
let benchInventory = [];
let benchTimer = null;
const CTX_CHOICES = [2048, 4096, 8192, 16384, 32768];
const MAX_BENCH_MODELS = 8;

const checked = (box) => [...box.querySelectorAll('input:checked')].map((i) => i.value);

async function loadBenchSetup() {
  if (!suiteInfo) {
    const s = await rawApi('/api/bench/suite');
    if (!s.ok) { $('b-error').hidden = false; $('b-error').textContent = s.data.error; return; }
    suiteInfo = s.data;
    buildCategoryBoxes();
    buildContextBoxes();
  }
  const r = await rawApi('/api/models/inspect');
  if (!r.ok) { banner(r.data.error); return; }
  banner('');
  const keep = new Set(checked($('bench-models')));
  benchInventory = r.data.models;
  const box = clear($('bench-models'));
  for (const m of benchInventory) {
    const text = !(m.capabilities || []).includes('completion');
    const input = h('input', { type: 'checkbox', value: m.name, disabled: text, checked: keep.has(m.name) });
    input.addEventListener('change', () => {
      if (checked(box).length > MAX_BENCH_MODELS) { input.checked = false; showBenchError(`Como máximo ${MAX_BENCH_MODELS} modelos por ejecución.`); }
      updateSummary();
    });
    box.append(h('label', { class: text ? 'disabled' : '', title: text ? 'No genera texto (sin capacidad «completion»): no se puede evaluar.' : '' }, input,
      h('span', {}, m.name, h('small', {}, [m.parameter_size, m.quantization.value, fmtBytes(m.disk_bytes), m.loaded ? 'cargado' : null].filter(Boolean).join(' · ') || 'sin datos'))));
  }
  updateSummary();
}

function buildCategoryBoxes() {
  const box = clear($('b-cats'));
  for (const [id, name] of Object.entries(suiteInfo.categories_names)) {
    const input = h('input', { type: 'checkbox', value: id, checked: true });
    input.addEventListener('change', updateSummary);
    box.append(h('label', {}, input, h('span', { 'data-cat': id }, name)));
  }
  refreshCategoryCounts();
}

function refreshCategoryCounts() {
  const mode = $('b-mode').value;
  for (const el of $('b-cats').querySelectorAll('span[data-cat]')) {
    const c = suiteInfo.categories[el.dataset.cat];
    const n = mode === 'quick' ? c.quick : c.total;
    clear(el).append(suiteInfo.categories_names[el.dataset.cat], h('small', {}, `${n} ${n === 1 ? 'prueba' : 'pruebas'}${c.context_sized ? ' · por tamaño de contexto' : ''}`));
  }
}

function buildContextBoxes() {
  const box = clear($('b-ctxsizes'));
  for (const s of CTX_CHOICES) {
    const input = h('input', { type: 'checkbox', value: String(s) });
    input.addEventListener('change', updateSummary);
    box.append(h('label', { class: s >= 16384 ? 'warn-size' : '', title: s >= 16384 ? 'Usa mucha VRAM y tarda bastante' : '' }, input, `${s.toLocaleString('es')} tokens${s >= 16384 ? ' ⚠' : ''}`));
  }
  applyDefaultSizes();
}

function applyDefaultSizes() {
  const defaults = suiteInfo.context_sizes[$('b-mode').value];
  $('b-ctxsizes').querySelectorAll('input').forEach((i) => { i.checked = defaults.includes(Number(i.value)); });
}

function plannedCounts() {
  const mode = $('b-mode').value, cats = checked($('b-cats')), sizes = checked($('b-ctxsizes')).length;
  let plain = 0, ctx = 0;
  for (const t of suiteInfo.tests) {
    if (!cats.includes(t.category) || (mode === 'quick' && !t.quick)) continue;
    if (t.context_sized) ctx += 1; else plain += 1;
  }
  return { plain, ctx, sizes, tests: plain + ctx * sizes };
}

function updateSummary() {
  if (!suiteInfo) return;
  const models = checked($('bench-models')).length, reps = Number($('b-reps').value) || 1, p = plannedCounts();
  $('b-summary').textContent = models
    ? `${models} modelo(s) × ${p.tests} pruebas × ${reps} rep. = ${models * p.tests * reps} ejecuciones` + (p.ctx && !p.sizes ? ' · elige al menos un tamaño de contexto' : '')
    : 'Elige al menos un modelo.';
  $('b-start').disabled = !models || !p.tests || (p.ctx && !p.sizes && p.plain === 0);
}

function showBenchError(text) { const e = $('b-error'); e.hidden = !text; e.textContent = text || ''; }

function configBody() {
  const cats = checked($('b-cats')), all = Object.keys(suiteInfo.categories_names);
  const num = (id) => ($(id).value === '' ? null : Number($(id).value));
  const think = $('b-think').value;
  return {
    models: checked($('bench-models')), mode: $('b-mode').value, categories: cats.length === all.length ? null : cats,
    repetitions: num('b-reps'), temperature: num('b-temp'), seed: num('b-seed'), num_ctx: num('b-ctx'), num_predict: num('b-pred'),
    think: think === '' ? null : think === 'true', timeout_s: num('b-timeout'), warmup: $('b-warmup').checked, unload_between: $('b-unload').checked,
    context_sizes: checked($('b-ctxsizes')).map(Number),
  };
}

async function startBench() {
  showBenchError('');
  let body = configBody();
  if (!plannedCounts().ctx || !body.context_sizes.length) delete body.context_sizes;
  let r = await postJson('/api/bench/start', body);
  if (r.status === 409 && r.data.code === 'models_loaded') {
    const ok = await askDialog({
      title: 'Descargar otros modelos de la VRAM',
      body: h('div', {}, h('p', {}, 'Para medir sin interferencias hay que liberar la VRAM. Se descargarán de memoria (no se borran del disco):'),
        h('ul', {}, r.data.loaded.map((n) => h('li', {}, n)))),
      okLabel: 'Descargar y empezar',
    });
    if (!ok) return;
    r = await postJson('/api/bench/start', { ...body, unload_others: true });
  }
  if (!r.ok) { showBenchError(r.data.error || `Error ${r.status}`); return; }
  $('bp-link').hidden = true;
  pollBench();
}

function renderProgress(st) {
  const card = $('bench-progress');
  if (!st.run_id) { card.hidden = true; return; }
  card.hidden = false;
  const frac = st.total ? st.done / st.total : 0;
  $('bp-state').textContent = `${STATE_LABEL[st.state] || st.state} · ${st.run_id}`;
  $('bp-fill').style.width = `${Math.round(frac * 100)}%`;
  $('bp-bar').setAttribute('aria-valuenow', String(Math.round(frac * 100)));
  const where = st.active
    ? `Modelo ${st.model_index}/${st.models.length}: ${st.current_model || '…'}${st.current_test ? ` · ${st.current_category} · ${st.current_test} (repetición ${st.current_repetition})` : ' · cargando/calentando'}`
    : (st.message || 'Terminada.');
  $('bp-info').textContent = `${where} · ${st.done}/${st.total} pruebas · ${fmtDur(st.elapsed_s)}${st.eta_s != null ? ` · quedan ≈ ${fmtDur(st.eta_s)} (estimación)` : ''}`;
  clear($('bp-counts')).append(...countsChips(st.counts, ['passed', 'failed', 'ungraded', 'error', 'timeout', 'cancelled', 'skipped']));
  const line = (r) => h('div', {}, r.error ? r.error : `${r.outcome === 'passed' ? '✓' : r.outcome === 'failed' ? '✗' : r.outcome === 'ungraded' ? '?' : '!'} ${r.model} · ${r.test}${r.reason ? ` — ${r.reason}` : ''}`);
  clear($('bp-recent')).append(h('div', { class: 'lines' }, [...st.recent].reverse().map(line)));
  clear($('bp-errors')).append(st.errors.length ? h('div', { class: 'lines' }, st.errors.map(line)) : h('p', { class: 'hint' }, 'Sin errores.'));
  $('bp-cancel').hidden = !st.active;
  $('bp-cancel').disabled = false;
  $('bp-link').hidden = st.active;
  $('bench-dot').hidden = !st.active;
  $('b-start').disabled = st.active;
  if (!st.active) $('bp-link').onclick = (e) => { e.preventDefault(); openRun(st.run_id); };
}

async function pollBench() {
  clearTimeout(benchTimer);
  const r = await rawApi('/api/bench/status').catch(() => null);
  if (!r?.ok) return;
  renderProgress(r.data);
  if (r.data.active) benchTimer = setTimeout(pollBench, 1000);
  else if (currentView === 'benchmark') updateSummary();
}

async function cancelBench() {
  $('bp-cancel').disabled = true;
  await postJson('/api/bench/cancel', {});
  pollBench();
}

$('b-mode').addEventListener('change', () => { refreshCategoryCounts(); applyDefaultSizes(); updateSummary(); });
['b-reps', 'b-temp', 'b-seed'].forEach((id) => $(id).addEventListener('input', updateSummary));
$('b-start').addEventListener('click', startBench);
$('bp-cancel').addEventListener('click', cancelBench);
onView('benchmark', async () => { await loadBenchSetup(); pollBench(); });

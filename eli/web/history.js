'use strict';
/* History: ejecuciones anteriores, detalle de cada resultado y exportacion. Los datos historicos son de solo lectura. */

let histDetail = null;
const OUTCOME_ICON = { passed: '✓', failed: '✗', ungraded: '?', error: '!', timeout: '!', cancelled: '–' };
const PROV_LABEL = { ollama: 'medido por Ollama', derivado: 'calculado de valores de Ollama', cliente: 'medido por el reloj de ELI', 'nvidia-smi': 'lecturas de nvidia-smi', estimado: 'estimación' };

async function loadHistory() {
  const r = await rawApi('/api/history');
  if (!r.ok) { banner(r.data.error || 'No se pudo leer el historial'); return; }
  banner('');
  $('hist-sub').textContent = `${r.data.runs.length} ejecuciones`;
  put(clear($('hist-runs')), runsTable(r.data.runs, (run) => [
    h('button', { type: 'button', onclick: () => openRunDetail(run.run_id) }, 'Ver'),
    h('a', { class: 'link', href: `/api/history/${run.run_id}/export?format=json`, download: '' }, 'JSON'),
    h('a', { class: 'link', href: `/api/history/${run.run_id}/export?format=csv`, download: '' }, 'CSV')]));
  const chats = r.data.chats;
  put(clear($('hist-chats')), chats.length ? h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['Fecha', 'Modelo', 'Estado', 'Tokens', 'tok/s'].map((c) => h('th', {}, c)))),
    h('tbody', {}, chats.map((c) => h('tr', {}, h('td', {}, fmtDate(c.started)), h('td', { class: 'name' }, c.model), h('td', {}, c.status),
      h('td', { class: 'num' }, c.gen_tokens ?? '—'), h('td', { class: 'num' }, c.decode_tok_s ?? '—'))))) : h('p', { class: 'hint' }, 'Todavía no hay chats guardados.'));
  if (window.pendingRun) { const id = window.pendingRun; window.pendingRun = null; openRunDetail(id); }
}

async function openRunDetail(runId) {
  const r = await rawApi('/api/history/' + encodeURIComponent(runId));
  if (!r.ok) { banner(r.data.error); return; }
  histDetail = r.data;
  $('hist-detail').hidden = false;
  $('hd-title').textContent = runId;
  renderRunDetail();
  $('hist-detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderRunDetail() {
  const { run, status, results, events } = histDetail, cfg = run.config, hw = run.hardware || {}, gpu = hw.gpu || {};
  const body = clear($('hd-body'));
  const dur = status.finished && run.created ? (new Date(status.finished) - new Date(run.created)) / 1000 : null;
  const facts = [
    ['Estado', STATE_LABEL[status.state] || status.state], ['Inicio', fmtDate(run.created)], ['Fin', unk(fmtDate(status.finished))], ['Duración', unk(fmtDur(dur))],
    ['Suite', `${run.suite.id} ${run.suite.version}`], ['Hash de la suite', run.suite.sha256.slice(0, 16)], ['ELI', run.eli_version], ['Ollama', unk(run.ollama_version)],
    ['GPU', unk(gpu.name, (v) => `${v} · driver ${gpu.driver || '?'} · SM ${gpu.compute_cap ?? '?'} · ${fmtMiB(gpu.vram_total_mib) || '?'}`)],
    ['CPU / RAM', `${hw.cpu?.model || 'desconocido'} · ${fmtMiB(hw.ram_total_mib) || '?'}`], ['Sandbox de código', run.sandbox?.available ? run.sandbox.kind : 'no disponible'],
    ['Configuración', `${cfg.mode === 'quick' ? 'rápido' : 'completo'} · ${cfg.repetitions} rep. · temperatura ${cfg.temperature} · semilla ${cfg.seed} · razonamiento ${cfg.think === null ? 'por defecto' : cfg.think ? 'activado' : 'desactivado'} · tiempo máx. ${cfg.timeout_s} s`
      + (cfg.num_ctx ? ` · contexto base ${cfg.num_ctx}` : '') + (cfg.num_predict ? ` · tokens máx. ${cfg.num_predict}` : '') + (cfg.context_sizes ? ` · contextos ${cfg.context_sizes.join(', ')}` : '')],
  ];
  const dl = h('dl', { class: 'facts' });
  facts.forEach(([k, v]) => dl.append(h('dt', {}, k), h('dd', {}, v)));
  put(body, dl, status.message ? h('p', { class: 'hint' }, status.message) : null,
    h('div', { class: 'chips' }, countsChips(status.counts || {}, ['passed', 'failed', 'ungraded', 'error', 'timeout', 'cancelled', 'skipped'])),
    h('div', { class: 'row' }, h('a', { class: 'link', href: `/api/history/${run.run_id}/export?format=json`, download: '' }, 'Exportar JSON'),
      h('a', { class: 'link', href: `/api/history/${run.run_id}/export?format=csv`, download: '' }, 'Exportar CSV')));

  put(body, h('h3', {}, 'Modelos evaluados (metadatos guardados en el momento de la prueba)'),
    h('div', { class: 'scroll' }, h('table', { class: 'data' },
      h('thead', {}, h('tr', {}, ['Modelo', 'Digest', 'Formato', 'Cuantización', 'Arquitectura', 'Parámetros', 'Disco', 'Carga (calentamiento)'].map((c) => h('th', {}, c)))),
      h('tbody', {}, run.models.map((m) => {
        const loads = events.filter((e) => e.type === 'warmup' && e.model === m.name && e.load_s != null).map((e) => `${fmt(e.load_s, 1)} s`);
        return h('tr', {}, h('td', { class: 'name' }, m.name), h('td', {}, unk(m.digest, (v) => v.slice(0, 12))), h('td', {}, unk(m.format)), h('td', {}, unk(m.quantization)),
          h('td', {}, unk(m.architecture)), h('td', {}, unk(m.parameter_size)), h('td', { class: 'num' }, unk(fmtBytes(m.disk_bytes))), h('td', {}, loads.length ? loads.join(' · ') : '—'));
      })))));

  const models = [...new Set(results.map((r) => r.model))], cats = [...new Set(results.map((r) => r.category))];
  const sel = (id, label, opts) => h('label', {}, label, h('select', { id, onchange: renderResults }, h('option', { value: '' }, 'Todos'), opts.map(([v, t]) => h('option', { value: v }, t))));
  put(body, h('h3', {}, `Resultados individuales (${results.length})`),
    h('div', { class: 'row toolbar' }, sel('hf-model', 'Modelo', models.map((m) => [m, m])), sel('hf-cat', 'Categoría', cats.map((c) => [c, (suiteInfo?.categories_names || {})[c] || c])),
      sel('hf-out', 'Resultado', Object.keys(OUTCOME_LABEL).map((o) => [o, OUTCOME_LABEL[o]]))),
    h('div', { id: 'hd-results' }));
  if (window.pendingModel) { $('hf-model').value = models.includes(window.pendingModel) ? window.pendingModel : ''; window.pendingModel = null; }
  renderResults();
}

function renderResults() {
  const f = { model: $('hf-model').value, cat: $('hf-cat').value, out: $('hf-out').value };
  const rows = histDetail.results.filter((r) => (!f.model || r.model === f.model) && (!f.cat || r.category === f.cat) && (!f.out || r.outcome === f.out));
  const box = clear($('hd-results'));
  if (!rows.length) { box.append(h('p', { class: 'hint' }, 'Ningún resultado coincide con el filtro.')); return; }
  for (const r of rows.slice(0, 400)) {
    const m = r.metrics || {};
    const det = h('details', { class: 'res' }, h('summary', {},
      outcomeChip(r.outcome), h('strong', {}, r.instance_id), h('span', { class: 'muted' }, `${r.model} · ${r.difficulty} · rep. ${r.repetition}`),
      h('span', { class: 'muted' }, [m.gen_tokens != null ? `${m.gen_tokens} tok` : null, m.decode_tok_s ? `${fmt(m.decode_tok_s, 1)} tok/s` : null, r.cold ? 'en frío' : null].filter(Boolean).join(' · '))));
    det.addEventListener('toggle', () => { if (det.open && !det.dataset.done) { det.dataset.done = '1'; det.append(resultBody(r)); } });
    box.append(det);
  }
  if (rows.length > 400) box.append(h('p', { class: 'hint' }, `Se muestran 400 de ${rows.length}. Usa los filtros o exporta la ejecución para verlos todos.`));
}

function prov(key, m) {
  const p = (m.provenance || {})[key] || (key.startsWith('gpu') ? 'nvidia-smi' : null);
  return p ? h('span', { class: 'prov', title: PROV_LABEL[p] || p }, p) : null;
}

function resultBody(r) {
  const m = r.metrics || {}, g = r.grade, gpu = (m.gpu || {}).request || {}, dec = (m.gpu || {}).decode || {};
  const metric = (label, key, value) => (value == null ? null : h('div', {}, `${label}: `, value, prov(key, m)));
  return h('div', {},
    h('p', {}, h('strong', {}, 'Esperado: '), r.expected), g ? h('p', {}, h('strong', {}, `Calificación (${g.status}): `), g.reason || 'cumple todos los criterios') : h('p', {}, h('strong', {}, `Sin calificar: `), r.error || r.gen_status),
    g?.checks?.length ? h('div', { class: 'lines' }, g.checks.map((c) => h('div', {}, `${c.ok === true ? '✓' : c.ok === false ? '✗' : '?'} ${c.check}${c.detail ? ` — ${c.detail}` : ''}`))) : null,
    h('h3', {}, 'Prompt'), h('pre', {}, r.prompt.length > 3000 ? `${r.prompt.slice(0, 1500)}\n… [${r.prompt.length - 3000} caracteres omitidos en pantalla; el JSON exportado los contiene] …\n${r.prompt.slice(-1500)}` : r.prompt),
    r.response.thinking ? [h('h3', {}, 'Razonamiento'), h('pre', {}, r.response.thinking)] : null,
    h('h3', {}, 'Respuesta'), h('pre', {}, r.response.content || '(vacía)'),
    h('h3', {}, 'Métricas'), h('div', { class: 'lines' },
      metric('Contexto pedido (num_ctx)', 'num_ctx', r.num_ctx_requested?.toLocaleString('es')), metric('Tokens de entrada procesados', 'prompt_tokens', m.prompt_tokens),
      metric('Tokens generados', 'gen_tokens', m.gen_tokens), metric('Velocidad de generación', 'decode_tok_s', m.decode_tok_s ? `${fmt(m.decode_tok_s, 1)} tok/s` : null),
      metric('Velocidad de lectura del prompt', 'prefill_tok_s', m.prefill_tok_s ? `${fmt(m.prefill_tok_s, 0)} tok/s` : null),
      metric('Primer token', 'ttft_s', m.ttft_s != null ? `${fmt(m.ttft_s, 2)} s` : null), metric('Latencia total', 'wall_s', m.wall_s != null ? `${fmt(m.wall_s, 2)} s` : null),
      metric('Carga del modelo', 'load_s', m.load_s != null ? `${fmt(m.load_s, 2)} s` : null),
      metric('VRAM pico (GPU completa)', 'gpu', gpu.vram_peak_mib != null ? fmtMiB(gpu.vram_peak_mib) : null), metric('Uso de GPU medio', 'gpu', gpu.util_mean != null ? `${fmt(gpu.util_mean, 0)} %` : null),
      metric('Consumo medio', 'gpu', gpu.power_mean_w != null ? `${fmt(gpu.power_mean_w, 0)} W` : null), metric('Temperatura máxima', 'gpu', gpu.temp_max_c != null ? `${fmt(gpu.temp_max_c, 0)} °C` : null),
      metric('Energía neta de la generación', 'energy_net_wh', dec.energy_net_wh != null ? `${fmt(dec.energy_net_wh, 4)} Wh` : null),
      m.warnings?.length ? h('div', {}, '⚠ ', m.warnings.join(' ')) : null,
      r.error ? h('div', {}, `Error: ${r.error}`) : null, r.think_fallback ? h('div', {}, 'El modelo no admitió el parámetro de razonamiento; se repitió sin él.') : null));
}

onView('history', async () => { if (!suiteInfo) { const s = await rawApi('/api/bench/suite'); if (s.ok) suiteInfo = s.data; } await loadHistory(); });

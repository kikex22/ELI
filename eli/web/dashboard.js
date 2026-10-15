'use strict';
/* Dashboard: resumen del sistema, de los modelos y de las evaluaciones. */

const STATE_LABEL = { running: 'en curso', completed: 'completada', cancelled: 'cancelada', interrupted: 'interrumpida', failed: 'fallida', unknown: 'desconocido' };

/* Tabla de ejecuciones, compartida con History. actions(run) devuelve nodos para la ultima columna. */
function runsTable(runs, actions) {
  if (!runs.length) return h('p', { class: 'hint' }, 'Todavía no hay evaluaciones. Ejecuta una desde Benchmark.');
  return h('table', { class: 'data' },
    h('thead', {}, h('tr', {}, ['Fecha', 'Estado', 'Modelos', 'Pruebas', 'Resultados', 'Suite', actions ? '' : null].filter((c) => c !== null).map((c) => h('th', {}, c)))),
    h('tbody', {}, runs.map((r) => h('tr', {},
      h('td', {}, fmtDate(r.created)),
      h('td', {}, STATE_LABEL[r.state] || r.state, r.message ? h('div', { class: 'hint' }, r.message) : null),
      h('td', { class: 'name' }, (r.models || []).join(', ')),
      h('td', { class: 'num' }, `${r.done}/${r.total}`),
      h('td', {}, h('div', { class: 'chips' }, countsChips(r.counts || {}))),
      h('td', {}, r.suite?.id ? `${r.suite.id} ${r.suite.version}` : '—', r.mode ? h('div', { class: 'hint' }, r.mode === 'quick' ? 'rápido' : 'completo') : null),
      actions ? h('td', {}, h('div', { class: 'btns' }, actions(r))) : null))));
}

async function renderDashboard() {
  const r = await rawApi('/api/dashboard');
  if (!r.ok) { banner(r.data.error || 'No se pudo leer el estado'); return; }
  const d = r.data;
  const hw = d.hardware || {}, gpu = hw.gpu || {}, sbx = d.sandbox || {};
  const facts = [
    ['GPU', unk(gpu.name)], ['Driver', unk(gpu.driver)], ['Capacidad de cálculo', unk(gpu.compute_cap, (v) => `SM ${v}`)],
    ['VRAM total', unk(fmtMiB(gpu.vram_total_mib))], ['CPU', unk(hw.cpu?.model, (v) => `${v}${hw.cpu.threads ? ` (${hw.cpu.threads} hilos)` : ''}`)],
    ['RAM', unk(fmtMiB(hw.ram_total_mib))], ['Sistema', unk(hw.os)], ['Python', unk(hw.python)],
    ['Ollama', unk(d.status.ollama.version)], ['ELI', `v${d.status.version}`],
    ['Sandbox de código', sbx.available ? `${sbx.kind}: ${sbx.isolation}` : h('span', { class: 'unknown' }, 'no disponible: las pruebas de programación quedarán sin calificar')],
  ];
  const dl = clear($('dash-system'));
  facts.forEach(([k, v]) => dl.append(h('dt', {}, k), h('dd', {}, v)));

  const m = d.models, box = clear($('dash-models'));
  if (d.ollama_error) put(box, h('p', { class: 'error' }, d.ollama_error));
  put(box, h('p', {}, `${m.count} modelos instalados · ${fmtBytes(m.disk_bytes) || '—'} en disco`),
    h('p', {}, 'En memoria ahora: ', m.loaded.length ? m.loaded.join(', ') : h('span', { class: 'muted' }, 'ninguno')),
    h('button', { type: 'button', onclick: () => showView('models') }, 'Ver inventario'));

  const b = d.bench, bb = clear($('dash-bench'));
  $('dash-bench-sub').textContent = `suite ${d.suite.id} ${d.suite.version} · ${d.suite.tests} pruebas (${d.suite.quick_tests} en el rápido)`;
  if (b.active) {
    const frac = b.total ? b.done / b.total : 0;
    put(bb, h('p', {}, `En curso: ${b.current_model || '…'} · ${b.done}/${b.total} pruebas · ${fmtDur(b.elapsed_s)}`),
      h('div', { class: 'bar' }, h('div', { style: `width:${Math.round(frac * 100)}%` })),
      h('button', { type: 'button', onclick: () => showView('benchmark') }, 'Ver progreso'));
  } else {
    put(bb, h('p', { class: 'muted' }, 'No hay ninguna evaluación en curso.'), h('button', { type: 'button', class: 'primary', onclick: () => showView('benchmark') }, 'Nueva evaluación'));
  }
  put(clear($('dash-runs')), runsTable(d.recent_runs, (run) => [h('button', { type: 'button', onclick: () => openRun(run.run_id) }, 'Abrir')]),
    d.runs_total > 5 ? h('p', { class: 'hint' }, `Mostrando 5 de ${d.runs_total}. El resto está en History.`) : null);
  $('bench-dot').hidden = !b.active;
}

function openRun(runId) {
  window.pendingRun = runId;
  showView('history');
}

onView('dashboard', renderDashboard);
setInterval(() => { if (currentView === 'dashboard') renderDashboard(); }, 4000);
setInterval(async () => {
  if (currentView === 'dashboard') return;
  const r = await rawApi('/api/status').catch(() => null);
  if (r?.ok) $('bench-dot').hidden = !r.data.bench_running;
}, 5000);

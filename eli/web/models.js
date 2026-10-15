'use strict';
/* Models: inventario, Model Inspector (incluye analisis TensorRT-LLM) y administracion. */

let inventory = [];
let inspected = null;

const VERDICT = {
  potencialmente_compatible: 'Arquitectura potencialmente compatible',
  potencialmente_compatible_requiere_checkpoint_original: 'Arquitectura potencialmente compatible, pero requiere el checkpoint original y conversión',
  no_determinada: 'Compatibilidad no determinada',
};
const DIM_LABEL = {
  potencialmente_compatible: 'Potencialmente compatible', no_determinada: 'No determinada', requiere_conversion: 'Requiere conversión',
  necesario: 'Necesario', informativo: 'Informativo', restringida: 'Restricción', validado_localmente: 'Validado localmente',
};

async function loadInventory() {
  $('models-sub').textContent = 'leyendo…';
  const r = await rawApi('/api/models/inspect');
  if (!r.ok) { banner(r.data.error || 'No se pudo leer Ollama'); $('models-sub').textContent = ''; return; }
  banner('');
  inventory = r.data.models;
  $('models-sub').textContent = `${inventory.length} instalados · reglas TensorRT-LLM ${r.data.rules_version}`;
  renderInventory();
}

function stateCell(m) {
  if (!m.loaded) return h('span', { class: 'muted' }, 'solo en disco');
  const l = m.loaded;
  return `cargado · ${fmtBytes(l.vram_bytes) || '?'} en GPU${l.context ? ` · ctx ${l.context}` : ''}`;
}

function renderInventory() {
  const q = $('models-filter').value.trim().toLowerCase();
  const rows = inventory.filter((m) => !q || [m.name, m.family, m.architecture, m.quantization?.value, m.format, m.parameter_size].some((v) => (v || '').toLowerCase().includes(q)));
  const head = ['Modelo', 'Familia', 'Arquitectura', 'Parámetros', 'Formato', 'Cuantización', 'Disco', 'Contexto declarado', 'Estado', 'VRAM en pruebas', ''];
  const table = clear($('models-table'));
  table.append(h('thead', {}, h('tr', {}, head.map((c) => h('th', {}, c)))));
  table.append(h('tbody', {}, rows.map((m) => h('tr', { class: 'clickable', onclick: (e) => { if (!e.target.closest('button')) showInspector(m.name); } },
    h('td', { class: 'name' }, m.name),
    h('td', {}, unk(m.family)), h('td', {}, unk(m.architecture)), h('td', {}, unk(m.parameter_size)), h('td', {}, unk(m.format)),
    h('td', {}, unk(m.quantization.value)), h('td', { class: 'num' }, unk(fmtBytes(m.disk_bytes))),
    h('td', { class: 'num' }, unk(m.context_length, (v) => v.toLocaleString('es'))), h('td', {}, stateCell(m)),
    h('td', { class: 'num' }, m.observed?.vram_used_mib != null ? `${fmtMiB(m.observed.vram_used_mib)} (${m.observed.runs} eval.)` : h('span', { class: 'muted' }, 'sin evaluar')),
    h('td', {}, h('div', { class: 'btns' },
      h('button', { type: 'button', onclick: () => showInspector(m.name) }, 'Detalles'),
      m.loaded ? h('button', { type: 'button', title: 'Libera la VRAM; el modelo sigue en disco', onclick: () => unloadModel(m.name) }, 'Descargar de VRAM') : null,
      h('button', { type: 'button', title: 'Borra el modelo del disco', onclick: () => deleteModel(m.name) }, 'Eliminar')))))));
  if (!rows.length) table.append(h('caption', { class: 'hint' }, inventory.length ? 'Ningún modelo coincide con el filtro.' : 'Ollama no tiene modelos instalados.'));
}

function group(title, hint, ...items) {
  return h('div', { class: 'chartbox' }, h('h3', {}, title), h('p', { class: 'hint' }, hint), h('dl', { class: 'facts' }, items.flat()));
}
const fact = (k, v) => [h('dt', {}, k), h('dd', {}, v)];

function showInspector(name) {
  const m = inventory.find((x) => x.name === name);
  if (!m) return;
  inspected = name;
  $('inspector').hidden = false;
  $('inspector-name').textContent = m.name;
  const q = m.quantization;
  const qkind = { float: 'precisión de punto flotante (sin cuantizar)', 'gguf-quant': 'cuantización propia de GGUF', other: 'otro (no clasificado por ELI)', unknown: 'desconocida' }[q.kind];
  const caps = h('div', { class: 'caps' }, m.capabilities.length ? m.capabilities.map((c) => h('span', { class: 'chip' }, c)) : h('span', { class: 'unknown' }, 'desconocidas'));
  const t = m.trtllm, dims = t.dimensions;

  const body = clear($('inspector-body'));
  put(body, 
    h('div', { class: 'grid2' },
      group('Identidad', 'Datos de identificación que informa Ollama.', fact('Nombre completo', m.name), fact('Digest', unk(m.digest, (v) => v.slice(0, 16))),
        fact('Modificado', unk(m.modified_at, fmtDate)), fact('Tamaño en disco', unk(fmtBytes(m.disk_bytes)))),
      group('Qué es', 'Tres propiedades distintas: no se deben confundir.', fact('Formato (cómo se guardan los pesos)', unk(m.format)),
        fact('Precisión / cuantización', unk(q.value, (v) => `${v} — ${qkind}`)), fact('Arquitectura (la red)', unk(m.architecture)),
        fact('Familia', unk(m.family)), fact('Parámetros', unk(m.parameter_size, (v) => `${v}${m.parameters_b ? ` (≈ ${fmt(m.parameters_b, 2)} mil millones)` : ''}`)),
        fact('Contexto declarado', unk(m.context_length, (v) => `${v.toLocaleString('es')} tokens (lo que el modelo soporta, no lo comprobado)`)))),
    h('div', { class: 'grid2' },
      group('Capacidades', 'Las informa Ollama.', fact('Capacidades', caps)),
      group('Memoria', 'Estado actual y lo observado en evaluaciones.',
        fact('Cargado ahora', m.loaded ? `sí · ${fmtBytes(m.loaded.vram_bytes) || '?'} en GPU de ${fmtBytes(m.loaded.bytes) || '?'}${m.loaded.context ? ` · contexto ${m.loaded.context}` : ''}` : 'no'),
        fact('VRAM usada en pruebas', m.observed?.vram_used_mib != null ? `${fmtMiB(m.observed.vram_used_mib)} sobre lo previo (pico de GPU ${fmtMiB(m.observed.vram_gpu_peak_mib)}) · ${m.observed.runs} evaluaciones` : h('span', { class: 'muted' }, 'sin evaluaciones todavía')))),
    m.unknown.length ? h('p', { class: 'hint' }, 'Ollama no informó: ', m.unknown.join(', '), '. Se muestran como desconocidos; ELI no los infiere.') : null,
    m.inspect_error ? h('p', { class: 'error' }, `No se pudo leer /api/show: ${m.inspect_error}`) : null,
    h('div', { class: 'chartbox' }, h('h3', {}, 'Compatibilidad con TensorRT-LLM (informativo)'),
      h('p', {}, h('strong', {}, VERDICT[t.verdict] || t.verdict)),
      h('dl', { class: 'facts' }, fact('Arquitectura', DIM_LABEL[dims.arquitectura] || dims.arquitectura), fact('Formato de los pesos', DIM_LABEL[dims.formato] || dims.formato),
        fact('Checkpoint original', DIM_LABEL[dims.checkpoint_original] || dims.checkpoint_original),
        fact('Restricciones', Array.isArray(dims.restricciones) ? dims.restricciones.map((r) => r.replaceAll('_', ' ')).join(', ') : 'ninguna detectada'),
        dims.memoria ? fact('Memoria de pesos (estimación)', Object.entries(dims.memoria.weights_gib_estimate).map(([k, v]) => `${k}: ${v} GiB${dims.memoria.fits[k] ? '' : ' (no cabe)'}`).join(' · ')) : null),
      h('h3', {}, 'Hallazgos y reglas aplicadas'),
      h('div', { class: 'lines' }, t.findings.map((f) => h('div', {}, h('span', { class: 'chip' }, f.rule), ' ', h('strong', {}, DIM_LABEL[f.status] || f.status), ': ', f.text, f.evidence ? h('div', { class: 'hint' }, `Fuente: ${f.evidence}`) : null))),
      h('p', { class: 'hint' }, `${t.notice} Reglas ${t.rules_version}.`)));
  $('inspector').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function unloadModel(name) {
  const r = await postJson('/api/unload', { model: name });
  banner(r.ok ? '' : r.data.error);
  await loadInventory();
}

async function deleteModel(name) {
  const pv = await rawApi('/api/models/delete-preview?name=' + encodeURIComponent(name));
  if (!pv.ok) { banner(pv.data.error); return; }
  const p = pv.data;
  const input = h('input', { type: 'text', autocomplete: 'off', spellcheck: 'false', 'aria-label': 'Nombre exacto del modelo' });
  const body = h('div', {},
    h('p', {}, 'Vas a eliminar del disco este modelo:'), h('p', {}, h('code', {}, p.name)),
    h('p', { class: 'hint' }, `Tamaño en disco: ${fmtBytes(p.disk_bytes) || 'desconocido'}.`),
    h('p', { class: 'hint' }, `Espacio recuperable (${p.reclaimable.kind}): ${p.reclaimable.note}`),
    p.loaded ? h('p', { class: 'error' }, 'Está cargado en la VRAM. Descárgalo primero (eso no lo borra) y vuelve a intentarlo.') : null,
    h('p', { class: 'hint' }, 'Los resultados de evaluaciones anteriores de este modelo se conservan. Esta acción no se puede deshacer.'),
    p.loaded ? null : h('label', {}, 'Escribe el nombre exacto para confirmar:', input));
  const ok = await askDialog({ title: 'Eliminar modelo', body, okLabel: 'Eliminar', check: () => !p.loaded && input.value === p.name });
  if (!ok) return;
  const r = await postJson('/api/models/delete', { model: p.name, confirm: input.value });
  banner(r.ok ? '' : r.data.error);
  if (r.ok) { $('inspector').hidden = inspected === name ? true : $('inspector').hidden; }
  await loadInventory();
}

$('models-refresh').addEventListener('click', loadInventory);
$('models-filter').addEventListener('input', renderInventory);
onView('models', loadInventory);

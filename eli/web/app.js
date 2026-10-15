'use strict';

const $ = (id) => document.getElementById(id);
const WINDOW_S = 120;
const MAX_POINTS = 600;

const fmt = (n, d = 1) =>
  n == null || Number.isNaN(n) ? '—' : n.toLocaleString('es', { minimumFractionDigits: d, maximumFractionDigits: d });
const gib = (bytes) => bytes / 2 ** 30;

const state = {
  status: {},
  samples: [], // {ts, util, mem_used, power, temp, ...}
  gens: [], // {t0, t1}: intervalos de generacion, para sombrear las graficas
  model: null,
  caps: [],
  messages: [],
  busy: false,
  abort: null,
  hover: {}, // id -> x en px dentro del canvas
};

const CHARTS = [
  { id: 'vram', key: 'mem_used', color: '--series-1', digits: 1, unit: 'GiB',
    scale: (v) => v / 1024, max: () => (state.status.gpu?.mem_total_mib || 16384) / 1024 },
  { id: 'util', key: 'util', color: '--series-2', digits: 0, unit: '%', max: () => 100 },
  { id: 'power', key: 'power', color: '--series-3', digits: 0, unit: 'W',
    max: () => state.status.gpu?.power_limit_w || 165 },
  { id: 'temp', key: 'temp', color: '--series-8', digits: 0, unit: '°C', min: () => 20, max: () => 100 },
];

/* ---------- almacenamiento local (opcional) ---------- */
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* sin almacenamiento */ } },
};

/* ---------- peticiones ---------- */
async function api(path, opts) {
  const resp = await fetch(path, opts);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `Error ${resp.status}`);
  return data;
}

function banner(text) {
  const el = $('banner');
  el.hidden = !text;
  el.textContent = text || '';
}

/* ---------- GIA: datos ---------- */
function pushSample(s) {
  state.samples.push(s);
  if (state.samples.length > MAX_POINTS) state.samples.splice(0, state.samples.length - MAX_POINTS);
}

async function startGpu() {
  try {
    state.status = await api('/api/status');
    const h = await api(`/api/gpu/history?seconds=${WINDOW_S}`);
    h.samples.forEach(pushSample);
  } catch (e) {
    banner(e.message);
  }
  const es = new EventSource('/api/gpu/stream');
  es.onmessage = (ev) => {
    pushSample(JSON.parse(ev.data));
    $('gia-status').textContent = 'muestreando 4 veces por segundo';
  };
  es.onerror = () => { $('gia-status').textContent = 'reconectando…'; };
  setInterval(pollProcesses, 2000);
  pollProcesses();
}

async function pollProcesses() {
  try {
    const { processes } = await api('/api/gpu/processes');
    $('procs').textContent = processes.length
      ? 'Procesos en la GPU: ' + processes.map((p) => `${p.name} (${p.mem_mib == null ? '?' : fmt(p.mem_mib / 1024, 1)} GiB)`).join(' · ')
      : 'Ningún proceso de cómputo usa la GPU ahora.';
  } catch { /* se reintenta */ }
}

/* ---------- GIA: graficas ---------- */
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawChart(cfg) {
  const canvas = $('c-' + cfg.id);
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 30, r: 6, t: 6, b: 16 };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const lo = cfg.min ? cfg.min() : 0, hi = cfg.max();
  const now = Date.now() / 1000, t0 = now - WINDOW_S;
  const X = (t) => pad.l + ((t - t0) / WINDOW_S) * pw;
  const Y = (v) => pad.t + ph - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * ph;

  // sombreado de las generaciones
  ctx.fillStyle = css('--text-primary');
  ctx.globalAlpha = 0.07;
  for (const g of state.gens) {
    const a = Math.max(X(g.t0), pad.l), b = Math.min(X(g.t1 ?? now), pad.l + pw);
    if (b > a) ctx.fillRect(a, pad.t, b - a, ph);
  }
  ctx.globalAlpha = 1;

  // rejilla y ejes (recesivos)
  ctx.strokeStyle = css('--grid');
  ctx.lineWidth = 1;
  ctx.fillStyle = css('--text-muted');
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const f of [0, 0.5, 1]) {
    const y = Math.round(Y(lo + (hi - lo) * f)) + 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    if (f !== 0.5) ctx.fillText(fmt(lo + (hi - lo) * f, 0), pad.l - 4, y);
  }
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.fillText('−2 min', pad.l, h - 3);
  ctx.textAlign = 'right';
  ctx.fillText('ahora', pad.l + pw, h - 3);

  // serie
  const pts = state.samples.filter((s) => s[cfg.key] != null && s.ts >= t0);
  ctx.strokeStyle = css(cfg.color);
  ctx.lineWidth = 2;
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.beginPath();
  pts.forEach((s, i) => {
    const v = cfg.scale ? cfg.scale(s[cfg.key]) : s[cfg.key];
    i ? ctx.lineTo(X(s.ts), Y(v)) : ctx.moveTo(X(s.ts), Y(v));
  });
  ctx.stroke();

  // cruz del cursor
  const hx = state.hover[cfg.id];
  if (hx != null && pts.length) {
    const t = t0 + ((hx - pad.l) / pw) * WINDOW_S;
    const near = pts.reduce((a, b) => (Math.abs(b.ts - t) < Math.abs(a.ts - t) ? b : a));
    const v = cfg.scale ? cfg.scale(near[cfg.key]) : near[cfg.key];
    ctx.strokeStyle = css('--text-secondary');
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(Math.round(X(near.ts)) + 0.5, pad.t); ctx.lineTo(Math.round(X(near.ts)) + 0.5, pad.t + ph); ctx.stroke();
    ctx.fillStyle = css(cfg.color);
    ctx.strokeStyle = css('--surface-1');
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(near.ts), Y(v), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    $('val-' + cfg.id).textContent = `${fmt(v, cfg.digits)} ${cfg.unit} · hace ${fmt(now - near.ts, 0)} s`;
  } else if (pts.length) {
    const last = pts[pts.length - 1];
    const v = cfg.scale ? cfg.scale(last[cfg.key]) : last[cfg.key];
    $('val-' + cfg.id).textContent = `${fmt(v, cfg.digits)} ${cfg.unit}`;
  }
}

function drawChips() {
  const last = state.samples[state.samples.length - 1];
  const box = $('chips');
  box.replaceChildren();
  const add = (text, warn) => {
    const c = document.createElement('span');
    c.className = 'chip' + (warn ? ' warn' : '');
    c.textContent = text;
    box.appendChild(c);
  };
  if (!last) { add(state.status.gpu?.error || 'Sin lecturas de GPU todavía'); return; }
  add(last.throttle.length ? '⚠ Frenada: ' + last.throttle.join(', ') : 'Sin frenado', last.throttle.length > 0);
  add(`Relojes: núcleo ${fmt(last.sm_clock, 0)} MHz · memoria ${fmt(last.mem_clock, 0)} MHz`);
  add(`Estado ${last.pstate ?? '—'}`);
  add(`Memoria usada ${fmt(last.mem_util, 0)} %`);
}

function drawAll() {
  CHARTS.forEach(drawChart);
  drawChips();
}

function bindHover() {
  for (const cfg of CHARTS) {
    const canvas = $('c-' + cfg.id);
    canvas.addEventListener('mousemove', (e) => { state.hover[cfg.id] = e.offsetX; drawChart(cfg); });
    canvas.addEventListener('mouseleave', () => { delete state.hover[cfg.id]; drawChart(cfg); });
  }
}

/* ---------- modelos y ficha ---------- */
async function loadModels(keep) {
  try {
    const { models } = await api('/api/models');
    const sel = $('model');
    sel.replaceChildren();
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m.name;
      o.textContent = (m.loaded ? '● ' : '') + m.name;
      sel.appendChild(o);
    }
    if (!models.length) { banner('Ollama no tiene modelos instalados.'); return; }
    banner('');
    const wanted = keep || store.get('eli.model');
    sel.value = models.some((m) => m.name === wanted) ? wanted : models[0].name;
    await selectModel(sel.value);
  } catch (e) {
    banner(e.message);
  }
}

async function selectModel(name) {
  state.model = name;
  store.set('eli.model', name);
  const dl = $('ficha');
  dl.replaceChildren();
  try {
    const [info, { models }] = await Promise.all([api('/api/model?name=' + encodeURIComponent(name)), api('/api/models')]);
    const m = models.find((x) => x.name === name) || {};
    state.caps = info.capabilities || [];
    const row = (k, v) => {
      const dt = document.createElement('dt'); dt.textContent = k;
      const dd = document.createElement('dd');
      if (v instanceof Node) dd.appendChild(v); else dd.textContent = v ?? '—';
      dl.append(dt, dd);
    };
    const caps = document.createElement('div');
    caps.className = 'caps';
    state.caps.forEach((c) => { const s = document.createElement('span'); s.className = 'chip'; s.textContent = c; caps.appendChild(s); });
    row('Modelo', name);
    row('Familia', info.family);
    row('Parámetros', info.parameter_size);
    row('Cuantización', info.quantization);
    row('Contexto anunciado', info.context_length ? info.context_length.toLocaleString('es') + ' tokens' : '—');
    row('Tamaño en disco', m.disk_bytes ? fmt(gib(m.disk_bytes), 1) + ' GiB' : '—');
    row('Estado', m.loaded
      ? `cargado: ${fmt(gib(m.vram_bytes), 1)} de ${fmt(gib(m.loaded_bytes), 1)} GiB en GPU · contexto ${m.loaded_context}`
      : 'no cargado');
    row('Capacidades', caps);
    $('think-row').hidden = !state.caps.includes('thinking');
  } catch (e) {
    banner(e.message);
  }
}

/* ---------- ajustes ---------- */
const FIELDS = { 's-ctx': 4096, 's-temp': 0, 's-seed': 42, 's-max': 512, 's-system': '', 's-think': '' };

function loadSettings() {
  const saved = store.get('eli.settings') || {};
  for (const [id, def] of Object.entries(FIELDS)) $(id).value = saved[id] ?? def;
  document.querySelectorAll('.settings input, .settings select, .settings textarea').forEach((el) =>
    el.addEventListener('change', saveSettings));
}
function saveSettings() {
  store.set('eli.settings', Object.fromEntries(Object.keys(FIELDS).map((id) => [id, $(id).value])));
}

/* ---------- chat ---------- */
function addMessage(role) {
  const box = document.createElement('div');
  box.className = 'msg ' + role;
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = role === 'user' ? 'Tú' : state.model;
  const text = document.createElement('div');
  text.className = 'text';
  box.append(who, text);
  $('log').appendChild(box);
  $('log').scrollTop = $('log').scrollHeight;
  return { box, text };
}

function line(parent, text, cls, title) {
  const d = document.createElement('div');
  if (cls) d.className = cls;
  if (title) d.title = title;
  d.textContent = text;
  parent.appendChild(d);
}

function renderMetrics(box, m) {
  const el = document.createElement('div');
  el.className = 'metrics';
  line(el,
    `${fmt(m.gen_tokens, 0)} tokens generados · ${fmt(m.decode_tok_s)} tok/s · primer token ${fmt(m.ttft_s, 2)} s` +
      (m.load_s > 0.5 ? ` (incluye ${fmt(m.load_s, 1)} s de carga del modelo)` : ''));
  line(el,
    `Prompt: ${fmt(m.prompt_tokens, 0)} tokens evaluados` + (m.prefill_tok_s ? ` a ${fmt(m.prefill_tok_s, 0)} tok/s` : '') +
      ` · total ${fmt(m.total_s, 2)} s`, '',
    'Si Ollama reutiliza la caché de la conversación, evalúa menos tokens que el contexto real.');
  const g = m.gpu?.request;
  if (g) {
    line(el,
      `GPU: VRAM pico ${fmt(g.vram_peak_mib / 1024, 1)} GiB · uso medio ${fmt(g.util_mean, 0)} % · ` +
        `${fmt(g.power_mean_w, 0)} W de media (pico ${fmt(g.power_peak_w, 0)} W) · ${fmt(g.temp_max_c, 0)} °C máx.`);
  }
  const d = m.gpu?.decode;
  if (d?.energy_net_wh != null) {
    line(el,
      `Energía de la generación: ${fmt(d.energy_net_wh, 3)} Wh sobre el reposo (${fmt(d.baseline_power_w, 0)} W)` +
        (m.wh_per_token != null ? ` · ${fmt(m.wh_per_token * 1000, 2)} mWh por token` : ''), '',
      'Energía de toda la GPU menos su consumo en reposo (mediana de las lecturas con la GPU casi inactiva en los últimos 10 min). No es solo la del modelo.');
  } else if (d?.energy_wh != null) {
    line(el, `Energía de la generación: ${fmt(d.energy_wh, 3)} Wh (sin referencia de reposo)`);
  }
  if (m.model_vram_bytes) {
    line(el,
      `Modelo en GPU: ${fmt(gib(m.model_vram_bytes), 1)} de ${fmt(gib(m.model_loaded_bytes), 1)} GiB ` +
        `(${fmt(m.gpu_fraction * 100, 0)} %) · contexto cargado ${m.loaded_context}`);
  }
  (m.warnings || []).forEach((w) => line(el, '⚠ ' + w, 'warn'));
  box.appendChild(el);
}

function setBusy(busy) {
  state.busy = busy;
  $('send').disabled = busy;
  $('stop').hidden = !busy;
  $('model').disabled = busy;
  $('unload').disabled = busy;
}

function requestBody() {
  const think = $('s-think').value;
  return {
    model: state.model,
    messages: state.messages,
    system: $('s-system').value,
    think: !$('think-row').hidden && think ? think === 'true' : null,
    options: {
      num_ctx: Number($('s-ctx').value), temperature: Number($('s-temp').value),
      seed: Number($('s-seed').value), num_predict: Number($('s-max').value),
    },
  };
}

async function send(text) {
  state.messages.push({ role: 'user', content: text });
  addMessage('user').text.textContent = text;
  const reply = addMessage('assistant');
  let thinkingEl = null, thinkingPre = null, content = '';
  setBusy(true);
  const gen = { t0: Date.now() / 1000, t1: null };
  state.gens.push(gen);
  state.abort = new AbortController();
  let finished = false;
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody()), signal: state.abort.signal,
    });
    if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).error || `Error ${resp.status}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!chunk.startsWith('data: ')) continue;
        const ev = JSON.parse(chunk.slice(6));
        if (ev.type === 'token' && ev.kind === 'content') {
          content += ev.text;
          reply.text.textContent = content;
        } else if (ev.type === 'token') {
          if (!thinkingEl) {
            thinkingEl = document.createElement('details');
            thinkingEl.open = true;
            const s = document.createElement('summary'); s.textContent = 'Razonamiento';
            thinkingPre = document.createElement('pre');
            thinkingEl.append(s, thinkingPre);
            reply.box.insertBefore(thinkingEl, reply.text);
          }
          thinkingPre.textContent += ev.text;
        } else if (ev.type === 'done') {
          finished = true;
          renderMetrics(reply.box, ev.metrics);
        } else if (ev.type === 'error') {
          throw new Error(ev.message);
        }
        $('log').scrollTop = $('log').scrollHeight;
      }
    }
    if (!finished) throw new Error('La conexión se cerró antes de terminar la respuesta.');
    state.messages.push({ role: 'assistant', content });
  } catch (e) {
    state.messages.pop(); // la pregunta sin respuesta no se arrastra al contexto
    if (e.name === 'AbortError') {
      line(reply.box, '(generación detenida; este intercambio no cuenta para el contexto)', 'hint');
    } else {
      line(reply.box, e.message, 'error');
    }
  } finally {
    gen.t1 = Date.now() / 1000;
    setBusy(false);
    state.abort = null;
    selectModel(state.model); // refresca el estado de carga en la ficha
  }
}

/* ---------- arranque ---------- */
function bind() {
  $('chat').addEventListener('submit', (e) => {
    e.preventDefault();
    const text = $('prompt').value.trim();
    if (!text || state.busy || !state.model) return;
    $('prompt').value = '';
    send(text);
  });
  $('prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('chat').requestSubmit(); }
  });
  $('stop').addEventListener('click', () => state.abort?.abort());
  $('reset').addEventListener('click', () => {
    if (state.busy) return;
    state.messages = [];
    $('log').replaceChildren();
  });
  $('refresh').addEventListener('click', () => loadModels(state.model));
  $('model').addEventListener('change', (e) => selectModel(e.target.value));
  $('unload').addEventListener('click', async () => {
    try { await api('/api/unload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }); }
    catch (e) { banner(e.message); }
    loadModels(state.model);
  });
  window.addEventListener('resize', drawAll);
}

loadSettings();
bind();
bindHover();
startGpu();
loadModels();
setInterval(drawAll, 250);

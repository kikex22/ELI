'use strict';
/* Navegacion, helpers de DOM y dialogos compartidos por todas las secciones. Se carga despues de app.js. */

const VIEWS = ['dashboard', 'models', 'benchmark', 'embeddings', 'compare', 'history', 'chat'];
const viewHooks = {};
let currentView = null;

function onView(name, fn) { viewHooks[name] = fn; }

function showView(name) {
  if (!VIEWS.includes(name)) name = 'dashboard';
  currentView = name;
  document.querySelectorAll('section[data-view]').forEach((s) => { s.hidden = s.dataset.view !== name; });
  document.querySelectorAll('#nav button').forEach((b) => {
    b.classList.toggle('active', b.dataset.nav === name);
    b.setAttribute('aria-current', b.dataset.nav === name ? 'page' : 'false');
  });
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  if (viewHooks[name]) viewHooks[name]();
}

/* ---------- DOM ---------- */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

/* append() que omite null/false (el append nativo escribiria el texto «null»). */
function put(parent, ...kids) {
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    parent.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return parent;
}

/* Un valor que Ollama no informo se muestra como desconocido, nunca se inventa. */
function unk(value, render) {
  if (value === null || value === undefined || value === '') return h('span', { class: 'unknown' }, 'desconocido');
  return render ? render(value) : String(value);
}

const fmtBytes = (n) => (n == null ? null : n >= 2 ** 30 ? `${fmt(n / 2 ** 30, 1)} GiB` : `${fmt(n / 2 ** 20, 0)} MiB`);
const fmtMiB = (m) => (m == null ? null : m >= 1024 ? `${fmt(m / 1024, 1)} GiB` : `${fmt(m, 0)} MiB`);
const fmtDur = (s) => {
  if (s == null) return null;
  s = Math.round(s);
  return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
};
const fmtDate = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('es', { dateStyle: 'short', timeStyle: 'short' });
};
const pct = (x, d = 0) => (x == null ? null : `${fmt(x * 100, d)} %`);

function clear(el) { el.replaceChildren(); return el; }

/* ---------- red ---------- */
async function rawApi(path, opts) {
  const resp = await fetch(path, opts);
  const data = await resp.json().catch(() => ({}));
  return { ok: resp.ok, status: resp.status, data };
}
const postJson = (path, body) => rawApi(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });

/* ---------- dialogo ---------- */
function askDialog({ title, body, okLabel = 'Aceptar', cancelLabel = 'Cancelar', check = null }) {
  const dlg = $('dlg');
  $('dlg-title').textContent = title;
  const box = clear($('dlg-body'));
  box.append(body);
  const ok = $('dlg-ok'), cancel = $('dlg-cancel');
  ok.textContent = okLabel;
  cancel.textContent = cancelLabel;
  ok.value = 'ok';
  const refresh = () => { ok.disabled = check ? !check() : false; };
  refresh();
  box.addEventListener('input', refresh);
  return new Promise((resolve) => {
    const done = () => { box.removeEventListener('input', refresh); dlg.removeEventListener('close', done); resolve(dlg.returnValue === 'ok'); };
    dlg.addEventListener('close', done);
    cancel.onclick = () => dlg.close('cancel');
    dlg.returnValue = 'cancel';
    dlg.showModal();
  });
}

/* ---------- tooltip ---------- */
function showTip(evt, text) {
  const tip = $('tip');
  tip.textContent = text;
  tip.hidden = false;
  const x = Math.min(evt.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = `${Math.max(4, x)}px`;
  tip.style.top = `${Math.max(4, y)}px`;
}
const hideTip = () => { $('tip').hidden = true; };
function tipOn(el, textFn) {
  el.addEventListener('mousemove', (e) => showTip(e, typeof textFn === 'function' ? textFn() : textFn));
  el.addEventListener('mouseleave', hideTip);
  el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); showTip({ clientX: r.left, clientY: r.top }, typeof textFn === 'function' ? textFn() : textFn); });
  el.addEventListener('blur', hideTip);
}

/* ---------- chips de resultados ---------- */
const OUTCOME_LABEL = { passed: 'superadas', failed: 'falladas', ungraded: 'sin calificar', error: 'errores', timeout: 'tiempo agotado', cancelled: 'canceladas', skipped: 'omitidas' };
const OUTCOME_CLASS = { passed: 'ok', failed: 'bad', ungraded: 'mid', error: 'err', timeout: 'err', cancelled: 'mid', skipped: 'mid' };
function outcomeChip(outcome, n) {
  return h('span', { class: `chip ${OUTCOME_CLASS[outcome] || ''}`, title: OUTCOME_LABEL[outcome] || outcome }, n == null ? (OUTCOME_LABEL[outcome] || outcome) : `${n} ${OUTCOME_LABEL[outcome] || outcome}`);
}
function countsChips(counts, keys = ['passed', 'failed', 'ungraded', 'error', 'timeout', 'skipped']) {
  return keys.filter((k) => counts[k]).map((k) => outcomeChip(k, counts[k]));
}

/* ---------- arranque ---------- */
function initNav() {
  document.querySelectorAll('#nav button').forEach((b) => b.addEventListener('click', () => showView(b.dataset.nav)));
  window.addEventListener('hashchange', () => showView(location.hash.slice(1)));
  $('gia-toggle').addEventListener('click', () => {
    const body = $('gia-body');
    body.hidden = !body.hidden;
    $('gia-toggle').textContent = body.hidden ? 'Mostrar gráficas' : 'Ocultar gráficas';
    $('gia-toggle').setAttribute('aria-expanded', String(!body.hidden));
  });
  rawApi('/api/version').then((r) => { if (r.ok) $('ver').textContent = `v${r.data.eli}`; });
  showView(location.hash.slice(1) || 'dashboard');
}
document.addEventListener('DOMContentLoaded', initNav);
if (document.readyState !== 'loading') initNav();

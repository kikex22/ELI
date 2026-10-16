"""Motor de la evaluacion de embeddings (un modelo tras otro, bajo el mismo bloqueo de GPU que chat y benchmark).

Pasos por modelo: warmup -> retrieval -> retrieval_user -> similarity -> stability -> performance.
- Cada paso se guarda al terminar (results.jsonl): una cancelacion o un fallo no pierde lo ya medido.
- Un paso que falla queda registrado con su error y no impide los demas; si el calentamiento falla se omite el modelo.
- `truncate=False`: un texto mas largo que el contexto del modelo es un error explicito, no un recorte silencioso.
- No se guarda el contenido de los documentos del usuario, solo rutas, consultas y posiciones.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .. import __version__, inspector
from ..generate import IDLE_WINDOW_SECONDS, gpu_summary
from ..gpu import GPUMonitor, idle_power
from ..ollama import Ollama, OllamaError
from ..bench.store import RunHandle, Store, now_iso
from . import bank, corpus as corpus_mod, metrics, subqueries

MAX_MODELS = 8
MAX_REPETITIONS = 30
EMBED_BATCH = 16
REFERENCE_TEXTS = 3
STEPS = ("warmup", "retrieval", "retrieval_user", "similarity", "stability", "performance")
MIN_FILES_FOR_SUBQUERIES = 2  # solo las preguntas con varios archivos necesitan descomposicion
MAX_ITER_ROUNDS = 4
REFINE_FILES, SNIPPET_CHARS = 6, 400  # archivos (y cuanto de cada uno) que el LLM ve en cada ronda


class EmbError(Exception):
    def __init__(self, message: str, code: str = "invalid_config", extra: dict | None = None):
        super().__init__(message)
        self.code = code
        self.extra = extra or {}


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class EmbConfig:
    models: tuple
    prefixes: str = "none"  # none | task
    corpus_folder: str | None = None
    corpus_mode: str = "docs"  # docs | code | both
    repetitions: int = 5
    timeout_s: float = 120.0
    unload_between: bool = True
    unload_others: bool = False
    subquery_llm: str | None = None  # LLM local que escribe subpreguntas (no se mide como embedding)
    subquery_lang: str = "en"  # en | same (el codigo esta en ingles)
    subquery_count: int = 6
    iter_rounds: int = 0  # rondas de busqueda iterativa (0 = apagada); el LLM decide que buscar en cada ronda

    def to_dict(self) -> dict:
        return {"models": list(self.models), "prefixes": self.prefixes, "corpus_folder": self.corpus_folder,
                "corpus_mode": self.corpus_mode, "repetitions": self.repetitions, "timeout_s": self.timeout_s, "unload_between": self.unload_between,
                "unload_others": self.unload_others, "subquery_llm": self.subquery_llm, "subquery_lang": self.subquery_lang,
                "subquery_count": self.subquery_count, "iter_rounds": self.iter_rounds}


def parse_config(body: dict, installed: dict[str, dict], suite: bank.EmbSuite, loaded_now: list[str]) -> EmbConfig:
    models = body.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
        raise EmbError("Elige al menos un modelo.")
    if len(set(models)) != len(models):
        raise EmbError("Hay modelos repetidos en la seleccion.")
    if len(models) > MAX_MODELS:
        raise EmbError(f"Como maximo {MAX_MODELS} modelos por ejecucion.")
    for m in models:
        info = installed.get(m)
        if info is None:
            raise EmbError(f"El modelo '{m}' no esta instalado en Ollama.")
        if "embedding" not in (info.get("capabilities") or []):
            raise EmbError(f"El modelo '{m}' no es de embeddings (capacidades: {info.get('capabilities') or 'desconocidas'}).")
    prefixes = body.get("prefixes", "none")
    if prefixes not in ("none", "task"):
        raise EmbError("'prefixes' debe ser 'none' o 'task'.")
    folder = body.get("corpus_folder")
    if folder is not None and (not isinstance(folder, str) or not folder.strip()):
        raise EmbError("'corpus_folder' debe ser una ruta de carpeta.")
    corpus_mode = body.get("corpus_mode", "docs")
    if corpus_mode not in corpus_mod.MODES:
        raise EmbError(f"'corpus_mode' debe ser uno de: {', '.join(corpus_mod.MODES)}.")
    reps = body.get("repetitions", suite.performance["default_repetitions"])
    if isinstance(reps, bool) or not isinstance(reps, int) or not 1 <= reps <= MAX_REPETITIONS:
        raise EmbError(f"'repetitions' debe estar entre 1 y {MAX_REPETITIONS}.")
    timeout_s = body.get("timeout_s", 120)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 10 <= timeout_s <= 900:
        raise EmbError("'timeout_s' debe estar entre 10 y 900 segundos.")
    unload_others = bool(body.get("unload_others", False))
    llm = body.get("subquery_llm")
    if llm is not None:
        if not isinstance(llm, str) or not llm:
            raise EmbError("'subquery_llm' debe ser el nombre de un modelo de texto o null.")
        if llm in models:
            raise EmbError("El LLM de subpreguntas no puede ser uno de los modelos de embeddings.")
        info = installed.get(llm)
        if info is None:
            raise EmbError(f"El LLM '{llm}' no esta instalado en Ollama.")
        if "completion" not in (info.get("capabilities") or []):
            raise EmbError(f"El modelo '{llm}' no genera texto (capacidades: {info.get('capabilities') or 'desconocidas'}).")
    lang = body.get("subquery_lang", "en")
    if lang not in subqueries.LANGS:
        raise EmbError("'subquery_lang' debe ser 'same' o 'en'.")
    count = body.get("subquery_count", subqueries.DEFAULT_SUBQUERIES)
    if isinstance(count, bool) or not isinstance(count, int) or not subqueries.MIN_SUBQUERIES <= count <= subqueries.MAX_SUBQUERIES:
        raise EmbError(f"'subquery_count' debe estar entre {subqueries.MIN_SUBQUERIES} y {subqueries.MAX_SUBQUERIES}.")
    iter_rounds = body.get("iter_rounds", 0)
    if isinstance(iter_rounds, bool) or not isinstance(iter_rounds, int) or not 0 <= iter_rounds <= MAX_ITER_ROUNDS:
        raise EmbError(f"'iter_rounds' debe estar entre 0 y {MAX_ITER_ROUNDS}.")
    if iter_rounds and not llm:
        raise EmbError("La busqueda iterativa necesita un LLM (subquery_llm).")
    others = [n for n in loaded_now if n not in models and n != llm]
    if others and not unload_others:
        raise EmbError("Hay otros modelos cargados en la VRAM: " + ", ".join(others)
                       + ". Para medir sin interferencias hay que descargarlos (no se borran del disco).",
                       code="models_loaded", extra={"loaded": others})
    return EmbConfig(tuple(models), prefixes, folder.strip() if folder else None, corpus_mode, reps, float(timeout_s),
                     bool(body.get("unload_between", True)), unload_others, llm, lang, count, iter_rounds)


def _fingerprint(vectors: list[list[float]]) -> str:
    return hashlib.sha256(json.dumps([[round(x, 5) for x in v] for v in vectors]).encode()).hexdigest()[:16]


@dataclass
class _State:
    run_id: str
    started: float
    data: dict = field(default_factory=dict)


class EmbManager:
    """Una evaluacion a la vez, en un hilo. Comparte el bloqueo con el chat y el benchmark."""

    def __init__(self, ollama: Ollama, gpu: GPUMonitor | None, store: Store, lock: threading.Lock, suite: bank.EmbSuite,
                 hardware: Callable[[], dict], inspect: Callable[[list[str]], dict[str, dict]],
                 sleep: Callable[[float], None] = time.sleep):
        self.ollama, self.gpu, self.store, self.lock, self.suite = ollama, gpu, store, lock, suite
        self._hardware, self._inspect, self._sleep = hardware, inspect, sleep
        self._plock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: _State | None = None
        self._rules = bank.load_prefix_rules()
        self.cache = subqueries.SubqueryCache(store.runs_dir / "subquery_cache.json")

    # --- API publica ---------------------------------------------------

    def start(self, body: dict) -> str:
        if not isinstance(body, dict):
            raise EmbError("Cuerpo invalido.")
        names = body.get("models") if isinstance(body.get("models"), list) else []
        if isinstance(body.get("subquery_llm"), str):
            names = names + [body["subquery_llm"]]
        try:
            installed = self._inspect([n for n in names if isinstance(n, str)])
            loaded_now = [m["name"] for m in self.ollama.loaded()]
        except OllamaError as exc:
            raise EmbError(str(exc), code="ollama_unavailable") from exc
        cfg = parse_config(body, installed, self.suite, loaded_now)
        user_corpus = None
        if cfg.corpus_folder:
            try:
                user_corpus = corpus_mod.build_corpus(cfg.corpus_folder, cfg.corpus_mode)
            except corpus_mod.CorpusError as exc:
                raise EmbError(str(exc), code="invalid_corpus") from exc
        if not self.lock.acquire(blocking=False):
            raise EmbError("Hay una generacion o evaluacion en curso: ELI ejecuta una a la vez.", code="busy")
        try:
            steps = [s for s in STEPS if s != "retrieval_user" or user_corpus]
            if cfg.iter_rounds and cfg.subquery_llm and user_corpus:
                steps.append("iterative")
            llm_jobs = self._llm_jobs(cfg, user_corpus)
            total = len(cfg.models) * len(steps) + (1 if llm_jobs else 0)
            handle = self.store.create_run({
                "eli_version": __version__, "kind": "embeddings", "config": cfg.to_dict(),
                "suite": {"id": self.suite.id, "version": self.suite.version, "sha256": self.suite.sha256,
                          "documents": len(self.suite.documents), "queries": len(self.suite.queries)},
                "corpus": {"fingerprint": user_corpus.fingerprint, "mode": user_corpus.mode, "stats": user_corpus.stats} if user_corpus else None,
                "prefix_rules_version": self._rules.get("rules_version"),
                "models": [inspector.snapshot(installed[m]) for m in cfg.models],
                "hardware": self._hardware(), "ollama_version": self.ollama.version(), "planned": {"total": total},
            })
            handle.set_status(total=total)
            self._cancel.clear()
            with self._plock:
                self._state = _State(handle.run_id, time.time(), {
                    "run_id": handle.run_id, "state": "running", "started": now_iso(), "models": list(cfg.models),
                    "model_index": 0, "current_model": None, "current_step": None, "done": 0, "total": total,
                    "counts": {"ok": 0, "error": 0, "skipped": 0}, "errors": [], "message": None,
                })
            self._thread = threading.Thread(target=self._execute, args=(handle, cfg, user_corpus, installed, steps), daemon=True)
            self._thread.start()
            return handle.run_id
        except BaseException:
            self.lock.release()
            raise

    def cancel(self) -> bool:
        with self._plock:
            running = self._state is not None and self._state.data["state"] == "running"
        if running:
            self._cancel.set()
        return running

    def status(self) -> dict:
        with self._plock:
            if self._state is None:
                return {"active": False, "run_id": None}
            data = dict(self._state.data)
            data["counts"], data["errors"] = dict(data["counts"]), list(data["errors"])
            data["elapsed_s"] = round(time.time() - self._state.started, 1)
        data["active"] = data["state"] == "running"
        return data

    def active_run_ids(self) -> set[str]:
        with self._plock:
            return {self._state.run_id} if self._state and self._state.data["state"] == "running" else set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # --- ejecucion -----------------------------------------------------

    def _set(self, **fields) -> None:
        with self._plock:
            self._state.data.update(fields)

    def _bump(self, outcome: str, error: dict | None = None) -> None:
        with self._plock:
            d = self._state.data
            d["counts"][outcome] += 1
            d["done"] += 1
            if error:
                d["errors"] = (d["errors"] + [error])[-20:]

    def _execute(self, handle: RunHandle, cfg: EmbConfig, user_corpus, installed: dict[str, dict], steps: list[str]) -> None:
        model_stats: dict[str, dict] = {}
        try:
            if cfg.unload_others:
                for m in self.ollama.loaded():
                    if m["name"] not in cfg.models:
                        self.ollama.unload(m["name"])
                self._wait_unloaded()
            llm_jobs = self._llm_jobs(cfg, user_corpus)
            if llm_jobs and not self._cancel.is_set():
                self._set(current_model=cfg.subquery_llm, current_step="subquery_gen")
                self._generate_subqueries(handle, cfg, llm_jobs)
            for index, model in enumerate(cfg.models):
                if self._cancel.is_set():
                    break
                self._set(model_index=index + 1, current_model=model)
                model_stats[model] = self._run_model(handle, cfg, user_corpus, model, installed[model], steps)
                handle.set_status(models=model_stats, done=self._state.data["done"], counts=dict(self._state.data["counts"]))
            state = "cancelled" if self._cancel.is_set() else "completed"
            message = "Cancelada por el usuario: lo medido hasta ahora se conserva." if state == "cancelled" else None
        except BaseException as exc:  # noqa: BLE001 - cualquier fallo debe quedar registrado, no perderse en el hilo
            state, message = "failed", f"{type(exc).__name__}: {exc}"
            with self._plock:
                self._state.data["errors"].append({"error": message})
        finally:
            with self._plock:
                d = self._state.data
                d["state"], d["message"], d["current_step"], d["finished"] = state, message, None, now_iso()
                final = {"state": state, "message": message, "finished": d["finished"], "done": d["done"],
                         "counts": dict(d["counts"]), "models": model_stats}
            handle.set_status(**final)
            self.lock.release()

    @staticmethod
    def _llm_jobs(cfg: EmbConfig, user_corpus) -> list[dict]:
        """Preguntas con varios archivos relevantes: son las que se descomponen en subpreguntas."""
        if not cfg.subquery_llm or not user_corpus:
            return []
        source = {d["id"]: d.get("source") for d in user_corpus.documents}
        return [q for q in user_corpus.queries
                if len({source.get(i) for i in q["relevant"]} - {None}) >= MIN_FILES_FOR_SUBQUERIES]

    def _generate_subqueries(self, handle: RunHandle, cfg: EmbConfig, jobs: list[dict]) -> None:
        """Subpreguntas del LLM: primero desde la cache (sin cargar el LLM); lo que falte lo escribe el LLM, una vez, sin ver los archivos."""
        t0, errors, examples, cached = time.perf_counter(), [], [], 0
        keys = {id(q): subqueries.cache_key(q["text"], cfg.subquery_llm, cfg.subquery_count, cfg.subquery_lang) for q in jobs}
        pending = []
        for q in jobs:
            hit = self.cache.get(keys[id(q)])
            if hit:
                q["subqueries_llm"], cached = hit, cached + 1
                if len(examples) < 3:
                    examples.append({"query": q["text"], "subqueries": hit})
            else:
                pending.append(q)
        if pending and not self._cancel.is_set():
            self._set(current_model=cfg.subquery_llm, current_step="subquery_gen")
            for q in pending:
                if self._cancel.is_set():
                    break
                try:
                    q["subqueries_llm"] = subqueries.generate_subqueries(
                        self.ollama, self.gpu, cfg.subquery_llm, q["text"], cfg.subquery_count, cfg.subquery_lang,
                        should_stop=self._cancel.is_set, timeout_s=cfg.timeout_s)
                    self.cache.put(keys[id(q)], q["subqueries_llm"])
                    if len(examples) < 3:
                        examples.append({"query": q["text"], "subqueries": q["subqueries_llm"]})
                except subqueries.SubqueryError as exc:
                    errors.append({"query": q["id"], "error": str(exc)})
            try:  # el LLM sale de la VRAM antes de medir los embeddings
                self.ollama.unload(cfg.subquery_llm)
            except OllamaError:
                pass
            self._wait_unloaded()
        ok = cached + sum(1 for q in pending if q.get("subqueries_llm"))
        failed = len(errors)
        self._bump("ok" if ok else "error", {"model": cfg.subquery_llm, "step": "subquery_gen", "error": errors[0]["error"]} if errors and not ok else None)
        handle.append({"type": "subquery_gen", "model": cfg.subquery_llm, "status": "ok" if ok else "error", "lang": cfg.subquery_lang,
                       "count": cfg.subquery_count, "queries": len(jobs), "generated": ok - cached, "cached": cached, "failed": failed,
                       "errors": errors[:5], "examples": examples, "duration_s": round(time.perf_counter() - t0, 3)})

    def _swap_out(self, name: str) -> None:
        """Descarga un modelo y espera a que Ollama lo confirme (para dejar sitio al otro en la GPU)."""
        try:
            self.ollama.unload(name)
        except OllamaError:
            pass
        self._wait_unloaded()

    @staticmethod
    def _absorb(state: dict, vec: list[float], doc_vecs: list[list[float]], sources: list, docs: list[dict]) -> None:
        """Actualiza el mejor fragmento de cada archivo con una busqueda nueva (el archivo vale su mejor fragmento)."""
        for i, dv in enumerate(doc_vecs):
            f = sources[i]
            if f is None:
                continue
            c = metrics.cosine(vec, dv)
            if c > state["best"].get(f, -2.0):
                state["best"][f], state["snip"][f] = c, docs[i]["text"][:SNIPPET_CHARS]

    def _iterative(self, cfg, user_corpus, model, pre, stats) -> dict:
        """Busqueda en rondas: el LLM ve la pregunta y los archivos ya encontrados, y escribe que buscar. Cada ronda alterna
        el LLM y el embedding en la GPU (no caben juntos en una 16 GB con modelos grandes)."""
        jobs, docs = self._llm_jobs(cfg, user_corpus), user_corpus.documents
        if not jobs:
            return {"metrics": None, "by_kind": {}, "queries": []}
        sources = [d.get("source") for d in docs]
        doc_vecs, _ = self._embed_many(cfg, model, [pre["document"] + d["text"] for d in docs])
        q_vecs, _ = self._embed_many(cfg, model, [pre["query"] + q["text"] for q in jobs])
        state = []
        for q, v in zip(jobs, q_vecs):
            s = {"q": q, "vecs": [v], "best": {}, "snip": {}, "phrases": [], "pending": [], "rounds": 0, "errors": 0, "done": False}
            self._absorb(s, v, doc_vecs, sources, docs)
            state.append(s)
        for _ in range(cfg.iter_rounds):
            active = [s for s in state if not s["done"]]
            if not active or self._cancel.is_set():
                break
            self._swap_out(model)  # la GPU es para el LLM durante la ronda
            for s in active:
                if self._cancel.is_set():
                    raise Cancelled()
                found = [(f, s["snip"][f]) for f in sorted(s["best"], key=s["best"].get, reverse=True)[:REFINE_FILES]]
                try:
                    done, phrases = subqueries.refine(self.ollama, self.gpu, cfg.subquery_llm, s["q"]["text"], found,
                                                      should_stop=self._cancel.is_set, timeout_s=cfg.timeout_s)
                except subqueries.SubqueryError:
                    done, phrases, s["errors"] = True, [], s["errors"] + 1  # una ronda rota cierra la pregunta, no el modelo
                s["rounds"] += 1
                s["pending"], s["phrases"] = phrases, s["phrases"] + phrases
                s["done"] = s["done"] or done or not phrases
            self._swap_out(cfg.subquery_llm)
            texts = [pre["query"] + p for s in active for p in s["pending"]]
            if texts:
                vecs, _ = self._embed_many(cfg, model, texts)
                k = 0
                for s in active:
                    for _ in s["pending"]:
                        s["vecs"].append(vecs[k])
                        self._absorb(s, vecs[k], doc_vecs, sources, docs)
                        k += 1
                    s["pending"] = []
        index = {d["id"]: i for i, d in enumerate(docs)}
        rows = []
        for s in state:
            relevant = {index[r] for r in s["q"]["relevant"]}
            cov = metrics.subquery_coverage(s["vecs"][0], s["vecs"], doc_vecs, sources, relevant)
            rows.append({"id": s["q"]["id"], "kind": s["q"].get("kind"), "rounds": s["rounds"], "phrases": s["phrases"],
                         "errors": s["errors"], "iterative": cov})
        kinds = sorted({r["kind"] for r in rows if r["kind"]})
        return {"metrics": metrics.subquery_metrics(rows, field="iterative"),
                "by_kind": {k: m for k in kinds if (m := metrics.subquery_metrics([r for r in rows if r["kind"] == k], field="iterative"))},
                "rounds_mean": round(statistics.fmean(r["rounds"] for r in rows), 2), "queries": rows}

    def _wait_unloaded(self, timeout: float = 15.0) -> bool:
        """Ollama descarga de forma asincrona: se espera a que /api/ps confirme que no queda nada cargado."""
        for _ in range(int(timeout / 0.25) + 1):
            if self._cancel.is_set():
                return False
            try:
                if not self.ollama.loaded():
                    return True
            except OllamaError:
                return False
            self._sleep(0.25)
        return False

    def _vram_idle(self) -> float | None:
        if self.gpu is None:
            return None
        self._sleep(1.5)
        latest = self.gpu.latest()
        return latest.mem_used if latest else None

    def _embed(self, cfg: EmbConfig, model: str, texts: list[str]) -> dict:
        if self._cancel.is_set():
            raise Cancelled()
        return self.ollama.embed(model, texts, timeout=cfg.timeout_s)

    def _embed_many(self, cfg: EmbConfig, model: str, texts: list[str]) -> tuple[list[list[float]], dict]:
        """Textos en lotes de EMBED_BATCH. Devuelve vectores y {tokens, wall_s}."""
        vectors, tokens, t0 = [], 0, time.perf_counter()
        for i in range(0, len(texts), EMBED_BATCH):
            out = self._embed(cfg, model, texts[i:i + EMBED_BATCH])
            vectors += out["embeddings"]
            tokens += out.get("prompt_tokens") or 0
        return vectors, {"tokens": tokens, "wall_s": round(time.perf_counter() - t0, 3)}

    def _run_model(self, handle: RunHandle, cfg: EmbConfig, user_corpus, model: str, info: dict, steps: list[str]) -> dict:
        stats = {"vram_used_mib": None, "vram_gpu_peak_mib": None, "model_vram_bytes": None, "aborted": False}
        pre = (bank.prefix_for(model, info.get("architecture"), self._rules) if cfg.prefixes == "task"
               else {"rule": None, "query": "", "document": "", "source": "prefijos desactivados"})
        if cfg.unload_between:
            self._wait_unloaded()  # sin esto la 'VRAM previa' podria incluir al modelo anterior todavia descargandose
        idle = self._vram_idle()
        handle.append({"type": "model_start", "model": model, "meta": inspector.snapshot(info), "vram_idle_mib": idle,
                       "prefixes": {"mode": cfg.prefixes, **pre}, "started": now_iso()})
        runners = {"warmup": self._warmup, "retrieval": self._retrieval, "retrieval_user": self._retrieval_user,
                   "similarity": self._similarity, "stability": self._stability, "performance": self._performance,
                   "iterative": self._iterative}
        for step in steps:
            if self._cancel.is_set():
                break
            if stats["aborted"]:
                self._bump("skipped")
                handle.append({"type": "step", "model": model, "step": step, "status": "skipped",
                               "error": "el calentamiento fallo: se omite el modelo."})
                continue
            self._set(current_step=step)
            t_start = time.time()
            baseline = idle_power(self.gpu.history(IDLE_WINDOW_SECONDS)) if self.gpu else None
            record = {"type": "step", "model": model, "step": step, "started": now_iso()}
            try:
                data = runners[step](cfg, user_corpus, model, pre, stats)
                record.update(status="ok", **data)
                self._bump("ok")
            except Cancelled:
                break
            except (OllamaError, ValueError) as exc:
                record.update(status="error", error=str(exc))
                self._bump("error", {"model": model, "step": step, "error": str(exc)})
                if step == "warmup":
                    stats["aborted"] = True
            t_end = time.time()
            _, gpu_stats = gpu_summary(self.gpu, t_start, None, t_end, baseline)
            record["gpu"] = gpu_stats["request"]
            record["duration_s"] = round(t_end - t_start, 3)
            self._track_vram(stats, record, idle)
            handle.append(record)
        handle.append({"type": "model_end", "model": model, "aborted": stats["aborted"], "finished": now_iso()})
        if cfg.unload_between:
            try:
                self.ollama.unload(model)
            except OllamaError:
                pass
        return stats

    @staticmethod
    def _track_vram(stats: dict, record: dict, idle: float | None) -> None:
        peak = (record.get("gpu") or {}).get("vram_peak_mib")
        if peak is not None:
            stats["vram_gpu_peak_mib"] = max(stats["vram_gpu_peak_mib"] or 0, peak)
            if idle is not None:
                stats["vram_used_mib"] = max(stats["vram_used_mib"] or 0, round(peak - idle, 1))

    # --- pasos ---------------------------------------------------------

    def _warmup(self, cfg, user_corpus, model, pre, stats) -> dict:
        out = self._embed(cfg, model, ["calentamiento"])
        try:  # cuanta VRAM reserva Ollama para este modelo segun /api/ps
            row = next((m for m in self.ollama.loaded() if m["name"] == model), None)
            if row and row.get("size_vram") is not None:
                stats["model_vram_bytes"] = row["size_vram"]
        except OllamaError:
            pass
        return {"load_s": out.get("load_s"), "dims": len(out["embeddings"][0]),
                "norm": round(metrics.norm(out["embeddings"][0]), 4), "model_vram_bytes": stats["model_vram_bytes"]}

    def _rank_queries(self, cfg, model, pre, documents: list[dict], queries: list[dict]) -> tuple[list[dict], dict]:
        doc_vecs, doc_info = self._embed_many(cfg, model, [pre["document"] + d["text"] for d in documents])
        q_vecs, q_info = self._embed_many(cfg, model, [pre["query"] + q["text"] for q in queries])
        # subpreguntas a mano (suite/eli_queries) y generadas por el LLM: se embeben juntas y se separan despues
        sub_texts = [t for q in queries for key in ("subqueries", "subqueries_llm") for t in q.get(key, [])]
        sub_vecs_all, _ = self._embed_many(cfg, model, [pre["query"] + t for t in sub_texts]) if sub_texts else ([], None)
        sub_at = 0
        index = {d["id"]: i for i, d in enumerate(documents)}
        sources = [d.get("source") for d in documents]
        rows = []
        for q, vec in zip(queries, q_vecs):
            relevant = {index[r] for r in q["relevant"]}
            rank = metrics.relevant_rank(vec, doc_vecs, relevant)
            coverage = metrics.file_coverage(vec, doc_vecs, sources, relevant)
            by_file = metrics.file_level(vec, doc_vecs, sources, relevant)
            picked = {}
            for key in ("subqueries", "subqueries_llm"):
                n_sub = len(q.get(key, []))
                sub_vecs, sub_at = sub_vecs_all[sub_at:sub_at + n_sub], sub_at + n_sub
                # la pregunta original cuenta como una consulta mas de la union (sus archivos ya estaban en la comparacion)
                extra = [vec] if key == "subqueries_llm" and n_sub else []
                picked[key] = metrics.subquery_coverage(vec, sub_vecs, doc_vecs, sources, relevant, extra_vecs=extra) if n_sub else None
            n_sub = len(q.get("subqueries", []))
            rows.append({"id": q["id"], "query": q["text"], "kind": q.get("kind"), "lang": q.get("lang"),
                         "source": q.get("source"), "relevant": list(q["relevant"]), "rank": rank,
                         "files": len({sources[i] for i in relevant if sources[i]}), "coverage": coverage, "file_level": by_file,
                         "subquery": picked["subqueries"], "subquery_llm": picked["subqueries_llm"],
                         **({"subqueries": q["subqueries"]} if n_sub else {}),
                         **({"subqueries_llm": q["subqueries_llm"]} if q.get("subqueries_llm") else {})})
        timing = {"documents": len(documents), "queries": len(queries), "tokens": doc_info["tokens"] + q_info["tokens"],
                  "docs_per_s": round(len(documents) / doc_info["wall_s"], 2) if doc_info["wall_s"] else None,
                  "dims": len(doc_vecs[0])}
        return rows, timing

    def _retrieval(self, cfg, user_corpus, model, pre, stats) -> dict:
        rows, timing = self._rank_queries(cfg, model, pre, list(self.suite.documents), list(self.suite.queries))
        by_kind = {k: metrics.retrieval_metrics([r["rank"] for r in rows if r["kind"] == k]) for k in ("mono", "cross")}
        by_lang = {k: metrics.retrieval_metrics([r["rank"] for r in rows if r["kind"] == "mono" and r["lang"] == k]) for k in ("es", "en")}
        return {"metrics": metrics.retrieval_metrics([r["rank"] for r in rows]), "by_kind": by_kind, "by_lang": by_lang,
                "timing": timing, "queries": rows}

    def _retrieval_user(self, cfg, user_corpus, model, pre, stats) -> dict:
        rows, timing = self._rank_queries(cfg, model, pre, user_corpus.documents, user_corpus.queries)
        for r in rows:  # solo rutas, consultas y posiciones: nunca el contenido de los documentos
            r.pop("lang", None)
        kinds = sorted({r["kind"] for r in rows if r["kind"]})
        by_kind = {k: metrics.retrieval_metrics([r["rank"] for r in rows if r["kind"] == k]) for k in kinds}
        return {"metrics": metrics.retrieval_metrics([r["rank"] for r in rows]), "by_kind": by_kind,
                "coverage": metrics.coverage_metrics(rows),
                "coverage_by_kind": {k: c for k in kinds if (c := metrics.coverage_metrics([r for r in rows if r["kind"] == k]))},
                "subquery": metrics.subquery_metrics(rows),
                "subquery_by_kind": {k: c for k in kinds if (c := metrics.subquery_metrics([r for r in rows if r["kind"] == k]))},
                "subquery_llm": metrics.subquery_metrics(rows, field="subquery_llm"),
                "subquery_llm_by_kind": {k: c for k in kinds if (c := metrics.subquery_metrics([r for r in rows if r["kind"] == k], field="subquery_llm"))},
                "file_level": metrics.file_level_metrics(rows),
                "file_level_by_kind": {k: f for k in kinds if (f := metrics.file_level_metrics([r for r in rows if r["kind"] == k]))},
                "timing": timing, "queries": rows, "corpus_fingerprint": user_corpus.fingerprint, "corpus_mode": user_corpus.mode}

    def _similarity(self, cfg, user_corpus, model, pre, stats) -> dict:
        texts = []
        for t in self.suite.triplets:
            texts += [pre["query"] + t["anchor"], pre["document"] + t["positive"], pre["document"] + t["negative"]]
        vecs, _ = self._embed_many(cfg, model, texts)
        rows = []
        for n, t in enumerate(self.suite.triplets):
            a, p, neg = vecs[3 * n:3 * n + 3]
            sp, sn = metrics.cosine(a, p), metrics.cosine(a, neg)
            rows.append({"id": t["id"], "kind": t["kind"], "sim_positive": round(sp, 4), "sim_negative": round(sn, 4), "ok": sp > sn})
        by_kind = {k: metrics.triplet_accuracy([r["ok"] for r in rows if r["kind"] == k]) for k in bank.TRIPLET_KINDS}
        return {"accuracy": metrics.triplet_accuracy([r["ok"] for r in rows]), "by_kind": by_kind, "triplets": rows}

    def _stability(self, cfg, user_corpus, model, pre, stats) -> dict:
        texts = [pre["document"] + t for t in self.suite.stability_texts]
        first, _ = self._embed_many(cfg, model, texts)
        again, _ = self._embed_many(cfg, model, texts)
        batch = self._embed(cfg, model, texts)["embeddings"]
        repeat = metrics.min_cosine(list(zip(first, again)))
        in_batch = metrics.min_cosine(list(zip(first, batch)))
        # vectores de referencia: tras convertir a TensorRT se compara coseno(ollama, trt) con estos mismos textos
        ref = first[:REFERENCE_TEXTS]
        return {"min_cos_repeat": repeat, "min_cos_batch": in_batch,
                "reference": {"texts": texts[:REFERENCE_TEXTS], "vectors": [[round(x, 6) for x in v] for v in ref],
                              "fingerprint": _fingerprint(ref)}}

    def _performance(self, cfg, user_corpus, model, pre, stats) -> dict:
        perf, reps = self.suite.performance, cfg.repetitions
        latency, count = {}, 0
        for label, words in (("short", perf["short_words"]), ("medium", perf["medium_words"]), ("long", perf["long_words"])):
            text = pre["document"] + self.suite.perf_text(words)
            walls, tokens = [], None
            for _ in range(reps):
                t0 = time.perf_counter()
                out = self._embed(cfg, model, [text])
                walls.append(time.perf_counter() - t0)
                tokens, count = out.get("prompt_tokens"), count + 1
            latency[label] = {"words": words, "tokens": tokens, "median_ms": round(statistics.median(walls) * 1000, 1),
                              "min_ms": round(min(walls) * 1000, 1), "max_ms": round(max(walls) * 1000, 1), "n": reps}
        throughput = {}
        for size in perf["batch_sizes"]:
            texts = [pre["document"] + self.suite.perf_text(perf["medium_words"], offset=i) for i in range(size)]
            walls = []
            for _ in range(max(3, reps // 2)):
                t0 = time.perf_counter()
                self._embed(cfg, model, texts)
                walls.append(time.perf_counter() - t0)
                count += size
            throughput[str(size)] = {"batch": size, "median_s": round(statistics.median(walls), 4),
                                     "emb_per_s": round(size / statistics.median(walls), 1), "n": len(walls)}
        return {"latency": latency, "throughput": throughput, "embeddings_done": count}

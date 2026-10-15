"""Motor del benchmark automatico.

- Los modelos se ejecutan uno tras otro (nunca compiten por la VRAM).
- Cada prueba usa una conversacion independiente: ningun historial pasa de una prueba o modelo a otra.
- Todos los modelos reciben exactamente las mismas pruebas y condiciones.
- El calentamiento va aparte: los tiempos de carga no se mezclan con la inferencia en caliente.
- Cada resultado se guarda al terminar (results.jsonl): una cancelacion o un fallo no pierde lo ya medido.
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .. import __version__, inspector
from ..generate import CANCELLED, ERROR, OK, TIMEOUT, generate
from ..gpu import GPUMonitor
from ..ollama import Ollama, OllamaError
from . import bank, graders, sandbox
from .store import RunHandle, Store, now_iso

MAX_MODELS = 8
MAX_REPETITIONS = 10
MAX_CONTEXT = 65536
CONSECUTIVE_ERRORS_LIMIT = 3


class BenchError(Exception):
    def __init__(self, message: str, code: str = "invalid_config", extra: dict | None = None):
        super().__init__(message)
        self.code = code
        self.extra = extra or {}


@dataclass(frozen=True)
class RunConfig:
    models: tuple
    mode: str = "quick"
    categories: tuple | None = None
    repetitions: int = 1
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int | None = None
    num_predict: int | None = None
    think: bool | None = False
    context_sizes: tuple | None = None
    timeout_s: float = 180.0
    warmup: bool = True
    unload_between: bool = True
    unload_others: bool = False

    def to_dict(self) -> dict:
        return {
            "models": list(self.models), "mode": self.mode, "categories": list(self.categories) if self.categories else None,
            "repetitions": self.repetitions, "temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx,
            "num_predict": self.num_predict, "think": self.think,
            "context_sizes": list(self.context_sizes) if self.context_sizes else None, "timeout_s": self.timeout_s,
            "warmup": self.warmup, "unload_between": self.unload_between, "unload_others": self.unload_others,
        }


def _int(body: dict, key: str, default, low: int, high: int, allow_none: bool = False):
    raw = body.get(key, default)
    if raw is None and allow_none:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) != raw:
        raise BenchError(f"'{key}' debe ser un entero.")
    if not low <= raw <= high:
        raise BenchError(f"'{key}' debe estar entre {low} y {high}.")
    return int(raw)


def parse_config(body: dict, installed: dict[str, dict], suite: bank.Suite, loaded_now: list[str]) -> RunConfig:
    """Valida la configuracion. `installed` mapea nombre -> informacion del Inspector de los modelos pedidos."""
    models = body.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
        raise BenchError("Elige al menos un modelo.")
    if len(set(models)) != len(models):
        raise BenchError("Hay modelos repetidos en la seleccion.")
    if len(models) > MAX_MODELS:
        raise BenchError(f"Como maximo {MAX_MODELS} modelos por ejecucion.")
    for m in models:
        info = installed.get(m)
        if info is None:
            raise BenchError(f"El modelo '{m}' no esta instalado en Ollama.")
        if "completion" not in (info.get("capabilities") or []):
            raise BenchError(f"El modelo '{m}' no genera texto (capacidades: {info.get('capabilities') or 'desconocidas'}); no se puede evaluar.")

    mode = body.get("mode", "quick")
    if mode not in ("quick", "full"):
        raise BenchError("'mode' debe ser 'quick' o 'full'.")
    categories = body.get("categories")
    if categories is not None:
        if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
            raise BenchError("'categories' debe ser una lista.")
        unknown = [c for c in categories if c not in suite.categories]
        if unknown:
            raise BenchError(f"Categorias desconocidas: {unknown}")
        categories = tuple(categories) or None

    temperature = body.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise BenchError("'temperature' debe estar entre 0 y 2.")
    think = body.get("think", False)
    if think is not None and not isinstance(think, bool):
        raise BenchError("'think' debe ser verdadero, falso o nulo.")
    timeout_s = body.get("timeout_s", 180)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 10 <= timeout_s <= 1800:
        raise BenchError("'timeout_s' debe estar entre 10 y 1800 segundos.")

    sizes = body.get("context_sizes")
    if sizes is not None:
        if not isinstance(sizes, list) or not 1 <= len(sizes) <= 6:
            raise BenchError("'context_sizes' debe ser una lista de 1 a 6 tamanos.")
        sizes = tuple(sorted({_int({"s": s}, "s", None, 512, MAX_CONTEXT) for s in sizes}))
    declared = [installed[m].get("context_length") for m in models if installed[m].get("context_length")]
    if sizes and declared and max(sizes) > min(declared):
        raise BenchError(f"Algun tamano de contexto supera el contexto declarado por un modelo (maximo comun: {min(declared)}).")
    unload_others = bool(body.get("unload_others", False))
    others = [n for n in loaded_now if n not in models]
    if others and not unload_others:
        raise BenchError("Hay otros modelos cargados en la VRAM: " + ", ".join(others)
                         + ". Para medir sin interferencias hay que descargarlos (no se borran del disco).",
                         code="models_loaded", extra={"loaded": others})

    return RunConfig(
        models=tuple(models), mode=mode, categories=categories,
        repetitions=_int(body, "repetitions", 1, 1, MAX_REPETITIONS), temperature=float(temperature),
        seed=_int(body, "seed", 42, 0, 2**31 - 1), num_ctx=_int(body, "num_ctx", None, 512, MAX_CONTEXT, allow_none=True),
        num_predict=_int(body, "num_predict", None, 16, 8192, allow_none=True), think=think, context_sizes=sizes,
        timeout_s=float(timeout_s), warmup=bool(body.get("warmup", True)),
        unload_between=bool(body.get("unload_between", True)), unload_others=unload_others,
    )


def outcome_of(gen_status: str, grade: graders.Grade | None) -> str:
    if gen_status == OK:
        return grade.status if grade else "ungraded"
    return {ERROR: "error", TIMEOUT: "timeout", CANCELLED: "cancelled"}[gen_status]


@dataclass
class _State:
    run_id: str
    started: float
    data: dict = field(default_factory=dict)


class BenchManager:
    """Una ejecucion a la vez, en un hilo. El bloqueo es el mismo que usa el chat: nunca compiten por la GPU."""

    def __init__(self, ollama: Ollama, gpu: GPUMonitor | None, store: Store, lock: threading.Lock,
                 suite: bank.Suite, hardware: Callable[[], dict], inspect: Callable[[list[str]], dict[str, dict]],
                 sleep: Callable[[float], None] = time.sleep):
        self.ollama, self.gpu, self.store, self.lock, self.suite = ollama, gpu, store, lock, suite
        self._hardware, self._inspect, self._sleep = hardware, inspect, sleep
        self._plock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: _State | None = None

    # --- API publica ---------------------------------------------------

    def start(self, body: dict) -> str:
        if not isinstance(body, dict):
            raise BenchError("Cuerpo invalido.")
        names = body.get("models") if isinstance(body.get("models"), list) else []
        try:
            installed = self._inspect([n for n in names if isinstance(n, str)])
            loaded_now = [m["name"] for m in self.ollama.loaded()]
        except OllamaError as exc:
            raise BenchError(str(exc), code="ollama_unavailable") from exc
        cfg = parse_config(body, installed, self.suite, loaded_now)
        tests = self.suite.select(cfg.mode, list(cfg.categories) if cfg.categories else None)
        if not tests:
            raise BenchError("La seleccion no contiene ninguna prueba.")
        if not self.lock.acquire(blocking=False):
            raise BenchError("Hay una generacion o evaluacion en curso: ELI ejecuta una a la vez.", code="busy")
        try:
            handle = self.store.create_run({
                "eli_version": __version__, "config": cfg.to_dict(),
                "suite": {"id": self.suite.id, "version": self.suite.version, "sha256": self.suite.sha256, "tests": len(self.suite.tests)},
                "models": [inspector.snapshot(installed[m]) for m in cfg.models],
                "hardware": self._hardware(), "ollama_version": self.ollama.version(), "sandbox": sandbox.info(),
                "planned": {"total": 0},
            })
            instances = self.suite.instances(tests, list(cfg.context_sizes) if cfg.context_sizes else None, cfg.num_ctx,
                                             marker_salt=handle.run_id, mode=cfg.mode)
            total = len(cfg.models) * len(instances) * cfg.repetitions
            handle.set_status(total=total)
            self._cancel.clear()
            with self._plock:
                self._state = _State(handle.run_id, time.time(), {
                    "run_id": handle.run_id, "state": "running", "started": now_iso(), "models": list(cfg.models),
                    "model_index": 0, "current_model": None, "current_test": None, "current_category": None,
                    "current_repetition": None, "done": 0, "total": total,
                    "counts": {"passed": 0, "failed": 0, "ungraded": 0, "error": 0, "timeout": 0, "cancelled": 0, "skipped": 0},
                    "recent": [], "errors": [], "message": None,
                })
            self._thread = threading.Thread(target=self._execute, args=(handle, cfg, instances, installed), daemon=True)
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
            data["counts"], data["recent"], data["errors"] = dict(data["counts"]), list(data["recent"]), list(data["errors"])
            data["elapsed_s"] = round(time.time() - self._state.started, 1)
        data["active"] = data["state"] == "running"
        done, total = data["done"], data["total"]
        data["eta_s"] = round(data["elapsed_s"] / done * (total - done)) if data["active"] and done >= 3 else None
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

    def _bump(self, outcome: str, summary: dict) -> None:
        with self._plock:
            d = self._state.data
            d["counts"][outcome] = d["counts"].get(outcome, 0) + 1
            d["done"] += 1
            d["recent"] = (d["recent"] + [summary])[-8:]
            if outcome in ("error", "timeout"):
                d["errors"] = (d["errors"] + [summary])[-20:]

    def _execute(self, handle: RunHandle, cfg: RunConfig, instances: list[bank.Instance], installed: dict[str, dict]) -> None:
        model_stats: dict[str, dict] = {}
        try:
            if cfg.unload_others:
                for m in self.ollama.loaded():
                    if m["name"] not in cfg.models:
                        self.ollama.unload(m["name"])
                self._wait_unloaded()
            for index, model in enumerate(cfg.models):
                if self._cancel.is_set():
                    break
                self._set(model_index=index + 1, current_model=model)
                model_stats[model] = self._run_model(handle, cfg, instances, model, installed[model], model_stats)
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
                d["state"], d["message"], d["current_test"], d["finished"] = state, message, None, now_iso()
                final = {"state": state, "message": message, "finished": d["finished"], "done": d["done"],
                         "counts": dict(d["counts"]), "models": model_stats}
            handle.set_status(**final)
            self.lock.release()

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

    def _run_model(self, handle: RunHandle, cfg: RunConfig, instances: list[bank.Instance], model: str, info: dict,
                   all_stats: dict) -> dict:
        thinks = "thinking" in (info.get("capabilities") or [])
        think = cfg.think if thinks else None
        stats = {"vram_used_mib": None, "vram_gpu_peak_mib": None, "model_vram_bytes": None, "aborted": False}
        if cfg.unload_between:
            self._wait_unloaded()  # sin esto el 'VRAM previa' podria incluir al modelo anterior todavia descargandose
        idle = self._vram_idle()
        handle.append({"type": "model_start", "model": model, "meta": inspector.snapshot(info), "vram_idle_mib": idle,
                       "think_effective": think, "started": now_iso()})
        consecutive_errors, last_ctx = 0, None
        remaining = len(instances) * cfg.repetitions
        for inst in instances:
            if self._cancel.is_set():
                break
            if stats["aborted"]:
                break
            ctx = inst.params["num_ctx"]
            if cfg.warmup and ctx != last_ctx:
                self._warmup(handle, cfg, model, ctx, think)
                last_ctx = ctx
            for rep in range(1, cfg.repetitions + 1):
                if self._cancel.is_set():
                    break
                self._set(current_test=inst.instance_id, current_category=inst.category, current_repetition=rep)
                record = self._run_one(cfg, model, inst, rep, think)
                handle.append(record)
                remaining -= 1
                outcome = record["outcome"]
                self._bump(outcome, {"model": model, "test": inst.instance_id, "outcome": outcome,
                                     "reason": (record.get("grade") or {}).get("reason") or record.get("error")})
                self._track_vram(stats, record, idle)
                consecutive_errors = consecutive_errors + 1 if outcome in ("error", "timeout") else 0
                if consecutive_errors >= CONSECUTIVE_ERRORS_LIMIT:
                    stats["aborted"] = True
                    break
        if stats["aborted"] and not self._cancel.is_set():
            skipped = remaining
            with self._plock:
                self._state.data["counts"]["skipped"] += skipped
                self._state.data["done"] += skipped
            handle.append({"type": "model_end", "model": model, "aborted": True, "skipped": skipped,
                           "reason": f"{CONSECUTIVE_ERRORS_LIMIT} errores seguidos: se omiten las pruebas restantes de este modelo."})
        else:
            handle.append({"type": "model_end", "model": model, "aborted": False, "finished": now_iso()})
        if cfg.unload_between:
            try:
                self.ollama.unload(model)
            except OllamaError:
                pass
        return stats

    @staticmethod
    def _track_vram(stats: dict, record: dict, idle: float | None) -> None:
        m = record.get("metrics") or {}
        req = (m.get("gpu") or {}).get("request") or {}
        peak = req.get("vram_peak_mib")
        if peak is not None:
            stats["vram_gpu_peak_mib"] = max(stats["vram_gpu_peak_mib"] or 0, peak)
            if idle is not None:
                stats["vram_used_mib"] = max(stats["vram_used_mib"] or 0, round(peak - idle, 1))
        if m.get("model_vram_bytes") is not None:
            stats["model_vram_bytes"] = max(stats["model_vram_bytes"] or 0, m["model_vram_bytes"])

    def _warmup(self, handle: RunHandle, cfg: RunConfig, model: str, num_ctx: int, think: bool | None) -> None:
        options = {"num_ctx": num_ctx, "num_predict": 8, "temperature": 0, "seed": cfg.seed}
        gen = generate(self.ollama, self.gpu, model, [{"role": "user", "content": "Responde solo: ok"}], options, think,
                       timeout_s=max(cfg.timeout_s, 300), should_stop=self._cancel.is_set)
        handle.append({"type": "warmup", "model": model, "num_ctx": num_ctx, "ok": gen.status == OK, "status": gen.status,
                       "load_s": gen.metrics.get("load_s"), "wall_s": gen.metrics.get("wall_s"), "error": gen.error})

    def _run_one(self, cfg: RunConfig, model: str, inst: bank.Instance, rep: int, think: bool | None) -> dict:
        options = {
            "num_ctx": inst.params["num_ctx"],
            "num_predict": cfg.num_predict or inst.params.get("num_predict", 512),
            "temperature": cfg.temperature,
            # con temperatura 0 la semilla no cambia nada; con temperatura > 0 cada repeticion es una muestra distinta
            "seed": cfg.seed + (rep - 1 if cfg.temperature > 0 else 0),
        }
        messages = ([{"role": "system", "content": inst.system}] if inst.system else []) + [{"role": "user", "content": inst.prompt}]
        started = now_iso()
        t0 = time.time()
        gen = generate(self.ollama, self.gpu, model, messages, options, think, timeout_s=cfg.timeout_s,
                       should_stop=self._cancel.is_set)
        fallback = False
        if gen.status == ERROR and think is not None and "think" in (gen.error or "").lower():
            fallback, think = True, None  # el modelo no admite el parametro: se repite sin el
            gen = generate(self.ollama, self.gpu, model, messages, options, None, timeout_s=cfg.timeout_s,
                           should_stop=self._cancel.is_set)
        grade = None
        if gen.status == OK:
            grade = graders.grade(inst.grader, gen.content, (gen.final or {}).get("done_reason"))
            if (grade.status == graders.FAILED and inst.context_size
                    and any("se lleno" in w for w in gen.metrics.get("warnings", []))):
                grade = graders.Grade(graders.UNGRADED, "el contexto se lleno: el prompt pudo recortarse y no se puede afirmar que el modelo falle",
                                      grade.checks)
        load_s = gen.metrics.get("load_s") or 0
        return {
            "type": "result", "model": model, "test_id": inst.test.id, "instance_id": inst.instance_id, "category": inst.category,
            "difficulty": inst.test.difficulty, "language": inst.test.language, "repetition": rep,
            "context_size": inst.context_size, "num_ctx_requested": options["num_ctx"],
            "options": {**options, "think": think}, "think_requested": cfg.think, "think_fallback": fallback,
            "prompt": inst.prompt, "system": inst.system, "expected": inst.test.expected,
            "response": {"content": gen.content, "thinking": gen.thinking},
            "outcome": outcome_of(gen.status, grade), "grade": grade.to_dict() if grade else None,
            "gen_status": gen.status, "error": gen.error, "metrics": gen.metrics, "cold": load_s > 1.0,
            "started": started, "duration_s": round(time.time() - t0, 3),
        }

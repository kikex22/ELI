"""Servidor local de ELI: pagina estatica + API JSON/SSE. Solo 127.0.0.1."""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from . import __version__, inspector, inventory
from .bench import bank, compare, sandbox
from .bench.runner import BenchError, BenchManager
from .bench.store import Store
from .emb import bank as emb_bank, corpus as emb_corpus
from .emb.runner import EmbError, EmbManager
from .generate import CANCELLED, OK, generate
from .gpu import GPUMonitor
from .hardware import hardware_info
from .ollama import Ollama, OllamaError

WEB_DIR = Path(__file__).resolve().parent / "web"
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
WEB_FILES = ["app.js", "style.css", "nav.js", "dashboard.js", "models.js", "bench.js", "charts.js", "compare.js", "history.js",
             "embeddings.js"]
STATIC = {"/": ("index.html", _TYPES[".html"]), **{f"/{n}": (n, _TYPES[Path(n).suffix]) for n in WEB_FILES}}
DEFAULTS = {"num_ctx": 4096, "temperature": 0.0, "seed": 42, "num_predict": 512}
MAX_MESSAGES = 200
MAX_CHARS = 2_000_000
BENCH_ERROR_STATUS = {"busy": 409, "models_loaded": 409, "invalid_config": 400, "invalid_corpus": 400, "ollama_unavailable": 502}


class BadRequest(Exception):
    pass


class NotFound(Exception):
    pass


class App:
    def __init__(self, ollama: Ollama, gpu: GPUMonitor, runs_dir: Path, port: int, suite: bank.Suite | None = None,
                 hardware_fn=hardware_info):
        self.ollama = ollama
        self.gpu = gpu
        self.runs_dir = runs_dir
        self.port = port
        self.hardware_fn = hardware_fn
        self.chat_lock = threading.Lock()  # una generacion (chat o benchmark) a la vez: nunca compiten por la GPU
        self.store = Store(runs_dir)
        self.emb_store = Store(runs_dir, "embeddings")
        self.suite = suite or bank.default_suite()
        self.bench = BenchManager(ollama, gpu, self.store, self.chat_lock, self.suite, hardware_fn, self._inspect)
        self.emb_suite = emb_bank.default_suite()
        self.emb = EmbManager(ollama, gpu, self.emb_store, self.chat_lock, self.emb_suite, hardware_fn, self._inspect)
        self.store.mark_interrupted(set())
        self.emb_store.mark_interrupted(set())  # ejecuciones que quedaron 'running' de una sesion anterior

    def _inspect(self, names: list[str] | None = None) -> dict[str, dict]:
        return {i["name"]: i for i in inventory.inspect_installed(self.ollama, self.store, self.hardware_fn(), names, (self.emb_store,))}

    def allowed_hosts(self) -> set[str]:
        return {f"127.0.0.1:{self.port}", f"localhost:{self.port}", "127.0.0.1", "localhost"}


def parse_chat_request(body: dict) -> dict:
    """Valida el cuerpo de /api/chat y devuelve mensajes y opciones listos para Ollama."""
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise BadRequest("Falta el modelo.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("Faltan los mensajes.")
    if len(messages) > MAX_MESSAGES:
        raise BadRequest(f"Demasiados mensajes (maximo {MAX_MESSAGES}).")
    clean, chars = [], 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "system"}:
            raise BadRequest("Mensaje con rol invalido.")
        if not isinstance(message.get("content"), str):
            raise BadRequest("Mensaje sin texto.")
        chars += len(message["content"])
        clean.append({"role": message["role"], "content": message["content"]})
    system = body.get("system")
    if system is not None and not isinstance(system, str):
        raise BadRequest("El mensaje de sistema debe ser texto.")
    if isinstance(system, str) and system.strip():
        chars += len(system)
        clean.insert(0, {"role": "system", "content": system.strip()})
    if chars > MAX_CHARS:
        raise BadRequest("La conversacion es demasiado grande.")

    given = body.get("options", {})
    if given is None:
        given = {}
    if not isinstance(given, dict):
        raise BadRequest("'options' debe ser un objeto.")

    def number(name: str, kind, low, high):
        raw = given.get(name, DEFAULTS[name])
        if isinstance(raw, bool):
            raise BadRequest(f"Valor invalido para {name}.")
        try:
            value = kind(raw)
        except (TypeError, ValueError):
            raise BadRequest(f"Valor invalido para {name}.") from None
        if value != value or not low <= value <= high:  # tambien rechaza NaN
            raise BadRequest(f"{name} debe estar entre {low} y {high}.")
        return value

    options = {
        "num_ctx": number("num_ctx", int, 512, 262144),
        "temperature": number("temperature", float, 0.0, 2.0),
        "seed": number("seed", int, 0, 2**31 - 1),
        "num_predict": number("num_predict", int, 1, 32768),
    }
    think = body.get("think")
    if think is not None and not isinstance(think, bool):
        raise BadRequest("think debe ser verdadero, falso o nulo.")
    return {"model": model.strip(), "messages": clean, "options": options, "think": think}


def _floats(query: dict, key: str) -> float | None:
    raw = (query.get(key) or [""])[0]
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        raise BadRequest(f"'{key}' debe ser un numero.") from None


class Handler(BaseHTTPRequestHandler):
    server_version = "ELI"
    protocol_version = "HTTP/1.0"  # una respuesta por conexion: el streaming no necesita chunked

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, *_args) -> None:
        pass

    # --- utilidades ----------------------------------------------------

    def _send(self, payload: bytes, content_type: str, status: int = 200, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, data, status: int = 200) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _sse(self, data: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
        self.wfile.flush()

    def _guard(self) -> bool:
        """Rechaza otros hosts y origenes (protege contra paginas ajenas del navegador)."""
        host = self.headers.get("Host", "")
        if host not in self.app.allowed_hosts():
            self._json({"error": "Host no permitido"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc not in self.app.allowed_hosts():
            self._json({"error": "Origen no permitido"}, 403)
            return False
        return True

    def _read_json(self) -> dict:
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            raise BadRequest("Se esperaba Content-Type: application/json.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise BadRequest("Content-Length invalido.") from None
        if not 0 < length <= 8_000_000:
            raise BadRequest("Cuerpo vacio o demasiado grande.")
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            raise BadRequest("JSON invalido.") from None
        if not isinstance(body, dict):
            raise BadRequest("Se esperaba un objeto JSON.")
        return body

    def _handle_errors(self, fn) -> None:
        try:
            fn()
        except BadRequest as exc:
            self._json({"error": str(exc)}, 400)
        except NotFound as exc:
            self._json({"error": str(exc) or "No existe"}, 404)
        except inventory.ModelNotFound as exc:
            self._json({"error": f"El modelo '{exc}' no esta instalado."}, 404)
        except inventory.DeleteRefused as exc:
            self._json({"error": str(exc)}, 400)
        except bank.SuiteError as exc:
            self._json({"error": str(exc)}, 500)
        except (BenchError, EmbError) as exc:
            self._json({"error": str(exc), "code": exc.code, **exc.extra}, BENCH_ERROR_STATUS.get(exc.code, 400))
        except OllamaError as exc:
            self._json({"error": str(exc)}, 502)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # --- rutas ---------------------------------------------------------

    def do_GET(self) -> None:
        if self._guard():
            self._handle_errors(self._route_get)

    def do_POST(self) -> None:
        if self._guard():
            self._handle_errors(self._route_post)

    def _route_get(self) -> None:
        url = urlparse(self.path)
        q, path = parse_qs(url.query), url.path
        if path in STATIC:
            return self._static(path)
        simple = {
            "/api/version": self._version, "/api/system": self._system, "/api/status": lambda: self._json(self._status()),
            "/api/dashboard": self._dashboard, "/api/gpu/stream": self._gpu_stream, "/api/bench/status": lambda: self._json(self.app.bench.status()),
            "/api/emb/status": lambda: self._json(self.app.emb.status()), "/api/emb/suite": lambda: self._json(self.app.emb_suite.info()),
            "/api/models": lambda: self._json({"models": self.app.ollama.models()}),
        }
        if path in simple:
            return simple[path]()
        if path == "/api/model":
            return self._json(self.app.ollama.show(self._need(q, "name")))
        if path == "/api/models/inspect":
            names = [q["name"][0]] if q.get("name") else None
            models = inventory.inspect_installed(self.app.ollama, self.app.store, self.app.hardware_fn(), names)
            if names and not models:
                raise NotFound(f"El modelo '{names[0]}' no esta instalado.")
            return self._json({"models": models, "rules_version": inspector.load_rules()["rules_version"]})
        if path == "/api/models/delete-preview":
            return self._json(inventory.delete_preview(self.app.ollama, self._need(q, "name")))
        if path == "/api/gpu/history":
            seconds = min(600.0, max(1.0, _floats(q, "seconds") or 120.0))
            return self._json({"samples": [s.to_dict() for s in self.app.gpu.history(seconds)]})
        if path == "/api/gpu/processes":
            return self._json({"processes": self.app.gpu.processes()})
        if path == "/api/emb/models":
            models = inventory.inspect_installed(self.app.ollama, self.app.store, self.app.hardware_fn(), None, (self.app.emb_store,))
            return self._json({"models": [m for m in models if "embedding" in (m.get("capabilities") or [])],
                               "llms": [m for m in models if "completion" in (m.get("capabilities") or []) and "embedding" not in (m.get("capabilities") or [])]})
        if path == "/api/emb/history":
            return self._json({"runs": self.app.emb_store.list_runs()})
        if path.startswith("/api/emb/history/"):
            return self._emb_history(path[len("/api/emb/history/"):], q)
        if path == "/api/bench/suite":
            return self._suite(q)
        if path == "/api/history":
            return self._json({"runs": self.app.store.list_runs(), "chats": self.app.store.list_chats()})
        if path.startswith("/api/history/"):
            return self._history(path[len("/api/history/"):], q)
        if path == "/api/compare/models":
            return self._json({"models": self.app.store.latest_models(),
                               "size_buckets": [{"id": b[0], "min": b[1], "max": b[2]} for b in compare.SIZE_BUCKETS]})
        if path == "/api/compare":
            return self._compare(q)
        raise NotFound()

    def _route_post(self) -> None:
        path = urlparse(self.path).path
        body = self._read_json()
        if path == "/api/chat":
            return self._chat(body)
        if path == "/api/unload":
            return self._unload(body)
        if path == "/api/models/delete":
            return self._delete(body)
        if path == "/api/bench/start":
            return self._json({"run_id": self.app.bench.start(body)}, 202)
        if path == "/api/bench/cancel":
            return self._json({"cancelling": self.app.bench.cancel()})
        if path == "/api/emb/start":
            return self._json({"run_id": self.app.emb.start(body)}, 202)
        if path == "/api/emb/cancel":
            return self._json({"cancelling": self.app.emb.cancel()})
        if path == "/api/emb/corpus-preview":
            folder = body.get("folder") if isinstance(body, dict) else None
            if not isinstance(folder, str) or not folder.strip():
                raise BadRequest("Falta la carpeta.")
            try:
                return self._json(emb_corpus.build_corpus(folder, body.get("mode", "docs")).preview())
            except emb_corpus.CorpusError as exc:
                raise BadRequest(str(exc)) from exc
        raise NotFound()

    @staticmethod
    def _need(query: dict, key: str) -> str:
        value = (query.get(key) or [""])[0]
        if not value:
            raise BadRequest(f"Falta {key}.")
        return value

    def _static(self, path: str) -> None:
        name, kind = STATIC[path]
        self._send((WEB_DIR / name).read_bytes(), kind)

    # --- informacion ---------------------------------------------------

    def _version(self) -> None:
        self._json({"eli": __version__, "suite": self.app.suite.info(), "trtllm_rules": inspector.load_rules()["rules_version"]})

    def _system(self) -> None:
        self._json({"eli": __version__, "hardware": self.app.hardware_fn(), "ollama": self._status()["ollama"],
                    "sandbox": sandbox.info(), "suite": self.app.suite.info()})

    def _status(self) -> dict:
        gpu = self.app.gpu
        latest = gpu.latest()
        return {
            "version": __version__,
            "gpu": {"available": gpu.available, "error": gpu.error, "mem_total_mib": latest.mem_total if latest else None,
                    "power_limit_w": latest.power_limit if latest else None},
            "ollama": {"version": self.app.ollama.version(), "url": self.app.ollama.base_url},
            "defaults": DEFAULTS,
            "busy": self.app.chat_lock.locked(),
            "bench_running": bool(self.app.bench.active_run_ids()),
            "emb_running": bool(self.app.emb.active_run_ids()),
        }

    def _dashboard(self) -> None:
        models, error = [], None
        try:
            models = self.app.ollama.models()
        except OllamaError as exc:
            error = str(exc)
        runs = self.app.store.list_runs()
        self._json({
            "status": self._status(), "hardware": self.app.hardware_fn(), "sandbox": sandbox.info(), "ollama_error": error,
            "models": {"count": len(models), "loaded": [m["name"] for m in models if m["loaded"]],
                       "disk_bytes": sum(m.get("disk_bytes") or 0 for m in models)},
            "recent_runs": runs[:5], "runs_total": len(runs), "bench": self.app.bench.status(),
            "suite": {k: v for k, v in self.app.suite.info().items() if k != "categories"},
        })

    def _gpu_stream(self) -> None:
        q = self.app.gpu.subscribe()
        try:
            self._sse_start()
            while True:
                try:
                    sample = q.get(timeout=5)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self._sse(sample.to_dict())
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.app.gpu.unsubscribe(q)

    def _suite(self, q: dict) -> None:
        suite = self.app.suite
        full = (q.get("prompts") or ["0"])[0] == "1"
        tests = [{"id": t.id, "category": t.category, "difficulty": t.difficulty, "language": t.language, "quick": t.quick,
                  "context_sized": t.context_sized, "expected": t.expected, **({"prompt": t.prompt} if full else {})}
                 for t in suite.tests]
        self._json({**suite.info(), "difficulties": list(bank.DIFFICULTIES), "context_sizes": {
            "quick": bank.DEFAULT_CONTEXT_SIZES_QUICK, "full": bank.DEFAULT_CONTEXT_SIZES_FULL}, "tests": tests,
            "categories_names": suite.categories})

    # --- historial y comparacion ---------------------------------------

    def _history(self, rest: str, q: dict) -> None:
        run_id, _, tail = rest.partition("/")
        if tail == "":
            doc = self.app.store.load_run(run_id)
            if doc is None:
                raise NotFound("No existe esa ejecucion.")
            return self._json(doc)
        if tail == "export":
            fmt = (q.get("format") or ["json"])[0]
            if fmt not in ("json", "csv"):
                raise BadRequest("format debe ser json o csv.")
            text = self.app.store.export_json(run_id) if fmt == "json" else self.app.store.export_csv(run_id)
            if text is None:
                raise NotFound("No existe esa ejecucion.")
            kind = "application/json; charset=utf-8" if fmt == "json" else "text/csv; charset=utf-8"
            return self._send(text.encode("utf-8"), kind, headers={
                "Content-Disposition": f'attachment; filename="eli-run-{quote(run_id)}.{fmt}"'})
        raise NotFound()

    def _emb_history(self, rest: str, q: dict) -> None:
        run_id, _, tail = rest.partition("/")
        store = self.app.emb_store
        if tail == "":
            doc = store.load_run(run_id)
            if doc is None:
                raise NotFound("No existe esa ejecucion.")
            return self._json(doc)
        if tail == "export":
            text = store.export_json(run_id)
            if text is None:
                raise NotFound("No existe esa ejecucion.")
            return self._send(text.encode("utf-8"), "application/json; charset=utf-8", headers={
                "Content-Disposition": f'attachment; filename="eli-emb-{quote(run_id)}.json"'})
        raise NotFound()

    def _compare(self, q: dict) -> None:
        def csv(key: str) -> list[str] | None:
            raw = (q.get(key) or [""])[0]
            return [p for p in raw.split(",") if p] or None
        self._json(compare.compare(self.app.store, csv("runs"), csv("models"), _floats(q, "param_min"), _floats(q, "param_max")))

    # --- administracion de modelos ---------------------------------------

    def _unload(self, body: dict) -> None:
        if self.app.chat_lock.locked():
            return self._json({"error": "Hay una generacion o evaluacion en curso."}, 409)
        model = body.get("model")
        if model is not None and not isinstance(model, str):
            raise BadRequest("'model' debe ser texto.")
        names = [model] if model else [m["name"] for m in self.app.ollama.loaded()]
        for name in names:
            self.app.ollama.unload(name)
        self._json({"unloaded": names})

    def _delete(self, body: dict) -> None:
        if self.app.chat_lock.locked():
            return self._json({"error": "Hay una generacion o evaluacion en curso."}, 409)
        result = inventory.delete_model(self.app.ollama, body.get("model"), body.get("confirm"))
        self._json(result)

    # --- chat ----------------------------------------------------------

    def _chat(self, body: dict) -> None:
        request = parse_chat_request(body)
        installed = {m["name"] for m in self.app.ollama.models()}
        if request["model"] not in installed:
            raise BadRequest(f"El modelo '{request['model']}' no esta instalado en Ollama.")
        if not self.app.chat_lock.acquire(blocking=False):
            return self._json({"error": "Ya hay una generacion o evaluacion en curso: ELI mide una a la vez."}, 409)
        try:
            self._sse_start()
            gen = generate(
                self.app.ollama, self.app.gpu, request["model"], request["messages"], request["options"], request["think"],
                on_token=lambda kind, text: self._sse({"type": "token", "kind": kind, "text": text}),
            )
            self._save_chat(request, gen)
            try:
                if gen.status == OK:
                    self._sse({"type": "done", "metrics": gen.metrics})
                elif gen.status != CANCELLED:
                    self._sse({"type": "error", "message": gen.error or "La generacion fallo."})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self.app.chat_lock.release()

    def _save_chat(self, request: dict, gen) -> None:
        record = {
            "kind": "chat", "status": gen.status, "started": datetime.fromtimestamp(gen.t_start).isoformat(timespec="seconds"),
            "model": request["model"], "options": request["options"], "think": request["think"], "messages": request["messages"],
            "response": {"thinking": gen.thinking, "content": gen.content}, "metrics": gen.metrics, "error": gen.error,
            "gpu_samples": [s.to_dict() for s in gen.samples],
        }
        self.app.store.save_chat(record, request["model"], gen.t_start)  # nunca pisa un archivo existente


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.app = app  # type: ignore[attr-defined]
    app.port = server.server_address[1]  # con puerto 0 el sistema elige uno libre
    return server

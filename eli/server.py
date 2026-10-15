"""Servidor local de ELI: pagina estatica + API JSON/SSE. Solo 127.0.0.1."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import metrics
from .gpu import GPUMonitor, idle_power, window_stats
from .ollama import Ollama, OllamaError

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
IDLE_WINDOW_SECONDS = 600.0
DEFAULTS = {"num_ctx": 4096, "temperature": 0.0, "seed": 42, "num_predict": 512}


class BadRequest(Exception):
    pass


class App:
    def __init__(self, ollama: Ollama, gpu: GPUMonitor, runs_dir: Path, port: int):
        self.ollama = ollama
        self.gpu = gpu
        self.runs_dir = runs_dir
        self.port = port
        self.chat_lock = threading.Lock()

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
    clean = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "system"}:
            raise BadRequest("Mensaje con rol invalido.")
        if not isinstance(message.get("content"), str):
            raise BadRequest("Mensaje sin texto.")
        clean.append({"role": message["role"], "content": message["content"]})
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        clean.insert(0, {"role": "system", "content": system.strip()})

    def number(name: str, kind, low, high):
        raw = body.get("options", {}).get(name, DEFAULTS[name])
        try:
            value = kind(raw)
        except (TypeError, ValueError):
            raise BadRequest(f"Valor invalido para {name}.") from None
        if not low <= value <= high:
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


class Handler(BaseHTTPRequestHandler):
    server_version = "ELI"
    protocol_version = "HTTP/1.0"  # una respuesta por conexion: el streaming no necesita chunked

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, *_args) -> None:
        pass

    # --- utilidades ----------------------------------------------------

    def _json(self, data, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

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

    # --- rutas ---------------------------------------------------------

    def do_GET(self) -> None:
        if not self._guard():
            return
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in STATIC:
                return self._static(url.path)
            if url.path == "/api/status":
                return self._json(self._status())
            if url.path == "/api/models":
                return self._json({"models": self.app.ollama.models()})
            if url.path == "/api/model":
                name = (query.get("name") or [""])[0]
                if not name:
                    raise BadRequest("Falta name.")
                return self._json(self.app.ollama.show(name))
            if url.path == "/api/gpu/history":
                seconds = min(600.0, max(1.0, float((query.get("seconds") or ["120"])[0])))
                return self._json({"samples": [s.to_dict() for s in self.app.gpu.history(seconds)]})
            if url.path == "/api/gpu/processes":
                return self._json({"processes": self.app.gpu.processes()})
            if url.path == "/api/gpu/stream":
                return self._gpu_stream()
            self._json({"error": "No existe"}, 404)
        except BadRequest as exc:
            self._json({"error": str(exc)}, 400)
        except ValueError:
            self._json({"error": "Parametro invalido"}, 400)
        except OllamaError as exc:
            self._json({"error": str(exc)}, 502)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        if not self._guard():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/chat":
                return self._chat(self._read_json())
            if path == "/api/unload":
                return self._unload(self._read_json())
            self._json({"error": "No existe"}, 404)
        except BadRequest as exc:
            self._json({"error": str(exc)}, 400)
        except OllamaError as exc:
            self._json({"error": str(exc)}, 502)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _static(self, path: str) -> None:
        name, kind = STATIC[path]
        payload = (WEB_DIR / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _status(self) -> dict:
        gpu = self.app.gpu
        latest = gpu.latest()
        return {
            "gpu": {
                "available": gpu.available,
                "error": gpu.error,
                "mem_total_mib": latest.mem_total if latest else None,
                "power_limit_w": latest.power_limit if latest else None,
            },
            "ollama": {"version": self.app.ollama.version(), "url": self.app.ollama.base_url},
            "defaults": DEFAULTS,
            "busy": self.app.chat_lock.locked(),
        }

    def _gpu_stream(self) -> None:
        import queue

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

    def _unload(self, body: dict) -> None:
        if self.app.chat_lock.locked():
            return self._json({"error": "Hay una generacion en curso."}, 409)
        names = [body["model"]] if isinstance(body.get("model"), str) else [m["name"] for m in self.app.ollama.loaded()]
        for name in names:
            self.app.ollama.unload(name)
        self._json({"unloaded": names})

    def _chat(self, body: dict) -> None:
        request = parse_chat_request(body)
        if not self.app.chat_lock.acquire(blocking=False):
            return self._json({"error": "Ya hay una generacion en curso: ELI mide una a la vez."}, 409)
        stream = None
        try:
            self._sse_start()
            gpu = self.app.gpu
            t_start = time.time()
            baseline = idle_power(gpu.history(IDLE_WINDOW_SECONDS))
            stream = self.app.ollama.chat_stream(
                request["model"], request["messages"], request["options"], request["think"]
            )
            t_first = t_answer = final = None
            thinking: list[str] = []
            content: list[str] = []
            for kind, payload in stream:
                if kind == "done":
                    final = payload
                    break
                now = time.time()
                t_first = t_first or now
                if kind == "content":
                    t_answer = t_answer or now
                (content if kind == "content" else thinking).append(payload)
                self._sse({"type": "token", "kind": kind, "text": payload})
            t_end = time.time()
            if final is None:
                raise OllamaError("Ollama cerro la conexion sin terminar la respuesta.")

            ps_entry = next((m for m in self.app.ollama.loaded() if m.get("name") == request["model"]), None)
            result = metrics.chat_metrics(final, t_start, t_first, t_answer, t_end, request["options"]["num_ctx"], ps_entry)
            margin = gpu.interval_ms / 1000
            samples = gpu.window(t_start - margin, t_end + margin)
            result["gpu"] = {
                "request": window_stats(samples, t_start, t_end, baseline),
                "decode": window_stats(
                    [s for s in samples if t_first and s.ts >= t_first - margin], t_first, t_end, baseline
                )
                if t_first
                else None,
            }
            decode = result["gpu"]["decode"]
            if decode and decode.get("energy_net_wh") is not None and result["gen_tokens"]:
                result["wh_per_token"] = round(decode["energy_net_wh"] / result["gen_tokens"], 6)
            if samples and samples[-1].throttle:
                result["warnings"].append("La GPU estaba frenada: " + ", ".join(samples[-1].throttle) + ".")
            self._save_run(request, "".join(thinking), "".join(content), result, samples, t_start)
            self._sse({"type": "done", "metrics": result})
        except OllamaError as exc:
            try:
                self._sse({"type": "error", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
        except (BrokenPipeError, ConnectionResetError):
            pass  # el usuario pulso Detener: cerrar el generador cancela la generacion
        finally:
            if stream is not None:
                stream.close()
            self.app.chat_lock.release()

    def _save_run(self, request: dict, thinking: str, content: str, result: dict, samples, t_start: float) -> None:
        try:
            self.app.runs_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.fromtimestamp(t_start).strftime("%Y%m%d-%H%M%S")
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", request["model"])[:60]
            record = {
                "kind": "chat",
                "started": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
                "model": request["model"],
                "options": request["options"],
                "think": request["think"],
                "messages": request["messages"],
                "response": {"thinking": thinking, "content": content},
                "metrics": result,
                "gpu_samples": [s.to_dict() for s in samples],
            }
            (self.app.runs_dir / f"{stamp}_{safe}_chat.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=1)
            )
        except OSError:
            pass  # guardar la corrida es un extra; no debe romper la respuesta


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.app = app  # type: ignore[attr-defined]
    app.port = server.server_address[1]  # con puerto 0 el sistema elige uno libre
    return server

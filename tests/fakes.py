"""Ollama y GPU falsos para probar ELI sin modelos ni hardware.

Nada de lo que producen estos falsos debe presentarse como una medicion real de un modelo o de la GPU.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# GPU en reposo al principio (para tener un consumo base medible) y luego con carga.
FAKE_GPU_COMMAND = [
    "python3",
    "-c",
    "import time\n"
    "for _ in range(30):\n"
    "    print('5, 2, 9000, 16380, 30.0, 165.00, 40, 300, 800, P8, 0x0000000000000004', flush=True)\n"
    "    time.sleep(0.05)\n"
    "while True:\n"
    "    print('50, 10, 9000, 16380, 100.0, 165.00, 60, 1800, 8000, P2, 0x0000000000000004', flush=True)\n"
    "    time.sleep(0.05)\n",
]

FINAL = {
    "model": "m1",
    "done": True,
    "done_reason": "stop",
    "total_duration": 2_000_000_000,
    "load_duration": 500_000_000,
    "prompt_eval_count": 20,
    "prompt_eval_duration": 100_000_000,
    "eval_count": 4,
    "eval_duration": 400_000_000,
}

DEFAULT_MODEL = {
    "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "modified_at": "2026-10-01T12:00:00Z", "size": 3_000_000_000,
    "format": "gguf", "family": "fam", "families": ["fam"], "parameter_size": "8B", "quantization_level": "Q4",
    "architecture": "qwen2", "context_length": 131072, "embedding_length": 2560, "parameter_count": 8_000_000_000,
    "capabilities": ["completion", "thinking"],
}


class FakeOllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_):
        pass

    def _send(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        owner = self.server.owner
        if self.path == "/api/tags":
            self._send({"models": [
                {"name": n, "model": n, "modified_at": m["modified_at"], "size": m["size"], "digest": m["digest"],
                 "details": {"format": m["format"], "family": m["family"], "families": m["families"],
                             "parameter_size": m["parameter_size"], "quantization_level": m["quantization_level"]}}
                for n, m in owner.models.items()]})
        elif self.path == "/api/ps":
            if owner.ps_error:
                return self._send({"error": "ps roto"}, 500)
            self._send({"models": [
                {"name": n, "model": n, "size": owner.size, "size_vram": owner.vram, "context_length": owner.loaded_context,
                 "expires_at": "2026-10-01T12:05:00Z"} for n in sorted(owner.loaded) if n in owner.models]})
        elif self.path == "/api/version":
            self._send({"version": "0.0-test"})
        else:
            self._send({"error": "no existe"}, 404)

    def do_DELETE(self):
        owner = self.server.owner
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/delete" and body["model"] in owner.models:
            del owner.models[body["model"]]
            owner.loaded.discard(body["model"])
            owner.deleted.append(body["model"])
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send({"error": f"model '{body.get('model')}' not found"}, 404)

    def do_POST(self):
        owner = self.server.owner
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            m = owner.models.get(body["model"])
            if m is None:
                return self._send({"error": f"model '{body['model']}' not found"}, 404)
            info = {"general.architecture": m["architecture"]}
            if m["architecture"]:
                info[f"{m['architecture']}.context_length"] = m["context_length"]
                info[f"{m['architecture']}.embedding_length"] = m["embedding_length"]
            if m.get("parameter_count"):
                info["general.parameter_count"] = m["parameter_count"]
            if not m["architecture"]:
                info.pop("general.architecture")
            self._send({"details": {"format": m["format"], "family": m["family"], "families": m["families"],
                                    "parameter_size": m["parameter_size"], "quantization_level": m["quantization_level"]},
                        "model_info": info, "capabilities": m["capabilities"], "modified_at": m["modified_at"], "parameters": ""})
        elif self.path == "/api/generate":
            owner.unloaded.append(body["model"])
            if owner.unload_delay:  # Ollama real descarga de forma asincrona
                threading.Timer(owner.unload_delay, owner.loaded.discard, args=(body["model"],)).start()
            else:
                owner.loaded.discard(body["model"])
            self._send({"done": True})
        elif self.path == "/api/chat":
            owner.chat_bodies.append(body)
            model = owner.models.get(body["model"])
            if model is None:
                return self._send({"error": f"model '{body['model']}' not found"}, 404)
            if body.get("think") is not None and (body["model"] in owner.reject_think or "thinking" not in model["capabilities"]):
                return self._send({"error": f"\"{body['model']}\" does not support thinking"}, 400)
            owner.loaded.add(body["model"])
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                if owner.chat_error:
                    self.wfile.write((json.dumps({"error": owner.chat_error}) + "\n").encode())
                    return
                tokens = owner.responder(body) if owner.responder else owner.tokens
                if isinstance(tokens, str):
                    tokens = [tokens[i:i + 4] for i in range(0, len(tokens), 4)] or [""]
                for text in tokens:
                    time.sleep(owner.chunk_delay)
                    self.wfile.write((json.dumps({"message": {"role": "assistant", "content": text}, "done": False}) + "\n").encode())
                    self.wfile.flush()
                final = {**FINAL, "model": body["model"], "eval_count": len(tokens), "message": {"role": "assistant", "content": ""}}
                if owner.done_reason:
                    final["done_reason"] = owner.done_reason
                final["load_duration"] = owner.load_duration
                self.wfile.write((json.dumps(final) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                owner.client_disconnected.set()


class FakeOllama:
    def __init__(self):
        self.models = {"m1": dict(DEFAULT_MODEL)}
        self.loaded = {"m1"}
        self.tokens = ["ho", "la", " ", "mundo"]
        self.responder = None  # callable(body) -> lista de trozos o texto
        self.chunk_delay = 0.0
        self.chat_error = None
        self.ps_error = False
        self.unload_delay = 0.0
        self.reject_think: set[str] = set()  # modelos que anuncian 'thinking' pero rechazan el parametro
        self.done_reason = None
        self.load_duration = FINAL["load_duration"]
        self.size = 3_000_000_000
        self.vram = 3_000_000_000
        self.loaded_context = 4096
        self.unloaded: list[str] = []
        self.deleted: list[str] = []
        self.chat_bodies: list[dict] = []
        self.client_disconnected = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        self.server.owner = self
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def add_model(self, name: str, **overrides) -> dict:
        info = {**DEFAULT_MODEL, "digest": f"sha256:{abs(hash(name)):032x}", **overrides}
        self.models[name] = info
        return info

    def close(self):
        self.server.shutdown()
        self.server.server_close()

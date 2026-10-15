"""Ollama y GPU falsos para probar ELI sin modelos ni hardware."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAKE_GPU_COMMAND = [
    "python3",
    "-c",
    "import time\n"
    "for _ in range(30):\n"  # primero GPU en reposo, para que haya un consumo base medible
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
            self._send({"models": [{"name": "m1", "size": 3_000_000_000,
                                    "details": {"family": "fam", "parameter_size": "8B", "quantization_level": "Q4"}}]})
        elif self.path == "/api/ps":
            self._send({"models": [{"name": "m1", "size": owner.size, "size_vram": owner.vram,
                                    "context_length": owner.loaded_context}]})
        elif self.path == "/api/version":
            self._send({"version": "0.0-test"})
        else:
            self._send({"error": "no existe"}, 404)

    def do_POST(self):
        owner = self.server.owner
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            if body["model"] == "missing":
                return self._send({"error": "model 'missing' not found"}, 404)
            self._send({"details": {"family": "fam", "parameter_size": "8B", "quantization_level": "Q4"},
                        "model_info": {"general.architecture": "fam", "fam.context_length": 131072,
                                       "fam.embedding_length": 2560},
                        "capabilities": ["completion", "thinking"], "parameters": ""})
        elif self.path == "/api/generate":
            owner.unloaded.append(body["model"])
            self._send({"done": True})
        elif self.path == "/api/chat":
            owner.chat_bodies.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            try:
                if owner.chat_error:
                    self.wfile.write((json.dumps({"error": owner.chat_error}) + "\n").encode())
                    return
                for text in owner.tokens:
                    time.sleep(owner.chunk_delay)
                    self.wfile.write((json.dumps({"message": {"role": "assistant", "content": text}, "done": False}) + "\n").encode())
                    self.wfile.flush()
                self.wfile.write((json.dumps({**FINAL, "message": {"role": "assistant", "content": ""}}) + "\n").encode())
            except (BrokenPipeError, ConnectionResetError):
                owner.client_disconnected.set()


class FakeOllama:
    def __init__(self):
        self.tokens = ["ho", "la", " ", "mundo"]
        self.chunk_delay = 0.0
        self.chat_error = None
        self.size = 3_000_000_000
        self.vram = 3_000_000_000
        self.loaded_context = 4096
        self.unloaded: list[str] = []
        self.chat_bodies: list[dict] = []
        self.client_disconnected = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        self.server.owner = self
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

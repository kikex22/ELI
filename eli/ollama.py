"""Cliente minimo de Ollama (solo libreria estandar)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaError(Exception):
    pass


class Ollama:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- transporte ----------------------------------------------------

    def _open(self, path: str, body: dict | None = None, timeout: float | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base_url + path, data=data, headers={"Content-Type": "application/json"} if data else {}
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")
            try:
                message = json.loads(message).get("error", message)
            except ValueError:
                pass
            raise OllamaError(f"Ollama respondio {exc.code}: {message}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise OllamaError(f"No se pudo conectar con Ollama en {self.base_url}: {getattr(exc, 'reason', exc)}") from exc

    def _json(self, path: str, body: dict | None = None) -> dict:
        with self._open(path, body) as resp:
            return json.load(resp)

    # --- consultas -----------------------------------------------------

    def version(self) -> str | None:
        try:
            return self._json("/api/version").get("version")
        except OllamaError:
            return None

    def loaded(self) -> list[dict]:
        return self._json("/api/ps").get("models", [])

    def models(self) -> list[dict]:
        tags = self._json("/api/tags").get("models", [])
        loaded = {m["name"]: m for m in self.loaded()}
        rows = []
        for tag in tags:
            details = tag.get("details", {})
            ps = loaded.get(tag["name"])
            rows.append(
                {
                    "name": tag["name"],
                    "disk_bytes": tag.get("size"),
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                    "loaded": ps is not None,
                    "loaded_bytes": ps.get("size") if ps else None,
                    "vram_bytes": ps.get("size_vram") if ps else None,
                    "loaded_context": ps.get("context_length") if ps else None,
                }
            )
        return sorted(rows, key=lambda r: (not r["loaded"], r["name"]))

    def show(self, name: str) -> dict:
        info = self._json("/api/show", {"model": name})
        details = info.get("details", {})
        model_info = info.get("model_info", {})
        arch = model_info.get("general.architecture", details.get("family"))
        return {
            "name": name,
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "architecture": arch,
            "context_length": model_info.get(f"{arch}.context_length"),
            "embedding_length": model_info.get(f"{arch}.embedding_length"),
            "capabilities": info.get("capabilities", []),
            "parameters": info.get("parameters", ""),
        }

    def unload(self, name: str) -> None:
        with self._open("/api/generate", {"model": name, "keep_alive": 0}) as resp:
            resp.read()

    # --- generacion ----------------------------------------------------

    def chat_stream(
        self, model: str, messages: list[dict], options: dict, think: bool | None = None
    ) -> Iterator[tuple[str, object]]:
        """Produce ('thinking'|'content', texto) y al final ('done', dict con los tiempos).

        Cerrar el generador cierra la conexion y Ollama cancela la generacion.
        """
        body: dict = {"model": model, "messages": messages, "stream": True, "options": options}
        if think is not None:
            body["think"] = think
        resp = self._open("/api/chat", body, timeout=600)
        try:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                if "error" in obj:
                    raise OllamaError(str(obj["error"]))
                message = obj.get("message") or {}
                if message.get("thinking"):
                    yield "thinking", message["thinking"]
                if message.get("content"):
                    yield "content", message["content"]
                if obj.get("done"):
                    yield "done", obj
                    return
        finally:
            resp.close()

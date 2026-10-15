"""Cliente minimo de Ollama (solo libreria estandar)."""
from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from typing import Iterator

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaError(Exception):
    pass


class ChatStream:
    """Respuesta en streaming de /api/chat.

    `events()` produce ('thinking'|'content', texto) y al final ('done', dict con
    los tiempos). `abort()` se puede llamar desde otro hilo: corta el socket, lo
    que desbloquea la lectura y hace que Ollama cancele la generacion.
    """

    def __init__(self, resp):
        self._resp = resp
        self._aborted = False

    def events(self) -> Iterator[tuple[str, object]]:
        try:
            for raw in self._resp:
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
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if self._aborted:
                return
            raise OllamaError(f"La conexion con Ollama se interrumpio: {exc}") from exc
        finally:
            self.close()

    def abort(self) -> None:
        self._aborted = True
        sock = getattr(getattr(getattr(self._resp, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.close()

    def close(self) -> None:
        try:
            self._resp.close()
        except Exception:  # noqa: BLE001 - cerrar nunca debe romper al llamador
            pass


class Ollama:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- transporte ----------------------------------------------------

    def _open(self, path: str, body: dict | None = None, timeout: float | None = None, method: str | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method=method,
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

    def _json(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        try:
            with self._open(path, body, timeout=timeout) as resp:
                return json.load(resp)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            if isinstance(exc, OllamaError):
                raise
            raise OllamaError(f"Respuesta no valida de Ollama en {path}: {exc}") from exc

    # --- consultas -----------------------------------------------------

    def version(self) -> str | None:
        try:
            return self._json("/api/version").get("version")
        except OllamaError:
            return None

    def tags(self) -> list[dict]:
        return self._json("/api/tags").get("models", [])

    def loaded(self) -> list[dict]:
        return self._json("/api/ps").get("models", [])

    def models(self) -> list[dict]:
        """Modelos instalados con su estado de carga. Si /api/ps falla se informa sin estado de carga."""
        tags = self.tags()
        try:
            loaded = {m["name"]: m for m in self.loaded()}
        except OllamaError:
            loaded = {}
        rows = []
        for tag in tags:
            details = tag.get("details") or {}
            ps = loaded.get(tag.get("name"))
            rows.append(
                {
                    "name": tag.get("name"),
                    "digest": tag.get("digest"),
                    "modified_at": tag.get("modified_at"),
                    "disk_bytes": tag.get("size"),
                    "format": details.get("format"),
                    "family": details.get("family"),
                    "families": details.get("families"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                    "loaded": ps is not None,
                    "loaded_bytes": ps.get("size") if ps else None,
                    "vram_bytes": ps.get("size_vram") if ps else None,
                    "loaded_context": ps.get("context_length") if ps else None,
                    "expires_at": ps.get("expires_at") if ps else None,
                }
            )
        return sorted((r for r in rows if r["name"]), key=lambda r: (not r["loaded"], r["name"]))

    def show(self, name: str) -> dict:
        info = self._json("/api/show", {"model": name})
        details = info.get("details") or {}
        model_info = info.get("model_info") or {}
        arch = model_info.get("general.architecture")
        return {
            "name": name,
            "format": details.get("format"),
            "family": details.get("family"),
            "families": details.get("families"),
            "parameter_size": details.get("parameter_size"),
            "parameter_count": model_info.get("general.parameter_count"),
            "quantization": details.get("quantization_level"),
            "architecture": arch,
            "context_length": model_info.get(f"{arch}.context_length") if arch else None,
            "embedding_length": model_info.get(f"{arch}.embedding_length") if arch else None,
            "capabilities": info.get("capabilities") or [],
            "modified_at": info.get("modified_at"),
            "parameters": info.get("parameters", ""),
        }

    def embed(self, model: str, inputs: list[str], timeout: float = 300) -> dict:
        """Embeddings de una lista de textos. `truncate=False`: un texto mas largo que el contexto es un error
        explicito, no un recorte silencioso que falsearia la calidad medida."""
        data = self._json("/api/embed", {"model": model, "input": inputs, "truncate": False}, timeout=timeout)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(inputs) or not all(isinstance(v, list) and v for v in vectors):
            raise OllamaError("Ollama no devolvio un embedding por cada texto enviado.")
        ns = lambda key: (data[key] / 1e9) if isinstance(data.get(key), (int, float)) else None  # noqa: E731
        return {"embeddings": vectors, "load_s": ns("load_duration"), "total_s": ns("total_duration"),
                "prompt_tokens": data.get("prompt_eval_count")}

    def unload(self, name: str) -> None:
        with self._open("/api/generate", {"model": name, "keep_alive": 0}) as resp:
            resp.read()

    def delete(self, name: str) -> None:
        """Elimina el modelo del disco (API oficial). No es lo mismo que descargarlo de la VRAM."""
        with self._open("/api/delete", {"model": name}, method="DELETE") as resp:
            resp.read()

    # --- generacion ----------------------------------------------------

    def open_chat(self, model: str, messages: list[dict], options: dict, think: bool | None = None) -> ChatStream:
        body: dict = {"model": model, "messages": messages, "stream": True, "options": options}
        if think is not None:
            body["think"] = think
        return ChatStream(self._open("/api/chat", body, timeout=600))

    def chat_stream(
        self, model: str, messages: list[dict], options: dict, think: bool | None = None
    ) -> Iterator[tuple[str, object]]:
        """Version generador de open_chat; cerrar el generador cierra la conexion."""
        stream = self.open_chat(model, messages, options, think)
        try:
            yield from stream.events()
        finally:
            stream.close()

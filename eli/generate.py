"""Una generacion medida: streaming, tiempos, GPU y energia.

La usan el chat y el benchmark, asi miden exactamente igual. La lectura del
stream ocurre en un hilo para poder cancelar o aplicar un tiempo maximo aunque
el modelo no este emitiendo tokens (por ejemplo mientras carga).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import metrics as metrics_mod
from .gpu import GPUMonitor, idle_power, window_stats
from .ollama import Ollama, OllamaError

IDLE_WINDOW_SECONDS = 600.0

OK, CANCELLED, TIMEOUT, ERROR = "ok", "cancelled", "timeout", "error"


@dataclass
class Generation:
    status: str
    model: str
    content: str
    thinking: str
    final: dict | None
    metrics: dict
    error: str | None
    t_start: float
    t_first: float | None
    t_end: float
    samples: list = field(default_factory=list)  # lecturas de GPU de la ventana, para guardar

    @property
    def truncated(self) -> bool:
        return bool(self.final and self.final.get("done_reason") == "length")


def gpu_summary(gpu: GPUMonitor | None, t_start: float, t_first: float | None, t_end: float, baseline: float | None):
    """Estadisticas de GPU de toda la peticion y solo de la fase de generacion."""
    if gpu is None:
        return [], {"request": None, "decode": None}
    margin = gpu.interval_ms / 1000
    samples = gpu.window(t_start - margin, t_end + margin)
    request = window_stats(samples, t_start, t_end, baseline)
    decode = None
    if t_first:
        decode = window_stats([s for s in samples if s.ts >= t_first - margin], t_first, t_end, baseline)
    return samples, {"request": request, "decode": decode}


def generate(
    ollama: Ollama,
    gpu: GPUMonitor | None,
    model: str,
    messages: list[dict],
    options: dict,
    think: bool | None = None,
    timeout_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_token: Callable[[str, str], None] | None = None,
) -> Generation:
    """Ejecuta una peticion independiente y devuelve siempre un Generation.

    - `should_stop()` verdadero cancela; `timeout_s` corta con estado 'timeout'.
    - Si `on_token` lanza BrokenPipeError/ConnectionResetError (el cliente se fue)
      se cancela la generacion y se devuelve lo parcial.
    - Los errores de Ollama no se propagan: quedan en `status='error'`.
    """
    t_start = time.time()
    baseline = idle_power(gpu.history(IDLE_WINDOW_SECONDS)) if gpu else None
    status, error = OK, None
    content: list[str] = []
    thinking: list[str] = []
    t_first = t_answer = None
    final = None
    stream = None
    thread = None

    try:
        stream = ollama.open_chat(model, messages, options, think)
    except OllamaError as exc:
        status, error = ERROR, str(exc)

    if stream is not None:
        q: queue.Queue = queue.Queue()

        def pump() -> None:
            try:
                for event in stream.events():
                    q.put(("event", event))
                q.put(("end", None))
            except BaseException as exc:  # noqa: BLE001 - se reenvia al hilo principal
                q.put(("error", exc))

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            while True:
                if should_stop and should_stop():
                    status = CANCELLED
                    break
                if timeout_s is not None and time.time() - t_start > timeout_s:
                    status, error = TIMEOUT, f"Tiempo maximo de {timeout_s:g} s superado."
                    break
                try:
                    kind, payload = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if kind == "end":
                    if final is None:
                        status, error = ERROR, "Ollama cerro la conexion sin terminar la respuesta."
                    break
                if kind == "error":
                    status, error = ERROR, str(payload)
                    break
                label, text = payload
                if label == "done":
                    final = text
                    break
                now = time.time()
                t_first = t_first or now
                if label == "content":
                    t_answer = t_answer or now
                (content if label == "content" else thinking).append(text)
                if on_token:
                    try:
                        on_token(label, text)
                    except (BrokenPipeError, ConnectionResetError):
                        status = CANCELLED
                        break
        except BaseException:
            stream.abort()
            raise
        finally:
            if status != OK or final is None:
                stream.abort()
            thread.join(timeout=2)

    t_end = time.time()
    num_ctx = options.get("num_ctx", 0)
    if final is not None and status == OK:
        ps_entry = None
        try:
            ps_entry = next((m for m in ollama.loaded() if m.get("name") == model), None)
        except OllamaError:
            pass
        result = metrics_mod.chat_metrics(final, t_start, t_first, t_answer, t_end, num_ctx, ps_entry)
    else:
        result = metrics_mod.partial_metrics(t_start, t_first, t_answer, t_end, num_ctx)

    samples, gpu_stats = gpu_summary(gpu, t_start, t_first, t_end, baseline)
    result["gpu"] = gpu_stats
    decode = gpu_stats["decode"]
    if status == OK and decode and decode.get("energy_net_wh") is not None and result.get("gen_tokens"):
        result["wh_per_token"] = round(decode["energy_net_wh"] / result["gen_tokens"], 6)
    if samples and samples[-1].throttle:
        result["warnings"].append("La GPU estaba frenada: " + ", ".join(samples[-1].throttle) + ".")

    return Generation(
        status=status,
        model=model,
        content="".join(content),
        thinking="".join(thinking),
        final=final,
        metrics=result,
        error=error,
        t_start=t_start,
        t_first=t_first,
        t_end=t_end,
        samples=samples,
    )

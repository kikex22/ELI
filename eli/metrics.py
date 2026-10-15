"""Metricas de una respuesta: tiempos de Ollama + medidas del cliente + GPU."""
from __future__ import annotations

NS = 1e9


def _rate(count: int | None, duration_ns: int | None) -> float | None:
    if not count or not duration_ns:
        return None
    return round(count / (duration_ns / NS), 2)


def _seconds(duration_ns: int | None) -> float | None:
    return None if duration_ns is None else round(duration_ns / NS, 3)


def chat_metrics(
    final: dict,
    t_start: float,
    t_first: float | None,
    t_answer: float | None,
    t_end: float,
    num_ctx: int,
    ps_entry: dict | None,
) -> dict:
    """`final` es el ultimo objeto de /api/chat; las t_* son segundos epoch del servidor.

    Ojo: `prompt_tokens` es lo que Ollama evaluo en esta peticion. Si reutilizo
    la cache de una conversacion anterior, es menor que el contexto real.
    """
    prompt_tokens = final.get("prompt_eval_count")
    gen_tokens = final.get("eval_count")
    metrics = {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "decode_tok_s": _rate(gen_tokens, final.get("eval_duration")),
        "prefill_tok_s": _rate(prompt_tokens, final.get("prompt_eval_duration")),
        "load_s": _seconds(final.get("load_duration")),
        "prefill_s": _seconds(final.get("prompt_eval_duration")),
        "decode_s": _seconds(final.get("eval_duration")),
        "total_s": _seconds(final.get("total_duration")),
        "ttft_s": round(t_first - t_start, 3) if t_first else None,
        "time_to_answer_s": round(t_answer - t_start, 3) if t_answer else None,
        "wall_s": round(t_end - t_start, 3),
        "done_reason": final.get("done_reason"),
        "num_ctx": num_ctx,
        "warnings": [],
    }
    if prompt_tokens is not None and gen_tokens is not None and prompt_tokens + gen_tokens >= num_ctx:
        metrics["warnings"].append(
            f"El contexto ({num_ctx}) se lleno: Ollama pudo recortar el inicio de la conversacion."
        )
    if final.get("done_reason") == "length":
        metrics["warnings"].append("La respuesta se corto por el limite de tokens maximos.")
    if ps_entry:
        size, vram = ps_entry.get("size"), ps_entry.get("size_vram")
        metrics["model_loaded_bytes"] = size
        metrics["model_vram_bytes"] = vram
        metrics["loaded_context"] = ps_entry.get("context_length")
        if size:
            metrics["gpu_fraction"] = round(vram / size, 3)
            if vram < size:
                metrics["warnings"].append(
                    f"El modelo no cabe entero en la GPU: {vram / size:.0%} en GPU, el resto en CPU. Se esperan velocidades mucho menores."
                )
        if ps_entry.get("context_length") and ps_entry["context_length"] != num_ctx:
            metrics["warnings"].append(
                f"Ollama tiene cargado un contexto de {ps_entry['context_length']}, no los {num_ctx} pedidos."
            )
    return metrics

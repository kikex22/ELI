"""Model Inspector: metadatos reales de Ollama, sin inferir lo que no se informa.

Se distinguen tres propiedades que a menudo se confunden:
  formato        como se guardan los pesos (GGUF, safetensors...)
  precision      FP16/BF16 o una cuantizacion (Q4_K_M, Q8_0...)
  arquitectura   la familia de red (qwen2, gemma4, llama...)
Lo que Ollama no entrega queda como desconocido (None).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RULES_PATH = Path(__file__).resolve().parent / "data" / "trtllm_rules.json"

_rules_cache: dict | None = None


def load_rules(path: Path = RULES_PATH) -> dict:
    global _rules_cache
    if path == RULES_PATH and _rules_cache is not None:
        return _rules_cache
    rules = json.loads(path.read_text(encoding="utf-8"))
    if path == RULES_PATH:
        _rules_cache = rules
    return rules


def parse_params_b(parameter_size: str | None, parameter_count: int | float | None = None) -> float | None:
    """Parametros en miles de millones. '8.0B' -> 8.0, '334M' -> 0.334; si falta, usa el conteo exacto."""
    if isinstance(parameter_size, str):
        m = re.fullmatch(r"\s*([\d.]+)\s*([KMBT])\s*", parameter_size.upper())
        if m:
            try:
                return round(float(m.group(1)) * {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}[m.group(2)], 4)
            except ValueError:
                pass
    if isinstance(parameter_count, (int, float)) and parameter_count > 0:
        return round(parameter_count / 1e9, 4)
    return None


def classify_quantization(value: str | None, rules: dict | None = None) -> dict:
    """{'value', 'kind'}: float, gguf-quant, other o unknown. Nunca inventa un valor."""
    rules = rules or load_rules()
    if not value:
        return {"value": None, "kind": "unknown"}
    upper = value.upper()
    if upper in {p.upper() for p in rules["precisions"]["float"]}:
        return {"value": value, "kind": "float"}
    if re.match(rules["precisions"]["gguf_quant_pattern"], upper):
        return {"value": value, "kind": "gguf-quant"}
    return {"value": value, "kind": "other"}


def size_bucket(params_b: float | None) -> str | None:
    if params_b is None:
        return None
    for limit, label in ((2, "<2B"), (5, "2-5B"), (10, "5-10B"), (20, "10-20B")):
        if params_b < limit:
            return label
    return "20B+"


def inspect_model(show: dict | None, tag: dict | None = None, ps: dict | None = None, observed: dict | None = None,
                  hardware: dict | None = None, rules: dict | None = None) -> dict:
    """Une /api/show, /api/tags y /api/ps (cualquiera puede faltar) y anade el analisis TRT-LLM."""
    rules = rules or load_rules()
    show, tag, ps = show or {}, tag or {}, ps or {}
    name = show.get("name") or tag.get("name")
    fmt = show.get("format") or tag.get("format")
    quant = classify_quantization(show.get("quantization") or tag.get("quantization"), rules)
    parameter_size = show.get("parameter_size") or tag.get("parameter_size")
    params_b = parse_params_b(parameter_size, show.get("parameter_count"))
    info = {
        "name": name,
        "digest": tag.get("digest"),
        "modified_at": show.get("modified_at") or tag.get("modified_at"),
        "disk_bytes": tag.get("disk_bytes"),
        "format": fmt.lower() if isinstance(fmt, str) else None,
        "quantization": quant,
        "family": show.get("family") or tag.get("family"),
        "families": show.get("families") or tag.get("families"),
        "architecture": show.get("architecture"),
        "parameter_size": parameter_size,
        "parameters_b": params_b,
        "size_bucket": size_bucket(params_b),
        "context_length": show.get("context_length"),
        "capabilities": show.get("capabilities") or [],
        "loaded": (
            {"bytes": ps.get("size") or tag.get("loaded_bytes"), "vram_bytes": ps.get("size_vram") or tag.get("vram_bytes"),
             "context": ps.get("context_length") or tag.get("loaded_context"), "expires_at": ps.get("expires_at") or tag.get("expires_at")}
            if (ps or tag.get("loaded")) else None
        ),
        "observed": observed,
    }
    info["unknown"] = sorted(k for k in ("format", "architecture", "parameter_size", "context_length", "digest", "modified_at", "disk_bytes")
                             if info.get(k) in (None, ""))
    if quant["value"] is None:
        info["unknown"].append("quantization")
    info["trtllm"] = trtllm_compatibility(info, hardware or {}, rules)
    return info


def trtllm_compatibility(info: dict, hardware: dict, rules: dict | None = None) -> dict:
    """Analisis informativo en cinco dimensiones. Nunca afirma 'compatible'; cada hallazgo cita su regla."""
    rules = rules or load_rules()
    arch, fmt = info.get("architecture"), info.get("format")
    quant = info["quantization"]
    findings: list[dict] = []

    def add(rule_id: str, dimension: str, status: str, text: str, evidence: str | None = None) -> None:
        findings.append({"rule": rule_id, "dimension": dimension, "status": status, "text": text, "evidence": evidence})

    # 1) arquitectura
    listed = rules["architectures_listed"]
    if arch is None:
        arch_status = "no_determinada"
        add(listed["rule_id"], "arquitectura", arch_status, "Ollama no informo la arquitectura del modelo; no se puede evaluar.")
    elif arch in rules["architectures_pending"]:
        arch_status = "no_determinada"
        add(listed["rule_id"], "arquitectura", arch_status, f"La arquitectura '{arch}' no figura entre las listadas.", rules["architectures_pending"][arch])
    elif arch in listed["ids"]:
        arch_status = "potencialmente_compatible"
        add(listed["rule_id"], "arquitectura", arch_status,
            f"La arquitectura '{arch}' figura en la lista informativa de ELI. Eso no garantiza soporte para esta variante ni para una version concreta.", listed["source"])
    else:
        arch_status = "no_determinada"
        add(listed["rule_id"], "arquitectura", arch_status,
            f"La arquitectura '{arch}' no figura en la lista de ELI; puede estar soportada en versiones mas nuevas. Hay que verificarlo.", listed["source"])

    # 2) formato / 3) checkpoint original
    src = rules["source_formats"]
    if fmt is None:
        conversion, checkpoint = "no_determinada", "no_determinada"
        add(src["rule_id"], "formato", conversion, "Ollama no informo el formato de los pesos.")
    elif fmt in src["unsupported_as_source"]:
        conversion, checkpoint = "requiere_conversion", "necesario"
        add(src["rule_id"], "formato", conversion,
            f"Los pesos estan en formato {fmt.upper()}, que TensorRT-LLM no carga directamente: habria que convertir o compilar desde otro formato.", src["evidence"])
        add(src["rule_id"], "checkpoint_original", checkpoint,
            "Hay que obtener el checkpoint original (por ejemplo, safetensors de Hugging Face) del mismo modelo; no se puede reconstruir desde el GGUF.", src["evidence"])
    else:
        conversion, checkpoint = "no_determinada", "no_determinada"
        add(src["rule_id"], "formato", conversion,
            f"El formato '{fmt}' no esta en las reglas de ELI; no se puede afirmar si requiere conversion.")

    # 4) restricciones de precision, cuantizacion, hardware y memoria
    restrictions: list[str] = []
    qr = rules["precisions"]["rule_id"]
    if quant["kind"] == "gguf-quant":
        restrictions.append("cuantizacion_gguf")
        add(qr, "restricciones", "restringida",
            f"La cuantizacion {quant['value']} es propia de GGUF (llama.cpp/Ollama). En TensorRT-LLM habria que aplicar otra cuantizacion (FP8, INT8, INT4...) al checkpoint original.")
    elif quant["kind"] == "float":
        add(qr, "restricciones", "informativo", f"La precision {quant['value']} es nativa (sin cuantizar); no impone restriccion de cuantizacion.")
    elif quant["kind"] == "unknown":
        add(qr, "restricciones", "no_determinada", "Ollama no informo la precision o cuantizacion.")
    else:
        add(qr, "restricciones", "no_determinada", f"La precision '{quant['value']}' no esta clasificada en las reglas de ELI.")

    cap = (hardware.get("gpu") or {}).get("compute_cap")
    fp8 = rules["hardware_requirements"]["FP8"]
    if cap is None:
        add(fp8["rule_id"], "restricciones", "no_determinada", "No se pudo leer la capacidad de calculo de la GPU; no se evalua FP8.")
    elif cap >= fp8["min_compute_cap"]:
        add(fp8["rule_id"], "restricciones", "informativo", f"La GPU (SM {cap:g}) cumple el requisito de hardware para FP8 (SM {fp8['min_compute_cap']:g}+). Es un requisito necesario, no una garantia de soporte del modelo.", fp8["note"])
    else:
        restrictions.append("fp8_no_soportado_por_hardware")
        add(fp8["rule_id"], "restricciones", "restringida", f"La GPU (SM {cap:g}) no cumple el requisito de FP8 (SM {fp8['min_compute_cap']:g}+).", fp8["note"])

    params_b, vram_mib = info.get("parameters_b"), (hardware.get("gpu") or {}).get("vram_total_mib")
    memory = None
    if params_b and vram_mib:
        vram_gib = vram_mib / 1024
        bpp = rules["weight_bytes_per_param"]
        sizes = {label: round(params_b * 1e9 * b / 2**30, 1) for label, b in bpp.items()}
        fits = {label: size <= vram_gib * rules["memory_fit_ratio"] for label, size in sizes.items()}
        memory = {"weights_gib_estimate": sizes, "fits": fits, "vram_gib": round(vram_gib, 1)}
        if not fits["BF16/FP16"]:
            restrictions.append("no_cabe_en_bf16")
        add(rules["memory_rule_id"], "restricciones", "restringida" if not fits["BF16/FP16"] else "informativo",
            f"Estimacion aritmetica de solo los pesos ({params_b:g}B parametros): BF16/FP16 ~{sizes['BF16/FP16']} GiB, FP8/INT8 ~{sizes['FP8/INT8']} GiB, INT4 ~{sizes['INT4']} GiB, "
            f"frente a {vram_gib:.1f} GiB de VRAM. No incluye cache KV ni activaciones.")
    else:
        add(rules["memory_rule_id"], "restricciones", "no_determinada", "Faltan los parametros o la VRAM para estimar la memoria.")

    # evidencia local
    local = []
    for ev in rules["local_evidence"]:
        if arch == ev["architecture"] and quant["value"] and quant["value"].upper() == ev["precision"].upper():
            local.append({"id": ev["id"], "summary": ev["summary"], "source": ev["source"]})
    if local:
        add("R-EVID-1", "evidencia_local", "validado_localmente",
            "Existe una prueba local con la misma arquitectura y precision: " + local[0]["summary"], local[0]["source"])

    if arch_status == "no_determinada":
        verdict = "no_determinada"
    elif checkpoint == "necesario":
        verdict = "potencialmente_compatible_requiere_checkpoint_original"
    else:
        verdict = "potencialmente_compatible"
    return {
        "rules_version": rules["rules_version"],
        "verdict": verdict,
        "dimensions": {"arquitectura": arch_status, "formato": conversion, "checkpoint_original": checkpoint,
                       "restricciones": restrictions or "ninguna_detectada", "memoria": memory},
        "findings": findings,
        "local_evidence": local,
        "notice": "Analisis informativo basado en reglas versionadas. No instala ni convierte nada, y no garantiza que el modelo funcione.",
    }


def snapshot(info: dict) -> dict:
    """Subconjunto estable de los metadatos para guardarlo en el historial (sirve aunque el modelo se borre)."""
    return {k: info.get(k) for k in ("name", "digest", "modified_at", "disk_bytes", "format", "family", "families", "architecture",
                                     "parameter_size", "parameters_b", "context_length", "capabilities")} | {
        "quantization": (info.get("quantization") or {}).get("value"),
        "quantization_kind": (info.get("quantization") or {}).get("kind"),
    }

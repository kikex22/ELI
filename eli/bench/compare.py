"""Comparacion de modelos a partir de ejecuciones guardadas.

Reglas:
- Solo se comparan las pruebas y condiciones que TODOS los modelos compararon (misma prueba, mismo contexto,
  mismos parametros). Si hay repeticiones distintas se usa el minimo comun.
- La calidad cuenta aciertos sobre pruebas calificadas; las no calificadas se muestran aparte, nunca como fallo.
- No hay un indicador unico: se muestran calidad, velocidad, VRAM, tamano y energia por separado, con sus
  frentes de Pareto. No se recomienda nada sin evaluaciones validas suficientes.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from .. import inspector
from .store import Store

OUTCOMES = ("passed", "failed", "ungraded", "error", "timeout", "cancelled")
MIN_GRADED_CATEGORY = 3  # pruebas calificadas minimas por categoria para clasificar
MIN_GRADED_OVERALL = 8
MIN_SPEED_SAMPLES = 3
MIN_SPEED_TOKENS = 16  # con menos tokens generados, tok/s es demasiado ruidoso
MIN_ENERGY_SAMPLES = 5
EFFICIENCY_MIN_RELATIVE_QUALITY = 0.6  # solo compiten por eficiencia los modelos con >= 60 % de la calidad del mejor
SIZE_BUCKETS = [("<2B", 0, 2), ("2-5B", 2, 5), ("5-10B", 5, 10), ("10-20B", 10, 20), ("20B+", 20, 10**6)]

THRESHOLDS = {"min_graded_category": MIN_GRADED_CATEGORY, "min_graded_overall": MIN_GRADED_OVERALL,
              "min_speed_samples": MIN_SPEED_SAMPLES, "min_speed_tokens": MIN_SPEED_TOKENS,
              "min_energy_samples": MIN_ENERGY_SAMPLES, "efficiency_min_relative_quality": EFFICIENCY_MIN_RELATIVE_QUALITY}


def cond_key(r: dict) -> tuple:
    """Condiciones de una prueba. El razonamiento que cuenta es el SOLICITADO: un modelo que no lo admite no recibe el
    parametro, pero eso no hace que su prueba sea distinta. (Registros antiguos sin 'think_requested' usan el efectivo.)"""
    o = r.get("options") or {}
    temp = o.get("temperature") or 0
    think = r["think_requested"] if "think_requested" in r else o.get("think")
    return (r.get("instance_id"), o.get("num_ctx"), temp, think, o.get("num_predict"), o.get("seed") if temp > 0 else None)


def tally(results: list[dict]) -> dict:
    counts = {k: 0 for k in OUTCOMES}
    for r in results:
        counts[r.get("outcome") if r.get("outcome") in counts else "ungraded"] += 1
    graded = counts["passed"] + counts["failed"]
    return {**counts, "total": len(results), "graded": graded, "score": round(counts["passed"] / graded, 4) if graded else None}


def _median(values: list[float], digits: int = 3) -> float | None:
    return round(statistics.median(values), digits) if values else None


def _cv(values: list[float]) -> float | None:
    """Coeficiente de variacion (desviacion / media): variabilidad entre repeticiones."""
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return round(statistics.stdev(values) / mean, 4) if mean else None


def performance(results: list[dict]) -> dict:
    """Rendimiento con el modelo caliente. Las respuestas en frio (cargando el modelo) no cuentan."""
    hot = [r for r in results if r.get("gen_status") == "ok" and not r.get("cold")]
    m = lambda r: r.get("metrics") or {}
    speed = [m(r)["decode_tok_s"] for r in hot if m(r).get("decode_tok_s") and (m(r).get("gen_tokens") or 0) >= MIN_SPEED_TOKENS]
    ttft = [m(r)["ttft_s"] for r in hot if m(r).get("ttft_s") is not None]
    total = [m(r)["total_s"] for r in hot if m(r).get("total_s") is not None]
    by_test: dict = defaultdict(list)
    for r in hot:
        if m(r).get("decode_tok_s") and (m(r).get("gen_tokens") or 0) >= MIN_SPEED_TOKENS:
            by_test[cond_key(r)].append(m(r)["decode_tok_s"])
    within = [cv for vals in by_test.values() if (cv := _cv(vals)) is not None]
    return {
        "decode_tok_s": _median(speed), "decode_samples": len(speed), "decode_cv": _median(within),
        "decode_cv_note": "variabilidad entre repeticiones de la misma prueba (None si no hubo repeticiones)",
        "ttft_s": _median(ttft), "total_s": _median(total),
        "errors": sum(1 for r in results if r.get("gen_status") in ("error", "timeout")),
        "cold_excluded": sum(1 for r in results if r.get("cold")),
    }


def energy(results: list[dict], passed: int) -> dict:
    vals = [r["metrics"]["wh_per_token"] for r in results if (r.get("metrics") or {}).get("wh_per_token") is not None and not r.get("cold")]
    totals = [((r["metrics"].get("gpu") or {}).get("decode") or {}).get("energy_net_wh") for r in results if r.get("metrics")]
    totals = [t for t in totals if t is not None]
    enough = len(vals) >= MIN_ENERGY_SAMPLES
    return {"valid": enough, "samples": len(vals), "estimated": True,
            "wh_per_token": _median(vals, 7) if enough else None,  # del orden de 0.0003 Wh: con 3 decimales se perderia
            "wh_per_correct": round(sum(totals) / passed, 5) if enough and passed else None,
            "note": "Estimacion: potencia media de toda la GPU x duracion, menos el reposo."}


def vram(results: list[dict], events: list[dict]) -> dict:
    idle = next((e.get("vram_idle_mib") for e in events if e.get("type") == "model_start"), None)
    peaks = [((r.get("metrics") or {}).get("gpu") or {}).get("request", {}) or {} for r in results]
    peak = max((p.get("vram_peak_mib") for p in peaks if p.get("vram_peak_mib") is not None), default=None)
    model_bytes = max(((r.get("metrics") or {}).get("model_vram_bytes") or 0 for r in results), default=0) or None
    used = round(peak - idle, 1) if peak is not None and idle is not None else None
    return {"gpu_peak_mib": peak, "idle_mib": idle, "used_mib": used, "model_bytes": model_bytes,
            "note": "gpu_peak_mib es la lectura de toda la GPU; used_mib resta lo que ya habia antes de cargar el modelo."}


def context_curve(results: list[dict]) -> dict:
    """Rendimiento por tamano de contexto pedido y capacidad realmente comprobada (tokens procesados con acierto)."""
    groups: dict = defaultdict(list)
    for r in results:
        if r.get("context_size"):
            groups[r["context_size"]].append(r)
    rows = []
    for size in sorted(groups):
        rs = groups[size]
        ok = [r for r in rs if r.get("gen_status") == "ok"]
        mt = lambda key: _median([r["metrics"][key] for r in ok if (r.get("metrics") or {}).get(key) is not None])
        peaks = [(((r.get("metrics") or {}).get("gpu") or {}).get("request") or {}).get("vram_peak_mib") for r in ok]
        rows.append({"num_ctx_requested": size, "prompt_tokens_processed": mt("prompt_tokens"), "ttft_s": mt("ttft_s"),
                     "prefill_tok_s": mt("prefill_tok_s"), "decode_tok_s": mt("decode_tok_s"),
                     "vram_peak_mib": max((p for p in peaks if p is not None), default=None), **tally(rs)})
    verified = [r["metrics"]["prompt_tokens"] for r in results
                if r.get("context_size") and r.get("outcome") == "passed" and (r.get("metrics") or {}).get("prompt_tokens")]
    return {"curve": rows, "verified_tokens": max(verified, default=None), "max_requested": max(groups, default=None),
            "note": "verified_tokens es el mayor prompt (en tokens procesados) con el que se recupero el dato; no es el num_ctx configurado."}


def pareto_front(points: dict[str, tuple[float, float]], x_lower_better: bool) -> list[str]:
    """Modelos no dominados en (x, calidad). Calidad: mas es mejor; x: menos o mas segun x_lower_better."""
    def dominates(a, b):
        ax, ay = a
        bx, by = b
        x_ok = ax <= bx if x_lower_better else ax >= bx
        x_strict = ax < bx if x_lower_better else ax > bx
        return x_ok and ay >= by and (x_strict or ay > by)
    return sorted(n for n, p in points.items() if not any(dominates(q, p) for m, q in points.items() if m != n))


def _rank(values: dict[str, float | None], higher_better: bool, basis: str, valid_note: str) -> dict:
    valid = {k: v for k, v in values.items() if v is not None}
    if len(valid) < 2:
        return {"status": "insufficient", "reason": f"hacen falta al menos 2 modelos con datos validos ({valid_note})", "basis": basis, "values": values}
    best = max(valid.values()) if higher_better else min(valid.values())
    winners = sorted(k for k, v in valid.items() if abs(v - best) < 1e-9)
    order = sorted(valid, key=lambda k: valid[k], reverse=higher_better)
    return {"status": "ok", "winners": winners, "order": order, "basis": basis, "values": values}


def select_runs(store: Store, run_ids: list[str] | None, models: list[str] | None) -> dict[str, dict]:
    """Para cada modelo, su ejecucion mas reciente (entre las elegidas) con resultados."""
    chosen: dict[str, dict] = {}
    rows = [r for r in store.list_runs() if run_ids is None or r["run_id"] in run_ids]  # del mas reciente al mas antiguo
    # Primero las ejecuciones completas; una cancelada o interrumpida solo se usa si no hay otra con ese modelo.
    for complete_only in (True, False):
        for row in rows:
            if complete_only != (row["state"] == "completed"):
                continue
            wanted = [m for m in row["models"] if (models is None or m in models) and m not in chosen]
            if not wanted:
                continue
            doc = store.load_run(row["run_id"])
            if doc is None:
                continue
            for name in wanted:
                results = [r for r in doc["results"] if r.get("model") == name]
                if not results:
                    continue
                events = [e for e in doc["events"] if e.get("model") == name]
                meta = next((m for m in doc["run"].get("models", []) if m.get("name") == name), {})
                chosen[name] = {"run_id": row["run_id"], "created": doc["run"].get("created"), "results": results, "events": events,
                                "meta": meta, "config": doc["run"].get("config", {}), "state": row["state"]}
    return chosen


def in_size_range(meta: dict, param_min: float | None, param_max: float | None) -> bool:
    if param_min is None and param_max is None:
        return True
    p = meta.get("parameters_b")
    if p is None:
        return False  # con un filtro de tamano, un modelo sin parametros conocidos no se puede comparar
    return (param_min is None or p >= param_min) and (param_max is None or p < param_max)


def compare(store: Store, run_ids: list[str] | None = None, models: list[str] | None = None,
            param_min: float | None = None, param_max: float | None = None) -> dict:
    chosen = select_runs(store, run_ids, models)
    warnings: list[str] = []
    for name in [n for n, d in chosen.items() if not in_size_range(d["meta"], param_min, param_max)]:
        del chosen[name]
    warnings += [f"'{n}': se usan resultados de una ejecucion {d['state']} (parcial); no hay una ejecucion completa de este modelo."
                 for n, d in chosen.items() if d["state"] != "completed"]
    if models:
        warnings += [f"'{m}' no tiene resultados en las ejecuciones elegidas o queda fuera del filtro de tamano." for m in models if m not in chosen]
    if not chosen:
        return {"models": [], "rankings": {}, "warnings": warnings or ["No hay resultados para comparar."], "thresholds": THRESHOLDS,
                "common": {"conditions": 0}, "pareto": {}, "quant_groups": [], "runs": {}}

    # 1) condiciones comunes a todos los modelos, con las mismas repeticiones
    by_model: dict[str, dict] = {}
    for name, d in chosen.items():
        groups: dict = defaultdict(list)
        for r in d["results"]:
            if r.get("outcome") != "cancelled":
                groups[cond_key(r)].append(r)
        by_model[name] = {k: sorted(v, key=lambda r: r.get("repetition", 0)) for k, v in groups.items()}
    common = set.intersection(*(set(g) for g in by_model.values())) if len(by_model) > 1 else set(next(iter(by_model.values())))
    reps = {k: min(len(by_model[n][k]) for n in by_model) for k in common}
    excluded = {n: sum(len(v) for k, v in g.items() if k not in common) for n, g in by_model.items()}
    if len(by_model) > 1 and any(excluded.values()):
        warnings.append("Algunas pruebas no se compararon porque no las ejecutaron todos los modelos con las mismas condiciones: "
                        + ", ".join(f"{n}: {c} resultados excluidos" for n, c in excluded.items() if c) + ".")
    if len(by_model) > 1 and not common:
        warnings.append("Los modelos no comparten ninguna prueba con las mismas condiciones: no hay comparacion posible.")

    # 2) agregados por modelo
    rows = []
    for name, d in chosen.items():
        used = [r for k in common for r in by_model[name][k][: reps[k]]]
        cats: dict = defaultdict(list)
        for r in used:
            cats[r["category"]].append(r)
        overall = tally(used)
        perf = performance(used)
        vr = vram(used, d["events"])
        en = energy(used, overall["passed"])
        meta = d["meta"]
        disk_gib = (meta.get("disk_bytes") or 0) / 2**30 or None
        vram_gib = ((vr["used_mib"] if vr["used_mib"] is not None else vr["gpu_peak_mib"]) or 0) / 1024 or None
        q = overall["score"] if overall["graded"] >= MIN_GRADED_OVERALL else None
        rows.append({
            "name": name, "run_id": d["run_id"], "run_state": d["state"], "created": d["created"], "meta": meta,
            "size_bucket": inspector.size_bucket(meta.get("parameters_b")),
            "overall": overall, "categories": {c: tally(rs) for c, rs in cats.items()},
            "performance": perf, "vram": vr, "energy": en, "context": context_curve(used),
            "indicators": {
                "quality": q, "speed_tok_s": perf["decode_tok_s"] if perf["decode_samples"] >= MIN_SPEED_SAMPLES else None,
                "vram_gib": round(vram_gib, 2) if vram_gib else None, "size_gib": round(disk_gib, 2) if disk_gib else None,
                "quality_per_gib_vram": round(q / vram_gib, 3) if q is not None and vram_gib else None,
                "quality_per_gib_disk": round(q / disk_gib, 3) if q is not None and disk_gib else None,
                "wh_per_token": en["wh_per_token"], "wh_per_correct": en["wh_per_correct"],
            },
        })

    # 3) frentes de Pareto y clasificaciones
    pareto: dict[str, list[str]] = {}
    fronts = {"calidad_vs_vram": ("vram_gib", True), "calidad_vs_velocidad": ("speed_tok_s", False),
              "calidad_vs_tamano": ("size_gib", True), "calidad_vs_energia": ("wh_per_token", True)}
    membership: dict[str, int] = defaultdict(int)
    for label, (key, lower) in fronts.items():
        pts = {r["name"]: (r["indicators"][key], r["indicators"]["quality"]) for r in rows
               if r["indicators"][key] is not None and r["indicators"]["quality"] is not None}
        if len(pts) >= 2:
            pareto[label] = pareto_front(pts, lower)
            for n in pareto[label]:
                membership[n] += 1
    for r in rows:
        r["pareto_fronts"] = membership.get(r["name"], 0)

    cat_score = lambda c: {r["name"]: (r["categories"][c]["score"] if c in r["categories"] and r["categories"][c]["graded"] >= MIN_GRADED_CATEGORY else None) for r in rows}
    rankings = {
        "reasoning": _rank(cat_score("logic"), True, "% de aciertos en 'Razonamiento logico'", f">= {MIN_GRADED_CATEGORY} pruebas calificadas"),
        "code": _rank(cat_score("code"), True, "% de aciertos en 'Programacion'", f">= {MIN_GRADED_CATEGORY} pruebas calificadas"),
        "instructions": _rank(cat_score("instructions"), True, "% de aciertos en 'Seguimiento de instrucciones'", f">= {MIN_GRADED_CATEGORY} pruebas calificadas"),
        "fastest": _rank({r["name"]: r["indicators"]["speed_tok_s"] for r in rows}, True, "tok/s de generacion (mediana, modelo caliente)", f">= {MIN_SPEED_SAMPLES} mediciones"),
        "lowest_vram": _rank({r["name"]: r["indicators"]["vram_gib"] for r in rows}, False, "VRAM usada (pico menos lo previo)", "medicion de VRAM"),
    }
    qualities = [r["indicators"]["quality"] for r in rows if r["indicators"]["quality"] is not None]
    floor = max(qualities) * EFFICIENCY_MIN_RELATIVE_QUALITY if qualities else None
    eff_ok = {r["name"]: (r["pareto_fronts"] if r["indicators"]["quality"] is not None and len(pareto) >= 2 and r["indicators"]["quality"] >= floor else None)
              for r in rows}
    eff = _rank(eff_ok, True, f"presencia en los frentes de Pareto de {', '.join(pareto) or 'calidad vs recursos'} (desempate: calidad por GiB de VRAM); "
                f"solo entre modelos con al menos {EFFICIENCY_MIN_RELATIVE_QUALITY:.0%} de la calidad del mejor, para que un modelo diminuto e inutil no gane por gastar poco",
                "calidad valida, calidad minima y al menos 2 frentes evaluables")
    if eff["status"] == "ok":
        best = [r for r in rows if r["name"] in eff["winners"]]
        top = max((r["indicators"]["quality_per_gib_vram"] or 0) for r in best)
        eff["winners"] = sorted(r["name"] for r in best if (r["indicators"]["quality_per_gib_vram"] or 0) >= top - 1e-9)
    rankings["efficiency"] = eff
    by_category = {c: _rank(cat_score(c), True, "% de aciertos", f">= {MIN_GRADED_CATEGORY} pruebas calificadas") for c in sorted({c for r in rows for c in r["categories"]})}

    # 4) variantes del mismo modelo con distinta cuantizacion
    groups: dict = defaultdict(list)
    for r in rows:
        key = (r["meta"].get("family"), r["meta"].get("parameter_size"))
        if key[0] and key[1]:
            groups[key].append({"name": r["name"], "quantization": r["meta"].get("quantization")})
    quant_groups = [{"family": f, "parameter_size": p, "models": ms} for (f, p), ms in groups.items()
                    if len(ms) > 1 and len({m["quantization"] for m in ms}) > 1]

    return {"models": rows, "rankings": rankings, "by_category": by_category, "pareto": pareto, "quant_groups": quant_groups,
            "common": {"conditions": len(common), "excluded": excluded}, "warnings": warnings, "thresholds": THRESHOLDS,
            "runs": {n: d["run_id"] for n, d in chosen.items()},
            "size_buckets": [{"id": b[0], "min": b[1], "max": b[2]} for b in SIZE_BUCKETS]}

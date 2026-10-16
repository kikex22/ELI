"""Metricas de embeddings, en Python puro (sin numpy)."""
from __future__ import annotations

import math
import statistics


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: list[float]) -> list[float]:
    n = norm(v)
    return [x / n for x in v] if n else list(v)


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = norm(a), norm(b)
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def relevant_rank(query: list[float], docs: list[list[float]], relevant: set[int]) -> int | None:
    """Posicion (1 = primero) del mejor documento relevante, de forma PESIMISTA con los empates.

    rank = 1 + documentos NO relevantes con similitud >= la del mejor relevante. Asi un modelo degenerado que da el mismo
    vector a todo no recibe credito por casualidad del orden.
    """
    if not relevant:
        return None
    sims = [cosine(query, d) for d in docs]
    best = max(sims[i] for i in relevant)
    return 1 + sum(1 for i, s in enumerate(sims) if i not in relevant and s >= best)


def retrieval_metrics(ranks: list[int | None], ks: tuple = (1, 3, 5)) -> dict:
    valid = [r for r in ranks if r is not None]
    n = len(valid)
    out = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = round(sum(1 for r in valid if r <= k) / n, 4) if n else None
    out["mrr"] = round(sum(1 / r for r in valid) / n, 4) if n else None
    return out


def triplet_accuracy(results: list[bool]) -> float | None:
    return round(sum(results) / len(results), 4) if results else None


def min_cosine(pairs: list[tuple[list[float], list[float]]]) -> float | None:
    return round(min(cosine(a, b) for a, b in pairs), 6) if pairs else None


def median(values: list[float], digits: int = 3) -> float | None:
    return round(statistics.median(values), digits) if values else None


def file_coverage(query: list[float], docs: list[list[float]], sources: list[str | None], relevant: set[int],
                  ks: tuple = (5, 10)) -> dict | None:
    """Que fraccion de los ARCHIVOS relevantes aparece entre los k mejores fragmentos.

    Solo tiene sentido si la respuesta esta repartida en varios archivos (preguntas de «como funciona»): con uno solo es
    igual a Recall@k y devuelve None. Con empates de similitud, lo relevante va al final (criterio pesimista).
    """
    wanted = {sources[i] for i in relevant if sources[i]}
    if len(wanted) < 2:
        return None
    sims = [cosine(query, d) for d in docs]
    order = sorted(range(len(docs)), key=lambda i: (-sims[i], i in relevant))
    return {str(k): round(len({sources[i] for i in order[:k] if i in relevant and sources[i]}) / len(wanted), 4) for k in ks}


def coverage_metrics(rows: list[dict], ks: tuple = (5, 10)) -> dict | None:
    """Resumen de `row['coverage']` de las consultas que la tienen: media por k y fraccion de consultas completas en el mayor k."""
    cov = [r["coverage"] for r in rows if r.get("coverage")]
    if not cov:
        return None
    out = {"n": len(cov), "files_mean": round(statistics.fmean(r["files"] for r in rows if r.get("coverage")), 2)}
    for k in ks:
        out[f"cov@{k}"] = round(statistics.fmean(c[str(k)] for c in cov), 4)
    top = str(ks[-1])
    out[f"full@{ks[-1]}"] = round(sum(1 for c in cov if c[top] >= 1.0) / len(cov), 4)
    return out


FILE_KS = (3, 5, 8)


def _file_order(sims: list[float], sources: list[str | None], wanted: set[str]) -> list[str]:
    """Archivos de mejor a peor: cada uno vale lo que su mejor fragmento. Con empates, lo relevante va al final."""
    best: dict[str, float] = {}
    for src, sim in zip(sources, sims):
        if src and sim > best.get(src, float("-inf")):
            best[src] = sim
    return sorted(best, key=lambda f: (-best[f], f in wanted))


def file_level(query: list[float], docs: list[list[float]], sources: list[str | None], relevant: set[int],
               ks: tuple = FILE_KS, sims: list[float] | None = None) -> dict | None:
    """Recuperacion AGRUPADA POR ARCHIVO: cada archivo vale lo que su mejor fragmento y se ordenan los archivos.

    Evita que un archivo grande (muchas ventanas) llene el ranking. Devuelve la posicion del mejor archivo relevante, cuantos
    archivos relevantes hay y la fraccion de ellos que cae entre los k primeros archivos. Con empates, lo relevante va al final
    (criterio pesimista). None si la consulta no tiene archivos conocidos (p. ej. el banco incluido).
    """
    wanted = {sources[i] for i in relevant if sources[i]}
    if not wanted:
        return None
    sims = sims if sims is not None else [cosine(query, d) for d in docs]
    order = _file_order(sims, sources, wanted)
    first = min(pos for pos, f in enumerate(order, 1) if f in wanted)
    return {"rank": first, "files": len(wanted),
            "coverage": {str(k): round(len(set(order[:k]) & wanted) / len(wanted), 4) for k in ks}}


SUB_MS = (1, 2, 3)


def subquery_coverage(query: list[float], sub_vecs: list[list[float]], docs: list[list[float]], sources: list[str | None],
                      relevant: set[int], ms: tuple = SUB_MS, extra_vecs: list[list[float]] | None = None) -> dict | None:
    """Cobertura de archivos al DESCOMPONER una pregunta en subpreguntas.

    Por cada m: se toman los m mejores archivos de cada subpregunta y se unen. `coverage` es la fraccion de archivos relevantes
    en esa union; `baseline` es la de la pregunta original con EL MISMO NUMERO de archivos (`union`), para que la ganancia no se
    deba solo a mirar mas archivos. Solo para preguntas con varios archivos y subpreguntas.
    """
    wanted = {sources[i] for i in relevant if sources[i]}
    if len(wanted) < 2 or not sub_vecs:
        return None
    base = _file_order([cosine(query, d) for d in docs], sources, wanted)
    orders = [_file_order([cosine(v, d) for d in docs], sources, wanted) for v in sub_vecs + list(extra_vecs or [])]
    by_m = {}
    for m in ms:
        union = {f for o in orders for f in o[:m]}
        by_m[str(m)] = {"union": len(union), "coverage": round(len(union & wanted) / len(wanted), 4),
                        "baseline": round(len(set(base[:len(union)]) & wanted) / len(wanted), 4)}
    return {"subqueries": len(sub_vecs), "files": len(wanted), "by_m": by_m}


def subquery_metrics(rows: list[dict], ms: tuple = SUB_MS, field: str = "subquery") -> dict | None:
    """Promedios de `row[field]`: cobertura con subpreguntas, de la pregunta sola con el mismo presupuesto, y la ganancia.
    `field` es "subquery" (escritas a mano) o "subquery_llm" (generadas por el LLM)."""
    sq = [r[field] for r in rows if r.get(field)]
    if not sq:
        return None
    out: dict = {"n": len(sq), "subqueries_mean": round(statistics.fmean(x["subqueries"] for x in sq), 1),
                 "files_mean": round(statistics.fmean(x["files"] for x in sq), 2), "by_m": {}}
    for m in ms:
        key = str(m)
        cov = [x["by_m"][key]["coverage"] for x in sq]
        base = [x["by_m"][key]["baseline"] for x in sq]
        out["by_m"][key] = {"union_mean": round(statistics.fmean(x["by_m"][key]["union"] for x in sq), 1),
                            "coverage": round(statistics.fmean(cov), 4), "baseline": round(statistics.fmean(base), 4),
                            "gain": round(statistics.fmean(c - b for c, b in zip(cov, base)), 4),
                            "full": round(sum(1 for c in cov if c >= 1.0) / len(sq), 4),
                            "full_baseline": round(sum(1 for b in base if b >= 1.0) / len(sq), 4)}
    return out


def file_level_metrics(rows: list[dict], ks: tuple = FILE_KS) -> dict | None:
    """Resumen de `row['file_level']`: acierto del mejor archivo (todas las consultas) y cobertura (consultas de varios archivos)."""
    fl = [r["file_level"] for r in rows if r.get("file_level")]
    if not fl:
        return None
    n = len(fl)
    out: dict = {"n": n}
    for k in (1, 3, 5):
        out[f"hit@{k}"] = round(sum(1 for f in fl if f["rank"] <= k) / n, 4)
    out["mrr"] = round(sum(1 / f["rank"] for f in fl) / n, 4)
    multi = [f for f in fl if f["files"] >= 2]
    if multi:
        top = str(ks[-1])
        out["multi"] = {"n": len(multi), "files_mean": round(statistics.fmean(f["files"] for f in multi), 2),
                        **{f"cov@{k}": round(statistics.fmean(f["coverage"][str(k)] for f in multi), 4) for k in ks},
                        f"full@{ks[-1]}": round(sum(1 for f in multi if f["coverage"][top] >= 1.0) / len(multi), 4)}
    return out

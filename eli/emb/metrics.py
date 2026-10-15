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

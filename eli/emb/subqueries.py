"""Subpreguntas generadas por un LLM local a partir de la pregunta, SIN ver los archivos que la responden.

El LLM solo recibe el texto de la pregunta. Lo que devuelve se usa como consultas adicionales: cada una se embebe y se
mide igual que las subpreguntas escritas a mano, pero sin el sesgo de conocer la respuesta.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ..generate import generate

MIN_SUBQUERIES, MAX_SUBQUERIES, DEFAULT_SUBQUERIES = 2, 8, 6
MAX_CHARS = 200
LANGS = ("same", "en")
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512}
TIMEOUT_S = 120.0

_LANG_NAME = {"same": "the same language as the question", "en": "English"}


class SubqueryError(Exception):
    pass


def build_messages(question: str, count: int, lang: str) -> list[dict]:
    """Pide frases de BUSQUEDA (no preguntas): el buscador de embeddings encuentra mejor palabras clave que interrogantes."""
    system = (
        "You write SEARCH PHRASES for a code search engine over a robotics project (C++, Python, ROS 2, CMake).\n"
        f"Given a question about the code, write {count} search phrases that would find the code needed to answer it.\n"
        "Rules:\n"
        "- 3 to 6 words each. Keywords, not sentences.\n"
        "- NO question marks and NO question words (what, how, which, where).\n"
        "- One concrete part per phrase: a component, a step, a topic, a parameter, a data type.\n"
        "- Use English code vocabulary (function, class, topic, message, config names), even if the question is in Spanish.\n"
        "- Do not invent exact function names; use plain descriptive words.\n"
        f"- Write the phrases in {_LANG_NAME[lang]}.\n"
        "Reply ONLY with a JSON array of strings, no text before or after. Example: [\"camera frame capture\", \"bounding box NMS\"]"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}"}]


def parse_subqueries(text: str, max_n: int = DEFAULT_SUBQUERIES) -> list[str]:
    """Extrae la lista del texto del modelo. Tolera <think>, cercas de codigo y listas con guiones o numeros."""
    body = text or ""
    if "</think>" in body:
        body = body.split("</think>")[-1]
    body = re.sub(r"<think>.*?</think>", "", body, flags=re.S).strip()
    body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body).strip()
    items = _from_json(body)
    if items is None:
        items = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln) for ln in body.splitlines()]
    return _clean(items, max_n)


def _from_json(body: str):
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return None
    return data


def _clean(items: list, max_n: int) -> list[str]:
    out = _normalize(items, max_n)
    if len(out) < MIN_SUBQUERIES:
        raise SubqueryError(f"El modelo dio {len(out)} subpreguntas validas; hacen falta al menos {MIN_SUBQUERIES}.")
    return out


def _normalize(items: list, max_n: int) -> list[str]:
    out, seen = [], set()
    for raw in items:
        text = re.sub(r"\s+", " ", str(raw)).strip().strip('"\'').replace("¿", "").replace("?", "").strip()  # si pregunta, se quita el signo
        if not text or len(text) > MAX_CHARS or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
    return out[:max_n]


def generate_subqueries(ollama, gpu, model: str, question: str, count: int = DEFAULT_SUBQUERIES, lang: str = "same",
                        should_stop=None, timeout_s: float = TIMEOUT_S) -> list[str]:
    """Una llamada al LLM por pregunta. Lanza SubqueryError si la respuesta no se puede usar."""
    gen = generate(ollama, gpu, model, build_messages(question, count, lang), dict(OPTIONS), think=False,
                   timeout_s=timeout_s, should_stop=should_stop)
    if gen.status != "ok":
        raise SubqueryError(gen.error or f"generacion {gen.status}")
    return parse_subqueries(gen.content, count)


PROMPT_VERSION = "search-phrases-1"  # cambia si cambia el prompt: las subpreguntas guardadas dejan de valer


REFINE_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 256}
REFINE_MAX = 3  # frases de busqueda por ronda


def build_refine_messages(question: str, found: list[tuple[str, str]]) -> list[dict]:
    """Ronda de refinamiento: el LLM ve la pregunta y los archivos ya encontrados (inicio de cada uno) y dice que falta."""
    listing = "\n\n".join(f"### {path}\n{snippet}" for path, snippet in found) or "(nothing found yet)"
    system = (
        "You help find code in a robotics project (C++, Python, ROS 2, CMake). You get a question about the code and the files "
        "found so far, with their first lines. Decide which part of the answer is still missing.\n"
        'Reply ONLY with JSON: {"done": true or false, "queries": ["search phrase", ...]}.\n'
        "done=true only if the files found already cover the whole answer. Otherwise give 1 to 3 search phrases for the missing parts.\n"
        "Rules for the phrases: 3 to 6 words, English code vocabulary (function, class, topic, message names), NO question marks."
    )
    user = f"Question: {question}\n\nFiles found so far:\n{listing}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_refine(text: str) -> tuple[bool, list[str]]:
    """(terminado, frases). Sin un objeto JSON util se lanza SubqueryError: la pregunta se da por cerrada sin mas busquedas."""
    body = text or ""
    if "</think>" in body:
        body = body.split("</think>")[-1]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise SubqueryError("La ronda no devolvio un objeto JSON.")
    try:
        data = json.loads(body[start:end + 1])
    except ValueError as exc:
        raise SubqueryError("La ronda devolvio JSON invalido.") from exc
    if not isinstance(data, dict):
        raise SubqueryError("La ronda devolvio algo que no es un objeto.")
    raw = data.get("queries") or []
    phrases = _normalize([x for x in raw if isinstance(x, str)], REFINE_MAX) if isinstance(raw, list) else []
    return bool(data.get("done")), phrases


def refine(ollama, gpu, model: str, question: str, found: list[tuple[str, str]], should_stop=None,
           timeout_s: float = TIMEOUT_S) -> tuple[bool, list[str]]:
    gen = generate(ollama, gpu, model, build_refine_messages(question, found), dict(REFINE_OPTIONS), think=False,
                   timeout_s=timeout_s, should_stop=should_stop)
    if gen.status != "ok":
        raise SubqueryError(gen.error or f"generacion {gen.status}")
    return parse_refine(gen.content)


def cache_key(question: str, llm: str, count: int, lang: str) -> str:
    raw = json.dumps([PROMPT_VERSION, llm, count, lang, question], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class SubqueryCache:
    """Subpreguntas ya generadas, por pregunta y LLM. Asi una corrida posterior (p. ej. el 8B) no vuelve a cargar el LLM."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}  # sin archivo o danado: se regenera, nunca se rompe la corrida

    def get(self, key: str) -> list[str] | None:
        items = self._load().get(key)
        return list(items) if isinstance(items, list) and items else None

    def put(self, key: str, items: list[str]) -> None:
        data = self._load()
        data[key] = list(items)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

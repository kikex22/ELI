"""Banco de pruebas de embeddings: versionado, reproducible y con respuesta correcta conocida por construccion."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SUITES_DIR = Path(__file__).resolve().parent / "suites"
PREFIX_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "embedding_prefixes.json"
DEFAULT_SUITE_ID = "emb-core"
LANGS = {"es", "en"}
TRIPLET_KINDS = {"es", "en", "cross"}


class EmbSuiteError(ValueError):
    pass


@dataclass(frozen=True)
class EmbSuite:
    id: str
    version: str
    description: str
    documents: tuple
    queries: tuple
    triplets: tuple
    stability_texts: tuple
    performance: dict
    sha256: str

    def info(self) -> dict:
        kinds: dict = {}
        for q in self.queries:
            kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
        tk: dict = {}
        for t in self.triplets:
            tk[t["kind"]] = tk.get(t["kind"], 0) + 1
        return {"id": self.id, "version": self.version, "description": self.description, "sha256": self.sha256,
                "documents": len(self.documents), "queries": len(self.queries), "query_kinds": kinds,
                "triplets": len(self.triplets), "triplet_kinds": tk, "performance": self.performance}

    def perf_text(self, words: int, offset: int = 0) -> str:
        """Texto determinista de ~`words` palabras formado con los documentos del banco."""
        pool = " ".join(d["text"] for d in self.documents[offset % len(self.documents):] + self.documents[:offset % len(self.documents)]).split()
        out: list[str] = []
        while len(out) < words:
            out += pool
        return " ".join(out[:words])


def parse_suite(data: dict) -> EmbSuite:
    for key in ("suite_id", "version", "documents", "queries", "triplets", "stability_texts", "performance"):
        if key not in data:
            raise EmbSuiteError(f"falta '{key}' en la suite")
    doc_ids, doc_lang = set(), {}
    for d in data["documents"]:
        if not d.get("id") or d["id"] in doc_ids:
            raise EmbSuiteError(f"id de documento invalido o repetido: {d.get('id')!r}")
        if d.get("lang") not in LANGS or not str(d.get("text", "")).strip():
            raise EmbSuiteError(f"documento {d['id']}: idioma o texto invalido")
        doc_ids.add(d["id"])
        doc_lang[d["id"]] = d["lang"]
    q_ids = set()
    for q in data["queries"]:
        if not q.get("id") or q["id"] in q_ids:
            raise EmbSuiteError(f"id de consulta invalido o repetido: {q.get('id')!r}")
        q_ids.add(q["id"])
        if q.get("lang") not in LANGS or not str(q.get("text", "")).strip():
            raise EmbSuiteError(f"consulta {q['id']}: idioma o texto invalido")
        rel = q.get("relevant")
        if not rel or not set(rel) <= doc_ids:
            raise EmbSuiteError(f"consulta {q['id']}: documento relevante inexistente")
        expected_kind = "mono" if all(doc_lang[r] == q["lang"] for r in rel) else "cross"
        if q.get("kind") != expected_kind:
            raise EmbSuiteError(f"consulta {q['id']}: 'kind' debe ser {expected_kind!r}")
    t_ids = set()
    for t in data["triplets"]:
        if not t.get("id") or t["id"] in t_ids or t.get("kind") not in TRIPLET_KINDS:
            raise EmbSuiteError(f"triplete invalido: {t.get('id')!r}")
        t_ids.add(t["id"])
        if not all(str(t.get(k, "")).strip() for k in ("anchor", "positive", "negative")):
            raise EmbSuiteError(f"triplete {t['id']}: faltan textos")
    perf = data["performance"]
    if not all(isinstance(perf.get(k), int) and perf[k] > 0 for k in ("short_words", "medium_words", "long_words", "default_repetitions")):
        raise EmbSuiteError("performance: palabras y repeticiones deben ser enteros positivos")
    if not perf.get("batch_sizes") or not all(isinstance(b, int) and b > 0 for b in perf["batch_sizes"]):
        raise EmbSuiteError("performance: batch_sizes invalido")
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return EmbSuite(data["suite_id"], data["version"], data.get("description", ""), tuple(data["documents"]), tuple(data["queries"]),
                    tuple(data["triplets"]), tuple(data["stability_texts"]), perf, digest)


def default_suite(suites_dir: Path = SUITES_DIR) -> EmbSuite:
    found = []
    for path in sorted(suites_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("suite_id") == DEFAULT_SUITE_ID:
            version = tuple(int(p) if p.isdigit() else 0 for p in str(data.get("version", "0")).split("."))
            found.append((version, data))
    if not found:
        raise EmbSuiteError(f"no hay suites '{DEFAULT_SUITE_ID}' en {suites_dir}")
    return parse_suite(max(found, key=lambda f: f[0])[1])


# --- prefijos de tarea ----------------------------------------------------------

def load_prefix_rules(path: Path = PREFIX_RULES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_for(model: str, architecture: str | None, rules: dict | None = None) -> dict:
    """{'rule', 'query', 'document', 'source'} de la primera regla que encaje, o sin prefijos."""
    rules = rules or load_prefix_rules()
    low = model.lower()
    for r in rules["rules"]:
        if any(s in low for s in r["name_contains"]) or (architecture and architecture in r["architectures"]):
            return {"rule": r["id"], "query": r["query"], "document": r["document"], "source": r["source"]}
    return {"rule": None, "query": "", "document": "", "source": "sin regla de prefijos para este modelo"}

"""Corpus propio para evaluar recuperacion con TUS documentos.

Sin modelo generador (no se mezcla otro modelo en la medicion). Dos fuentes de consultas, ambas verificables:
  - Titulos de seccion Markdown: la consulta es el titulo; los documentos correctos son los fragmentos de esa seccion,
    ya sin el titulo. Se descartan los titulos repetidos o genericos porque serian ambiguos.
  - Opcional, `eli_queries.jsonl` en la carpeta: {"query": "...", "relevant_files": ["ruta/relativa.md"]} por linea.
Solo lee texto (.md .txt .rst), nunca sigue enlaces simbolicos, ignora carpetas ocultas y nombres sensibles, y tiene topes
de tamano. No se guarda el contenido de los documentos en los resultados, solo rutas, consultas y posiciones.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import codecorpus

EXTS = {".md", ".markdown", ".txt", ".rst"}
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "env", "site-packages", "dist", "build", "target"}
SENSITIVE = re.compile(r"(secret|password|passwd|token|credential|id_rsa|\.env|api[_-]?key|private[_-]?key)", re.I)
FORBIDDEN_ROOTS = {Path("/"), Path("/proc"), Path("/sys"), Path("/dev")}
GENERIC_HEADINGS = {"introduccion", "introduction", "overview", "resumen", "summary", "contenido", "contents", "indice", "index",
                    "licencia", "license", "notas", "notes", "ejemplo", "ejemplos", "example", "examples", "uso", "usage", "todo",
                    "referencias", "references", "changelog", "readme", "descripcion", "description"}
MANUAL_FILE = "eli_queries.jsonl"  # opcional: "kind" por linea (p. ej. "code-en") separa las metricas por tipo de pregunta
KIND_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")

MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 300, 400_000, 5_000_000
MAX_DOCS, MAX_QUERIES, MAX_DIRS, MAX_DEPTH = 600, 60, 4000, 8
MIN_BODY_WORDS, SPLIT_OVER_WORDS, CHUNK_WORDS = 30, 220, 160
MIN_DOCS, MIN_QUERIES = 10, 5
MODES = ("docs", "code", "both")
MAX_SUBQUERIES = 12
MAX_MANUAL_LINES = 200
MAX_CODE_FILES, MAX_CODE_DOCS, MAX_CODE_QUERIES = 600, 2500, 80


class CorpusError(ValueError):
    pass


@dataclass
class Corpus:
    folder: str
    mode: str = "docs"
    documents: list = field(default_factory=list)  # {id, source, text}
    queries: list = field(default_factory=list)  # {id, text, relevant:[doc ids], kind, source}
    fingerprint: str = ""
    stats: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def preview(self) -> dict:
        return {"folder": self.folder, "mode": self.mode, "fingerprint": self.fingerprint, "stats": self.stats, "skipped": self.skipped,
                "warnings": self.warnings, "sample_queries": [q["text"] for q in self.queries[:6]]}


def _norm(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFD", text.lower())
    return re.sub(r"\s+", " ", "".join(c for c in t if unicodedata.category(c) != "Mn")).strip(" .:-")


def _clean_heading(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"^\s*[\d.]+\s+", "", text)  # numeracion «1.2 »
    return re.sub(r"\s+", " ", text).strip(" :#")


def _clean_body(lines: list[str]) -> str:
    body = "\n".join(lines)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"^\s*[-*+]\s+", "", body, flags=re.M)
    body = re.sub(r"[*_`>|]+", " ", body)
    return re.sub(r"\s+", " ", body).strip()


def split_sections(text: str, markdown: bool) -> list[tuple[str | None, str]]:
    """[(titulo|None, cuerpo)]. Ignora lo que parece titulo dentro de bloques de codigo."""
    if not markdown:
        return [(None, _clean_body(text.splitlines()))]
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if m:
            sections.append((_clean_heading(m.group(1)), []))
        else:
            sections[-1][1].append(line)
    return [(h, _clean_body(body)) for h, body in sections]


def chunk(text: str) -> list[str]:
    words = text.split()
    if len(words) <= SPLIT_OVER_WORDS:
        return [text]
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS) if len(words[i:i + CHUNK_WORDS]) >= 20]


def _iter_files(root: Path, skipped: dict, exts: set = EXTS, skip_dirs: set = SKIP_DIRS,
                names: frozenset = frozenset()) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    dirs_seen = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirs_seen += 1
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in skip_dirs and not (Path(dirpath) / d).is_symlink())
        if dirs_seen > MAX_DIRS or depth >= MAX_DEPTH:
            dirnames[:] = []
            skipped["carpetas_no_recorridas"] = skipped.get("carpetas_no_recorridas", 0) + 1
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = str(path.relative_to(root))
            if name.startswith("."):
                skipped["ocultos"] = skipped.get("ocultos", 0) + 1
            elif path.suffix.lower() not in exts and name not in names:
                continue
            elif path.is_symlink():
                skipped["enlaces_simbolicos"] = skipped.get("enlaces_simbolicos", 0) + 1
            elif SENSITIVE.search(rel):
                skipped["nombres_sensibles"] = skipped.get("nombres_sensibles", 0) + 1
            else:
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        skipped["demasiado_grandes"] = skipped.get("demasiado_grandes", 0) + 1
                        continue
                except OSError:
                    continue
                found.append((path, rel))
    return found


def _read_files(files: list, corpus: Corpus, digests: list, budget: dict) -> list[tuple[str, Path, str]]:
    """[(ruta relativa, ruta, texto)] respetando el tope de tamano total compartido (`budget`)."""
    out = []
    for path, rel in files:
        try:
            raw = path.read_bytes()
        except OSError:
            corpus.skipped["ilegibles"] = corpus.skipped.get("ilegibles", 0) + 1
            continue
        budget["bytes"] += len(raw)
        if budget["bytes"] > MAX_TOTAL_BYTES:
            corpus.warnings.append("Se alcanzo el limite de tamano total; el resto de archivos no se uso.")
            break
        digests.append((rel, hashlib.sha256(raw).hexdigest()))
        out.append((rel, path, raw.decode("utf-8", errors="replace")))
    return out


def _add_docs(root: Path, corpus: Corpus, digests: list, budget: dict) -> list[dict]:
    """Documentos de texto (.md/.txt/.rst). Devuelve las consultas automaticas (titulos de seccion)."""
    files = _iter_files(root, corpus.skipped)
    if len(files) > MAX_FILES:
        corpus.warnings.append(f"Hay {len(files)} archivos de texto; se usan los primeros {MAX_FILES} por orden alfabetico.")
        files = files[:MAX_FILES]
    sections_by_source = [(rel, split_sections(text, path.suffix.lower() in {".md", ".markdown"}))
                          for rel, path, text in _read_files(files, corpus, digests, budget)]
    corpus.stats["files"] = len(sections_by_source)

    section_docs: list[tuple[str, str | None, list[str]]] = []  # (fuente, titulo, ids de documentos)
    for rel, sections in sections_by_source:
        for heading, body in sections:
            if len(body.split()) < MIN_BODY_WORDS:
                continue
            ids = []
            for piece in chunk(body):
                if len(corpus.documents) >= MAX_DOCS:
                    break
                doc_id = f"c{len(corpus.documents) + 1:04d}"
                corpus.documents.append({"id": doc_id, "source": rel, "text": piece})
                ids.append(doc_id)
            if ids:
                section_docs.append((rel, heading, ids))
    if len(corpus.documents) >= MAX_DOCS:
        corpus.warnings.append(f"Se alcanzo el limite de {MAX_DOCS} fragmentos; el resto de texto no se uso.")

    # consultas por titulo de seccion (descartando titulos repetidos o genericos)
    counts: dict[str, int] = {}
    for _, heading, _ in section_docs:
        if heading:
            counts[_norm(heading)] = counts.get(_norm(heading), 0) + 1
    auto = []
    for rel, heading, ids in section_docs:
        key = _norm(heading) if heading else ""
        if (not heading or counts.get(key) != 1 or key in GENERIC_HEADINGS or len(heading) < 8 or len(heading.split()) < 2
                or len(corpus.documents) < 2):
            continue
        auto.append({"text": heading, "relevant": ids, "kind": "titulo", "source": rel})
    return _spread(auto, MAX_QUERIES, corpus, "titulos")


def _spread(items: list, limit: int, corpus: Corpus, what: str) -> list:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    corpus.warnings.append(f"Habia mas de {limit} {what} usables; se uso una muestra repartida.")
    return [items[int(i * step)] for i in range(limit)]  # muestra repartida y determinista


def _add_code(root: Path, corpus: Corpus, digests: list, budget: dict) -> list[dict]:
    """Fragmentos de codigo (funciones, clases, bloques). Devuelve las consultas automaticas (docstrings y comentarios)."""
    files = _iter_files(root, corpus.skipped, codecorpus.CODE_EXTS, SKIP_DIRS | codecorpus.CODE_SKIP_DIRS, frozenset(codecorpus.CODE_NAMES))
    if len(files) > MAX_CODE_FILES:
        corpus.warnings.append(f"Hay {len(files)} archivos de codigo; se usan los primeros {MAX_CODE_FILES} por orden alfabetico.")
        files = files[:MAX_CODE_FILES]
    by_lang: dict[str, int] = {}
    seen_text: set[str] = set()
    auto, seen_query, code_docs, n_files = [], set(), 0, 0
    for rel, path, text in _read_files(files, corpus, digests, budget):
        n_files += 1
        for unit in codecorpus.extract_units(path.name, path.suffix.lower(), text):
            ids = []
            for piece in unit["pieces"]:
                digest = hashlib.sha256(piece.encode()).hexdigest()
                if digest in seen_text:
                    continue  # el mismo codigo copiado en varios archivos no debe contar como respuesta distinta
                if code_docs >= MAX_CODE_DOCS:
                    break
                seen_text.add(digest)
                doc_id = f"k{code_docs + 1:04d}"
                corpus.documents.append({"id": doc_id, "source": rel, "text": piece, "symbol": unit["symbol"], "lang": unit["lang"]})
                ids.append(doc_id)
                code_docs += 1
            if not ids:
                continue
            by_lang[unit["lang"]] = by_lang.get(unit["lang"], 0) + len(ids)
            key = _norm(unit["query"]) if unit["query"] else None
            if key and key not in seen_query:
                seen_query.add(key)
                auto.append({"text": unit["query"], "relevant": ids, "kind": unit["query_kind"], "source": f"{rel}::{unit['symbol']}"})
    if code_docs >= MAX_CODE_DOCS:
        corpus.warnings.append(f"Se alcanzo el limite de {MAX_CODE_DOCS} fragmentos de codigo; el resto no se uso.")
    corpus.stats.update(code_files=n_files, code_documents=code_docs, code_by_language=by_lang)
    return _spread(auto, MAX_CODE_QUERIES, corpus, "descripciones")


def _manual_queries(root: Path, corpus: Corpus) -> list[dict]:
    by_source: dict[str, list[str]] = {}
    for d in corpus.documents:
        by_source.setdefault(d["source"], []).append(d["id"])
    manual, manual_path = [], root / MANUAL_FILE
    if manual_path.is_file() and not manual_path.is_symlink():
        lines = manual_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > MAX_MANUAL_LINES:
            corpus.warnings.append(f"{MANUAL_FILE} tiene {len(lines)} lineas; se usan las primeras {MAX_MANUAL_LINES}.")
        for n, line in enumerate(lines[:MAX_MANUAL_LINES], 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                query, files_rel = str(item["query"]).strip(), [str(f) for f in item["relevant_files"]]
            except (ValueError, KeyError, TypeError):
                corpus.warnings.append(f"{MANUAL_FILE} linea {n}: formato invalido (se esperaba query y relevant_files).")
                continue
            ids = [i for f in files_rel for i in by_source.get(f, [])]
            if not query or not ids:
                corpus.warnings.append(f"{MANUAL_FILE} linea {n}: sin consulta o con archivos que no estan en el corpus.")
                continue
            kind = str(item.get("kind") or "manual")
            if not KIND_RE.match(kind):
                corpus.warnings.append(f"{MANUAL_FILE} linea {n}: 'kind' invalido (usa letras, numeros, - o _); se usa 'manual'.")
                kind = "manual"
            entry = {"text": query, "relevant": ids, "kind": kind, "source": ", ".join(files_rel)}
            subs = item.get("subqueries")
            if subs is not None:
                if (not isinstance(subs, list) or not 1 <= len(subs) <= MAX_SUBQUERIES
                        or not all(isinstance(x, str) and x.strip() for x in subs)):
                    corpus.warnings.append(f"{MANUAL_FILE} linea {n}: 'subqueries' debe ser una lista de 1 a {MAX_SUBQUERIES} textos; se ignora.")
                else:
                    entry["subqueries"] = [x.strip() for x in subs]
            manual.append(entry)
    return manual


def build_corpus(folder: str | Path, mode: str = "docs") -> Corpus:
    """mode: 'docs' (Markdown/texto, como siempre), 'code' (fuentes) o 'both' (todo junto: busqueda mas realista y mas dificil)."""
    if mode not in MODES:
        raise CorpusError(f"Tipo de corpus desconocido: {mode!r} (usa {', '.join(MODES)}).")
    try:
        root = Path(folder).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CorpusError(f"No se pudo abrir la carpeta: {exc}") from exc
    if not root.is_dir():
        raise CorpusError("La ruta no es una carpeta.")
    if root in FORBIDDEN_ROOTS:
        raise CorpusError("Esa carpeta es demasiado amplia; elige la de tus documentos.")

    corpus = Corpus(folder=str(root), mode=mode)
    digests: list = []
    budget = {"bytes": 0}
    auto: list[dict] = []
    if mode in ("docs", "both"):
        auto += _add_docs(root, corpus, digests, budget)
    if mode in ("code", "both"):
        auto += _add_code(root, corpus, digests, budget)
    manual_file = root / MANUAL_FILE
    manual_digest = hashlib.sha256(manual_file.read_bytes()).hexdigest() if manual_file.is_file() and not manual_file.is_symlink() else None
    corpus.fingerprint = hashlib.sha256(json.dumps([mode, digests, manual_digest], ensure_ascii=False).encode()).hexdigest()

    manual = _manual_queries(root, corpus)
    for n, q in enumerate(manual + auto, 1):
        corpus.queries.append({"id": f"u{n:03d}", **q})
    kinds: dict[str, int] = {}
    for q in corpus.queries:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    corpus.stats.update(documents=len(corpus.documents), queries=len(corpus.queries), queries_manual=len(manual),
                        queries_titles=kinds.get("titulo", 0), queries_by_kind=kinds,
                        words=sum(len(d["text"].split()) for d in corpus.documents))
    corpus.stats.setdefault("files", 0)
    if len(corpus.documents) < MIN_DOCS or len(corpus.queries) < MIN_QUERIES:
        raise CorpusError(
            f"Muy poco material: {len(corpus.documents)} fragmentos y {len(corpus.queries)} consultas (minimo {MIN_DOCS} y {MIN_QUERIES}). "
            + ("Hacen falta fuentes con comentarios o docstrings, o un archivo " if mode != "docs"
               else "Hacen falta documentos Markdown con secciones tituladas, o un archivo ") + f"{MANUAL_FILE}.")
    return corpus

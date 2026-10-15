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

EXTS = {".md", ".markdown", ".txt", ".rst"}
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "env", "site-packages", "dist", "build", "target"}
SENSITIVE = re.compile(r"(secret|password|passwd|token|credential|id_rsa|\.env|api[_-]?key|private[_-]?key)", re.I)
FORBIDDEN_ROOTS = {Path("/"), Path("/proc"), Path("/sys"), Path("/dev")}
GENERIC_HEADINGS = {"introduccion", "introduction", "overview", "resumen", "summary", "contenido", "contents", "indice", "index",
                    "licencia", "license", "notas", "notes", "ejemplo", "ejemplos", "example", "examples", "uso", "usage", "todo",
                    "referencias", "references", "changelog", "readme", "descripcion", "description"}
MANUAL_FILE = "eli_queries.jsonl"

MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES = 300, 400_000, 5_000_000
MAX_DOCS, MAX_QUERIES, MAX_DIRS, MAX_DEPTH = 600, 60, 4000, 8
MIN_BODY_WORDS, SPLIT_OVER_WORDS, CHUNK_WORDS = 30, 220, 160
MIN_DOCS, MIN_QUERIES = 10, 5


class CorpusError(ValueError):
    pass


@dataclass
class Corpus:
    folder: str
    documents: list = field(default_factory=list)  # {id, source, text}
    queries: list = field(default_factory=list)  # {id, text, relevant:[doc ids], kind, source}
    fingerprint: str = ""
    stats: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def preview(self) -> dict:
        return {"folder": self.folder, "fingerprint": self.fingerprint, "stats": self.stats, "skipped": self.skipped,
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


def _iter_files(root: Path, skipped: dict) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    dirs_seen = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirs_seen += 1
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS and not (Path(dirpath) / d).is_symlink())
        if dirs_seen > MAX_DIRS or depth >= MAX_DEPTH:
            dirnames[:] = []
            skipped["carpetas_no_recorridas"] = skipped.get("carpetas_no_recorridas", 0) + 1
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = str(path.relative_to(root))
            if name.startswith("."):
                skipped["ocultos"] = skipped.get("ocultos", 0) + 1
            elif path.suffix.lower() not in EXTS:
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


def build_corpus(folder: str | Path) -> Corpus:
    try:
        root = Path(folder).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CorpusError(f"No se pudo abrir la carpeta: {exc}") from exc
    if not root.is_dir():
        raise CorpusError("La ruta no es una carpeta.")
    if root in FORBIDDEN_ROOTS:
        raise CorpusError("Esa carpeta es demasiado amplia; elige la de tus documentos.")

    corpus = Corpus(folder=str(root))
    files = _iter_files(root, corpus.skipped)
    if len(files) > MAX_FILES:
        corpus.warnings.append(f"Hay {len(files)} archivos de texto; se usan los primeros {MAX_FILES} por orden alfabetico.")
        files = files[:MAX_FILES]

    digests, total_bytes, sections_by_source = [], 0, []
    for path, rel in files:
        try:
            raw = path.read_bytes()
        except OSError:
            corpus.skipped["ilegibles"] = corpus.skipped.get("ilegibles", 0) + 1
            continue
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_BYTES:
            corpus.warnings.append("Se alcanzo el limite de tamano total; el resto de archivos no se uso.")
            break
        digests.append((rel, hashlib.sha256(raw).hexdigest()))
        text = raw.decode("utf-8", errors="replace")
        sections_by_source.append((rel, split_sections(text, path.suffix.lower() in {".md", ".markdown"})))
    corpus.fingerprint = hashlib.sha256(json.dumps(digests, ensure_ascii=False).encode()).hexdigest()

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
    if len(auto) > MAX_QUERIES:
        step = len(auto) / MAX_QUERIES
        auto = [auto[int(i * step)] for i in range(MAX_QUERIES)]  # muestra repartida y determinista
        corpus.warnings.append(f"Habia mas de {MAX_QUERIES} titulos usables; se uso una muestra repartida.")

    # consultas manuales
    manual, by_source = [], {}
    for d in corpus.documents:
        by_source.setdefault(d["source"], []).append(d["id"])
    manual_path = root / MANUAL_FILE
    if manual_path.is_file() and not manual_path.is_symlink():
        for n, line in enumerate(manual_path.read_text(encoding="utf-8", errors="replace").splitlines()[:100], 1):
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
            manual.append({"text": query, "relevant": ids, "kind": "manual", "source": ", ".join(files_rel)})

    for n, q in enumerate(manual + auto, 1):
        corpus.queries.append({"id": f"u{n:03d}", **q})
    corpus.stats = {"files": len(digests), "documents": len(corpus.documents), "queries": len(corpus.queries),
                    "queries_manual": len(manual), "queries_titles": len(auto),
                    "words": sum(len(d["text"].split()) for d in corpus.documents)}
    if len(corpus.documents) < MIN_DOCS or len(corpus.queries) < MIN_QUERIES:
        raise CorpusError(
            f"Muy poco material: {len(corpus.documents)} fragmentos y {len(corpus.queries)} consultas (minimo {MIN_DOCS} y {MIN_QUERIES}). "
            f"Hacen falta documentos Markdown con secciones tituladas, o un archivo {MANUAL_FILE}.")
    return corpus

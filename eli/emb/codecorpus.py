"""Extraccion de fragmentos de codigo para evaluar recuperacion de codigo (sin dependencias externas).

Una *unidad* es una funcion, metodo, clase/struct o bloque de un archivo. Si es muy larga se parte en ventanas (varios
fragmentos de la misma unidad). Consultas verificables por construccion: el docstring (Python) o el comentario justo encima
(C/C++/CUDA) describe la unidad y se QUITA del fragmento, para que la consulta no este copiada en la respuesta.
Unidades sin descripcion siguen siendo documentos (distractores realistas). No ejecuta ni importa el codigo del usuario.
"""
from __future__ import annotations

import ast
import re

PY_EXTS = {".py"}
CPP_EXTS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh"}
CMAKE_EXTS = {".cmake"}
MSG_EXTS = {".msg", ".srv"}
CODE_EXTS = PY_EXTS | CPP_EXTS | CMAKE_EXTS | MSG_EXTS
CODE_NAMES = {"CMakeLists.txt"}
CODE_SKIP_DIRS = {"install", "log", "logs", "CMakeFiles", "third_party", "external", "vendor", "thirdparty", "3rdparty"}

MAX_CHUNK_LINES, WINDOW_LINES = 80, 60
MIN_CHUNK_CHARS, MIN_CHUNK_LINES = 80, 3
MIN_QUERY_WORDS, MAX_QUERY_CHARS = 4, 300
BANNER_RE = re.compile(r"[=\-*#_~/]{4,}")
JUNK_RE = re.compile(r"copyright|license|spdx|skip` decorator|noqa|pylint|type: ignore|nolint", re.I)
TYPE_RE = re.compile(r"^\s*(?:template\s*<.*?>\s*)?(?:class|struct|union|enum)\b")
NS_RE = re.compile(r'^\s*(?:namespace\b[^;{]*|extern\s+"C(?:\+\+)?"\s*)\{\s*$')


def language_of(name: str, suffix: str) -> str | None:
    if suffix in PY_EXTS:
        return "python"
    if suffix in CPP_EXTS:
        return "cuda" if suffix in {".cu", ".cuh"} else "c/c++"
    if suffix in CMAKE_EXTS or name in CODE_NAMES:
        return "cmake"
    if suffix in MSG_EXTS:
        return "ros-msg"
    return None


def _windows(lines: list[str]) -> list[str]:
    if len(lines) <= MAX_CHUNK_LINES:
        return ["\n".join(lines)]
    return ["\n".join(lines[i:i + WINDOW_LINES]) for i in range(0, len(lines), WINDOW_LINES)
            if len([ln for ln in lines[i:i + WINDOW_LINES] if ln.strip()]) >= MIN_CHUNK_LINES]


def _query_from(text: str | None) -> str | None:
    """Primer parrafo de una descripcion, sin marcas de documentacion; None si es muy corta o solo lista parametros."""
    if not text:
        return None
    kept = []
    for line in text.splitlines():
        line = BANNER_RE.sub(" ", line).strip()
        if not line:
            if kept:
                break
            continue
        if re.match(r"^[@\\](param|return|returns|throws|tparam|see|note|todo)\b|^(Args|Returns|Raises|Parameters):", line, re.I):
            break
        line = re.sub(r"^[@\\](brief|details)\s*", "", line)
        kept.append(line)
    out = re.sub(r"\s+", " ", " ".join(kept)).strip()[:MAX_QUERY_CHARS]
    if len(out.split()) < MIN_QUERY_WORDS or not re.search(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]{3}", out):
        return None
    if JUNK_RE.search(out):
        return None
    return out


def _unit(symbol: str, lang: str, lines: list[str], query: str | None, kind: str | None) -> dict | None:
    body = [ln.rstrip() for ln in lines]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    text = "\n".join(body)
    if len([ln for ln in body if ln.strip()]) < MIN_CHUNK_LINES and len(text) < MIN_CHUNK_CHARS:
        return None
    pieces = _windows(body)
    if not pieces:
        return None
    return {"symbol": symbol, "lang": lang, "pieces": pieces, "query": query, "query_kind": kind if query else None}


# --- Python ---------------------------------------------------------------------------

def _py_comment_above(lines: list[str], start: int) -> str | None:
    """Comentarios `#` contiguos justo encima de la linea `start` (indice base 0)."""
    out, i = [], start - 1
    while i >= 0 and lines[i].lstrip().startswith("#") and not lines[i].lstrip().startswith("#!"):
        out.append(lines[i].lstrip().lstrip("#").strip())
        i -= 1
    return "\n".join(reversed(out)) or None


def _py_units(text: str) -> list[dict]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _generic_units(text, "python")
    lines = text.splitlines()
    units: list[dict] = []

    def visit(node, prefix: str) -> None:
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = (child.decorator_list[0].lineno if child.decorator_list else child.lineno) - 1
                end = child.end_lineno or child.lineno
                doc = ast.get_docstring(child, clean=True)
                body = lines[start:end]
                kind = "docstring"
                first = child.body[0] if child.body else None
                if doc and isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                    a, b = first.lineno - 1 - start, (first.end_lineno or first.lineno) - start
                    body = body[:a] + body[b:]  # el docstring no puede quedar en la respuesta
                else:
                    doc, kind = _py_comment_above(lines, start), "comentario"
                unit = _unit(f"{prefix}{child.name}", "python", body, _query_from(doc), kind)
                if unit:
                    units.append(unit)
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")

    visit(tree, "")
    return units


# --- C / C++ / CUDA -------------------------------------------------------------------

def _strip_code(line: str, state: dict) -> str:
    """La linea sin comentarios, cadenas ni literales de caracter (para contar llaves). `state['block']` sigue /* */."""
    out, i, n = [], 0, len(line)
    while i < n:
        if state["block"]:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out)
            state["block"], i = False, j + 2
            continue
        two = line[i:i + 2]
        if two == "/*":
            state["block"], i = True, i + 2
        elif two == "//":
            break
        elif line[i] in "\"'":
            q, i = line[i], i + 1
            while i < n and line[i] != q:
                i += 2 if line[i] == "\\" else 1
            i += 1
        else:
            out.append(line[i])
            i += 1
    return "".join(out)


def _comment_block_above(lines: list[str], start: int) -> tuple[int, str | None]:
    """(primera linea del comentario, texto) del comentario contiguo encima de `start`. Sin comentario: (start, None)."""
    i, found = start - 1, []
    if i >= 0 and lines[i].strip().endswith("*/"):
        j = i
        while j >= 0 and "/*" not in lines[j]:
            j -= 1
        if j >= 0:
            raw = lines[j:i + 1]
            cleaned = [re.sub(r"^\s*(/\*+!?|\*+/?|\*)\s?", "", re.sub(r"\*/\s*$", "", ln)) for ln in raw]
            return j, "\n".join(cleaned)
    while i >= 0 and re.match(r"^\s*//", lines[i]):
        found.append(re.sub(r"^\s*//[/!<]?\s?", "", lines[i]))
        i -= 1
    if not found:
        return start, None
    return i + 1, "\n".join(reversed(found))


def _cpp_units(text: str, lang: str) -> list[dict]:
    lines = text.splitlines()
    state = {"block": False}
    units: list[dict] = []
    depth, start, head, i = 0, None, [], 0
    continued_define = False
    while i < len(lines):
        raw = lines[i]
        code = _strip_code(raw, state)
        stripped = code.strip()
        if depth == 0 and start is None:
            if continued_define or stripped.startswith("#"):
                continued_define = raw.rstrip().endswith("\\")
                i += 1
                continue
            if not stripped:
                i += 1
                continue
            if NS_RE.match(code) or (stripped.endswith("{") and re.match(r'^(?:namespace\b|extern\s+"C)', stripped)):
                i += 1  # los espacios de nombres son transparentes: lo de dentro es de nivel superior
                continue
            if stripped == "}" or stripped.startswith("}") and not stripped.strip("};, "):
                i += 1  # cierre de un espacio de nombres
                continue
            start, head = i, []
        if start is not None:
            if not head or "{" not in "".join(head):
                head.append(code)
            depth += code.count("{") - code.count("}")
            depth = max(depth, 0)
            ended = depth == 0 and (stripped.endswith(";") or stripped.endswith("}") or stripped.endswith("};"))
            if ended:
                _finish_cpp_unit(lines, start, i, head, lang, units)
                start, head = None, []
        i += 1
    return units


def _finish_cpp_unit(lines: list[str], start: int, end: int, head: list[str], lang: str, units: list[dict]) -> None:
    head_text = " ".join(head)
    before = head_text.split("{", 1)[0]
    if "{" not in head_text:
        return  # declaracion sin cuerpo
    is_type = bool(TYPE_RE.match(before))
    if not is_type and "(" not in before:
        return  # variable global inicializada, enum anonimo, etc.
    first, comment = _comment_block_above(lines, start)
    symbol = _cpp_symbol(before, is_type)
    unit = _unit(symbol, lang, lines[start:end + 1], _query_from(comment), "comentario")
    if unit:
        units.append(unit)


def _cpp_symbol(before: str, is_type: bool) -> str:
    before = re.sub(r"\s+", " ", before).strip()
    if is_type:
        m = re.search(r"\b(?:class|struct|union|enum)\s+(?:class\s+)?(\w+)", before)
        return m.group(1) if m else "tipo"
    m = re.search(r"([~\w:<>,]+)\s*\(", before)
    return m.group(1) if m else "funcion"


# --- CMake, mensajes ROS y respaldo generico --------------------------------------------

def _generic_units(text: str, lang: str, target_chars: int = 500) -> list[dict]:
    """Bloques separados por lineas en blanco, juntados hasta ~`target_chars`. Sin consulta automatica."""
    units, cur = [], []
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        cur.append(block)
        if sum(len(b) for b in cur) >= target_chars:
            unit = _unit(f"bloque{len(units) + 1}", lang, "\n\n".join(cur).splitlines(), None, None)
            if unit:
                units.append(unit)
            cur = []
    if cur:
        unit = _unit(f"bloque{len(units) + 1}", lang, "\n\n".join(cur).splitlines(), None, None)
        if unit:
            units.append(unit)
    return units


def extract_units(name: str, suffix: str, text: str) -> list[dict]:
    """Unidades de un archivo de codigo. `suffix` en minusculas. Lista vacia si no es un tipo conocido."""
    lang = language_of(name, suffix)
    if lang is None:
        return []
    if lang == "python":
        return _py_units(text)
    if lang in ("c/c++", "cuda"):
        return _cpp_units(text, lang)
    if lang == "ros-msg":
        unit = _unit(name, lang, text.splitlines(), None, None)
        return [unit] if unit else []
    return _generic_units(text, lang)

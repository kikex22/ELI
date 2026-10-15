"""Calificadores con criterio verificable. Ningun calificador usa un modelo como juez.

Estados de una calificacion:
  passed    cumple todos los criterios.
  failed    se pudo evaluar y no cumple.
  ungraded  no se puede evaluar con fiabilidad (respuesta cortada, ambigua, sin
            aislamiento para ejecutar codigo...). No cuenta como acierto ni fallo.

Una respuesta que falla pero fue cortada por el limite de tokens pasa a
`ungraded`: no se puede afirmar que sea incorrecta.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from . import sandbox

PASSED, FAILED, UNGRADED = "passed", "failed", "ungraded"


@dataclass
class Grade:
    status: str
    reason: str = ""
    checks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "checks": self.checks}


def _check(name: str, ok: bool | None, detail: str = "", **extra) -> dict:
    return {"check": name, "ok": ok, "detail": detail, **extra}


def _grade_from_checks(checks: list[dict]) -> Grade:
    """Falla si algun criterio falla; sin calificar si alguno no se pudo evaluar; si no, pasa."""
    failed = [c for c in checks if c["ok"] is False]
    if failed:
        return Grade(FAILED, "; ".join(c["detail"] or c["check"] for c in failed), checks)
    unknown = [c for c in checks if c["ok"] is None]
    if unknown:
        return Grade(UNGRADED, "; ".join(c["detail"] or c["check"] for c in unknown), checks)
    return Grade(PASSED, "", checks)


# --- utilidades de texto ---------------------------------------------------

def strip_thinking(text: str) -> tuple[str, bool]:
    """Quita bloques <think>...</think>. Devuelve (respuesta, razonamiento_sin_cerrar)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    unclosed = bool(re.search(r"<think>", text, re.I))
    if unclosed:
        text = re.split(r"<think>", text, flags=re.I)[0]
    return text.strip(), unclosed


def fold(text: str) -> str:
    """Minusculas, sin acentos, sin marcas de Markdown y con espacios colapsados."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[*_`#>]+", "", text.casefold())
    return " ".join(text.split())


_TRIM = " .,;:!?¡¿\"'“”«»()[]{}"


def trim(text: str) -> str:
    return fold(text).strip(_TRIM)


def words(text: str) -> list[str]:
    return re.findall(r"\w+(?:['’-]\w+)*", text, flags=re.U)


def nonblank_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def unfence(text: str) -> str:
    """Si toda la respuesta es un unico bloque de codigo, devuelve su contenido."""
    m = re.fullmatch(r"```[\w+-]*[ \t]*\n?(.*?)\n?```", text.strip(), re.S)
    return m.group(1).strip() if m else text.strip()


# --- idioma (heuristica por palabras funcionales) ---------------------------

_ES = set("de la que el en y los del se las por un para con una su al lo como mas pero sus le ya o este si porque esta entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante es son esta estan fue ser tiene tienen hacer cada otro otra".split())
_EN = set("the of and to in is that it was for on are as with his they at be this have from or one had by but not what all were we when your can there an which their if do will each about how up out them then she many some so these would other into has her two like him see time could make than first been its who now people my made over did only way find use may long little very after where most know".split())
_COMMON = _ES & _EN
_ES -= _COMMON
_EN -= _COMMON


def detect_language(text: str) -> dict:
    tokens = [fold(w) for w in words(text)]
    n = len(tokens)
    es = sum(1 for t in tokens if t in _ES)
    en = sum(1 for t in tokens if t in _EN)
    ratio_es, ratio_en = (es / n, en / n) if n else (0.0, 0.0)
    lang = "es" if ratio_es > ratio_en else "en" if ratio_en > ratio_es else None
    return {"words": n, "es": round(ratio_es, 3), "en": round(ratio_en, 3), "lang": lang}


# --- numeros ----------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")


def parse_number(token: str) -> float | None:
    s = token
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thousands = "." if dec == "," else ","
        s = s.replace(thousands, "").replace(dec, ".")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


# --- JSON -------------------------------------------------------------------

def parse_json_answer(answer: str, allow_fence: bool = True):
    text = unfence(answer) if allow_fence else answer.strip()
    try:
        return json.loads(text), None
    except ValueError as exc:
        return None, f"no es JSON valido o tiene texto adicional ({str(exc)[:60]})"


def _type_ok(value, expected) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    for kind in kinds:
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "object" and isinstance(value, dict):
            return True
        if kind == "null" and value is None:
            return True
    return False


def validate_schema(schema: dict, value, path: str = "$") -> list[str]:
    """Subconjunto de JSON Schema: type, required, properties, additionalProperties, items, enum, min/max."""
    errors: list[str] = []
    if "type" in schema and not _type_ok(value, schema["type"]):
        return [f"{path}: se esperaba {schema['type']}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: valor fuera de {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: menor que {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: mayor que {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: demasiado corto")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: demasiado largo")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: falta la clave '{key}'")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                errors += validate_schema(sub, value[key], f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: clave no permitida '{key}'")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: menos de {schema['minItems']} elementos")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: mas de {schema['maxItems']} elementos")
        if "items" in schema:
            for i, item in enumerate(value):
                errors += validate_schema(schema["items"], item, f"{path}[{i}]")
    return errors


def get_path(value, path: str):
    """'a.b.0.c' sobre dicts y listas; '$' es la raiz. Lanza KeyError/IndexError si no existe."""
    if path in ("$", ""):
        return value
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _equal(got, expected, fold_strings: bool) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return got is expected
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return abs(got - expected) < 1e-9
    if isinstance(expected, str) and isinstance(got, str):
        return fold(got) == fold(expected) if fold_strings else got == expected
    if isinstance(expected, list) and isinstance(got, list):
        return len(got) == len(expected) and all(_equal(g, e, fold_strings) for g, e in zip(got, expected))
    return got == expected


# --- calificadores ----------------------------------------------------------

def g_exact(spec: dict, answer: str) -> Grade:
    expected = spec["expected"]
    expected = [expected] if isinstance(expected, str) else list(expected)
    strip_spaces = spec.get("strip_spaces", False)

    def prep(text: str) -> str:
        if spec.get("case_sensitive"):
            text = text.strip().strip("`*\"'“”").strip()
            text = " ".join(text.split())
        else:
            text = trim(text)
        return re.sub(r"\s+", "", text) if strip_spaces else text

    got = prep(answer)
    ok = any(got == prep(e) or (not e.endswith(".") and got.rstrip(".") == prep(e)) for e in expected)
    return _grade_from_checks([_check("exact", ok, f"se esperaba {expected[0]!r}, se obtuvo {answer.strip()[:80]!r}")])


def g_candidates(spec: dict, answer: str) -> Grade:
    """Respuesta de una palabra entre opciones conocidas; tolera una frase alrededor."""
    candidates = [fold(c) for c in spec["candidates"]]
    expected = [fold(e) for e in ([spec["expected"]] if isinstance(spec["expected"], str) else spec["expected"])]
    text = fold(answer)
    text = re.sub(r"^(respuesta|answer|solucion|resultado)\s*[:\-]\s*", "", text)
    tokens = re.findall(r"[\w']+", text)
    picked = tokens[0] if tokens and tokens[0] in candidates else None
    if picked is None:
        present = sorted({c for c in candidates if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text)})
        if len(present) > 1:
            return Grade(UNGRADED, f"respuesta ambigua: menciona {present}", [_check("candidates", None, "ambigua")])
        if not present:
            return Grade(FAILED, "no dio ninguna de las respuestas posibles", [_check("candidates", False, "sin respuesta reconocible")])
        picked = present[0]
    ok = picked in expected
    return _grade_from_checks([_check("candidates", ok, f"se esperaba {expected[0]!r}, eligio {picked!r}")])


def g_letters(spec: dict, answer: str) -> Grade:
    """Conjunto de letras (A, B, C...) dado en la ultima linea; 'ninguno' si no hay ninguna."""
    candidates = spec["candidates"]
    expected = set(spec["expected"])
    last = nonblank_lines(answer)[-1]
    found = set(re.findall(rf"(?<![\w])([{''.join(candidates)}])(?![\w])", last))
    if not found and re.search(r"ninguno|nadie|none|no one", fold(last)):
        found = set()
    elif not found:
        found = set(re.findall(rf"(?<![\w])([{''.join(candidates)}])(?![\w])", answer))
        if not found:
            return Grade(FAILED, "no se identifico ninguna letra", [_check("letters", False, "sin letras")])
    return _grade_from_checks([_check("letters", found == expected, f"se esperaba {sorted(expected)}, se obtuvo {sorted(found)}")])


def g_number(spec: dict, answer: str) -> Grade:
    expected, tol = float(spec["expected"]), float(spec.get("tolerance", 0))
    nums = [parse_number(t) for t in _NUM.findall(answer)]
    nums = [n for n in nums if n is not None]
    if not nums:
        return Grade(FAILED, "la respuesta no contiene ningun numero", [_check("number", False, "sin numero")])
    got = nums[-1] if spec.get("mode", "last") == "last" else nums[0]
    ok = abs(got - expected) <= tol
    return _grade_from_checks([_check("number", ok, f"se esperaba {expected:g}, se obtuvo {got:g}")])


def _contains(spec: dict, answer: str, mode: str) -> Grade:
    items = spec["expected"] if isinstance(spec["expected"], list) else [spec["expected"]]
    case = spec.get("case_sensitive", False)
    hay = answer if case else fold(answer)
    needles = items if case else [fold(i) for i in items]
    hits = [n in hay for n in needles]
    ok = {"all": all(hits), "any": any(hits), "none": not any(hits)}[mode]
    label = {"all": "debe contener todo", "any": "debe contener alguno de", "none": "no debe contener"}[mode]
    return _grade_from_checks([_check(f"contains_{mode}", ok, f"{label} {items}")])


def g_contains_all(spec, answer): return _contains(spec, answer, "all")
def g_contains_any(spec, answer): return _contains(spec, answer, "any")
def g_not_contains(spec, answer): return _contains(spec, answer, "none")


def g_regex(spec: dict, answer: str) -> Grade:
    flags = sum({"i": re.I, "m": re.M, "s": re.S}.get(f, 0) for f in spec.get("flags", ""))
    search = re.fullmatch if spec.get("mode") == "fullmatch" else re.search
    ok = search(spec["pattern"], answer.strip(), flags) is not None
    return _grade_from_checks([_check("regex", ok, f"debe coincidir con /{spec['pattern']}/")])


def g_json(spec: dict, answer: str) -> Grade:
    value, err = parse_json_answer(answer, spec.get("allow_fence", True))
    if err:
        return Grade(FAILED, err, [_check("json", False, err)])
    checks = []
    if "schema" in spec:
        errors = validate_schema(spec["schema"], value)
        checks.append(_check("schema", not errors, "; ".join(errors[:3])))
    for path, expected in spec.get("equals", {}).items():
        try:
            got = get_path(value, path)
            ok = _equal(got, expected, spec.get("fold_strings", True))
            checks.append(_check(f"equals {path}", ok, f"{path}: se esperaba {expected!r}, se obtuvo {got!r}"))
        except (KeyError, IndexError, ValueError, TypeError):
            checks.append(_check(f"equals {path}", False, f"{path}: no existe"))
    return _grade_from_checks(checks)


def g_lines_equal(spec: dict, answer: str) -> Grade:
    text = unfence(answer) if spec.get("allow_fence", True) else answer
    got = nonblank_lines(text)
    expected = spec["lines"]
    ok = got == expected
    return _grade_from_checks([_check("lines_equal", ok, f"se esperaban {expected}, se obtuvo {got[:6]}")])


def g_table_md(spec: dict, answer: str) -> Grade:
    text = unfence(answer)
    rows = []
    for line in nonblank_lines(text):
        if not line.startswith("|"):
            return Grade(FAILED, f"linea fuera de la tabla: {line[:40]!r}", [_check("table_md", False, "texto adicional")])
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        rows.append([fold(c) for c in cells])
    expected = [[fold(c) for c in spec["header"]]] + [[fold(str(c)) for c in r] for r in spec["rows"]]
    return _grade_from_checks([_check("table_md", rows == expected, f"se esperaba {expected}, se obtuvo {rows}")])


def _count(name: str, got: int, spec: dict, what: str) -> list[dict]:
    checks = []
    if "count" in spec:
        checks.append(_check(name, got == spec["count"], f"{what}: se esperaban {spec['count']}, hay {got}"))
    if "min" in spec:
        checks.append(_check(name + "_min", got >= spec["min"], f"{what}: minimo {spec['min']}, hay {got}"))
    if "max" in spec:
        checks.append(_check(name + "_max", got <= spec["max"], f"{what}: maximo {spec['max']}, hay {got}"))
    return checks


def run_check(spec: dict, answer: str) -> list[dict]:
    """Un criterio de instruccion verificable. Devuelve una o mas comprobaciones."""
    kind = spec["type"]
    if kind == "words":
        return _count("words", len(words(answer)), spec, "palabras")
    if kind == "chars":
        return _count("chars", len(answer.strip()), spec, "caracteres")
    if kind == "lines":
        return _count("lines", len(nonblank_lines(answer)), spec, "lineas")
    if kind == "paragraphs":
        return _count("paragraphs", len(paragraphs(answer)), spec, "parrafos")
    if kind == "sentences":
        n = len([s for s in re.split(r"[.!?…]+(?:\s|$)", answer.strip()) if s.strip()])
        return _count("sentences", n, spec, "frases")
    if kind in ("starts_with", "ends_with"):
        text, want = answer.strip(), spec["text"]
        if not spec.get("case_sensitive", True):
            text, want = fold(text), fold(want)
        ok = text.startswith(want) if kind == "starts_with" else text.endswith(want)
        return [_check(kind, ok, f"debe {'empezar' if kind == 'starts_with' else 'terminar'} con {spec['text']!r}")]
    if kind == "forbidden_chars":
        bad = sorted({c for c in spec["chars"] if c.casefold() in answer.casefold()})
        return [_check(kind, not bad, f"usa letras prohibidas {bad}")]
    if kind == "forbidden_words":
        toks = {fold(w) for w in words(answer)}
        bad = sorted(w for w in spec["words"] if fold(w) in toks)
        return [_check(kind, not bad, f"usa palabras prohibidas {bad}")]
    if kind == "required_words":
        toks = {fold(w) for w in words(answer)}
        missing = [w for w in spec["words"] if fold(w) not in toks]
        return [_check(kind, not missing, f"faltan palabras {missing}")]
    if kind == "paragraph_contains":
        paras = paragraphs(answer)
        if spec["index"] >= len(paras):
            return [_check(kind, False, f"no existe el parrafo {spec['index'] + 1}")]
        toks = {fold(w) for w in words(paras[spec["index"]])}
        missing = [w for w in spec["words"] if fold(w) not in toks]
        return [_check(kind, not missing, f"parrafo {spec['index'] + 1}: faltan {missing}")]
    if kind == "uppercase":
        letters = [c for c in answer if c.isalpha()]
        return [_check(kind, bool(letters) and all(c.isupper() for c in letters), "todo debe estar en mayusculas")]
    if kind == "line_regex":
        bad = [ln for ln in nonblank_lines(answer) if not re.search(spec["pattern"], ln)]
        return [_check(kind, not bad, f"lineas que no cumplen /{spec['pattern']}/: {[b[:30] for b in bad[:3]]}")]
    if kind == "no_list_markers":
        bad = [ln for ln in nonblank_lines(answer) if re.match(r"^(?:[-*•–]|\d+[.)])\s", ln)]
        return [_check(kind, not bad, "hay lineas con vinetas o numeracion")]
    if kind == "language":
        info = detect_language(answer)
        if info["words"] < spec.get("min_words", 4):
            return [_check(kind, None, "texto demasiado corto para detectar el idioma", info=info)]
        other = info["en"] if spec["lang"] == "es" else info["es"]
        mine = info[spec["lang"]]
        return [_check(kind, mine > other, f"idioma esperado {spec['lang']}, detectado {info['lang']}", info=info)]
    raise ValueError(f"criterio desconocido: {kind}")


def g_checks(spec: dict, answer: str) -> Grade:
    checks: list[dict] = []
    for sub in spec["checks"]:
        checks += run_check(sub, answer)
    return _grade_from_checks(checks)


def g_all(spec: dict, answer: str) -> Grade:
    checks: list[dict] = []
    for sub in spec["graders"]:
        g = GRADERS[sub["type"]](sub, answer)
        for c in g.checks:
            checks.append(c)
        if not g.checks:
            checks.append(_check(sub["type"], {"passed": True, "failed": False}.get(g.status), g.reason))
    return _grade_from_checks(checks)


def extract_code(answer: str, function: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", answer, re.S | re.I)
    with_fn = [b for b in blocks if f"def {function}" in b]
    if with_fn:
        return with_fn[-1]
    if blocks:
        return blocks[-1]
    return answer if f"def {function}" in answer else None


def g_code(spec: dict, answer: str) -> Grade:
    code = extract_code(answer, spec["function"])
    if code is None:
        return Grade(FAILED, "no se encontro codigo en la respuesta", [_check("code", False, "sin codigo")])
    result = sandbox.run_function_tests(
        code, spec["function"], spec["tests"], timeout_s=spec.get("timeout_s", 10.0)
    )
    if not result.ran:
        return Grade(UNGRADED, result.error or "el codigo no se pudo ejecutar", [_check("code", None, result.error or "")])
    shown = result.cases[:8]
    if result.timed_out or result.error:
        return Grade(FAILED, result.error or "error de ejecucion", [_check("code", False, result.error or "", cases=shown)])
    ok = result.passed == result.total
    detail = f"{result.passed}/{result.total} casos correctos"
    if not ok:
        first = next((c for c in result.cases if not c.get("ok")), None)
        if first:
            detail += f"; primer fallo: {first['args']!r} -> {first.get('got')!r} (esperado {first['expected']!r})" + (
                f" [{first['error']}]" if first.get("error") else ""
            )
    return _grade_from_checks([_check("code", ok, detail, cases=shown)])


GRADERS = {
    "exact": g_exact,
    "candidates": g_candidates,
    "letters": g_letters,
    "number": g_number,
    "contains_all": g_contains_all,
    "contains_any": g_contains_any,
    "not_contains": g_not_contains,
    "regex": g_regex,
    "json": g_json,
    "lines_equal": g_lines_equal,
    "table_md": g_table_md,
    "checks": g_checks,
    "all": g_all,
    "code": g_code,
}

REQUIRED_KEYS = {
    "exact": ["expected"],
    "candidates": ["candidates", "expected"],
    "letters": ["candidates", "expected"],
    "number": ["expected"],
    "contains_all": ["expected"],
    "contains_any": ["expected"],
    "not_contains": ["expected"],
    "regex": ["pattern"],
    "json": [],
    "lines_equal": ["lines"],
    "table_md": ["header", "rows"],
    "checks": ["checks"],
    "all": ["graders"],
    "code": ["function", "tests"],
}


def validate_spec(spec: dict) -> list[str]:
    """Errores de forma de un calificador (para validar el banco al cargarlo)."""
    kind = spec.get("type")
    if kind not in GRADERS:
        return [f"tipo de calificador desconocido: {kind!r}"]
    errors = [f"{kind}: falta '{k}'" for k in REQUIRED_KEYS[kind] if k not in spec]
    if kind == "all":
        for sub in spec.get("graders", []):
            errors += validate_spec(sub)
    return errors


def grade(spec: dict, response: str, done_reason: str | None = None) -> Grade:
    """Califica una respuesta completa (con razonamiento incluido si lo hubo)."""
    answer, unclosed = strip_thinking(response)
    truncated = done_reason == "length"
    if not answer.strip():
        if unclosed:
            return Grade(UNGRADED, "el razonamiento no termino: no hay respuesta final")
        if truncated:
            return Grade(UNGRADED, "respuesta vacia por el limite de tokens")
        return Grade(FAILED, "respuesta vacia")
    fn = GRADERS.get(spec.get("type"))
    if fn is None:
        return Grade(UNGRADED, f"tipo de calificador desconocido: {spec.get('type')!r}")
    try:
        result = fn(spec, answer)
    except Exception as exc:  # noqa: BLE001 - un calificador roto no debe tumbar la ejecucion
        return Grade(UNGRADED, f"error del calificador: {type(exc).__name__}: {str(exc)[:80]}")
    if result.status == FAILED and truncated:
        return Grade(
            UNGRADED, "respuesta cortada por el limite de tokens: no se puede afirmar que sea incorrecta", result.checks
        )
    return result

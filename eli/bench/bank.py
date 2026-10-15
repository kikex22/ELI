"""Banco de pruebas versionado y reproducible.

Una suite es un JSON en `suites/` con version y categorias. Cada prueba lleva
id unico, categoria, dificultad, prompt, criterio verificable (calificador),
parametros de ejecucion y si pertenece al benchmark rapido. El hash SHA-256 de
la suite entera se guarda en cada ejecucion.

Las pruebas de contexto son plantillas (`generator`): el prompt se construye de
forma determinista para cada tamano de contexto pedido.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from . import graders

SUITES_DIR = Path(__file__).resolve().parent / "suites"
DEFAULT_SUITE_ID = "eli-core"

DIFFICULTIES = ("easy", "medium", "hard")
PARAM_KEYS = {"num_ctx", "num_predict", "temperature", "seed"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

DEFAULT_CONTEXT_SIZES_FULL = [2048, 4096, 8192]
DEFAULT_CONTEXT_SIZES_QUICK = [2048]
FILL_RATIO = 0.6  # fraccion del contexto que se intenta llenar con el texto
CHARS_PER_TOKEN = 3.3  # estimacion conservadora; los tokens reales se miden


class SuiteError(ValueError):
    pass


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # no es una prueba de pytest

    id: str
    category: str
    difficulty: str
    prompt: str
    grader: dict
    params: dict
    language: str
    quick: bool
    expected: str
    system: str | None = None
    generator: dict | None = None
    context_sized: bool = False
    reference: str | None = None


@dataclass(frozen=True)
class Instance:
    """Una prueba lista para ejecutar (con el prompt ya construido)."""

    test: TestCase
    instance_id: str
    prompt: str
    system: str | None
    grader: dict
    params: dict
    context_size: int | None

    @property
    def category(self) -> str:
        return self.test.category


@dataclass(frozen=True)
class Suite:
    id: str
    version: str
    description: str
    categories: dict
    tests: tuple
    sha256: str
    path: Path

    def info(self) -> dict:
        by_cat: dict = {}
        for t in self.tests:
            c = by_cat.setdefault(t.category, {"name": self.categories[t.category], "total": 0, "quick": 0,
                                               "easy": 0, "medium": 0, "hard": 0, "context_sized": 0})
            c["total"] += 1
            c["quick"] += 1 if t.quick else 0
            c[t.difficulty] += 1
            c["context_sized"] += 1 if t.context_sized else 0
        return {"id": self.id, "version": self.version, "description": self.description, "sha256": self.sha256,
                "tests": len(self.tests), "quick_tests": sum(1 for t in self.tests if t.quick), "categories": by_cat}

    def get(self, test_id: str) -> TestCase | None:
        return next((t for t in self.tests if t.id == test_id), None)

    def select(self, mode: str = "quick", categories: list[str] | None = None) -> list[TestCase]:
        if mode not in ("quick", "full"):
            raise SuiteError("mode debe ser 'quick' o 'full'")
        if categories:
            unknown = [c for c in categories if c not in self.categories]
            if unknown:
                raise SuiteError(f"categorias desconocidas: {unknown}")
        chosen = [t for t in self.tests if (mode == "full" or t.quick) and (not categories or t.category in categories)]
        order = list(self.categories)
        return sorted(chosen, key=lambda t: (order.index(t.category), t.id))

    def instances(
        self,
        tests: list[TestCase],
        context_sizes: list[int] | None = None,
        num_ctx: int | None = None,
        marker_salt: str = "",
        mode: str = "quick",
    ) -> list[Instance]:
        """Expande las pruebas: las de contexto una vez por tamano. Orden: por num_ctx y luego categoria."""
        if context_sizes is None:
            context_sizes = DEFAULT_CONTEXT_SIZES_QUICK if mode == "quick" else DEFAULT_CONTEXT_SIZES_FULL
        out: list[Instance] = []
        for t in tests:
            if t.context_sized:
                for size in sorted(set(context_sizes)):
                    out.append(build_context_instance(self, t, size, marker_salt))
            else:
                params = dict(t.params)
                if num_ctx:
                    params["num_ctx"] = num_ctx
                out.append(Instance(t, t.id, t.prompt, t.system, t.grader, params, None))
        order = list(self.categories)
        return sorted(out, key=lambda i: (i.params.get("num_ctx", 4096), order.index(i.category), i.instance_id))


# --- carga y validacion -------------------------------------------------------

def _canonical_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def parse_suite(data: dict, path: Path | None = None) -> Suite:
    for key in ("suite_id", "version", "categories", "tests"):
        if key not in data:
            raise SuiteError(f"falta '{key}' en la suite")
    categories = data["categories"]
    if not isinstance(categories, dict) or not categories:
        raise SuiteError("'categories' debe ser un objeto no vacio")
    tests, seen = [], set()
    for raw in data["tests"]:
        tid = raw.get("id")
        if not isinstance(tid, str) or not ID_RE.match(tid):
            raise SuiteError(f"id de prueba invalido: {tid!r}")
        if tid in seen:
            raise SuiteError(f"id duplicado: {tid}")
        seen.add(tid)
        if raw.get("category") not in categories:
            raise SuiteError(f"{tid}: categoria desconocida {raw.get('category')!r}")
        if raw.get("difficulty") not in DIFFICULTIES:
            raise SuiteError(f"{tid}: dificultad invalida {raw.get('difficulty')!r}")
        generator = raw.get("generator")
        prompt = raw.get("prompt", "")
        if not generator and not (isinstance(prompt, str) and prompt.strip()):
            raise SuiteError(f"{tid}: falta el prompt")
        if generator and generator.get("type") != "needle":
            raise SuiteError(f"{tid}: generador desconocido {generator.get('type')!r}")
        errors = graders.validate_spec(raw.get("grader") or {})
        if errors:
            raise SuiteError(f"{tid}: {'; '.join(errors)}")
        params = raw.get("params", {})
        if not set(params) <= PARAM_KEYS:
            raise SuiteError(f"{tid}: parametros no permitidos {sorted(set(params) - PARAM_KEYS)}")
        if not raw.get("expected"):
            raise SuiteError(f"{tid}: falta la descripcion de la respuesta esperada ('expected')")
        tests.append(TestCase(
            id=tid, category=raw["category"], difficulty=raw["difficulty"], prompt=prompt,
            grader=raw["grader"], params=params, language=raw.get("language", "es"), quick=bool(raw.get("quick", False)),
            expected=raw["expected"], system=raw.get("system"), generator=generator,
            context_sized=bool(raw.get("context_sized", False)), reference=raw.get("reference"),
        ))
    return Suite(
        id=data["suite_id"], version=data["version"], description=data.get("description", ""),
        categories=categories, tests=tuple(tests), sha256=_canonical_hash(data), path=path or Path("<memoria>"),
    )


def load_suite(path: Path) -> Suite:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SuiteError(f"no se pudo leer la suite {path.name}: {exc}") from exc
    return parse_suite(data, path)


def default_suite(suites_dir: Path = SUITES_DIR) -> Suite:
    """La version mas reciente de la suite por defecto (orden por numero de version)."""
    candidates = []
    for path in sorted(suites_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("suite_id") == DEFAULT_SUITE_ID:
            version = tuple(int(p) if p.isdigit() else 0 for p in str(data.get("version", "0")).split("."))
            candidates.append((version, path))
    if not candidates:
        raise SuiteError(f"no hay suites '{DEFAULT_SUITE_ID}' en {suites_dir}")
    return load_suite(max(candidates)[1])


# --- pruebas de contexto largo --------------------------------------------------

_FILLER = {
    "es": [
        "El almacén {n} registró {k} paquetes el día {d} del mes sin incidencias.",
        "La reunión del equipo {t} comenzó a las {h}:{m} y terminó sin acuerdos pendientes.",
        "El informe número {k} describe el estado de la sala {r} como normal.",
        "La ruta {n} conecta la estación {a} con el puerto {b} todos los días laborables.",
        "El técnico {t} revisó {k} equipos del edificio {r} durante la mañana.",
        "Según el inventario {n}, quedan {k} cajas en el pasillo {a}.",
    ],
    "en": [
        "Warehouse {n} logged {k} parcels on day {d} of the month without incident.",
        "The {t} team meeting started at {h}:{m} and ended with no pending items.",
        "Report number {k} describes the condition of room {r} as normal.",
        "Route {n} links station {a} with harbour {b} on every working day.",
        "Technician {t} inspected {k} machines in building {r} during the morning.",
        "According to inventory {n}, {k} boxes remain in aisle {a}.",
    ],
}
_NAMES = ["Norte", "Sur", "Este", "Oeste", "Central", "Delta", "Omega", "Alfa", "Bravo", "Gamma", "Sigma", "Kappa"]
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sin 0/O ni 1/I para no confundir
_NEEDLE = {
    "es": ("La contraseña secreta de la sala {room} es {code}.",
           "Pregunta: según el texto anterior, ¿cuál es la contraseña secreta de la sala {room}? Responde solo con el código.",
           "Lee el texto siguiente con atencion."),
    "en": ("The secret password of room {room} is {code}.",
           "Question: according to the text above, what is the secret password of room {room}? Answer with the code only.",
           "Read the following text carefully."),
}


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def build_needle(generator: dict, target_tokens: int, seed_parts: tuple) -> tuple[str, str]:
    """Devuelve (prompt, codigo_esperado). Determinista para los mismos argumentos."""
    lang = generator.get("language", "es")
    rng = random.Random(_seed(*seed_parts))
    code = "".join(rng.choice(_ALPHABET) for _ in range(6))
    room = f"{rng.choice(_NAMES)}-{rng.randint(10, 99)}"
    needle_t, question_t, intro = _NEEDLE[lang]
    needle = needle_t.format(room=room, code=code)
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    sentences, total = [], 0
    while total < target_chars:
        template = rng.choice(_FILLER[lang])
        s = template.format(n=rng.randint(100, 999), k=rng.randint(2, 480), d=rng.randint(1, 28),
                            t=rng.choice(_NAMES), h=rng.randint(7, 18), m=rng.choice(["00", "15", "30", "45"]),
                            r=f"{rng.choice(_NAMES)}-{rng.randint(10, 99)}", a=rng.choice(_NAMES), b=rng.choice(_NAMES))
        sentences.append(s)
        total += len(s) + 1
    position = min(max(float(generator.get("position", 0.5)), 0.0), 1.0)
    sentences.insert(int(position * len(sentences)), needle)
    return f"{intro}\n\n{' '.join(sentences)}\n\n{question_t.format(room=room)}", code


def build_context_instance(suite: Suite, test: TestCase, size: int, marker_salt: str = "") -> Instance:
    prompt, code = build_needle(test.generator, int(size * FILL_RATIO), (suite.sha256[:12], test.id, size))
    if marker_salt:
        # Marca unica por ejecucion al principio: evita que la cache de prompts de Ollama
        # reutilice el prefijo de otra ejecucion y falsee los tokens procesados.
        marker = hashlib.sha256(f"{marker_salt}|{test.id}|{size}".encode()).hexdigest()[:8]
        prompt = f"[ref {marker}]\n{prompt}"
    params = dict(test.params)
    params["num_ctx"] = size
    grader = dict(test.grader)
    grader["expected"] = code
    return Instance(test, f"{test.id}@{size}", prompt, test.system, grader, params, size)

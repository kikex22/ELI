import os
import tempfile
import unittest
from pathlib import Path

from eli.emb import codecorpus, corpus
from eli.emb.corpus import CorpusError, build_corpus

BODY = " ".join(f"palabra{i}" for i in range(45))


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


PY_SRC = '''
def calcular_iou(a, b):
    """Calcula la interseccion sobre union entre dos cajas.

    Args:
        a: caja
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    return x1 + y1


# Convierte grados a radianes para el brazo
def a_radianes(grados):
    import math
    valor = grados * math.pi / 180
    return valor


class Robot:
    def mover(self, x):
        """Mueve la base del robot hacia la posicion indicada por el usuario."""
        self.x = x
        self.y = x
        return self.x
'''

CPP_SRC = '''#include <vector>
#define LARGO(x) \\
   (x + 1)

namespace od {

/// Aplica supresion de no maximos sobre las detecciones de un frame.
std::vector<int> nms(const std::vector<float>& scores, float thr) {
    std::vector<int> keep;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] > thr) { keep.push_back(i); }  // } comentario con llave
    }
    const char* s = "}}}";
    return keep;
}

int sin_comentario(int a,
                   int b) {
    int c = a + b;
    return c * 2;
}

void solo_declaracion(int x);

struct Caja {
    float x, y;
    float area() const { return x * y; }
};

}  // namespace od
'''


class CodeUnitTests(unittest.TestCase):
    def test_python_docstring_is_removed_and_becomes_the_query(self):
        units = {u["symbol"]: u for u in codecorpus.extract_units("m.py", ".py", PY_SRC)}
        iou = units["calcular_iou"]
        self.assertEqual(iou["query"], "Calcula la interseccion sobre union entre dos cajas.")
        self.assertEqual(iou["query_kind"], "docstring")
        self.assertNotIn("interseccion", iou["pieces"][0])
        self.assertIn("def calcular_iou", iou["pieces"][0])
        self.assertIn("Mueve la base del robot", units["Robot.mover"]["query"])
        self.assertNotIn("Mueve la base", units["Robot.mover"]["pieces"][0])

    def test_python_comment_above_is_used_when_there_is_no_docstring(self):
        unit = {u["symbol"]: u for u in codecorpus.extract_units("m.py", ".py", PY_SRC)}["a_radianes"]
        self.assertEqual(unit["query"], "Convierte grados a radianes para el brazo")
        self.assertEqual(unit["query_kind"], "comentario")

    def test_python_syntax_error_falls_back_to_blocks(self):
        units = codecorpus.extract_units("roto.py", ".py", "def f(:\n" + "x = 1\n" * 200)
        self.assertTrue(units)
        self.assertTrue(all(u["query"] is None for u in units))

    def test_cpp_functions_types_and_namespaces(self):
        units = {u["symbol"]: u for u in codecorpus.extract_units("a.cpp", ".cpp", CPP_SRC)}
        self.assertEqual(set(units), {"nms", "sin_comentario", "Caja"})  # la declaracion y el #define no cuentan
        nms = units["nms"]
        self.assertEqual(nms["query"], "Aplica supresion de no maximos sobre las detecciones de un frame.")
        self.assertNotIn("supresion", nms["pieces"][0])  # el comentario no queda en la respuesta
        self.assertIn('"}}}"', nms["pieces"][0])  # llaves dentro de cadenas no rompen el conteo
        self.assertIn("return c * 2;", units["sin_comentario"]["pieces"][0])  # firma en dos lineas
        self.assertIsNone(units["sin_comentario"]["query"])
        self.assertIn("float area()", units["Caja"]["pieces"][0])

    def test_banners_and_boilerplate_are_not_queries(self):
        banner = "// ======================\n// PERFIL POR CLASE\n// ======================\n"
        fn = "int f(int a) {\n  int b = a;\n  return b + 1;\n}\n"
        unit = codecorpus.extract_units("a.cpp", ".cpp", banner + fn)[0]
        self.assertIsNone(unit["query"])  # 3 palabras: demasiado corto
        lic = "// Copyright 2026 Alguien. Licensed under the Apache License.\n" + fn
        self.assertIsNone(codecorpus.extract_units("a.cpp", ".cpp", lic)[0]["query"])

    def test_long_functions_are_split_into_windows_of_the_same_unit(self):
        src = "void grande() {\n" + "".join(f"    int v{i} = {i};\n" for i in range(200)) + "}\n"
        unit = codecorpus.extract_units("g.cpp", ".cpp", src)[0]
        self.assertGreater(len(unit["pieces"]), 2)
        self.assertTrue(all(p.count("\n") < codecorpus.MAX_CHUNK_LINES for p in unit["pieces"]))

    def test_unknown_extension_and_msg_files(self):
        self.assertEqual(codecorpus.extract_units("a.txt", ".txt", "hola"), [])
        msg = codecorpus.extract_units("D.msg", ".msg", "float32 x\nfloat32 y\nstring clase\nfloat32 confianza\n")
        self.assertEqual(len(msg), 1)
        self.assertEqual(msg[0]["lang"], "ros-msg")


class CorpusFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def docs(self, n=6):
        for i in range(n):
            write(self.root, f"doc{i}.md", f"# Titulo numero {i} largo\n\n{BODY} unico{i}\n\n## Seccion extra {i} detalle\n\n{BODY} otro{i}\n")

    def code(self, n=6):
        for i in range(n):
            write(self.root, f"src/m{i}.py", PY_SRC.replace("calcular_iou", f"calcular_iou{i}").replace("a_radianes", f"a_radianes{i}").replace(
                "Calcula la interseccion", f"Calcula la interseccion numero {i}").replace("Convierte grados", f"Convierte grados tipo {i}").replace(
                "Mueve la base", f"Mueve la base {i}").replace("mover", f"mover{i}"))

    # --- documentos -------------------------------------------------------------

    def test_docs_mode_builds_documents_and_title_queries(self):
        self.docs()
        c = build_corpus(self.root)
        self.assertEqual(c.mode, "docs")
        self.assertEqual(c.stats["queries_titles"], 12)
        self.assertTrue(all(q["kind"] == "titulo" for q in c.queries))
        self.assertEqual(len(c.fingerprint), 64)

    def test_docs_mode_ignores_code_files(self):
        self.docs()
        self.code()
        self.assertTrue(all(d["source"].endswith(".md") for d in build_corpus(self.root).documents))

    def test_skips_hidden_symlinks_sensitive_and_big(self):
        self.docs()
        write(self.root, ".oculta/x.md", f"# Secreto escondido titulo\n\n{BODY}")
        write(self.root, "password_notes.md", f"# Notas con clave aqui\n\n{BODY}")
        (self.root / "enlace.md").symlink_to(self.root / "doc0.md")
        (self.root / "enorme.md").write_text("x" * (corpus.MAX_FILE_BYTES + 1))
        c = build_corpus(self.root)
        sources = {d["source"] for d in c.documents}
        self.assertFalse(any(s in sources for s in ("enlace.md", "enorme.md", "password_notes.md")))
        self.assertFalse(any(".oculta" in s for s in sources))
        for key in ("enlaces_simbolicos", "nombres_sensibles", "demasiado_grandes"):
            self.assertEqual(c.skipped.get(key), 1, key)

    def test_duplicated_and_generic_headings_make_no_query(self):
        for i in range(6):
            write(self.root, f"d{i}.md", f"# Introduccion\n\n{BODY}\n\n## Seccion repetida aqui\n\n{BODY} {i}\n\n## Unica seccion numero {i}\n\n{BODY}\n")
        queries = {q["text"] for q in build_corpus(self.root).queries}
        self.assertNotIn("Introduccion", queries)
        self.assertNotIn("Seccion repetida aqui", queries)
        self.assertIn("Unica seccion numero 3", queries)

    def test_too_little_material_and_bad_folders_are_rejected(self):
        write(self.root, "a.md", "# Un titulo corto\n\nhola")
        with self.assertRaisesRegex(CorpusError, "Muy poco material"):
            build_corpus(self.root)
        for bad in ("/", "/proc", str(self.root / "no_existe"), str(self.root / "a.md")):
            with self.assertRaises(CorpusError):
                build_corpus(bad)
        with self.assertRaisesRegex(CorpusError, "Tipo de corpus"):
            build_corpus(self.root, "otro")

    def test_manual_queries_map_files_to_fragments_and_report_problems(self):
        self.docs()
        lines = ['{"query": "pregunta uno sobre doc1", "relevant_files": ["doc1.md"]}', "esto no es json",
                 '{"query": "archivo inexistente", "relevant_files": ["nada.md"]}', ""]
        write(self.root, "eli_queries.jsonl", "\n".join(lines))
        c = build_corpus(self.root)
        manual = [q for q in c.queries if q["kind"] == "manual"]
        self.assertEqual(len(manual), 1)
        self.assertTrue(all(c.documents[int(i[1:]) - 1]["source"] == "doc1.md" for i in manual[0]["relevant"]))
        self.assertEqual(len([w for w in c.warnings if "eli_queries.jsonl" in w]), 2)

    def test_manual_queries_can_carry_a_kind_and_it_changes_the_fingerprint(self):
        self.docs()
        write(self.root, "eli_queries.jsonl", '{"query": "pregunta en ingles uno", "relevant_files": ["doc1.md"], "kind": "docs-en"}\n'
              '{"query": "otra sin tipo", "relevant_files": ["doc2.md"]}\n'
              '{"query": "tipo invalido", "relevant_files": ["doc3.md"], "kind": "Con Espacios!"}\n')
        c = build_corpus(self.root)
        kinds = [q["kind"] for q in c.queries if q["source"].startswith("doc") and q["kind"] != "titulo"]
        self.assertEqual(kinds, ["docs-en", "manual", "manual"])
        self.assertEqual(len([w for w in c.warnings if "kind" in w]), 1)
        before = c.fingerprint
        write(self.root, "eli_queries.jsonl", '{"query": "pregunta en ingles uno cambiada", "relevant_files": ["doc1.md"]}\n')
        self.assertNotEqual(before, build_corpus(self.root).fingerprint)  # cambiar las preguntas cambia la huella

    def test_manual_file_beyond_100_lines_is_read_up_to_the_limit(self):
        self.docs()
        lines = [f'{{"query": "pregunta numero {i}", "relevant_files": ["doc1.md"]}}' for i in range(corpus.MAX_MANUAL_LINES + 5)]
        write(self.root, "eli_queries.jsonl", "\n".join(lines))
        c = build_corpus(self.root)
        self.assertEqual(c.stats["queries_manual"], corpus.MAX_MANUAL_LINES)
        self.assertGreater(corpus.MAX_MANUAL_LINES, 108)  # cabe el banco de EMMA (63 + 45 de OD45)
        self.assertEqual(len([w for w in c.warnings if "se usan las primeras" in w]), 1)

    def test_subqueries_are_validated(self):
        self.docs()
        write(self.root, "eli_queries.jsonl",
              '{"query": "buena", "relevant_files": ["doc1.md"], "subqueries": ["uno", " dos "]}\n'
              '{"query": "mala lista", "relevant_files": ["doc2.md"], "subqueries": "no es lista"}\n'
              '{"query": "vacia", "relevant_files": ["doc3.md"], "subqueries": ["ok", ""]}\n'
              '{"query": "demasiadas", "relevant_files": ["doc4.md"], "subqueries": ' + str(["x"] * 13).replace("'", '"') + '}\n')
        c = build_corpus(self.root)
        manual = {q["text"]: q for q in c.queries if q["kind"] == "manual"}
        self.assertEqual(manual["buena"]["subqueries"], ["uno", "dos"])
        for name in ("mala lista", "vacia", "demasiadas"):
            self.assertNotIn("subqueries", manual[name])
        self.assertEqual(len([w for w in c.warnings if "subqueries" in w]), 3)

    def test_fingerprint_changes_with_content_and_is_stable_otherwise(self):
        self.docs()
        first = build_corpus(self.root).fingerprint
        self.assertEqual(first, build_corpus(self.root).fingerprint)
        write(self.root, "doc0.md", f"# Titulo numero 0 largo\n\n{BODY} cambio\n")
        self.assertNotEqual(first, build_corpus(self.root).fingerprint)

    # --- codigo -----------------------------------------------------------------

    def test_code_mode_indexes_sources_and_skips_build_dirs(self):
        self.code()
        write(self.root, "install/lib/x.py", PY_SRC.replace("calcular_iou", "instalado"))
        write(self.root, "build/y.cpp", CPP_SRC)
        write(self.root, "src/CMakeLists.txt", "cmake_minimum_required(VERSION 3.8)\nproject(x)\n" + "add_executable(a a.cpp)\n" * 30)
        c = build_corpus(self.root, "code")
        self.assertEqual(c.mode, "code")
        sources = {d["source"] for d in c.documents}
        self.assertFalse(any(s.startswith(("install/", "build/")) for s in sources))
        self.assertIn("src/CMakeLists.txt", sources)
        self.assertEqual(c.stats["code_by_language"]["python"], 18)
        self.assertEqual({q["kind"] for q in c.queries}, {"docstring", "comentario"})
        self.assertTrue(all(d["id"].startswith("k") for d in c.documents))

    def test_code_queries_do_not_leak_into_their_answer(self):
        self.code()
        c = build_corpus(self.root, "code")
        by_id = {d["id"]: d for d in c.documents}
        for q in c.queries:
            for doc_id in q["relevant"]:
                self.assertNotIn(q["text"], by_id[doc_id]["text"])

    def test_identical_code_in_two_files_is_counted_once(self):
        self.code()
        write(self.root, "copia/m0.py", (self.root / "src/m0.py").read_text())
        c = build_corpus(self.root, "code")
        self.assertEqual(c.stats["code_documents"], 18)

    def test_both_mode_merges_docs_and_code_and_manual_queries_see_both(self):
        self.docs()
        self.code()
        write(self.root, "eli_queries.jsonl", '{"query": "como se mueve la base", "relevant_files": ["src/m2.py", "doc1.md"]}\n')
        c = build_corpus(self.root, "both")
        kinds = {q["kind"] for q in c.queries}
        self.assertTrue({"titulo", "docstring", "manual"} <= kinds)
        manual = next(q for q in c.queries if q["kind"] == "manual")
        self.assertEqual({c.documents[[d["id"] for d in c.documents].index(i)]["source"] for i in manual["relevant"]}, {"src/m2.py", "doc1.md"})
        self.assertEqual(len({d["id"] for d in c.documents}), len(c.documents))
        self.assertNotEqual(c.fingerprint, build_corpus(self.root, "docs").fingerprint)

    def test_code_mode_without_descriptions_needs_manual_queries(self):
        for i in range(8):
            write(self.root, f"s{i}.py", f"def f{i}(a, b):\n    c = a + b + {i}\n    d = c * 2\n    return d\n")
        with self.assertRaisesRegex(CorpusError, "docstrings"):
            build_corpus(self.root, "code")


if __name__ == "__main__":
    unittest.main()

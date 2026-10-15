import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from eli.bench import graders as g
from eli.bench import sandbox

PASSED, FAILED, UNGRADED = g.PASSED, g.FAILED, g.UNGRADED
needs_sandbox = unittest.skipUnless(sandbox.available(), "bubblewrap no disponible")


def status(spec, text, done_reason=None):
    return g.grade(spec, text, done_reason).status


class TextHelperTests(unittest.TestCase):
    def test_strip_thinking(self):
        self.assertEqual(g.strip_thinking("<think>razono</think>Respuesta"), ("Respuesta", False))
        self.assertEqual(g.strip_thinking("hola <think>sin cerrar"), ("hola", True))

    def test_fold_removes_accents_markdown_and_case(self):
        self.assertEqual(g.fold("**Sábado**  Noche"), "sabado noche")

    def test_parse_number_handles_separators(self):
        self.assertEqual(g.parse_number("391"), 391)
        self.assertEqual(g.parse_number("1,234"), 1234)
        self.assertEqual(g.parse_number("3,5"), 3.5)
        self.assertEqual(g.parse_number("1.234,5"), 1234.5)
        self.assertEqual(g.parse_number("1.000"), 1000)

    def test_language_detection(self):
        self.assertEqual(g.detect_language("El perro corre por el parque con los niños")["lang"], "es")
        self.assertEqual(g.detect_language("The dog runs in the park with the children")["lang"], "en")


class SimpleGraderTests(unittest.TestCase):
    def test_exact_ignores_case_accents_and_punctuation_unless_case_sensitive(self):
        spec = {"type": "exact", "expected": "sabado"}
        self.assertEqual(status(spec, "**Sábado.**"), PASSED)
        self.assertEqual(status(spec, "domingo"), FAILED)
        strict = {"type": "exact", "expected": "OK", "case_sensitive": True}
        self.assertEqual(status(strict, "OK"), PASSED)
        self.assertEqual(status(strict, "ok"), FAILED)
        self.assertEqual(status({"type": "exact", "expected": "lapices,canciones", "strip_spaces": True}, "lápices, canciones"), PASSED)

    def test_candidates_accept_a_sentence_but_not_ambiguity(self):
        spec = {"type": "candidates", "candidates": ["si", "no"], "expected": "no"}
        self.assertEqual(status(spec, "No."), PASSED)
        self.assertEqual(status(spec, "Respuesta: No, no se puede."), PASSED)
        self.assertEqual(status(spec, "Sí."), FAILED)
        self.assertEqual(status(spec, "Quizá, depende de si es lunes o no."), UNGRADED)  # menciona si y no: ambigua
        self.assertEqual(status(spec, "Tal vez"), FAILED)

    def test_letters_use_the_last_line_and_none_word(self):
        spec = {"type": "letters", "candidates": ["A", "B", "C"], "expected": ["B"]}
        self.assertEqual(status(spec, "A dice algo.\nB"), PASSED)
        self.assertEqual(status(spec, "A, C"), FAILED)
        self.assertEqual(status({**spec, "expected": []}, "ninguno"), PASSED)
        self.assertEqual(status(spec, "caballero,mentiroso,caballero"), FAILED)

    def test_number_takes_the_last_number(self):
        spec = {"type": "number", "expected": 240}
        self.assertEqual(status(spec, "90 x 2,67 = 240"), PASSED)
        self.assertEqual(status(spec, "La respuesta es 600"), FAILED)
        self.assertEqual(status(spec, "no lo sé"), FAILED)
        self.assertEqual(status({"type": "number", "expected": 3.14, "tolerance": 0.01}, "3,14"), PASSED)

    def test_contains_and_regex(self):
        self.assertEqual(status({"type": "contains_all", "expected": ["muy cansados"]}, "…nos vimos MUY cansados."), PASSED)
        self.assertEqual(status({"type": "not_contains", "expected": ["mui"]}, "nos vimos mui cansados"), FAILED)
        self.assertEqual(status({"type": "contains_any", "expected": "AB12", "case_sensitive": True}, "es ab12"), FAILED)
        self.assertEqual(status({"type": "regex", "pattern": r"\b20 de marzo\b", "flags": "i"}, "El 20 de Marzo"), PASSED)


class StructuredGraderTests(unittest.TestCase):
    SPEC = {"type": "json", "schema": {"type": "object", "required": ["nombre", "edad"], "additionalProperties": False,
                                       "properties": {"nombre": {"type": "string"}, "edad": {"type": "integer"}}},
            "equals": {"nombre": "Lucía", "edad": 29}}

    def test_json_accepts_a_fenced_block_but_not_extra_text(self):
        self.assertEqual(status(self.SPEC, '{"nombre": "Lucía", "edad": 29}'), PASSED)
        self.assertEqual(status(self.SPEC, '```json\n{"nombre": "Lucia", "edad": 29}\n```'), PASSED)
        self.assertEqual(status(self.SPEC, 'Claro: {"nombre": "Lucía", "edad": 29}'), FAILED)
        self.assertEqual(status(self.SPEC, '{"nombre": "Lucía", "edad": "29"}'), FAILED)
        self.assertEqual(status(self.SPEC, '{"nombre": "Lucía", "edad": 29, "extra": 1}'), FAILED)
        self.assertEqual(status(self.SPEC, '{"nombre": "Lucía", "edad": true}'), FAILED)  # bool no es entero

    def test_json_paths_into_lists_and_root(self):
        spec = {"type": "json", "equals": {"$": ["a", "b"]}}
        self.assertEqual(status(spec, '["a", "b"]'), PASSED)
        self.assertEqual(status(spec, '["b", "a"]'), FAILED)
        nested = {"type": "json", "equals": {"r.1.id": 2}}
        self.assertEqual(status(nested, '{"r": [{"id": 1}, {"id": 2}]}'), PASSED)
        self.assertEqual(status(nested, '{"r": []}'), FAILED)

    def test_lines_and_markdown_table(self):
        self.assertEqual(status({"type": "lines_equal", "lines": ["a,b", "1,2"]}, "```\na,b\n1,2\n```"), PASSED)
        self.assertEqual(status({"type": "lines_equal", "lines": ["a,b", "1,2"]}, "a,b\n1,3"), FAILED)
        table = {"type": "table_md", "header": ["Lenguaje", "Año"], "rows": [["Python", "1991"]]}
        self.assertEqual(status(table, "| Lenguaje | Año |\n|---|---|\n| Python | 1991 |"), PASSED)
        self.assertEqual(status(table, "Aquí va:\n| Lenguaje | Año |\n|---|---|\n| Python | 1991 |"), FAILED)

    def test_instruction_checks(self):
        spec = {"type": "checks", "checks": [{"type": "words", "min": 3, "max": 6}, {"type": "forbidden_chars", "chars": ["z"]}, {"type": "sentences", "count": 1}]}
        self.assertEqual(status(spec, "El mar es muy grande."), PASSED)
        self.assertEqual(status(spec, "El mar es muy azul."), FAILED)  # contiene la z de azul
        self.assertEqual(status({"type": "checks", "checks": [{"type": "uppercase"}]}, "HOLA MUNDO"), PASSED)
        self.assertEqual(status({"type": "checks", "checks": [{"type": "lines", "count": 3}, {"type": "no_list_markers"}]}, "pera\nuva\nfresa"), PASSED)
        self.assertEqual(status({"type": "checks", "checks": [{"type": "no_list_markers"}]}, "- pera\n- uva"), FAILED)
        para = {"type": "checks", "checks": [{"type": "paragraphs", "count": 2}, {"type": "paragraph_contains", "index": 1, "words": ["montaña"]}]}
        self.assertEqual(status(para, "El río corre.\n\nLa montaña es alta."), PASSED)
        self.assertEqual(status(para, "El río corre. La montaña es alta."), FAILED)
        self.assertEqual(status({"type": "checks", "checks": [{"type": "ends_with", "text": "FIN."}]}, "Un átomo es pequeño. FIN."), PASSED)

    def test_language_check_is_ungraded_when_text_is_too_short(self):
        spec = {"type": "checks", "checks": [{"type": "language", "lang": "en", "min_words": 4}]}
        self.assertEqual(status(spec, "Jupiter"), UNGRADED)
        self.assertEqual(status(spec, "Jupiter is the largest planet in the solar system"), PASSED)
        self.assertEqual(status(spec, "Júpiter es el planeta más grande de todos los del sistema"), FAILED)

    def test_all_combines_graders(self):
        spec = {"type": "all", "graders": [{"type": "contains_any", "expected": ["paris"]}, {"type": "checks", "checks": [{"type": "language", "lang": "es"}]}]}
        self.assertEqual(status(spec, "La capital de Francia es París y es muy bonita."), PASSED)
        self.assertEqual(status(spec, "The capital of France is Paris and it is very nice."), FAILED)


class UngradedPolicyTests(unittest.TestCase):
    def test_empty_response_fails_unless_truncated_or_unclosed_thinking(self):
        spec = {"type": "exact", "expected": "x"}
        self.assertEqual(status(spec, ""), FAILED)
        self.assertEqual(status(spec, "", "length"), UNGRADED)
        self.assertEqual(status(spec, "<think>aún pensando"), UNGRADED)

    def test_a_failure_on_a_truncated_answer_is_not_a_failure(self):
        spec = {"type": "number", "expected": 42}
        self.assertEqual(status(spec, "Paso 1: tenemos 6 x 7 y entonces el res", "length"), UNGRADED)
        self.assertEqual(status(spec, "El resultado es 42", "length"), PASSED)  # un acierto sigue siendo un acierto

    def test_broken_or_unknown_grader_is_ungraded_not_a_crash(self):
        self.assertEqual(status({"type": "inexistente"}, "x"), UNGRADED)
        self.assertEqual(status({"type": "number", "expected": "no-numero"}, "42"), UNGRADED)

    def test_validate_spec(self):
        self.assertTrue(g.validate_spec({"type": "number"}))
        self.assertTrue(g.validate_spec({"type": "zzz"}))
        self.assertEqual(g.validate_spec({"type": "number", "expected": 1}), [])
        self.assertTrue(g.validate_spec({"type": "all", "graders": [{"type": "exact"}]}))

    def test_schema_validator(self):
        schema = {"type": "array", "minItems": 2, "items": {"type": "integer", "minimum": 0}}
        self.assertEqual(g.validate_schema(schema, [1, 2]), [])
        self.assertTrue(g.validate_schema(schema, [1]))
        self.assertTrue(g.validate_schema(schema, [1, -2]))
        self.assertTrue(g.validate_schema({"enum": ["a"]}, "b"))


CODE_SPEC = {"type": "code", "function": "doble", "tests": [{"args": [2], "expected": 4}, {"args": [0], "expected": 0}], "timeout_s": 5}


class CodeGraderTests(unittest.TestCase):
    @needs_sandbox
    def test_correct_and_wrong_code(self):
        self.assertEqual(status(CODE_SPEC, "```python\ndef doble(x):\n    return x * 2\n```"), PASSED)
        wrong = g.grade(CODE_SPEC, "```python\ndef doble(x):\n    return x + 2\n```")
        self.assertEqual(wrong.status, FAILED)
        self.assertIn("1/2", wrong.checks[0]["detail"])

    @needs_sandbox
    def test_missing_code_function_and_syntax_errors_are_failures(self):
        self.assertEqual(status(CODE_SPEC, "Aquí tienes: usa x*2"), FAILED)
        self.assertEqual(status(CODE_SPEC, "```python\ndef otra(x):\n    return x\n```"), FAILED)
        self.assertEqual(status(CODE_SPEC, "```python\ndef doble(x) return x\n```"), FAILED)

    @needs_sandbox
    def test_infinite_loop_is_stopped_by_the_timeout(self):
        spec = {**CODE_SPEC, "timeout_s": 2}
        res = g.grade(spec, "```python\ndef doble(x):\n    while True: pass\n```")
        self.assertEqual(res.status, FAILED)

    def test_without_bubblewrap_nothing_is_executed(self):
        marker = tempfile.NamedTemporaryFile(delete=False)
        marker.close()
        os.unlink(marker.name)
        code = f"```python\nopen({marker.name!r}, 'w').write('ejecutado')\ndef doble(x):\n    return x * 2\n```"
        with patch.object(sandbox, "available", return_value=None):
            res = g.grade(CODE_SPEC, code)
        self.assertEqual(res.status, UNGRADED)
        self.assertFalse(os.path.exists(marker.name), "el codigo se ejecuto sin aislamiento")


@needs_sandbox
class SandboxIsolationTests(unittest.TestCase):
    def run_code(self, body: str, **kw):
        return sandbox.run_function_tests(f"def f():\n{body}\n", "f", [{"args": [], "expected": True}], **kw)

    def test_no_network_even_to_localhost_services(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            res = self.run_code(f"    import socket\n    try:\n        socket.create_connection(('127.0.0.1', {port}), timeout=2)\n        return False\n    except OSError:\n        return True")
        finally:
            server.close()
        self.assertEqual((res.passed, res.total), (1, 1), res.cases)

    def test_host_files_are_invisible_and_system_is_read_only(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("secreto")
        try:
            home = os.path.expanduser("~")
            res = self.run_code(f"    import os\n    return (not os.path.exists({f.name!r})) and (not os.path.exists({home!r}))")
            self.assertEqual(res.passed, 1, res.cases)
            res = self.run_code("    try:\n        open('/usr/eli_x', 'w')\n        return False\n    except OSError:\n        return True")
            self.assertEqual(res.passed, 1, res.cases)
        finally:
            os.unlink(f.name)

    def test_environment_is_empty(self):
        res = self.run_code("    import os\n    return not any(k in os.environ for k in ('HOME_SECRET', 'SSH_AUTH_SOCK', 'USER', 'DISPLAY'))")
        self.assertEqual(res.passed, 1, res.cases)

    def test_memory_bomb_is_contained(self):
        res = self.run_code("    x = bytearray(3 * 1024 * 1024 * 1024)\n    return True", memory_mib=256)
        self.assertEqual(res.passed, 0)

    def test_timeout_kills_the_process_tree(self):
        res = self.run_code("    import time\n    time.sleep(60)\n    return True", timeout_s=2)
        self.assertTrue(res.timed_out)
        # los sandbox llevan 'eli-sbx-' en su linea de comandos; el corchete evita que pgrep se encuentre a si mismo
        # el kernel desmonta el espacio de nombres del sandbox de forma asincrona: se espera a que termine
        deadline = time.time() + 3
        while True:
            left = subprocess.run(["pgrep", "-f", "[e]li-sbx-"], capture_output=True, text=True).stdout.split()
            if not left or time.time() > deadline:
                break
            time.sleep(0.05)
        self.assertEqual(left, [], "quedaron procesos del sandbox")


if __name__ == "__main__":
    unittest.main()

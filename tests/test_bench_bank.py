import copy
import json
import unittest

from eli.bench import bank, graders, sandbox

SUITE = bank.default_suite()


def known_good(test: bank.TestCase) -> str | None:
    """Una respuesta correcta construida solo a partir del criterio de la prueba (None si no se puede derivar)."""
    spec = test.grader
    kind = spec["type"]
    if kind in ("number", "candidates", "exact", "regex") and kind != "regex":
        exp = spec["expected"]
        return str(exp[0] if isinstance(exp, list) else exp)
    if kind == "letters":
        return ", ".join(spec["expected"]) or "ninguno"
    if kind == "lines_equal":
        return "\n".join(spec["lines"])
    if kind == "table_md":
        rows = [spec["header"]] + spec["rows"]
        return "\n".join("| " + " | ".join(r) + " |" if i != 1 else "|---|---|\n| " + " | ".join(r) + " |" for i, r in enumerate(rows))
    return None


class SuiteStructureTests(unittest.TestCase):
    def test_suite_is_versioned_and_hashed(self):
        self.assertEqual((SUITE.id, SUITE.version), ("eli-core", "1.0.0"))
        self.assertRegex(SUITE.sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(bank.default_suite().sha256, SUITE.sha256)  # estable entre cargas

    def test_every_required_category_exists_with_several_tests_and_difficulties(self):
        required = {"logic", "math", "code", "reading", "instructions", "format", "language", "context"}
        self.assertTrue(required <= set(SUITE.categories))
        for cat in required:
            tests = [t for t in SUITE.tests if t.category == cat]
            self.assertGreaterEqual(len(tests), 3, cat)
            if cat != "context":
                self.assertGreaterEqual(len({t.difficulty for t in tests}), 2, f"{cat} necesita varios niveles")

    def test_both_languages_are_covered(self):
        self.assertEqual({t.language for t in SUITE.tests}, {"es", "en"})

    def test_ids_unique_and_every_test_has_a_verifiable_criterion(self):
        ids = [t.id for t in SUITE.tests]
        self.assertEqual(len(ids), len(set(ids)))
        for t in SUITE.tests:
            self.assertEqual(graders.validate_spec(t.grader), [], t.id)
            self.assertTrue(t.expected, t.id)
            self.assertIn(t.difficulty, bank.DIFFICULTIES)
            self.assertTrue(set(t.params) <= bank.PARAM_KEYS)

    def test_quick_mode_is_a_proper_subset_that_touches_every_category(self):
        quick = SUITE.select("quick")
        self.assertLess(len(quick), len(SUITE.select("full")))
        self.assertEqual({t.category for t in quick}, set(SUITE.categories))

    def test_select_filters_by_category_and_rejects_unknown(self):
        self.assertEqual({t.category for t in SUITE.select("full", ["math"])}, {"math"})
        with self.assertRaises(bank.SuiteError):
            SUITE.select("full", ["inexistente"])
        with self.assertRaises(bank.SuiteError):
            SUITE.select("turbo")

    def test_known_good_answers_pass_and_known_bad_ones_fail(self):
        checked = 0
        for t in SUITE.tests:
            if t.context_sized:
                continue
            good = known_good(t)
            if good is None:
                continue
            checked += 1
            self.assertEqual(graders.grade(t.grader, good).status, graders.PASSED, f"{t.id}: {good!r}")
            self.assertNotEqual(graders.grade(t.grader, "zzz respuesta equivocada 987654").status, graders.PASSED, t.id)
        self.assertGreater(checked, 20)

    def test_sample_answers_for_structured_tests(self):
        samples = {
            "format-001": '{"nombre": "Lucía", "edad": 29}',
            "format-004": '{"resenas": [{"id": 1, "sentimiento": "positivo"}, {"id": 2, "sentimiento": "negativo"}, {"id": 3, "sentimiento": "neutro"}]}',
            "format-007": '["banano", "cereza", "manzana", "pera", "uva"]',
            "instr-003": "pera\nuva\nfresa",
            "instr-005": "PORQUE LA ATMOSFERA DISPERSA LA LUZ AZUL MAS QUE LAS DEMAS",
            "instr-006": "Los dias pasan lentos\nlos sueños son eternos\ncantan los vientos\nvuelan los cuervos",
            "lang-001": "The dog runs in the park.",
            "lang-007": "lápices, canciones",
            "speed-001": " ".join(["agua"] * 120),
        }
        for tid, text in samples.items():
            t = SUITE.get(tid)
            self.assertEqual(graders.grade(t.grader, text).status, graders.PASSED, f"{tid}: {graders.grade(t.grader, text).reason}")

    @unittest.skipUnless(sandbox.available(), "bubblewrap no disponible")
    def test_reference_solutions_pass_inside_the_sandbox(self):
        for t in [t for t in SUITE.tests if t.category == "code"]:
            res = graders.grade(t.grader, "```python\n" + t.reference + "```")
            self.assertEqual(res.status, graders.PASSED, f"{t.id}: {res.reason} {res.checks}")
            wrong = graders.grade(t.grader, "```python\ndef " + t.grader["function"] + "(*a):\n    return None\n```")
            self.assertEqual(wrong.status, graders.FAILED, t.id)


class SuiteValidationTests(unittest.TestCase):
    BASE = {"suite_id": "x", "version": "1", "categories": {"c": "C"}, "tests": [
        {"id": "t-1", "category": "c", "difficulty": "easy", "prompt": "p", "expected": "e", "grader": {"type": "number", "expected": 1}}]}

    def mutate(self, fn):
        data = copy.deepcopy(self.BASE)
        fn(data)
        return data

    def test_valid_suite_parses(self):
        self.assertEqual(len(bank.parse_suite(self.BASE).tests), 1)

    def test_rejects_malformed_suites(self):
        cases = {
            "id duplicado": lambda d: d["tests"].append(dict(d["tests"][0])),
            "id invalido": lambda d: d["tests"][0].update(id="Mal Id"),
            "dificultad": lambda d: d["tests"][0].update(difficulty="extreme"),
            "categoria": lambda d: d["tests"][0].update(category="zzz"),
            "sin prompt": lambda d: d["tests"][0].update(prompt=""),
            "calificador": lambda d: d["tests"][0].update(grader={"type": "nope"}),
            "calificador incompleto": lambda d: d["tests"][0].update(grader={"type": "number"}),
            "parametros": lambda d: d["tests"][0].update(params={"hack": 1}),
            "sin esperado": lambda d: d["tests"][0].update(expected=""),
            "falta version": lambda d: d.pop("version"),
        }
        for label, fn in cases.items():
            with self.assertRaises(bank.SuiteError, msg=label):
                bank.parse_suite(self.mutate(fn))

    def test_hash_changes_when_a_test_changes(self):
        other = self.mutate(lambda d: d["tests"][0].update(prompt="otro"))
        self.assertNotEqual(bank.parse_suite(self.BASE).sha256, bank.parse_suite(other).sha256)


class ContextGeneratorTests(unittest.TestCase):
    GEN = {"type": "needle", "position": 0.5, "language": "es"}

    def test_prompt_is_deterministic_and_contains_the_needle(self):
        a, code = bank.build_needle(self.GEN, 2000, ("s", "t", 2048))
        b, code2 = bank.build_needle(self.GEN, 2000, ("s", "t", 2048))
        self.assertEqual((a, code), (b, code2))
        self.assertIn(code, a)
        self.assertEqual(len(code), 6)
        other, other_code = bank.build_needle(self.GEN, 2000, ("s", "t", 4096))
        self.assertNotEqual(code, other_code)

    def test_needle_position_follows_the_setting(self):
        for pos in (0.05, 0.5, 0.95):
            prompt, code = bank.build_needle({**self.GEN, "position": pos}, 6000, ("s", "t", 8192))
            text = prompt.split("\n\n")[1]
            rel = text.index(code) / len(text)
            self.assertAlmostEqual(rel, pos, delta=0.06)

    def test_size_scales_with_the_requested_context(self):
        short, _ = bank.build_needle(self.GEN, 600, ("s", "t", 1000))
        long, _ = bank.build_needle(self.GEN, 6000, ("s", "t", 10000))
        self.assertGreater(len(long), len(short) * 5)

    def test_english_variant(self):
        prompt, code = bank.build_needle({**self.GEN, "language": "en"}, 800, ("s", "t", 1000))
        self.assertIn("secret password", prompt)

    def test_instances_expand_per_context_size_and_run_in_num_ctx_order(self):
        tests = SUITE.select("full", ["context", "math"])
        inst = SUITE.instances(tests, [8192, 2048], mode="full", marker_salt="run-1")
        ctx = [i for i in inst if i.context_size]
        self.assertEqual(sorted({i.context_size for i in ctx}), [2048, 8192])
        self.assertEqual(len(ctx), 4 * 2)
        self.assertEqual([i.params["num_ctx"] for i in inst], sorted(i.params["num_ctx"] for i in inst))
        for i in ctx:
            self.assertEqual(i.params["num_ctx"], i.context_size)
            self.assertTrue(i.prompt.startswith("[ref "))
            self.assertIn(i.grader["expected"], i.prompt)
        again = SUITE.instances(tests, [8192, 2048], mode="full", marker_salt="run-1")
        self.assertEqual([i.prompt for i in inst], [i.prompt for i in again])  # reproducible con la misma ejecucion
        other = SUITE.instances(tests, [8192, 2048], mode="full", marker_salt="run-2")
        self.assertNotEqual([i.prompt for i in inst if i.context_size], [i.prompt for i in other if i.context_size])

    def test_base_context_override_applies_to_plain_tests_only(self):
        inst = SUITE.instances(SUITE.select("quick", ["math", "context"]), None, num_ctx=8192, mode="quick")
        for i in inst:
            self.assertEqual(i.params["num_ctx"], 2048 if i.context_size else 8192)


if __name__ == "__main__":
    unittest.main()

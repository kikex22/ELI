import tempfile
import unittest
from pathlib import Path

from eli.bench import compare
from eli.bench.store import Store


def result(model, instance, outcome, cat="logic", rep=1, tok_s=50.0, gen=40, ttft=0.3, cold=False, vram=9000.0, wh=None,
           temp=0, ctx=4096, think=False, predict=512, context_size=None, prompt_tokens=100, status="ok", requested="unset"):
    return {
        "type": "result", "model": model, "test_id": instance.split("@")[0], "instance_id": instance, "category": cat, "repetition": rep,
        "options": {"num_ctx": ctx, "temperature": temp, "think": think, "num_predict": predict, "seed": 42},
        "outcome": outcome, "gen_status": status, "cold": cold, "context_size": context_size,
        **({} if requested == "unset" else {"think_requested": requested}),
        "metrics": {"decode_tok_s": tok_s, "gen_tokens": gen, "ttft_s": ttft, "total_s": 1.0, "prompt_tokens": prompt_tokens,
                    "prefill_tok_s": 800.0, "wh_per_token": wh, "model_vram_bytes": 3_000_000_000,
                    "gpu": {"request": {"vram_peak_mib": vram}, "decode": {"energy_net_wh": (wh or 0) * gen if wh else None}}},
    }


def meta(name, params="8B", quant="Q4_K_M", family="fam", disk=5 * 2**30, pb=None):
    return {"name": name, "parameter_size": params, "parameters_b": pb if pb is not None else (float(params.rstrip("B")) if params.rstrip("B").replace(".", "").isdigit() else None),
            "quantization": quant, "family": family, "disk_bytes": disk}


class CompareBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def add_run(self, models: dict[str, list[dict]], metas: dict[str, dict] | None = None, idle=500.0) -> str:
        metas = metas or {}
        handle = self.store.create_run({"config": {}, "models": [metas.get(m, meta(m)) for m in models], "planned": {"total": 0}})
        for name, results in models.items():
            handle.append({"type": "model_start", "model": name, "vram_idle_mib": idle})
            for r in results:
                handle.append(r)
            handle.append({"type": "model_end", "model": name})
        handle.set_status(state="completed", done=sum(len(v) for v in models.values()))
        return handle.run_id

    @staticmethod
    def batch(model, outcomes, cat="logic", **kw):
        return [result(model, f"{cat}-{i}", o, cat=cat, **kw) for i, o in enumerate(outcomes)]

    def model(self, data, name):
        return next(m for m in data["models"] if m["name"] == name)


class CommonConditionsTests(CompareBase):
    def test_only_tests_that_every_model_ran_under_the_same_conditions_count(self):
        a = self.batch("a", ["passed"] * 4)  # logic-0..3
        b = self.batch("b", ["failed"] * 2)  # solo logic-0..1
        self.add_run({"a": a, "b": b})
        data = compare.compare(self.store)
        self.assertEqual(data["common"]["conditions"], 2)
        self.assertEqual(self.model(data, "a")["overall"]["total"], 2)
        self.assertEqual(data["common"]["excluded"], {"a": 2, "b": 0})
        self.assertTrue(any("no se compararon" in w for w in data["warnings"]))

    def test_repetitions_are_trimmed_to_the_common_minimum(self):
        a = [result("a", "logic-0", "passed", rep=r) for r in (1, 2, 3)]
        b = [result("b", "logic-0", "failed", rep=1)]
        self.add_run({"a": a, "b": b})
        data = compare.compare(self.store)
        self.assertEqual((self.model(data, "a")["overall"]["total"], self.model(data, "b")["overall"]["total"]), (1, 1))

    def test_different_conditions_are_not_comparable(self):
        self.add_run({"a": [result("a", "logic-0", "passed", temp=0)], "b": [result("b", "logic-0", "passed", temp=0.7)]})
        data = compare.compare(self.store)
        self.assertEqual(data["common"]["conditions"], 0)
        self.assertTrue(any("ninguna prueba" in w for w in data["warnings"]))
        self.add_run({"c": [result("c", "logic-0", "passed", think=True)], "d": [result("d", "logic-0", "passed", think=False)]})
        self.assertEqual(compare.compare(self.store, models=["c", "d"])["common"]["conditions"], 0)

    def test_a_model_without_thinking_is_still_comparable_with_one_that_has_it(self):
        """El parametro solo se envia a quien lo admite, pero la condicion solicitada es la misma."""
        thinker = [result("t", f"logic-{i}", "passed", think=False, requested=False) for i in range(3)]
        plain = [result("p", f"logic-{i}", "failed", think=None, requested=False) for i in range(3)]
        self.add_run({"t": thinker, "p": plain})
        self.assertEqual(compare.compare(self.store)["common"]["conditions"], 3)

    def test_a_different_requested_thinking_mode_is_not_comparable(self):
        a = [result("a", "logic-0", "passed", think=False, requested=False)]
        b = [result("b", "logic-0", "passed", think=True, requested=True)]
        self.add_run({"a": a, "b": b})
        self.assertEqual(compare.compare(self.store)["common"]["conditions"], 0)

    def test_complete_runs_are_preferred_over_newer_partial_ones(self):
        full = self.add_run({"a": self.batch("a", ["passed"] * 4)})
        handle = self.store.create_run({"config": {}, "models": [meta("a")], "planned": {"total": 4}})
        for r in self.batch("a", ["failed"] * 2):
            handle.append(r)
        handle.set_status(state="cancelled", done=2)
        data = compare.compare(self.store)
        self.assertEqual(self.model(data, "a")["run_id"], full)
        self.assertEqual(self.model(data, "a")["run_state"], "completed")
        self.assertFalse(any("parcial" in w for w in data["warnings"]))

    def test_a_partial_run_is_used_with_a_warning_when_it_is_all_there_is(self):
        handle = self.store.create_run({"config": {}, "models": [meta("a")], "planned": {"total": 4}})
        for r in self.batch("a", ["passed"] * 2):
            handle.append(r)
        handle.set_status(state="cancelled", done=2)
        data = compare.compare(self.store)
        self.assertEqual(self.model(data, "a")["overall"]["total"], 2)
        self.assertTrue(any("parcial" in w and "'a'" in w for w in data["warnings"]))

    def test_cancelled_results_are_ignored(self):
        self.add_run({"a": self.batch("a", ["passed", "passed"]) + [result("a", "logic-9", "cancelled")]})
        self.assertEqual(self.model(compare.compare(self.store), "a")["overall"]["total"], 2)

    def test_latest_run_per_model_and_explicit_run_selection(self):
        old = self.add_run({"a": self.batch("a", ["failed"] * 4)})
        new = self.add_run({"a": self.batch("a", ["passed"] * 4)})
        self.assertEqual(self.model(compare.compare(self.store), "a")["run_id"], new)
        self.assertEqual(self.model(compare.compare(self.store, run_ids=[old]), "a")["run_id"], old)

    def test_models_from_different_runs_are_combined(self):
        self.add_run({"a": self.batch("a", ["passed"] * 3)})
        self.add_run({"b": self.batch("b", ["failed"] * 3)})
        data = compare.compare(self.store)
        self.assertEqual({m["name"] for m in data["models"]}, {"a", "b"})
        self.assertEqual(data["common"]["conditions"], 3)


class ScoringTests(CompareBase):
    def test_ungraded_is_neither_pass_nor_fail(self):
        self.add_run({"a": self.batch("a", ["passed", "failed", "ungraded", "ungraded", "error"])})
        o = self.model(compare.compare(self.store), "a")["overall"]
        self.assertEqual((o["passed"], o["failed"], o["ungraded"], o["error"], o["graded"], o["score"]), (1, 1, 2, 1, 2, 0.5))

    def test_quality_needs_enough_graded_tests(self):
        self.add_run({"a": self.batch("a", ["passed"] * 3)})
        self.assertIsNone(self.model(compare.compare(self.store), "a")["indicators"]["quality"])
        self.add_run({"b": self.batch("b", ["passed"] * 9)})
        self.assertEqual(self.model(compare.compare(self.store, models=["b"]), "b")["indicators"]["quality"], 1.0)


class RankingTests(CompareBase):
    def setUp(self):
        super().setUp()
        logic_a, logic_b = self.batch("a", ["passed"] * 4), self.batch("b", ["passed", "passed", "failed", "failed"])
        code_a, code_b = self.batch("a", ["failed"] * 4, cat="code"), self.batch("b", ["passed"] * 4, cat="code")
        ins_a, ins_b = self.batch("a", ["passed"] * 3, cat="instructions"), self.batch("b", ["passed"] * 3, cat="instructions")
        self.add_run({"a": logic_a + code_a + ins_a, "b": logic_b + code_b + ins_b},
                     {"a": meta("a", "8B", disk=8 * 2**30), "b": meta("b", "8B", disk=2 * 2**30)})

    def test_category_winners_and_ties(self):
        r = compare.compare(self.store)["rankings"]
        self.assertEqual(r["reasoning"]["winners"], ["a"])
        self.assertEqual(r["code"]["winners"], ["b"])
        self.assertEqual(r["instructions"]["winners"], ["a", "b"])  # empate: se muestran ambos

    def test_no_recommendation_without_enough_graded_tests(self):
        store = Store(Path(self.tmp.name) / "otra")
        h = store.create_run({"models": [meta("x"), meta("y")], "planned": {"total": 0}})
        for m in ("x", "y"):
            for r in self.batch(m, ["passed", "failed"]):
                h.append(r)
        h.set_status(state="completed")
        rank = compare.compare(store)["rankings"]["reasoning"]
        self.assertEqual(rank["status"], "insufficient")
        self.assertIn("al menos 2 modelos", rank["reason"])

    def test_single_model_cannot_win_anything(self):
        data = compare.compare(self.store, models=["a"])
        self.assertTrue(all(v["status"] == "insufficient" for v in data["rankings"].values()))

    def test_fastest_uses_hot_long_responses_only(self):
        self.add_run({"f": self.batch("f", ["passed"] * 4, tok_s=90) + self.batch("f", ["passed"] * 2, cat="m", tok_s=500, gen=4)
                      + self.batch("f", ["passed"] * 2, cat="n", tok_s=900, cold=True),
                      "s": self.batch("s", ["passed"] * 4, tok_s=40) + self.batch("s", ["passed"] * 2, cat="m", tok_s=10, gen=4)
                      + self.batch("s", ["passed"] * 2, cat="n", tok_s=1, cold=True)})
        data = compare.compare(self.store, models=["f", "s"])
        self.assertEqual((self.model(data, "f")["indicators"]["speed_tok_s"], self.model(data, "s")["indicators"]["speed_tok_s"]), (90.0, 40.0))
        self.assertEqual(data["rankings"]["fastest"]["winners"], ["f"])
        self.assertEqual(self.model(data, "f")["performance"]["cold_excluded"], 2)

    def test_vram_is_peak_minus_what_was_there_before_loading(self):
        self.add_run({"v": self.batch("v", ["passed"] * 3, vram=9000), "w": self.batch("w", ["passed"] * 3, vram=6000)}, idle=1000)
        data = compare.compare(self.store, models=["v", "w"])
        self.assertEqual(self.model(data, "v")["vram"]["used_mib"], 8000)
        self.assertEqual(data["rankings"]["lowest_vram"]["winners"], ["w"])

    def test_energy_is_reported_only_with_enough_measurements(self):
        self.add_run({"e": self.batch("e", ["passed"] * 6, wh=0.0004), "g": self.batch("g", ["passed"] * 3, wh=0.0008)})
        e = self.model(compare.compare(self.store, models=["e"]), "e")["energy"]
        self.assertTrue(e["valid"])
        self.assertAlmostEqual(e["wh_per_token"], 0.0004)
        self.assertTrue(e["estimated"])
        g = self.model(compare.compare(self.store, models=["g"]), "g")["energy"]
        self.assertFalse(g["valid"])  # solo 3 < 5 mediciones
        self.assertIsNone(g["wh_per_token"])
        # al comparar ambos solo cuentan las 3 pruebas comunes: tampoco alcanza para estimar energia
        both = compare.compare(self.store, models=["e", "g"])
        self.assertFalse(self.model(both, "e")["energy"]["valid"])

    def test_variability_between_repetitions(self):
        rs = [result("z", "logic-0", "passed", rep=r, tok_s=v) for r, v in ((1, 40.0), (2, 60.0), (3, 50.0))]
        self.add_run({"z": rs})
        self.assertAlmostEqual(self.model(compare.compare(self.store, models=["z"]), "z")["performance"]["decode_cv"], 0.2, places=2)

    def test_pareto_fronts_and_efficiency(self):
        pareto = compare.pareto_front({"a": (8, 0.9), "b": (2, 0.6), "c": (9, 0.5)}, x_lower_better=True)
        self.assertEqual(pareto, ["a", "b"])  # c esta dominado por a y por b
        self.assertEqual(compare.pareto_front({"a": (100, 0.5), "b": (50, 0.9)}, x_lower_better=False), ["a", "b"])
        data = compare.compare(self.store)
        self.assertIn("calidad_vs_tamano", data["pareto"])
        self.assertEqual(self.model(data, "b")["pareto_fronts"] >= 1, True)


class EfficiencyFloorTests(CompareBase):
    def test_a_tiny_useless_model_does_not_win_efficiency(self):
        tiny = self.batch("tiny", ["failed"] * 7 + ["passed"] * 3, vram=1500, tok_s=170)  # 30 %, minimo de recursos
        good = self.batch("good", ["passed"] * 9 + ["failed"], vram=4500, tok_s=70)  # 90 %
        self.add_run({"tiny": tiny, "good": good}, {"tiny": meta("tiny", "1B", disk=1 * 2**30), "good": meta("good", "8B", disk=8 * 2**30)})
        data = compare.compare(self.store)
        self.assertEqual(data["rankings"]["fastest"]["winners"], ["tiny"])
        self.assertEqual(data["rankings"]["lowest_vram"]["winners"], ["tiny"])
        eff = data["rankings"]["efficiency"]
        self.assertEqual(eff["status"], "insufficient")  # solo un modelo supera el minimo de calidad: no hay con quien comparar
        self.assertIsNone(eff["values"]["tiny"])
        self.assertIn("diminuto", eff["basis"])

    def test_models_above_the_floor_compete_normally(self):
        a = self.batch("a", ["passed"] * 9 + ["failed"], vram=4500, tok_s=70)
        b = self.batch("b", ["passed"] * 8 + ["failed"] * 2, vram=9000, tok_s=30)
        self.add_run({"a": a, "b": b}, {"a": meta("a", "8B", disk=8 * 2**30), "b": meta("b", "14B", disk=14 * 2**30)})
        eff = compare.compare(self.store)["rankings"]["efficiency"]
        self.assertEqual((eff["status"], eff["winners"]), ("ok", ["a"]))


class FiltersAndGroupsTests(CompareBase):
    def test_parameter_size_filter(self):
        self.add_run({"s": self.batch("s", ["passed"] * 3), "m": self.batch("m", ["passed"] * 3), "u": self.batch("u", ["passed"] * 3)},
                     {"s": meta("s", "1B"), "m": meta("m", "8B"), "u": meta("u", "?")})
        names = lambda **kw: sorted(m["name"] for m in compare.compare(self.store, **kw)["models"])  # noqa: E731
        self.assertEqual(names(), ["m", "s", "u"])
        self.assertEqual(names(param_min=5, param_max=10), ["m"])
        self.assertEqual(names(param_max=2), ["s"])
        self.assertNotIn("u", names(param_min=0, param_max=100), "sin parametros conocidos no se puede aplicar un filtro de tamano")

    def test_same_family_different_quantization_is_flagged(self):
        self.add_run({"q4": self.batch("q4", ["passed"] * 3), "q8": self.batch("q8", ["passed"] * 3), "other": self.batch("other", ["passed"] * 3)},
                     {"q4": meta("q4", "8B", "Q4_K_M"), "q8": meta("q8", "8B", "Q8_0"), "other": meta("other", "7B", "Q4_K_M", family="otra")})
        groups = compare.compare(self.store)["quant_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual({m["quantization"] for m in groups[0]["models"]}, {"Q4_K_M", "Q8_0"})


class ContextTests(CompareBase):
    def test_context_curve_and_verified_capacity(self):
        rs = []
        for size, passed, tokens in ((2048, "passed", 1100), (4096, "passed", 2400), (8192, "failed", 4900)):
            rs.append(result("c", f"ctx-needle-middle-es@{size}", passed, cat="context", ctx=size, context_size=size, prompt_tokens=tokens, ttft=size / 4000))
        self.add_run({"c": rs})
        ctx = self.model(compare.compare(self.store, models=["c"]), "c")["context"]
        self.assertEqual([r["num_ctx_requested"] for r in ctx["curve"]], [2048, 4096, 8192])
        self.assertEqual(ctx["verified_tokens"], 2400)  # lo comprobado no es el num_ctx pedido
        self.assertEqual(ctx["max_requested"], 8192)
        self.assertEqual(ctx["curve"][2]["failed"], 1)


if __name__ == "__main__":
    unittest.main()

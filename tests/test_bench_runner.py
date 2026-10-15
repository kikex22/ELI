import json
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

from eli import inventory
from eli.bench import bank
from eli.bench.runner import BenchError, BenchManager, RunConfig, parse_config
from eli.bench.store import Store, write_unique
from eli.gpu import GPUMonitor
from eli.ollama import Ollama

from fakes import FAKE_GPU_COMMAND, FakeOllama
from test_bench_bank import known_good

SUITE = bank.default_suite()
GOOD = {t.prompt: known_good(t) for t in SUITE.tests if not t.context_sized and known_good(t)}
HW = {"gpu": {"name": "gpu-falsa", "compute_cap": 8.9, "vram_total_mib": 16380}}


def answers(good_models):
    def respond(body):
        prompt = body["messages"][-1]["content"]
        if body["model"] in good_models and prompt in GOOD:
            return GOOD[prompt]
        m = re.search(r"\b(?:es|is) ([A-Z0-9]{6})\.", prompt)  # prueba de contexto: acierta si el modelo es 'bueno'
        if m and body["model"] in good_models:
            return m.group(1)
        return "zzz"  # nunca coincide por casualidad con una respuesta valida
    return respond


class RunnerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = FakeOllama()
        self.fake.add_model("m2", capabilities=["completion"])
        self.fake.add_model("emb", capabilities=["embedding"])
        self.fake.loaded = set()
        self.gpu = GPUMonitor(command=FAKE_GPU_COMMAND)
        self.gpu.start()
        deadline = time.time() + 5
        while len(self.gpu.history(5)) < 10 and time.time() < deadline:
            time.sleep(0.05)
        self.ollama = Ollama(self.fake.url, timeout=5)
        self.store = Store(Path(self.tmp.name))
        self.lock = threading.Lock()
        inspect = lambda names=None: {i["name"]: i for i in inventory.inspect_installed(self.ollama, self.store, HW, names)}  # noqa: E731
        self.mgr = BenchManager(self.ollama, self.gpu, self.store, self.lock, SUITE, lambda: HW, inspect, sleep=lambda s: None)

    def tearDown(self):
        self.mgr.cancel()
        self.mgr.join(10)
        self.gpu.stop()
        self.fake.close()
        self.tmp.cleanup()

    def run_bench(self, body, timeout=60):
        run_id = self.mgr.start(body)
        self.mgr.join(timeout)
        return run_id, self.store.load_run(run_id)

    def chats(self, model=None, warmups=False):
        return [b for b in self.fake.chat_bodies if (model is None or b["model"] == model) and (warmups or b["options"]["num_predict"] != 8)]


class ConfigValidationTests(unittest.TestCase):
    INSTALLED = {"a": {"capabilities": ["completion"], "context_length": 8192}, "e": {"capabilities": ["embedding"], "context_length": 2048}}

    def parse(self, body, loaded=()):
        return parse_config(body, self.INSTALLED, SUITE, list(loaded))

    def test_valid_defaults(self):
        cfg = self.parse({"models": ["a"]})
        self.assertEqual((cfg.mode, cfg.repetitions, cfg.temperature, cfg.think, cfg.warmup), ("quick", 1, 0.0, False, True))

    def test_rejections(self):
        bad = [
            {}, {"models": []}, {"models": ["a", "a"]}, {"models": ["zzz"]}, {"models": ["e"]}, {"models": [1]},
            {"models": ["a"], "mode": "turbo"}, {"models": ["a"], "categories": ["zzz"]}, {"models": ["a"], "repetitions": 0},
            {"models": ["a"], "repetitions": 2.5}, {"models": ["a"], "repetitions": True}, {"models": ["a"], "temperature": 5},
            {"models": ["a"], "temperature": "x"}, {"models": ["a"], "think": "si"}, {"models": ["a"], "timeout_s": 1},
            {"models": ["a"], "context_sizes": []}, {"models": ["a"], "context_sizes": [100]}, {"models": ["a"], "context_sizes": [16384]},
            {"models": ["a"], "num_ctx": 10**7}, {"models": ["a"], "num_predict": 1}, {"models": [str(i) for i in range(20)]},
        ]
        for body in bad:
            with self.assertRaises(BenchError, msg=str(body)):
                self.parse(body)

    def test_other_loaded_models_need_explicit_permission(self):
        with self.assertRaises(BenchError) as ctx:
            self.parse({"models": ["a"]}, loaded=["otro:1b"])
        self.assertEqual((ctx.exception.code, ctx.exception.extra["loaded"]), ("models_loaded", ["otro:1b"]))
        self.assertTrue(self.parse({"models": ["a"], "unload_others": True}, loaded=["otro:1b"]).unload_others)


class EndToEndTests(RunnerBase):
    BODY = {"models": ["m1", "m2"], "mode": "quick", "categories": ["logic", "math"], "repetitions": 2, "unload_others": True}

    def test_full_run_is_persisted_and_counted(self):
        self.fake.responder = answers({"m1"})
        run_id, doc = self.run_bench(self.BODY)
        self.assertEqual(doc["status"]["state"], "completed")
        self.assertEqual(len(doc["results"]), 2 * 8 * 2)
        self.assertEqual(doc["status"]["done"], doc["status"]["total"])
        by_model = {m: [r for r in doc["results"] if r["model"] == m] for m in ("m1", "m2")}
        self.assertTrue(all(r["outcome"] == "passed" for r in by_model["m1"]))
        self.assertTrue(all(r["outcome"] == "failed" for r in by_model["m2"]))
        self.assertEqual(doc["status"]["counts"]["passed"], 16)
        self.assertEqual(doc["status"]["counts"]["failed"], 16)
        types = [e["type"] for e in doc["events"]]
        self.assertEqual((types.count("model_start"), types.count("model_end")), (2, 2))
        self.assertGreaterEqual(types.count("warmup"), 2)
        lines = (Path(self.tmp.name) / "benchmarks" / run_id / "results.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), len(doc["results"]) + len(doc["events"]))

    def test_record_contains_everything_needed_to_reproduce(self):
        _, doc = self.run_bench({**self.BODY, "models": ["m1"], "repetitions": 1})
        run = doc["run"]
        self.assertEqual(run["suite"]["sha256"], SUITE.sha256)
        self.assertEqual(run["models"][0]["name"], "m1")
        self.assertEqual(run["models"][0]["quantization"], "Q4")
        self.assertEqual(run["models"][0]["format"], "gguf")
        self.assertEqual(run["hardware"]["gpu"]["name"], "gpu-falsa")
        self.assertIn("sandbox", run)
        r = doc["results"][0]
        for key in ("prompt", "expected", "response", "options", "grade", "metrics", "num_ctx_requested", "started", "duration_s", "cold"):
            self.assertIn(key, r)
        self.assertIn("provenance", r["metrics"])
        self.assertEqual(r["metrics"]["provenance"]["decode_tok_s"], "derivado")

    def test_every_test_is_an_independent_conversation(self):
        self.fake.responder = answers({"m1", "m2"})
        self.run_bench(self.BODY)
        for body in self.fake.chat_bodies:
            self.assertEqual([m["role"] for m in body["messages"]], ["user"], "se arrastro historial de otra prueba o modelo")
        self.assertEqual(len({b["messages"][0]["content"] for b in self.chats("m1")}), 8)

    def test_all_models_get_the_same_tests_and_conditions(self):
        self.run_bench(self.BODY)
        sig = lambda model: [(b["messages"][0]["content"], b["options"]) for b in self.chats(model)]  # noqa: E731
        self.assertEqual(sig("m1"), sig("m2"))

    def test_models_run_one_after_another_and_are_unloaded_in_between(self):
        _, doc = self.run_bench(self.BODY)
        order = [(e["type"], e["model"]) for e in doc["events"] if e["type"] in ("model_start", "model_end")]
        self.assertEqual(order, [("model_start", "m1"), ("model_end", "m1"), ("model_start", "m2"), ("model_end", "m2")])
        self.assertEqual(self.fake.unloaded.count("m1"), 1)
        first_m2 = next(i for i, b in enumerate(self.fake.chat_bodies) if b["model"] == "m2")
        self.assertTrue(all(b["model"] == "m1" for b in self.fake.chat_bodies[:first_m2]))

    def test_warmup_is_separate_from_hot_measurements(self):
        self.fake.load_duration = 4_000_000_000
        _, doc = self.run_bench({**self.BODY, "models": ["m1"], "repetitions": 1})
        warm = [e for e in doc["events"] if e["type"] == "warmup"]
        self.assertTrue(warm and warm[0]["load_s"] == 4.0)
        self.assertTrue(all(r["cold"] for r in doc["results"]))  # el fake siempre informa carga alta: se marcan en frio
        self.fake.load_duration = 1_000_000
        _, doc2 = self.run_bench({**self.BODY, "models": ["m1"], "repetitions": 1})
        self.assertTrue(all(not r["cold"] for r in doc2["results"]))

    def test_without_warmup_there_are_no_warmup_events(self):
        _, doc = self.run_bench({**self.BODY, "models": ["m1"], "warmup": False})
        self.assertFalse([e for e in doc["events"] if e["type"] == "warmup"])

    def test_think_parameter_is_only_sent_to_models_that_support_it(self):
        self.run_bench({**self.BODY, "repetitions": 1, "think": False})
        self.assertTrue(all(b.get("think") is False for b in self.chats("m1")))
        self.assertTrue(all("think" not in b for b in self.chats("m2")))

    def test_think_rejected_by_the_model_falls_back_and_is_recorded(self):
        self.fake.reject_think = {"m1"}
        _, doc = self.run_bench({**self.BODY, "models": ["m1"], "repetitions": 1})
        self.assertTrue(all(r["think_fallback"] for r in doc["results"]))
        self.assertTrue(all(r["outcome"] in ("passed", "failed") for r in doc["results"]))

    def test_the_requested_thinking_mode_is_recorded_even_when_the_model_cannot_use_it(self):
        _, doc = self.run_bench({**self.BODY, "repetitions": 1, "think": False})
        for r in doc["results"]:
            self.assertIs(r["think_requested"], False)
            self.assertIs(r["options"]["think"], False if r["model"] == "m1" else None)

    def test_seed_is_constant_at_temperature_zero_and_varies_otherwise(self):
        self.run_bench({**self.BODY, "models": ["m2"], "categories": ["math"]})
        self.assertEqual({b["options"]["seed"] for b in self.chats("m2")}, {42})
        self.fake.chat_bodies.clear()
        self.run_bench({**self.BODY, "models": ["m2"], "categories": ["math"], "temperature": 0.7})
        self.assertEqual({b["options"]["seed"] for b in self.chats("m2")}, {42, 43})

    def test_truncated_wrong_answers_are_ungraded_not_failed(self):
        self.fake.done_reason = "length"
        _, doc = self.run_bench({**self.BODY, "models": ["m2"], "repetitions": 1})
        self.assertTrue(all(r["outcome"] == "ungraded" for r in doc["results"]))
        self.assertEqual(doc["status"]["counts"]["failed"], 0)

    def test_context_category_uses_each_requested_size(self):
        self.fake.responder = answers({"m1"})
        _, doc = self.run_bench({"models": ["m1"], "mode": "full", "categories": ["context"], "context_sizes": [2048, 4096], "unload_others": True})
        sizes = sorted({r["num_ctx_requested"] for r in doc["results"]})
        self.assertEqual(sizes, [2048, 4096])
        self.assertEqual(len(doc["results"]), 4 * 2)
        self.assertTrue(all(r["outcome"] == "passed" and r["metrics"]["prompt_tokens"] == 20 for r in doc["results"]))
        warm_ctx = sorted({e["num_ctx"] for e in doc["events"] if e["type"] == "warmup"})
        self.assertEqual(warm_ctx, [2048, 4096])  # un calentamiento por cada num_ctx distinto

    def test_other_models_are_unloaded_only_with_permission(self):
        self.fake.loaded = {"m2"}
        with self.assertRaises(BenchError) as ctx:
            self.mgr.start({"models": ["m1"]})
        self.assertEqual(ctx.exception.code, "models_loaded")
        self.assertEqual(self.fake.unloaded, [])
        self.assertFalse(self.lock.locked(), "el bloqueo se filtro tras un inicio rechazado")
        self.run_bench({"models": ["m1"], "categories": ["math"], "unload_others": True})
        self.assertIn("m2", self.fake.unloaded)

    def test_it_waits_for_ollama_to_finish_unloading_before_measuring_idle_vram(self):
        self.fake.loaded = {"m1"}
        self.fake.unload_delay = 0.5
        self.ollama.unload("m1")
        self.assertTrue(self.fake.loaded, "el fake debia seguir mostrandolo cargado un instante")
        self.mgr._sleep = time.sleep  # en los demas tests el sleep esta anulado; aqui hay que esperar de verdad
        started = time.time()
        self.assertTrue(self.mgr._wait_unloaded(timeout=5))
        self.assertGreaterEqual(time.time() - started, 0.25)
        self.assertFalse(self.fake.loaded)

    def test_waiting_for_unload_gives_up_instead_of_hanging(self):
        self.fake.loaded = {"m1"}  # nunca se descarga
        started = time.time()
        self.assertFalse(self.mgr._wait_unloaded(timeout=0.5))
        self.assertLess(time.time() - started, 3)

    def test_second_run_while_busy_is_rejected(self):
        self.fake.chunk_delay = 0.05
        self.fake.tokens = ["a"] * 6
        self.mgr.start({**self.BODY, "models": ["m1"]})
        with self.assertRaises(BenchError) as ctx:
            self.mgr.start({**self.BODY, "models": ["m2"]})
        self.assertEqual(ctx.exception.code, "busy")
        self.mgr.cancel()
        self.mgr.join(20)

    def test_status_reports_real_progress(self):
        self.fake.chunk_delay = 0.03
        self.fake.tokens = ["x"] * 5
        self.mgr.start({**self.BODY, "models": ["m1"], "repetitions": 1})
        seen = []
        deadline = time.time() + 30
        while time.time() < deadline:
            st = self.mgr.status()
            seen.append(st["done"])
            if not st["active"]:
                break
            time.sleep(0.05)
        st = self.mgr.status()
        self.assertEqual((st["state"], st["done"], st["total"]), ("completed", 8, 8))
        self.assertEqual(seen, sorted(seen))
        self.assertTrue(any(0 < d < 8 for d in seen), "no se observo progreso intermedio")
        self.assertEqual(sum(st["counts"][k] for k in ("passed", "failed", "ungraded", "error", "timeout")), 8)


class CancellationAndErrorTests(RunnerBase):
    def test_cancel_keeps_what_was_measured_and_frees_the_gpu_lock(self):
        self.fake.chunk_delay = 0.1
        self.fake.tokens = ["x"] * 5
        self.mgr.start({"models": ["m1", "m2"], "mode": "quick", "categories": ["math"], "repetitions": 3})
        deadline = time.time() + 20
        while self.mgr.status()["done"] < 1 and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(self.mgr.cancel())
        self.mgr.join(20)
        st = self.mgr.status()
        run_id = st["run_id"]
        doc = self.store.load_run(run_id)
        self.assertEqual(doc["status"]["state"], "cancelled")
        self.assertGreaterEqual(len(doc["results"]), 1)
        self.assertLess(len(doc["results"]), doc["status"]["total"])
        self.assertFalse(self.lock.locked())
        self.assertFalse(self.mgr.status()["active"])
        # se puede empezar otra evaluacion despues de cancelar
        self.fake.chunk_delay = 0
        self.mgr.start({"models": ["m1"], "mode": "quick", "categories": ["math"]})
        self.mgr.join(20)
        self.assertEqual(self.mgr.status()["state"], "completed")

    def test_three_errors_in_a_row_abort_that_model_only(self):
        self.fake.chat_error = "out of memory"
        _, doc = self.run_bench({"models": ["m1"], "mode": "quick", "categories": ["math"], "warmup": False})
        st = doc["status"]
        self.assertEqual(st["state"], "completed")
        self.assertEqual(st["counts"]["error"], 3)
        self.assertEqual(st["counts"]["skipped"], 1)
        self.assertEqual(st["done"], st["total"])
        self.assertTrue(any(e.get("aborted") for e in doc["events"] if e["type"] == "model_end"))
        self.assertTrue(all("out of memory" in r["error"] for r in doc["results"]))

    def test_one_failing_model_does_not_stop_the_next(self):
        original = self.fake.chat_error
        self.fake.chat_error = "boom"
        self.mgr.start({"models": ["m1", "m2"], "mode": "quick", "categories": ["math"], "warmup": False})
        deadline = time.time() + 30
        while self.mgr.status()["model_index"] < 2 and time.time() < deadline:
            time.sleep(0.02)
        self.fake.chat_error = original  # el segundo modelo ya no falla
        self.mgr.join(30)
        doc = self.store.load_run(self.mgr.status()["run_id"])
        self.assertEqual({r["model"] for r in doc["results"]}, {"m1", "m2"})
        self.assertEqual(doc["status"]["state"], "completed")

    def test_timeout_is_recorded_as_such(self):
        self.fake.chunk_delay = 0.5
        self.fake.tokens = ["x"] * 20
        cfg = RunConfig(models=("m1",), timeout_s=0.6)
        inst = SUITE.instances(SUITE.select("quick", ["math"]), mode="quick")[0]
        record = self.mgr._run_one(cfg, "m1", inst, 1, None)
        self.assertEqual((record["outcome"], record["gen_status"]), ("timeout", "timeout"))
        self.assertIn("Tiempo maximo", record["error"])
        self.assertIsNone(record["grade"])


class PersistenceTests(RunnerBase):
    def test_runs_never_share_or_overwrite_a_folder(self):
        ids = {self.store.create_run({"planned": {"total": 0}}).run_id for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_write_unique_never_overwrites(self):
        d = Path(self.tmp.name) / "u"
        first = write_unique(d, "chat", ".json", "uno")
        second = write_unique(d, "chat", ".json", "dos")
        third = write_unique(d, "chat", ".json", "tres")
        self.assertEqual((first.read_text(), second.read_text(), third.read_text()), ("uno", "dos", "tres"))
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual([p.name for p in d.iterdir() if p.name.startswith(".")], [], "quedaron temporales")

    def test_run_json_is_immutable_while_status_changes(self):
        run_id, _ = self.run_bench({"models": ["m1"], "mode": "quick", "categories": ["math"]})
        run_json = (Path(self.tmp.name) / "benchmarks" / run_id / "run.json").read_bytes()
        handle_status = self.store.load_run(run_id)["status"]
        self.assertEqual(handle_status["state"], "completed")
        self.mgr.start({"models": ["m1"], "mode": "quick", "categories": ["math"]})
        self.mgr.join(20)
        self.assertEqual((Path(self.tmp.name) / "benchmarks" / run_id / "run.json").read_bytes(), run_json)
        self.assertEqual(len(self.store.list_runs()), 2)

    def test_runs_left_running_by_a_crash_are_marked_interrupted_and_nothing_else_changes(self):
        handle = self.store.create_run({"planned": {"total": 5}})
        handle.append({"type": "result", "model": "x", "outcome": "passed"})
        before = (handle.dir / "results.jsonl").read_text()
        self.assertEqual(self.store.mark_interrupted({"otro-id"}), [handle.run_id])
        self.assertEqual(self.store.load_run(handle.run_id)["status"]["state"], "interrupted")
        self.assertEqual((handle.dir / "results.jsonl").read_text(), before)
        self.assertEqual(self.store.mark_interrupted(set()), [])  # idempotente
        live = self.store.create_run({"planned": {"total": 1}})
        self.assertEqual(self.store.mark_interrupted({live.run_id}), [])  # una activa no se toca

    def test_a_half_written_last_line_is_skipped(self):
        handle = self.store.create_run({"planned": {"total": 2}})
        handle.append({"type": "result", "model": "x", "outcome": "passed"})
        with open(handle.dir / "results.jsonl", "a") as f:
            f.write('{"type": "result", "mod')
        self.assertEqual(len(self.store.load_run(handle.run_id)["results"]), 1)

    def test_path_traversal_ids_are_rejected(self):
        for bad in ("../etc", "..", "20261015-000000-zzzzzz", "", "a/b"):
            self.assertIsNone(self.store.load_run(bad))
            self.assertIsNone(self.store.export_json(bad))

    def test_exports_json_and_csv_and_neutralize_formulas(self):
        run_id, doc = self.run_bench({"models": ["m1"], "mode": "quick", "categories": ["math"]})
        exported = json.loads(self.store.export_json(run_id))
        self.assertEqual(len(exported["results"]), len(doc["results"]))
        csv_text = self.store.export_csv(run_id)
        self.assertEqual(len(csv_text.strip().splitlines()), len(doc["results"]) + 1)
        self.assertIn("decode_tok_s", csv_text.splitlines()[0])
        handle = self.store.create_run({"planned": {"total": 1}})
        handle.append({"type": "result", "model": "=cmd|' /C calc'!A0", "test_id": "t", "instance_id": "t", "outcome": "failed",
                       "grade": {"reason": "+SUM(1+1)"}, "metrics": {}})
        csv_text = self.store.export_csv(handle.run_id)
        self.assertNotIn(",=cmd", csv_text)
        self.assertIn("'=cmd", csv_text)
        self.assertIn("'+SUM", csv_text)

    def test_history_survives_deleting_the_model(self):
        run_id, _ = self.run_bench({"models": ["m2"], "mode": "quick", "categories": ["math"]})
        self.ollama.unload("m2")
        inventory.delete_model(self.ollama, "m2", "m2")
        self.assertNotIn("m2", [m["name"] for m in self.ollama.models()])
        doc = self.store.load_run(run_id)
        self.assertEqual(doc["run"]["models"][0]["name"], "m2")
        self.assertTrue(doc["results"])
        self.assertEqual([m["name"] for m in self.store.latest_models()], ["m2"])
        self.assertIn("m2", self.store.observed_vram())


if __name__ == "__main__":
    unittest.main()

import tempfile
import threading
import time
import unittest
from pathlib import Path

from eli import inventory
from eli.bench.store import Store
from eli.emb import bank, metrics
from eli.emb.runner import EmbError, EmbManager, parse_config
from eli.gpu import GPUMonitor
from eli.ollama import Ollama

from fakes import FAKE_GPU_COMMAND, FakeOllama

SUITE = bank.default_suite()
HW = {"gpu": {"name": "gpu-falsa", "compute_cap": 8.9, "vram_total_mib": 16380}}


class EmbBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = FakeOllama()
        self.fake.loaded = set()
        self.fake.add_model("emb", capabilities=["embedding"], architecture="nomic-bert", context_length=2048)
        self.fake.add_model("emb2", capabilities=["embedding"], architecture="bert")
        self.fake.add_model("chat", capabilities=["completion"])
        self.gpu = GPUMonitor(command=FAKE_GPU_COMMAND)
        self.gpu.start()
        deadline = time.time() + 5
        while len(self.gpu.history(5)) < 10 and time.time() < deadline:
            time.sleep(0.05)
        self.ollama = Ollama(self.fake.url, timeout=5)
        self.store = Store(Path(self.tmp.name), "embeddings")
        self.lock = threading.Lock()
        inspect = lambda names=None: {i["name"]: i for i in inventory.inspect_installed(self.ollama, None, HW, names)}  # noqa: E731
        self.mgr = EmbManager(self.ollama, self.gpu, self.store, self.lock, SUITE, lambda: HW, inspect, sleep=lambda s: None)

    def tearDown(self):
        self.mgr.cancel()
        self.mgr.join(10)
        self.gpu.stop()
        self.fake.close()
        self.tmp.cleanup()

    def run_and_wait(self, body):
        run_id = self.mgr.start(body)
        self.mgr.join(60)
        return run_id, self.store.load_run(run_id)

    @staticmethod
    def steps(doc, model="emb"):
        return {r["step"]: r for r in doc["events"] if r.get("type") == "step" and r["model"] == model}


class ParseConfigTests(EmbBase):
    def info(self, *names):
        return {i["name"]: i for i in inventory.inspect_installed(self.ollama, None, HW, list(names))}

    def test_rejects_chat_model(self):
        with self.assertRaisesRegex(EmbError, "no es de embeddings"):
            parse_config({"models": ["chat"]}, self.info("chat"), SUITE, [])

    def test_rejects_unknown_empty_and_duplicates(self):
        for body in ({}, {"models": []}, {"models": ["emb", "emb"]}, {"models": ["nope"]}):
            with self.assertRaises(EmbError):
                parse_config(body, self.info("emb"), SUITE, [])

    def test_rejects_bad_options(self):
        for extra in ({"prefixes": "x"}, {"repetitions": 0}, {"repetitions": True}, {"timeout_s": 1}, {"corpus_folder": ""}):
            with self.assertRaises(EmbError):
                parse_config({"models": ["emb"], **extra}, self.info("emb"), SUITE, [])

    def test_other_loaded_models_need_confirmation(self):
        with self.assertRaises(EmbError) as cm:
            parse_config({"models": ["emb"]}, self.info("emb"), SUITE, ["chat"])
        self.assertEqual(cm.exception.code, "models_loaded")
        cfg = parse_config({"models": ["emb"], "unload_others": True}, self.info("emb"), SUITE, ["chat"])
        self.assertTrue(cfg.unload_others)


class CoverageMetricTests(unittest.TestCase):
    DOCS = [[1, 0], [0, 1], [1, 1], [0.9, 0.1], [0.8, 0.2]]
    SRC = ["a", "b", "c", "a", "a"]

    def test_single_file_answers_have_no_coverage(self):
        self.assertIsNone(metrics.file_coverage([1, 0], self.DOCS, self.SRC, {0, 3, 4}))

    def test_fraction_of_files_in_the_top_k(self):
        got = metrics.file_coverage([1, 0], self.DOCS, self.SRC, {0, 1, 3}, ks=(1, 3, 5))
        self.assertEqual(got, {"1": 0.5, "3": 0.5, "5": 1.0})  # 'b' solo aparece al final del ranking

    def test_irrelevant_fragments_do_not_count_and_ties_are_pessimistic(self):
        same = [[1, 1]] * 4
        got = metrics.file_coverage([1, 1], same, ["a", "b", "c", "d"], {0, 1}, ks=(2,))
        self.assertEqual(got, {"2": 0.0})  # empate total: lo relevante queda detras de lo irrelevante

    def test_summary_over_rows(self):
        rows = [{"coverage": {"5": 0.5, "10": 1.0}, "files": 4}, {"coverage": {"5": 0.0, "10": 0.5}, "files": 2}, {"coverage": None, "files": 1}]
        self.assertEqual(metrics.coverage_metrics(rows), {"n": 2, "files_mean": 3.0, "cov@5": 0.25, "cov@10": 0.75, "full@10": 0.5})
        self.assertIsNone(metrics.coverage_metrics([{"coverage": None, "files": 1}]))


class FileLevelMetricTests(unittest.TestCase):
    # un archivo grande ('big') con 3 ventanas casi iguales llena el top de fragmentos, pero cuenta una sola vez por archivo
    DOCS = [[1, 0], [0.99, 0.1], [0.98, 0.1], [0, 1], [0.5, 0.5]]
    SRC = ["big", "big", "big", "b", "c"]

    def test_big_files_do_not_crowd_out_other_files(self):
        chunk = metrics.file_coverage([1, 0], self.DOCS, self.SRC, {0, 1, 2, 3}, ks=(3,))
        by_file = metrics.file_level([1, 0], self.DOCS, self.SRC, {0, 1, 2, 3}, ks=(1, 2, 3))
        self.assertEqual(chunk, {"3": 0.5})  # 3 fragmentos, todos de 'big'
        self.assertEqual(by_file, {"rank": 1, "files": 2, "coverage": {"1": 0.5, "2": 0.5, "3": 1.0}})

    def test_ties_are_pessimistic_and_unknown_sources_give_none(self):
        same = [[1, 1]] * 3
        got = metrics.file_level([1, 1], same, ["a", "b", "c"], {0}, ks=(1,))
        self.assertEqual((got["rank"], got["coverage"]), (3, {"1": 0.0}))
        self.assertIsNone(metrics.file_level([1, 0], self.DOCS, [None] * 5, {0}))

    def test_summary_separates_hit_rate_and_multi_file_coverage(self):
        a = metrics.file_level([1, 0], self.DOCS, self.SRC, {0, 1, 2, 3})
        b = metrics.file_level([1, 0], self.DOCS, self.SRC, {4})
        got = metrics.file_level_metrics([{"file_level": a}, {"file_level": b}, {"file_level": None}])
        self.assertEqual(got["n"], 2)
        self.assertEqual(got["hit@1"], 0.5)  # 'big' es el primero para a; 'c' es el 3.o para b
        self.assertEqual(got["multi"]["n"], 1)
        self.assertEqual(got["multi"]["cov@3"], 1.0)
        self.assertIsNone(metrics.file_level_metrics([{"file_level": None}]))


class SubqueryMetricTests(unittest.TestCase):
    DOCS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]]
    SRC = ["a", "b", "c", "z"]

    def test_decomposition_reaches_files_the_original_question_misses(self):
        got = metrics.subquery_coverage([1, 0.05, 0.05], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], self.DOCS, self.SRC, {0, 1, 2})
        self.assertEqual(got["by_m"]["1"], {"union": 3, "coverage": 1.0, "baseline": 0.6667})

    def test_baseline_gets_the_same_number_of_files_as_the_union(self):
        # con m=2 la union ya tiene 4 archivos (todos): la pregunta sola tambien mira 4 y no hay ganancia que reclamar
        got = metrics.subquery_coverage([1, 0.05, 0.05], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], self.DOCS, self.SRC, {0, 1, 2})
        self.assertEqual(got["by_m"]["2"], {"union": 4, "coverage": 1.0, "baseline": 1.0})

    def test_needs_several_files_and_subqueries(self):
        self.assertIsNone(metrics.subquery_coverage([1, 0, 0], [[1, 0, 0]], self.DOCS, self.SRC, {0}))
        self.assertIsNone(metrics.subquery_coverage([1, 0, 0], [], self.DOCS, self.SRC, {0, 1}))

    def test_summary_reports_gain_and_full_fractions(self):
        row = {"subquery": metrics.subquery_coverage([1, 0.05, 0.05], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], self.DOCS, self.SRC, {0, 1, 2})}
        got = metrics.subquery_metrics([row, {"subquery": None}])
        self.assertEqual(got["n"], 1)
        self.assertEqual(got["by_m"]["1"]["gain"], 0.3333)
        self.assertEqual((got["by_m"]["1"]["full"], got["by_m"]["1"]["full_baseline"]), (1.0, 0.0))
        self.assertIsNone(metrics.subquery_metrics([{"subquery": None}]))


class PrefixRuleTests(unittest.TestCase):
    def rule(self, name, arch=None):
        return bank.prefix_for(name, arch)["rule"]

    def test_code_and_new_models_have_their_own_rules(self):
        self.assertEqual(self.rule("qwen3-embedding:0.6b"), "qwen3-embedding")
        self.assertEqual(self.rule("embeddinggemma:latest"), "embeddinggemma")
        self.assertEqual(self.rule("hf.co/awhiteside/CodeRankEmbed-Q8_0-GGUF:latest"), "code-query-prefix")
        self.assertEqual(self.rule("jina-code-embeddings-0.5b:q8"), "jina-code")

    def test_nomic_embed_code_does_not_fall_into_the_nomic_text_rule(self):
        self.assertEqual(self.rule("nomic-embed-code:latest"), "code-query-prefix")
        self.assertEqual(self.rule("nomic-embed-text:latest"), "nomic")

    def test_unknown_model_gets_no_prefixes(self):
        pre = bank.prefix_for("bge-m3:latest", None)
        self.assertEqual((pre["rule"], pre["query"], pre["document"]), (None, "", ""))

    def test_qwen3_query_has_no_space_after_query_colon_and_documents_none(self):
        pre = bank.prefix_for("qwen3-embedding:4b", None)
        self.assertTrue(pre["query"].startswith("Instruct: ") and pre["query"].endswith("\nQuery:"))
        self.assertEqual(pre["document"], "")


class RunnerTests(EmbBase):
    def test_full_run_records_every_step(self):
        run_id, doc = self.run_and_wait({"models": ["emb"], "repetitions": 2})
        self.assertEqual(doc["status"]["state"], "completed")
        steps = self.steps(doc)
        self.assertEqual(list(steps), ["warmup", "retrieval", "similarity", "stability", "performance"])
        self.assertTrue(all(s["status"] == "ok" for s in steps.values()), steps)
        self.assertEqual(steps["warmup"]["dims"], 64)
        self.assertEqual(steps["retrieval"]["metrics"]["n"], len(SUITE.queries))
        self.assertIn("recall@1", steps["retrieval"]["metrics"])
        self.assertEqual(len(steps["similarity"]["triplets"]), len(SUITE.triplets))
        self.assertEqual(steps["stability"]["min_cos_repeat"], 1.0)
        self.assertEqual(steps["stability"]["min_cos_batch"], 1.0)
        self.assertEqual(len(steps["stability"]["reference"]["vectors"]), 3)
        perf = steps["performance"]
        self.assertEqual(set(perf["latency"]), {"short", "medium", "long"})
        self.assertEqual(set(perf["throughput"]), {"1", "8", "32"})
        self.assertGreater(perf["throughput"]["8"]["emb_per_s"], 0)
        self.assertIsNotNone(perf["gpu"])
        self.assertEqual(doc["status"]["models"]["emb"]["aborted"], False)

    def test_always_sends_truncate_false(self):
        self.run_and_wait({"models": ["emb"], "repetitions": 1})
        self.assertTrue(self.fake.embed_bodies)
        self.assertTrue(all(b["truncate"] is False for b in self.fake.embed_bodies))

    def test_prefixes_recorded_and_applied_only_when_requested(self):
        _, none_doc = self.run_and_wait({"models": ["emb"], "repetitions": 1})
        start = next(e for e in none_doc["events"] if e["type"] == "model_start")
        self.assertEqual(start["prefixes"]["mode"], "none")
        self.assertFalse(any(t.startswith("search_") for b in self.fake.embed_bodies for t in b["input"]))
        self.fake.embed_bodies.clear()
        _, task_doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "prefixes": "task"})
        start = next(e for e in task_doc["events"] if e["type"] == "model_start")
        self.assertEqual(start["prefixes"]["rule"], "nomic")
        sent = [t for b in self.fake.embed_bodies for t in b["input"]]
        self.assertTrue(any(t.startswith("search_query: ") for t in sent))
        self.assertTrue(any(t.startswith("search_document: ") for t in sent))

    def test_degenerate_model_gets_no_credit(self):
        self.fake.embedder = lambda model, text: [1.0, 2.0, 3.0]  # mismo vector para todo
        _, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1})
        m = self.steps(doc)["retrieval"]["metrics"]
        self.assertLess(m["recall@1"], 0.1)
        self.assertEqual(self.steps(doc)["similarity"]["accuracy"], 0.0)  # sim(+) == sim(-): no es acierto

    def test_step_error_is_recorded_and_others_continue(self):
        self.fake.embed_max_words = 200  # el texto largo (250 palabras) supera el contexto
        _, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1})
        steps = self.steps(doc)
        self.assertEqual(steps["performance"]["status"], "error")
        self.assertIn("context length", steps["performance"]["error"])
        self.assertEqual(steps["retrieval"]["status"], "ok")
        self.assertEqual(doc["status"]["state"], "completed")

    def test_warmup_failure_skips_model(self):
        self.fake.embed_error = "boom"
        _, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1})
        steps = self.steps(doc)
        self.assertEqual(steps["warmup"]["status"], "error")
        self.assertTrue(all(s["status"] == "skipped" for k, s in steps.items() if k != "warmup"))
        self.assertTrue(doc["status"]["models"]["emb"]["aborted"])

    def test_models_run_one_after_another_and_unload(self):
        _, doc = self.run_and_wait({"models": ["emb", "emb2"], "repetitions": 1})
        order = [e["model"] for e in doc["events"] if e["type"] == "model_start"]
        self.assertEqual(order, ["emb", "emb2"])
        self.assertEqual(self.fake.unloaded, ["emb", "emb2"])

    def test_cancel_keeps_what_was_measured(self):
        slow = threading.Event()

        def embedder(model, text):
            slow.set()
            time.sleep(0.01)
            return [1.0, float(len(text) % 7), 2.0]

        self.fake.embedder = embedder
        run_id = self.mgr.start({"models": ["emb"], "repetitions": 30})
        self.assertTrue(slow.wait(5))
        self.assertTrue(self.mgr.cancel())
        self.mgr.join(30)
        doc = self.store.load_run(run_id)
        self.assertEqual(doc["status"]["state"], "cancelled")
        self.assertFalse(self.lock.locked())

    def test_busy_when_lock_is_held(self):
        self.lock.acquire()
        try:
            with self.assertRaises(EmbError) as cm:
                self.mgr.start({"models": ["emb"]})
            self.assertEqual(cm.exception.code, "busy")
        finally:
            self.lock.release()

    def test_user_corpus_step_stores_no_document_text(self):
        folder = Path(self.tmp.name) / "docs"
        folder.mkdir()
        body = " ".join(f"palabra{i}" for i in range(60))
        for n in range(6):
            (folder / f"a{n}.md").write_text(f"# Titulo numero {n} largo\n\n{body} unico{n} secreto{n}\n\n## Seccion extra {n} detalle\n\n{body} otro{n}\n")
        run_id, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder)})
        steps = self.steps(doc)
        self.assertEqual(steps["retrieval_user"]["status"], "ok", steps["retrieval_user"])
        self.assertTrue(doc["run"]["corpus"]["fingerprint"])
        raw = (Path(self.tmp.name) / "embeddings" / run_id / "results.jsonl").read_text()
        self.assertNotIn("palabra5", raw)
        self.assertNotIn("secreto", raw)

    def test_multi_file_manual_queries_get_coverage_in_the_run(self):
        folder = Path(self.tmp.name) / "docs2"
        folder.mkdir()
        body = " ".join(f"palabra{i}" for i in range(60))
        for n in range(6):
            (folder / f"a{n}.md").write_text(f"# Titulo numero {n} largo\n\n{body} unico{n}\n\n## Seccion extra {n} detalle\n\n{body} otro{n}\n")
        (folder / "eli_queries.jsonl").write_text(
            '{"query": "unico1 unico2 unico3", "relevant_files": ["a1.md", "a2.md", "a3.md"], "kind": "how-es"}\n'
            '{"query": "unico4", "relevant_files": ["a4.md"], "kind": "code-es"}\n')
        run_id, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder)})
        user = self.steps(doc)["retrieval_user"]
        multi = [q for q in user["queries"] if q["kind"] == "how-es"]
        self.assertEqual(multi[0]["files"], 3)
        self.assertEqual(set(multi[0]["coverage"]), {"5", "10"})
        self.assertIsNone(next(q for q in user["queries"] if q["kind"] == "code-es")["coverage"])
        self.assertEqual(user["coverage"]["n"], 1)
        self.assertEqual(set(user["coverage_by_kind"]), {"how-es"})
        self.assertGreater(user["coverage"]["cov@10"], 0)
        self.assertEqual(multi[0]["file_level"]["files"], 3)
        self.assertEqual(set(multi[0]["file_level"]["coverage"]), {"3", "5", "8"})
        single = next(q for q in user["queries"] if q["kind"] == "code-es")
        self.assertEqual(single["file_level"]["files"], 1)  # una sola respuesta tambien tiene recuperacion por archivo
        self.assertEqual(user["file_level"]["n"], 14)  # 12 titulos + 2 manuales: todas tienen archivo conocido
        self.assertEqual(user["file_level"]["multi"]["n"], 1)
        self.assertEqual(set(user["file_level_by_kind"]), {"how-es", "code-es", "titulo"})
        self.assertIsNone(multi[0]["subquery"])  # sin subqueries en el archivo de preguntas no hay metrica de descomposicion
        self.assertIsNone(user["subquery"])

    def test_subqueries_are_embedded_with_the_query_prefix_and_measured(self):
        folder = Path(self.tmp.name) / "docs3"
        folder.mkdir()
        body = " ".join(f"palabra{i}" for i in range(60))
        for n in range(6):
            (folder / f"a{n}.md").write_text(f"# Titulo numero {n} largo\n\n{body} unico{n}\n\n## Seccion extra {n} detalle\n\n{body} otro{n}\n")
        (folder / "eli_queries.jsonl").write_text(
            '{"query": "como funciona todo", "relevant_files": ["a1.md", "a2.md", "a3.md"], "kind": "how-es", '
            '"subqueries": ["unico1", "unico2", "unico3"]}\n'
            '{"query": "pregunta simple", "relevant_files": ["a4.md"], "subqueries": ["unico4"]}\n')
        self.fake.embed_bodies.clear()
        run_id, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder), "prefixes": "task"})
        user = self.steps(doc)["retrieval_user"]
        how = next(q for q in user["queries"] if q["kind"] == "how-es")
        self.assertEqual(how["subqueries"], ["unico1", "unico2", "unico3"])
        self.assertEqual(how["subquery"]["subqueries"], 3)
        self.assertEqual(how["subquery"]["by_m"]["1"]["coverage"], 1.0)  # cada subpregunta acierta su archivo
        self.assertIsNone(next(q for q in user["queries"] if q["query"] == "pregunta simple")["subquery"])  # un solo archivo
        self.assertEqual(user["subquery"]["n"], 1)
        self.assertEqual(set(user["subquery_by_kind"]), {"how-es"})
        sent = [t for b in self.fake.embed_bodies for t in b["input"]]
        self.assertTrue(any(t == "search_query: unico2" for t in sent))

    def test_bad_corpus_folder_is_rejected_before_running(self):
        with self.assertRaises(EmbError) as cm:
            self.mgr.start({"models": ["emb"], "corpus_folder": "/no/existe/nada"})
        self.assertEqual(cm.exception.code, "invalid_corpus")
        self.assertFalse(self.lock.locked())


if __name__ == "__main__":
    unittest.main()


class LlmSubqueryTests(EmbBase):
    def setUp(self):
        super().setUp()
        self.fake.models["chat"]["capabilities"] = ["completion", "thinking"]  # generate() envia think=False

    def corpus(self, name="docs4"):
        folder = Path(self.tmp.name) / name
        folder.mkdir()
        body = " ".join(f"palabra{i}" for i in range(60))
        for n in range(6):
            (folder / f"a{n}.md").write_text(f"# Titulo numero {n} largo\n\n{body} unico{n}\n\n## Seccion extra {n} detalle\n\n{body} otro{n}\n")
        (folder / "eli_queries.jsonl").write_text(
            '{"query": "como funciona todo", "relevant_files": ["a1.md", "a2.md", "a3.md"], "kind": "how-es"}\n'
            '{"query": "pregunta simple", "relevant_files": ["a4.md"]}\n')
        return folder

    def test_config_rejects_bad_llm_choices(self):
        for body, text in (
            ({"models": ["emb"], "subquery_llm": "emb2"}, "no genera texto"),  # modelo de embeddings, no de texto
            ({"subquery_llm": "nope"}, "no esta instalado"),
            ({"subquery_llm": "chat", "subquery_count": 9}, "subquery_count"),
            ({"subquery_llm": "chat", "subquery_lang": "fr"}, "subquery_lang"),
            ({"models": ["emb"], "subquery_llm": "emb"}, "no puede ser uno de los modelos"),
        ):
            with self.subTest(body=body):
                with self.assertRaises(EmbError) as cm:
                    self.mgr.start({"models": body.get("models", ["emb"]), **body})
                self.assertIn(text, str(cm.exception))

    def test_llm_writes_subqueries_once_and_they_are_measured_next_to_manual_ones(self):
        folder = self.corpus()
        self.fake.responder = lambda body: '<think>x</think>["unico1 palabra", "unico2 palabra", "unico3 palabra"]' if body["model"] == "chat" else ["ok"]
        run_id, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder),
                                         "subquery_llm": "chat", "subquery_count": 3})
        gen = [e for e in doc["events"] if e.get("type") == "subquery_gen"]
        self.assertEqual(len(gen), 1)
        self.assertEqual((gen[0]["status"], gen[0]["queries"], gen[0]["generated"], gen[0]["failed"]), ("ok", 1, 1, 0))
        self.assertEqual(gen[0]["examples"][0]["subqueries"], ["unico1 palabra", "unico2 palabra", "unico3 palabra"])
        self.assertEqual(doc["status"]["done"], doc["status"]["total"])  # 6 pasos + 1 de subpreguntas
        user = self.steps(doc)["retrieval_user"]
        how = next(q for q in user["queries"] if q["query"] == "como funciona todo")
        self.assertEqual(how["subqueries_llm"], ["unico1 palabra", "unico2 palabra", "unico3 palabra"])
        self.assertEqual(how["subquery_llm"]["subqueries"], 3)
        self.assertIsNone(how["subquery"])  # sin subpreguntas a mano: no se mezclan
        self.assertEqual(user["subquery_llm"]["n"], 1)
        self.assertEqual(set(user["subquery_llm_by_kind"]), {"how-es"})
        self.assertIsNone(user["subquery"])
        self.assertIsNotNone(user["subquery_llm"]["by_m"]["1"]["coverage"])

    def test_llm_sees_only_the_question_and_runs_with_fixed_options(self):
        folder = self.corpus("docs5")
        self.fake.responder = lambda body: '["unico1 palabra", "unico2 palabra"]' if body["model"] == "chat" else ["ok"]
        self.fake.chat_bodies.clear()
        self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder), "subquery_llm": "chat"})
        self.assertEqual(len(self.fake.chat_bodies), 1)  # una sola pregunta de varios archivos
        body = self.fake.chat_bodies[0]
        self.assertFalse(body.get("think"))
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertEqual(body["options"]["seed"], 42)
        text = " ".join(m["content"] for m in body["messages"])
        self.assertIn("como funciona todo", text)
        self.assertNotIn("a1.md", text)
        self.assertNotIn("unico1", text)  # ni el contenido de los documentos

    def test_bad_llm_output_is_recorded_and_the_run_still_completes(self):
        folder = self.corpus("docs6")
        self.fake.responder = lambda body: "no se me ocurre nada" if body["model"] == "chat" else ["ok"]
        run_id, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder), "subquery_llm": "chat"})
        gen = next(e for e in doc["events"] if e.get("type") == "subquery_gen")
        self.assertEqual((gen["status"], gen["generated"], gen["failed"]), ("error", 0, 1))
        self.assertIn("subpreguntas validas", gen["errors"][0]["error"])
        self.assertEqual(doc["status"]["state"], "completed")
        self.assertIsNone(self.steps(doc)["retrieval_user"]["subquery_llm"])
        self.assertNotIn("chat", self.fake.loaded)  # el LLM sale de la VRAM aunque haya fallado

    def test_second_run_reuses_cached_subqueries_without_loading_the_llm(self):
        folder = self.corpus("docs7")
        self.fake.responder = lambda body: '["camera frame capture", "bounding box NMS"]' if body["model"] == "chat" else ["ok"]
        self.fake.chat_bodies.clear()
        self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder), "subquery_llm": "chat", "subquery_count": 2})
        self.assertEqual(len(self.fake.chat_bodies), 1)
        _, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder), "subquery_llm": "chat", "subquery_count": 2})
        self.assertEqual(len(self.fake.chat_bodies), 1)  # la segunda corrida no pregunta al LLM
        gen = next(e for e in doc["events"] if e.get("type") == "subquery_gen")
        self.assertEqual((gen["cached"], gen["generated"], gen["status"]), (1, 0, "ok"))
        how = next(q for q in self.steps(doc)["retrieval_user"]["queries"] if q["query"] == "como funciona todo")
        self.assertEqual(how["subqueries_llm"], ["camera frame capture", "bounding box NMS"])

    def test_cache_key_depends_on_llm_count_and_language_and_damaged_cache_is_ignored(self):
        from eli.emb import subqueries as sq
        base = sq.cache_key("pregunta", "chat", 6, "en")
        self.assertNotEqual(base, sq.cache_key("pregunta", "otro", 6, "en"))
        self.assertNotEqual(base, sq.cache_key("pregunta", "chat", 5, "en"))
        self.assertNotEqual(base, sq.cache_key("pregunta", "chat", 6, "same"))
        cache = sq.SubqueryCache(Path(self.tmp.name) / "c.json")
        (Path(self.tmp.name) / "c.json").write_text("{no es json")
        self.assertIsNone(cache.get(base))
        cache.put(base, ["a b", "c d"])
        self.assertEqual(cache.get(base), ["a b", "c d"])


class IterativeTests(EmbBase):
    def setUp(self):
        super().setUp()
        self.fake.models["chat"]["capabilities"] = ["completion", "thinking"]

    corpus = LlmSubqueryTests.corpus

    def test_config_rules_for_iterative_rounds(self):
        with self.assertRaises(EmbError) as cm:
            self.mgr.start({"models": ["emb"], "iter_rounds": 2})
        self.assertIn("necesita un LLM", str(cm.exception))
        with self.assertRaises(EmbError):
            self.mgr.start({"models": ["emb"], "subquery_llm": "chat", "iter_rounds": 9})

    def test_a_round_adds_the_phrase_it_asked_for_and_stops_when_done(self):
        folder = self.corpus("docs8")
        calls = []

        def responder(body):
            text = " ".join(m["content"] for m in body["messages"])
            if body["model"] != "chat":
                return ["ok"]
            if "Files found so far" in text:
                calls.append(text)
                if len(calls) == 1:
                    return '{"done": false, "queries": ["unico2 palabra"]}'
                return '{"done": true, "queries": []}'
            return '["x y", "z w"]'
        self.fake.responder = responder
        _, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder),
                                    "subquery_llm": "chat", "subquery_count": 2, "iter_rounds": 3})
        self.assertEqual(doc["status"]["state"], "completed")
        it = self.steps(doc)["iterative"]
        self.assertEqual(it["status"], "ok")
        how = next(q for q in it["queries"] if q["kind"] == "how-es")
        self.assertEqual(how["rounds"], 2)  # la ronda 2 dijo done: no hay una tercera
        self.assertEqual(how["phrases"], ["unico2 palabra"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(it["metrics"]["n"], 1)
        self.assertIsNotNone(it["metrics"]["by_m"]["2"]["coverage"])

    def test_broken_round_closes_the_question_without_failing_the_model(self):
        folder = self.corpus("docs9")
        def responder(body):
            text = " ".join(m["content"] for m in body["messages"])
            if body["model"] != "chat":
                return ["ok"]
            return "no es json" if "Files found so far" in text else '["x y", "z w"]'
        self.fake.responder = responder
        _, doc = self.run_and_wait({"models": ["emb"], "repetitions": 1, "corpus_folder": str(folder),
                                    "subquery_llm": "chat", "subquery_count": 2, "iter_rounds": 2})
        it = self.steps(doc)["iterative"]
        self.assertEqual(it["status"], "ok")
        how = next(q for q in it["queries"] if q["kind"] == "how-es")
        self.assertEqual((how["rounds"], how["errors"]), (1, 1))

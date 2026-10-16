import http.client
import json
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path

from eli import server as srv
from eli.gpu import GPUMonitor
from eli.ollama import Ollama

from fakes import DEFAULT_MODEL, FAKE_GPU_COMMAND, FakeOllama
from test_bench_bank import known_good

HW = {"gpu": {"name": "gpu-falsa", "driver": "0", "compute_cap": 8.9, "vram_total_mib": 16380}, "cpu": {"model": "cpu", "threads": 8},
      "ram_total_mib": 32000.0, "os": "test", "python": "3"}
WEB = Path(srv.WEB_DIR)


class ServerBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fake = FakeOllama()
        cls.gpu = GPUMonitor(command=FAKE_GPU_COMMAND)
        cls.gpu.start()
        cls.app = srv.App(Ollama(cls.fake.url, timeout=5), cls.gpu, Path(cls.tmp.name), 0, hardware_fn=lambda: HW)
        cls.server = srv.make_server(cls.app, 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        deadline = time.time() + 5
        while len(cls.gpu.history(5)) < 10 and time.time() < deadline:
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.app.bench.cancel()
        cls.app.bench.join(10)
        cls.server.shutdown()
        cls.server.server_close()
        cls.gpu.stop()
        cls.fake.close()
        cls.tmp.cleanup()

    def setUp(self):
        self.fake.models = {"m1": dict(DEFAULT_MODEL), "m2": {**DEFAULT_MODEL, "digest": "sha256:m2", "capabilities": ["completion"]},
                            "emb": {**DEFAULT_MODEL, "digest": "sha256:emb", "capabilities": ["embedding"]}}
        self.fake.loaded = set()
        for attr, val in (("chunk_delay", 0.0), ("chat_error", None), ("responder", None), ("done_reason", None)):
            setattr(self.fake, attr, val)
        self.fake.tokens = ["ho", "la", " ", "mundo"]
        self.fake.unloaded.clear()
        self.fake.deleted.clear()
        self.fake.chat_bodies.clear()
        self.fake.client_disconnected.clear()

    def request(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        return resp.status, payload, resp

    def api(self, method, path, body=None, **kw):
        status, payload, _ = self.request(method, path, body, **kw)
        try:
            return status, json.loads(payload or b"{}")
        except ValueError:
            return status, payload.decode()

    def wait_bench(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.api("GET", "/api/bench/status")[1]
            if not st.get("active"):
                return st
            time.sleep(0.05)
        self.fail("el benchmark no termino")

    def finish_bench(self, body):
        status, data = self.api("POST", "/api/bench/start", body)
        self.assertEqual(status, 202, data)
        return data["run_id"], self.wait_bench()


class InfoEndpointTests(ServerBase):
    def test_version_system_and_dashboard(self):
        v = self.api("GET", "/api/version")[1]
        self.assertEqual(v["eli"], "0.2.0")
        self.assertEqual(v["suite"]["id"], "eli-core")
        self.assertRegex(v["suite"]["sha256"], r"^[0-9a-f]{64}$")
        system = self.api("GET", "/api/system")[1]
        self.assertEqual(system["hardware"]["gpu"]["name"], "gpu-falsa")
        self.assertIn("available", system["sandbox"])
        d = self.api("GET", "/api/dashboard")[1]
        self.assertEqual(d["models"]["count"], 3)
        self.assertEqual(d["suite"]["tests"], len(self.app.suite.tests))
        self.assertIn("recent_runs", d)
        self.assertFalse(d["bench"]["active"] if "active" in d["bench"] else False)

    def test_suite_endpoint_describes_the_bank(self):
        s = self.api("GET", "/api/bench/suite")[1]
        self.assertEqual(s["tests"][0].keys() >= {"id", "category", "difficulty", "quick"}, True)
        self.assertNotIn("prompt", s["tests"][0])
        self.assertIn("prompt", self.api("GET", "/api/bench/suite?prompts=1")[1]["tests"][0])
        self.assertEqual(s["context_sizes"]["quick"], [2048])


class InspectorEndpointTests(ServerBase):
    def test_inspect_lists_every_model_with_metadata_and_trtllm_analysis(self):
        data = self.api("GET", "/api/models/inspect")[1]
        by = {m["name"]: m for m in data["models"]}
        self.assertEqual(set(by), {"m1", "m2", "emb"})
        m1 = by["m1"]
        self.assertEqual((m1["format"], m1["quantization"]["value"], m1["architecture"], m1["parameters_b"]), ("gguf", "Q4", "qwen2", 8.0))
        self.assertEqual(m1["context_length"], 131072)
        self.assertEqual(m1["trtllm"]["dimensions"]["formato"], "requiere_conversion")
        self.assertRegex(data["rules_version"], r"^\d{4}-")
        self.assertEqual(by["emb"]["capabilities"], ["embedding"])

    def test_incomplete_metadata_does_not_break_the_inventory(self):
        self.fake.models["raro"] = {**DEFAULT_MODEL, "architecture": None, "format": None, "family": None, "parameter_size": None,
                                    "parameter_count": None, "quantization_level": None, "capabilities": []}
        data = self.api("GET", "/api/models/inspect")[1]
        raro = next(m for m in data["models"] if m["name"] == "raro")
        for field in ("format", "architecture", "parameter_size", "quantization"):
            self.assertIn(field, raro["unknown"])
        self.assertEqual(raro["trtllm"]["verdict"], "no_determinada")
        self.assertEqual(len(data["models"]), 4)

    def test_inspect_one_and_unknown_model(self):
        self.assertEqual(self.api("GET", "/api/models/inspect?name=m2")[1]["models"][0]["name"], "m2")
        self.assertEqual(self.api("GET", "/api/models/inspect?name=nada")[0], 404)

    def test_ps_failure_still_lists_models_without_load_state(self):
        self.fake.ps_error = True
        try:
            models = self.api("GET", "/api/models")[1]["models"]
            self.assertEqual(len(models), 3)
            self.assertTrue(all(not m["loaded"] for m in models))
        finally:
            self.fake.ps_error = False


class ModelManagerTests(ServerBase):
    def test_unload_keeps_the_model_on_disk(self):
        self.fake.loaded = {"m1"}
        status, data = self.api("POST", "/api/unload", {"model": "m1"})
        self.assertEqual((status, data["unloaded"]), (200, ["m1"]))
        self.assertIn("m1", self.fake.models)
        self.assertEqual(self.fake.deleted, [])

    def test_delete_requires_the_exact_name(self):
        for body in ({"model": "m2"}, {"model": "m2", "confirm": "m"}, {"model": "m2", "confirm": "M2"}, {"model": "m2", "confirm": ""}):
            status, data = self.api("POST", "/api/models/delete", body)
            self.assertEqual(status, 400, body)
        self.assertEqual(self.fake.deleted, [])
        self.assertIn("m2", self.fake.models)

    def test_exact_confirmation_deletes_only_that_model(self):
        status, data = self.api("POST", "/api/models/delete", {"model": "m2", "confirm": "m2"})
        self.assertEqual((status, data["deleted"]), (200, "m2"))
        self.assertEqual(self.fake.deleted, ["m2"])
        self.assertEqual(set(self.fake.models), {"m1", "emb"})

    def test_bulk_or_wildcard_deletes_are_refused(self):
        for body in ({"model": ["m1", "m2"], "confirm": ["m1", "m2"]}, {"model": "*", "confirm": "*"}, {"model": "m1,m2", "confirm": "m1,m2"},
                     {"model": "m*", "confirm": "m*"}, {"models": ["m1"], "confirm": "m1"}, {"model": None, "confirm": None}):
            self.assertEqual(self.api("POST", "/api/models/delete", body)[0], 400, body)
        self.assertEqual(self.fake.deleted, [])

    def test_loaded_model_must_be_unloaded_first(self):
        self.fake.loaded = {"m2"}
        status, data = self.api("POST", "/api/models/delete", {"model": "m2", "confirm": "m2"})
        self.assertEqual(status, 400)
        self.assertIn("VRAM", data["error"])
        self.assertEqual(self.fake.deleted, [])
        self.api("POST", "/api/unload", {"model": "m2"})
        self.assertEqual(self.api("POST", "/api/models/delete", {"model": "m2", "confirm": "m2"})[0], 200)

    def test_unknown_model_is_404_and_foreign_origin_is_blocked(self):
        self.assertEqual(self.api("POST", "/api/models/delete", {"model": "nada", "confirm": "nada"})[0], 404)
        status, _ = self.api("POST", "/api/models/delete", {"model": "m2", "confirm": "m2"}, headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.fake.deleted, [])

    def test_preview_shows_the_exact_name_and_does_not_count_shared_weights(self):
        self.fake.models["alias"] = dict(self.fake.models["m2"])  # mismo digest que m2
        shared = self.api("GET", "/api/models/delete-preview?name=m2")[1]
        self.assertEqual((shared["name"], shared["reclaimable"]["kind"], shared["reclaimable"]["bytes"]), ("m2", "ninguno", 0))
        self.assertEqual(shared["shares_weights_with"], ["alias"])
        alone = self.api("GET", "/api/models/delete-preview?name=m1")[1]
        self.assertEqual(alone["reclaimable"]["kind"], "estimacion")
        self.assertIn("no es lo mismo", alone["note"].lower())
        self.assertEqual(self.api("GET", "/api/models/delete-preview?name=nada")[0], 404)

    def test_no_destructive_operation_while_a_generation_is_running(self):
        self.app.chat_lock.acquire()
        try:
            self.assertEqual(self.api("POST", "/api/models/delete", {"model": "m2", "confirm": "m2"})[0], 409)
            self.assertEqual(self.api("POST", "/api/unload", {})[0], 409)
        finally:
            self.app.chat_lock.release()
        self.assertEqual(self.fake.deleted, [])


class BenchmarkApiTests(ServerBase):
    GOOD = {t.prompt: known_good(t) for t in srv.bank.default_suite().tests if not t.context_sized and known_good(t)}
    BODY = {"models": ["m1"], "mode": "quick", "categories": ["math"], "unload_others": True}

    def test_invalid_configuration_is_rejected_with_a_clear_message(self):
        for body, fragment in (({}, "al menos un modelo"), ({"models": ["nada"]}, "no esta instalado"), ({"models": ["emb"]}, "no genera texto"),
                               ({"models": ["m1"], "repetitions": 50}, "repetitions")):
            status, data = self.api("POST", "/api/bench/start", body)
            self.assertEqual(status, 400, body)
            self.assertIn(fragment, data["error"])
        self.assertEqual(self.api("POST", "/api/bench/start", raw=b"[1,2]")[0], 400)

    def test_loaded_models_require_confirmation_over_the_api(self):
        self.fake.loaded = {"m2"}
        status, data = self.api("POST", "/api/bench/start", {"models": ["m1"], "categories": ["math"]})
        self.assertEqual((status, data["code"], data["loaded"]), (409, "models_loaded", ["m2"]))
        self.assertEqual(self.fake.unloaded, [])

    def test_run_history_detail_exports_and_comparison(self):
        self.fake.responder = lambda body: self.GOOD.get(body["messages"][-1]["content"], "zzz")
        run_id, st = self.finish_bench(self.BODY)
        self.assertEqual((st["state"], st["done"], st["total"]), ("completed", 4, 4))
        listing = self.api("GET", "/api/history")[1]
        self.assertEqual(listing["runs"][0]["run_id"], run_id)
        self.assertEqual(listing["runs"][0]["counts"]["passed"], 4)
        detail = self.api("GET", f"/api/history/{run_id}")[1]
        self.assertEqual(len(detail["results"]), 4)
        self.assertEqual(detail["run"]["suite"]["id"], "eli-core")
        status, payload, resp = self.request("GET", f"/api/history/{run_id}/export?format=json")
        self.assertEqual(status, 200)
        self.assertIn(f"eli-run-{run_id}.json", resp.getheader("Content-Disposition"))
        self.assertEqual(json.loads(payload)["run"]["run_id"], run_id)
        status, payload, resp = self.request("GET", f"/api/history/{run_id}/export?format=csv")
        self.assertEqual((status, resp.getheader("Content-Type")), (200, "text/csv; charset=utf-8"))
        self.assertEqual(len(payload.decode().strip().splitlines()), 5)
        cmp_data = self.api("GET", "/api/compare?models=m1")[1]
        self.assertEqual(cmp_data["models"][0]["name"], "m1")
        self.assertEqual(cmp_data["models"][0]["overall"]["passed"], 4)
        latest = self.api("GET", "/api/compare/models")[1]
        self.assertEqual([m["name"] for m in latest["models"]], ["m1"])
        self.assertEqual(latest["models"][0]["meta"]["quantization"], "Q4")

    def test_history_is_read_only_and_ids_cannot_escape(self):
        run_id, _ = self.finish_bench(self.BODY)
        self.assertEqual(self.request("DELETE", f"/api/history/{run_id}")[0], 501)
        self.assertEqual(self.api("GET", f"/api/history/{run_id}")[0], 200)
        for bad in ("..", "..%2F..%2Fetc", "20261015-000000-zzzzzz"):
            self.assertEqual(self.api("GET", f"/api/history/{bad}")[0], 404)
        self.assertEqual(self.api("GET", f"/api/history/{run_id}/export?format=xml")[0], 400)

    def test_chat_is_refused_while_a_benchmark_runs_and_cancel_works(self):
        self.fake.chunk_delay = 0.1
        self.fake.tokens = ["x"] * 5
        status, data = self.api("POST", "/api/bench/start", {**self.BODY, "categories": ["math", "logic"], "repetitions": 3})
        self.assertEqual(status, 202)
        time.sleep(0.4)
        self.assertTrue(self.api("GET", "/api/status")[1]["bench_running"])
        chat = self.api("POST", "/api/chat", {"model": "m1", "messages": [{"role": "user", "content": "hola"}]}, raw=None)
        self.assertEqual(chat[0], 409)
        self.assertEqual(self.api("POST", "/api/bench/cancel", {})[1], {"cancelling": True})
        st = self.wait_bench()
        self.assertEqual(st["state"], "cancelled")
        self.assertFalse(self.app.chat_lock.locked())


class ChatReliabilityTests(ServerBase):
    CHAT = {"model": "m1", "messages": [{"role": "user", "content": "hola"}]}

    def chat_files(self):
        return sorted(Path(self.tmp.name).glob("*_chat*.json"))

    def test_malformed_options_are_a_400_not_a_crash(self):
        for opts in ("abc", [1], 7, {"num_ctx": True}, {"temperature": float("nan")}, {"num_ctx": "x"}):
            raw = json.dumps({**self.CHAT, "options": opts}).encode()
            self.assertEqual(self.api("POST", "/api/chat", raw=raw)[0], 400, opts)
        for bad in ({**self.CHAT, "system": 5}, {**self.CHAT, "think": "si"}, {"model": "m1", "messages": [{"role": "user"}]},
                    {"model": "m1", "messages": [{"role": "user", "content": "x"}] * (srv.MAX_MESSAGES + 1)}):
            self.assertEqual(self.api("POST", "/api/chat", bad)[0], 400)

    def test_model_must_be_installed_and_post_must_be_json(self):
        self.assertEqual(self.api("POST", "/api/chat", {**self.CHAT, "model": "nada"})[0], 400)
        status, data = self.api("POST", "/api/chat", raw=json.dumps(self.CHAT).encode(), headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 400)
        self.assertIn("application/json", data["error"])

    def test_two_chats_in_the_same_second_never_overwrite_each_other(self):
        before = set(self.chat_files())
        for _ in range(3):
            self.api("POST", "/api/chat", self.CHAT)
        new = set(self.chat_files()) - before
        self.assertEqual(len(new), 3)
        self.assertEqual(len({json.loads(p.read_text())["started"] + p.name for p in new}), 3)

    def test_a_cancelled_chat_keeps_its_partial_record(self):
        self.fake.chunk_delay = 0.1
        self.fake.tokens = ["x"] * 40
        before = set(self.chat_files())
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/api/chat", body=json.dumps(self.CHAT), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.fp.readline()
        resp.fp.readline()
        resp.close()
        conn.close()
        self.assertTrue(self.fake.client_disconnected.wait(5))
        deadline = time.time() + 5
        while self.app.chat_lock.locked() and time.time() < deadline:
            time.sleep(0.05)
        new = set(self.chat_files()) - before
        self.assertEqual(len(new), 1)
        record = json.loads(new.pop().read_text())
        self.assertEqual(record["status"], "cancelled")
        self.assertTrue(record["response"]["content"].startswith("x"))

    def test_ollama_error_is_recorded_with_its_message(self):
        self.fake.chat_error = "out of memory"
        before = set(self.chat_files())
        _, payload, _ = self.request("POST", "/api/chat", self.CHAT)
        self.assertIn(b'"type": "error"', payload)
        record = json.loads((set(self.chat_files()) - before).pop().read_text())
        self.assertEqual(record["status"], "error")
        self.assertIn("out of memory", record["error"])


class UiCompatibilityTests(ServerBase):
    """La interfaz solo usa elementos y rutas que existen."""

    JS = {p.name: p.read_text() for p in WEB.glob("*.js")}
    HTML = (WEB / "index.html").read_text()
    DYNAMIC_IDS = {"hd-results", "hf-model", "hf-cat", "hf-out"}  # los crea el propio JS

    def test_every_served_file_exists_and_every_script_is_loaded(self):
        for path, (name, kind) in srv.STATIC.items():
            status, payload, resp = self.request("GET", path)
            self.assertEqual(status, 200, path)
            self.assertEqual(resp.getheader("Content-Type"), kind)
            self.assertTrue(payload)
        for src in re.findall(r'<script src="(/[^"]+)"', self.HTML):
            self.assertIn(src, srv.STATIC, f"{src} se carga en la pagina pero el servidor no lo sirve")
        for js in WEB.glob("*.js"):
            self.assertIn(f'src="/{js.name}"', self.HTML, f"{js.name} no se carga en index.html")

    def test_every_element_id_used_by_the_scripts_exists_in_the_page(self):
        ids = set(re.findall(r'\bid="([^"]+)"', self.HTML)) | self.DYNAMIC_IDS
        missing = []
        for name, code in self.JS.items():
            for ref in set(re.findall(r"\$\('([A-Za-z][\w-]*)'\)", code)) | set(re.findall(r"getElementById\('([^']+)'\)", code)):
                if ref not in ids:
                    missing.append((name, ref))
        self.assertEqual(missing, [])

    def test_every_view_has_a_nav_button_and_a_section(self):
        nav = set(re.findall(r'data-nav="(\w+)"', self.HTML))
        views = set(re.findall(r'<section data-view="(\w+)"', self.HTML))
        self.assertEqual(nav, views)
        self.assertEqual(set(re.search(r"const VIEWS = \[([^\]]+)\]", self.JS["nav.js"]).group(1).replace("'", "").replace(" ", "").split(",")), views)

    def test_every_api_route_called_by_the_scripts_exists(self):
        calls = set()
        for code in self.JS.values():
            calls |= set(re.findall(r"""['"`](/api/[A-Za-z0-9/_-]*)""", code))
        self.assertGreater(len(calls), 12)
        post_only = {"/api/chat", "/api/unload", "/api/models/delete", "/api/bench/start", "/api/bench/cancel",
                     "/api/emb/start", "/api/emb/cancel", "/api/emb/corpus-preview"}
        for path in sorted(calls):
            target = path.rstrip("/") if path.endswith("/") and path != "/api/history/" else path
            if path == "/api/history/":
                target = "/api/history/20261015-000000-aaaaaa"  # el JS le anade el id de la ejecucion
            method = "POST" if path in post_only else "GET"
            if path == "/api/gpu/stream":
                continue  # flujo SSE infinito: se comprueba en test_server.py
            status, payload, _ = self.request(method, target, {} if method == "POST" else None)
            error = json.loads(payload).get("error") if payload[:1] == b"{" else ""
            self.assertFalse(status == 404 and error == "No existe", f"{method} {path}: la interfaz llama a una ruta inexistente")

    @staticmethod
    def top_level_args(text):
        args, depth, cur = [], 0, ""
        for ch in text:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                args.append(cur)
                cur = ""
            else:
                cur += ch
        return args + [cur]

    def test_append_is_never_called_with_a_nullable_argument(self):
        """append(null) escribe el texto «null» en pantalla: un argumento opcional debe pasar por put() o por h()."""
        offenders = []
        for name, code in self.JS.items():
            for m in re.finditer(r"\.append\(", code):
                depth, i = 1, m.end()
                while i < len(code) and depth:
                    depth += {"(": 1, ")": -1}.get(code[i], 0)
                    i += 1
                for arg in self.top_level_args(code[m.end():i - 1]):
                    if re.search(r"\?\s*null\s*:|:\s*null\s*$|&&\s*h\(", arg.strip()):
                        offenders.append((name, arg.strip()[:60].replace("\n", " ")))
        self.assertEqual(offenders, [])

    def test_new_pages_do_not_use_unsafe_html_insertion(self):
        for name, code in self.JS.items():
            self.assertNotRegex(code, r"\.innerHTML\s*=", f"{name} inserta HTML sin escapar")
            self.assertNotIn("insertAdjacentHTML", code)
            self.assertNotIn("eval(", code)


if __name__ == "__main__":
    unittest.main()

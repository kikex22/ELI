import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from eli.gpu import GPUMonitor
from eli.ollama import Ollama
from eli.server import App, BadRequest, make_server, parse_chat_request

from fakes import FAKE_GPU_COMMAND, FakeOllama


class ParseChatRequestTests(unittest.TestCase):
    def base(self, **extra):
        return {"model": "m1", "messages": [{"role": "user", "content": "hola"}], **extra}

    def test_defaults(self):
        req = parse_chat_request(self.base())
        self.assertEqual(req["options"], {"num_ctx": 4096, "temperature": 0.0, "seed": 42, "num_predict": 512})
        self.assertIsNone(req["think"])

    def test_system_prompt_is_prepended(self):
        req = parse_chat_request(self.base(system="  Responde corto  "))
        self.assertEqual(req["messages"][0], {"role": "system", "content": "Responde corto"})

    def test_rejects_bad_input(self):
        for body in (
            {}, {"model": "", "messages": [{"role": "user", "content": "x"}]}, {"model": "m", "messages": []},
            self.base(messages=[{"role": "bot", "content": "x"}]), self.base(messages=[{"role": "user", "content": 5}]),
            self.base(options={"num_ctx": 10}), self.base(options={"num_ctx": "abc"}), self.base(think="si"),
        ):
            with self.assertRaises(BadRequest, msg=str(body)):
                parse_chat_request(body)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fake = FakeOllama()
        cls.gpu = GPUMonitor(command=FAKE_GPU_COMMAND)
        cls.gpu.start()
        cls.app = App(Ollama(cls.fake.url, timeout=5), cls.gpu, Path(cls.tmp.name), 0)
        cls.server = make_server(cls.app, 0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        deadline = time.time() + 5
        while len(cls.gpu.history(5)) < 10 and time.time() < deadline:
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.gpu.stop()
        cls.fake.close()
        cls.tmp.cleanup()

    def setUp(self):
        self.fake.chunk_delay = 0.0
        self.fake.chat_error = None
        self.fake.unloaded.clear()
        self.fake.chat_bodies.clear()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        return conn, conn.getresponse()

    def get_json(self, path):
        conn, resp = self.request("GET", path)
        data = json.load(resp)
        conn.close()
        return resp.status, data

    def chat(self, body):
        conn, resp = self.request("POST", "/api/chat", body)
        raw = resp.read().decode()
        conn.close()
        events = [json.loads(chunk[6:]) for chunk in raw.split("\n\n") if chunk.startswith("data: ")]
        return resp.status, events, raw

    CHAT = {"model": "m1", "messages": [{"role": "user", "content": "hola"}]}

    def test_static_page_is_served(self):
        conn, resp = self.request("GET", "/")
        html = resp.read().decode()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertIn("ELI", html)
        self.assertIn("text/html", resp.getheader("Content-Type"))

    def test_foreign_host_and_origin_are_rejected(self):
        conn, resp = self.request("GET", "/api/status", headers={"Host": "evil.example"})
        self.assertEqual(resp.status, 403)
        conn.close()
        conn, resp = self.request("POST", "/api/chat", self.CHAT, headers={"Origin": "http://evil.example"})
        self.assertEqual(resp.status, 403)
        conn.close()
        self.assertEqual(self.fake.chat_bodies, [])

    def test_status_models_model_and_gpu_history(self):
        status, data = self.get_json("/api/status")
        self.assertEqual(status, 200)
        self.assertTrue(data["gpu"]["available"])
        self.assertEqual(data["gpu"]["mem_total_mib"], 16380.0)
        self.assertEqual(data["ollama"]["version"], "0.0-test")
        self.assertEqual(self.get_json("/api/models")[1]["models"][0]["name"], "m1")
        self.assertEqual(self.get_json("/api/model?name=m1")[1]["context_length"], 131072)
        self.assertEqual(self.get_json("/api/model")[0], 400)
        self.assertEqual(self.get_json("/api/model?name=missing")[0], 502)
        self.assertGreater(len(self.get_json("/api/gpu/history?seconds=30")[1]["samples"]), 0)
        self.assertEqual(self.get_json("/api/nada")[0], 404)

    def test_chat_streams_tokens_and_final_metrics(self):
        status, events, _ = self.chat({**self.CHAT, "options": {"num_ctx": 4096}, "think": None})
        self.assertEqual(status, 200)
        text = "".join(e["text"] for e in events if e["type"] == "token" and e["kind"] == "content")
        self.assertEqual(text, "hola mundo")
        done = events[-1]
        self.assertEqual(done["type"], "done")
        m = done["metrics"]
        self.assertEqual((m["gen_tokens"], m["decode_tok_s"]), (4, 10.0))
        self.assertEqual(m["gpu"]["request"]["vram_peak_mib"], 9000.0)
        self.assertEqual(m["gpu"]["request"]["throttle"], ["limite de potencia"])
        self.assertTrue(any("frenada" in w for w in m["warnings"]))
        self.assertEqual(m["gpu"]["decode"]["baseline_power_w"], 30.0)  # reposo, no la carga
        self.assertIsNotNone(m["gpu"]["decode"]["energy_net_wh"])
        self.assertIn("wh_per_token", m)

    def test_chat_run_is_saved(self):
        before = set(Path(self.tmp.name).glob("*_chat.json"))
        self.chat(self.CHAT)
        new = set(Path(self.tmp.name).glob("*_chat.json")) - before
        self.assertEqual(len(new), 1)
        record = json.loads(new.pop().read_text())
        self.assertEqual((record["model"], record["response"]["content"]), ("m1", "hola mundo"))
        self.assertGreater(len(record["gpu_samples"]), 0)

    def test_chat_validation_and_ollama_errors(self):
        status, _, _ = self.chat({"model": "m1"})
        self.assertEqual(status, 400)
        self.fake.chat_error = "out of memory"
        status, events, _ = self.chat(self.CHAT)
        self.assertEqual(status, 200)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("out of memory", events[-1]["message"])

    def test_second_chat_is_rejected_while_one_runs(self):
        self.fake.chunk_delay = 0.4
        results = {}

        def first():
            results["first"] = self.chat(self.CHAT)

        t = threading.Thread(target=first)
        t.start()
        time.sleep(0.5)
        status, _, _ = self.chat(self.CHAT)
        t.join(timeout=10)
        self.assertEqual(status, 409)
        self.assertEqual(results["first"][1][-1]["type"], "done")

    def test_disconnect_cancels_generation_and_releases_lock(self):
        self.fake.chunk_delay = 0.1
        self.fake.tokens = ["x"] * 40
        self.fake.client_disconnected.clear()
        try:
            conn, resp = self.request("POST", "/api/chat", self.CHAT)
            resp.fp.readline()  # espera el primer evento
            resp.close()  # http.client ya solto el socket de la conexion; cerrar la respuesta lo cierra
            self.assertTrue(self.fake.client_disconnected.wait(5), "Ollama no vio la desconexion")
        finally:
            self.fake.tokens = ["ho", "la", " ", "mundo"]
        deadline = time.time() + 5
        while self.app.chat_lock.locked() and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(self.app.chat_lock.locked())

    def test_unload(self):
        conn, resp = self.request("POST", "/api/unload", {})
        data = json.load(resp)
        conn.close()
        self.assertEqual(data["unloaded"], ["m1"])
        self.assertEqual(self.fake.unloaded, ["m1"])


if __name__ == "__main__":
    unittest.main()

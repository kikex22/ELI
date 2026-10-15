import unittest

from eli.metrics import chat_metrics
from eli.ollama import Ollama, OllamaError

from fakes import FINAL, FakeOllama


class OllamaClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.client = Ollama(self.fake.url, timeout=5)

    def tearDown(self):
        self.fake.close()

    def test_models_merge_loaded_state(self):
        (m,) = self.client.models()
        self.assertEqual((m["name"], m["parameter_size"], m["quantization"], m["loaded"]), ("m1", "8B", "Q4", True))
        self.assertEqual((m["vram_bytes"], m["loaded_context"]), (3_000_000_000, 4096))

    def test_show_normalizes_context_and_capabilities(self):
        info = self.client.show("m1")
        self.assertEqual((info["context_length"], info["embedding_length"]), (131072, 2560))
        self.assertIn("thinking", info["capabilities"])

    def test_show_missing_model_raises_with_server_message(self):
        with self.assertRaises(OllamaError) as ctx:
            self.client.show("missing")
        self.assertIn("not found", str(ctx.exception))

    def test_chat_stream_yields_tokens_then_done(self):
        events = list(self.client.chat_stream("m1", [{"role": "user", "content": "hi"}], {"num_ctx": 4096}, think=False))
        self.assertEqual([e for k, e in events if k == "content"], ["ho", "la", " ", "mundo"])
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(events[-1][1]["eval_count"], 4)
        body = self.fake.chat_bodies[0]
        self.assertEqual((body["stream"], body["think"], body["options"]), (True, False, {"num_ctx": 4096}))

    def test_think_is_omitted_when_none(self):
        list(self.client.chat_stream("m1", [{"role": "user", "content": "hi"}], {}))
        self.assertNotIn("think", self.fake.chat_bodies[0])

    def test_stream_error_becomes_ollama_error(self):
        self.fake.chat_error = "out of memory"
        with self.assertRaises(OllamaError) as ctx:
            list(self.client.chat_stream("m1", [{"role": "user", "content": "hi"}], {}))
        self.assertIn("out of memory", str(ctx.exception))

    def test_unload_sends_keep_alive_zero(self):
        self.client.unload("m1")
        self.assertEqual(self.fake.unloaded, ["m1"])

    def test_connection_refused_is_friendly(self):
        with self.assertRaises(OllamaError) as ctx:
            Ollama("http://127.0.0.1:1", timeout=2).models()
        self.assertIn("No se pudo conectar", str(ctx.exception))


class MetricsTests(unittest.TestCase):
    def ps(self, size=3000, vram=3000, ctx=4096):
        return {"size": size, "size_vram": vram, "context_length": ctx}

    def test_rates_and_times(self):
        m = chat_metrics(FINAL, 100.0, 100.6, 100.7, 102.0, 4096, self.ps())
        self.assertEqual(m["decode_tok_s"], 10.0)  # 4 tokens / 0.4 s
        self.assertEqual(m["prefill_tok_s"], 200.0)  # 20 tokens / 0.1 s
        self.assertEqual((m["load_s"], m["total_s"]), (0.5, 2.0))
        self.assertEqual((m["ttft_s"], m["time_to_answer_s"], m["wall_s"]), (0.6, 0.7, 2.0))
        self.assertEqual(m["gpu_fraction"], 1.0)
        self.assertEqual(m["warnings"], [])

    def test_partial_gpu_offload_warns(self):
        m = chat_metrics(FINAL, 0, 1, 1, 2, 4096, self.ps(size=18000, vram=12000))
        self.assertEqual(m["gpu_fraction"], 0.667)
        self.assertTrue(any("no cabe entero" in w for w in m["warnings"]))

    def test_context_mismatch_and_full_context_warn(self):
        m = chat_metrics({**FINAL, "prompt_eval_count": 4095}, 0, 1, 1, 2, 4096, self.ps(ctx=2048))
        text = " ".join(m["warnings"])
        self.assertIn("se lleno", text)
        self.assertIn("2048", text)

    def test_length_stop_warns(self):
        m = chat_metrics({**FINAL, "done_reason": "length"}, 0, 1, 1, 2, 4096, None)
        self.assertTrue(any("limite de tokens" in w for w in m["warnings"]))

    def test_missing_durations_do_not_crash(self):
        m = chat_metrics({"done": True}, 0, None, None, 1, 4096, None)
        self.assertIsNone(m["decode_tok_s"])
        self.assertIsNone(m["ttft_s"])


if __name__ == "__main__":
    unittest.main()

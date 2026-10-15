"""Integracion con Ollama y la GPU REALES. Apagada por defecto:

    ELI_INTEGRATION=1 PYTHONPATH=tests python3 -m unittest tests.test_integration_ollama -v

Mide de verdad con el modelo instalado mas pequeno que genere texto (categoria de matematicas, modo rapido).
No descarga modelos, no borra nada y guarda las ejecuciones en una carpeta temporal.
"""
import os
import tempfile
import time
import unittest
from pathlib import Path

from eli import server as srv
from eli.gpu import GPUMonitor
from eli.ollama import Ollama, OllamaError

ENABLED = os.environ.get("ELI_INTEGRATION") == "1"


@unittest.skipUnless(ENABLED, "integracion real desactivada (ELI_INTEGRATION=1 para activarla)")
class RealOllamaTests(unittest.TestCase):
    def test_a_tiny_real_benchmark_has_real_measurements(self):
        ollama = Ollama()
        try:
            rows = ollama.models()
        except OllamaError as exc:
            self.skipTest(f"Ollama no responde: {exc}")
        candidates = [r for r in rows if "completion" in ollama.show(r["name"])["capabilities"] and r.get("disk_bytes")]
        if not candidates:
            self.skipTest("no hay modelos que generen texto")
        model = min(candidates, key=lambda r: r["disk_bytes"])["name"]
        gpu = GPUMonitor()
        gpu.start()
        try:
            deadline = time.time() + 10
            while len(gpu.history(10)) < 8 and time.time() < deadline:
                time.sleep(0.1)
            self.assertTrue(gpu.available, "nvidia-smi no entrega lecturas")
            with tempfile.TemporaryDirectory() as tmp:
                app = srv.App(ollama, gpu, Path(tmp), 0)
                run_id = app.bench.start({"models": [model], "mode": "quick", "categories": ["math"], "unload_others": True})
                app.bench.join(300)
                doc = app.store.load_run(run_id)
            self.assertEqual(doc["status"]["state"], "completed", doc["status"])
            self.assertEqual(len(doc["results"]), 4)
            self.assertEqual(doc["run"]["models"][0]["name"], model)
            for r in doc["results"]:
                m = r["metrics"]
                self.assertEqual(r["gen_status"], "ok", r["error"])
                self.assertGreater(m["gen_tokens"], 0)
                self.assertGreater(m["prompt_tokens"], 0)
                self.assertGreater(m["gpu"]["request"]["vram_peak_mib"], 0)  # lectura real de nvidia-smi
                self.assertGreater(m["gpu"]["request"]["samples"], 0)
                self.assertEqual(m["provenance"]["gpu"], "nvidia-smi")
            print(f"\n[integracion real] {model}: {[(r['instance_id'], r['outcome'], r['metrics']['decode_tok_s']) for r in doc['results']]}")
            deadline = time.time() + 10  # Ollama descarga de forma asincrona tras la orden
            while [m for m in ollama.loaded() if m["name"] == model] and time.time() < deadline:
                time.sleep(0.25)
            self.assertEqual([m for m in ollama.loaded() if m["name"] == model], [], "el modelo debia descargarse de la VRAM al terminar")
        finally:
            gpu.stop()


if __name__ == "__main__":
    unittest.main()

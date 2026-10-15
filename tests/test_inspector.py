import unittest

from eli import inspector as ins

HW_4060 = {"gpu": {"name": "RTX 4060 Ti", "compute_cap": 8.9, "vram_total_mib": 16380}}
HW_OLD = {"gpu": {"name": "GPU vieja", "compute_cap": 8.0, "vram_total_mib": 8192}}


def show(**kw):
    base = {"name": "m", "format": "gguf", "family": "qwen2", "architecture": "qwen2", "parameter_size": "7.6B", "quantization": "Q4_K_M",
            "context_length": 32768, "capabilities": ["completion"], "modified_at": "2026-10-01T00:00:00Z", "parameter_count": None}
    return {**base, **kw}


def dim(info, name):
    return info["trtllm"]["dimensions"][name]


class ParsingTests(unittest.TestCase):
    def test_parameter_parsing(self):
        self.assertEqual(ins.parse_params_b("8.0B"), 8.0)
        self.assertEqual(ins.parse_params_b("334M"), 0.334)
        self.assertEqual(ins.parse_params_b("25.2B"), 25.2)
        self.assertEqual(ins.parse_params_b("???", 7_000_000_000), 7.0)  # cae al conteo exacto
        self.assertIsNone(ins.parse_params_b(None, None))
        self.assertIsNone(ins.parse_params_b("grande"))

    def test_quantization_kinds_are_not_inferred(self):
        self.assertEqual(ins.classify_quantization("Q4_K_M")["kind"], "gguf-quant")
        self.assertEqual(ins.classify_quantization("IQ3_M")["kind"], "gguf-quant")
        self.assertEqual(ins.classify_quantization("Q8_0")["kind"], "gguf-quant")
        self.assertEqual(ins.classify_quantization("BF16")["kind"], "float")
        self.assertEqual(ins.classify_quantization("F16")["kind"], "float")
        self.assertEqual(ins.classify_quantization("FP8")["kind"], "other")  # no se da por conocido lo que no esta en las reglas
        self.assertEqual(ins.classify_quantization(None), {"value": None, "kind": "unknown"})
        self.assertEqual(ins.classify_quantization(""), {"value": None, "kind": "unknown"})

    def test_size_buckets(self):
        self.assertEqual([ins.size_bucket(x) for x in (0.5, 3, 8, 14, 30, None)], ["<2B", "2-5B", "5-10B", "10-20B", "20B+", None])


class InspectionTests(unittest.TestCase):
    def test_format_precision_and_architecture_stay_separate(self):
        info = ins.inspect_model(show(), {"name": "m", "digest": "sha256:abc", "disk_bytes": 4_700_000_000}, None, None, HW_4060)
        self.assertEqual((info["format"], info["quantization"]["value"], info["architecture"], info["family"]), ("gguf", "Q4_K_M", "qwen2", "qwen2"))
        self.assertEqual((info["parameters_b"], info["context_length"], info["digest"]), (7.6, 32768, "sha256:abc"))
        self.assertEqual(info["unknown"], [])

    def test_missing_metadata_is_reported_as_unknown_not_invented(self):
        info = ins.inspect_model({"name": "raro", "capabilities": []}, None, None, None, HW_4060)
        self.assertIsNone(info["format"])
        self.assertIsNone(info["architecture"])
        self.assertIsNone(info["quantization"]["value"])
        self.assertIsNone(info["parameters_b"])
        for field in ("format", "architecture", "parameter_size", "context_length", "digest", "quantization", "disk_bytes"):
            self.assertIn(field, info["unknown"])
        self.assertEqual(dim(info, "arquitectura"), "no_determinada")
        self.assertEqual(info["trtllm"]["verdict"], "no_determinada")

    def test_works_with_only_tag_data_when_show_failed(self):
        info = ins.inspect_model(None, {"name": "m", "format": "gguf", "family": "llama", "parameter_size": "8B", "quantization": "Q4_0",
                                        "digest": "sha256:x", "disk_bytes": 1}, None, None, HW_4060)
        self.assertEqual((info["name"], info["format"], info["quantization"]["value"]), ("m", "gguf", "Q4_0"))
        self.assertIn("architecture", info["unknown"])

    def test_loaded_state_and_observed_vram_are_attached(self):
        info = ins.inspect_model(show(), {"name": "m", "loaded": True, "loaded_bytes": 100, "vram_bytes": 90, "loaded_context": 4096},
                                 None, {"vram_used_mib": 5000, "runs": 2}, HW_4060)
        self.assertEqual((info["loaded"]["vram_bytes"], info["loaded"]["context"]), (90, 4096))
        self.assertEqual(info["observed"]["runs"], 2)
        self.assertIsNone(ins.inspect_model(show(), {"name": "m", "loaded": False}, None, None, HW_4060)["loaded"])

    def test_snapshot_is_stable_and_complete(self):
        snap = ins.snapshot(ins.inspect_model(show(), {"name": "m", "digest": "d", "disk_bytes": 5}, None, None, HW_4060))
        for key in ("name", "digest", "format", "quantization", "quantization_kind", "architecture", "parameters_b", "context_length", "capabilities", "disk_bytes"):
            self.assertIn(key, snap)


class TrtLlmRulesTests(unittest.TestCase):
    def info(self, **kw):
        tag = kw.pop("tag", {"name": "m", "digest": "d"})
        return ins.inspect_model(show(**kw), tag, None, None, kw.pop("hw", HW_4060) if False else HW_4060)

    def test_gguf_needs_conversion_and_the_original_checkpoint(self):
        info = self.info()
        self.assertEqual(dim(info, "arquitectura"), "potencialmente_compatible")
        self.assertEqual(dim(info, "formato"), "requiere_conversion")
        self.assertEqual(dim(info, "checkpoint_original"), "necesario")
        self.assertIn("cuantizacion_gguf", dim(info, "restricciones"))
        self.assertEqual(info["trtllm"]["verdict"], "potencialmente_compatible_requiere_checkpoint_original")

    def test_a_known_family_is_never_declared_simply_compatible(self):
        for arch in ("llama", "qwen3moe", "gemma3"):
            info = self.info(architecture=arch, family=arch)
            self.assertNotEqual(info["trtllm"]["verdict"], "compatible")
            self.assertIn("potencialmente", info["trtllm"]["verdict"])
            self.assertTrue(all(f["rule"] for f in info["trtllm"]["findings"]), "cada hallazgo cita su regla")

    def test_fp8_does_not_imply_compatibility(self):
        info = self.info(quantization="FP8", architecture="gemma99", family="gemma99")
        self.assertEqual(info["quantization"]["kind"], "other")
        self.assertEqual(info["trtllm"]["verdict"], "no_determinada")

    def test_unlisted_architecture_is_undetermined_not_incompatible(self):
        info = self.info(architecture="arquitectura-nueva")
        self.assertEqual(dim(info, "arquitectura"), "no_determinada")
        self.assertNotIn("incompatible", str(info["trtllm"]).lower())

    def test_gemma4_cites_the_local_finding(self):
        info = self.info(architecture="gemma4", family="gemma4")
        f = next(f for f in info["trtllm"]["findings"] if f["dimension"] == "arquitectura")
        self.assertEqual(f["status"], "no_determinada")
        self.assertIn("CUDA 13", f["evidence"])

    def test_unknown_or_non_gguf_format_is_not_assumed(self):
        self.assertEqual(dim(self.info(format=None), "formato"), "no_determinada")
        info = self.info(format="safetensors")
        self.assertEqual(dim(info, "formato"), "no_determinada")  # no hay regla para afirmar nada
        self.assertEqual(dim(info, "checkpoint_original"), "no_determinada")

    def test_local_evidence_matches_only_architecture_and_precision(self):
        ok = self.info(architecture="qwen2", quantization="BF16", format="safetensors")
        self.assertEqual([e["id"] for e in ok["trtllm"]["local_evidence"]], ["EV-2026-10-08-qwen25-1.5b"])
        self.assertEqual(self.info(architecture="qwen2", quantization="Q4_K_M")["trtllm"]["local_evidence"], [])
        self.assertEqual(self.info(architecture="llama", quantization="BF16")["trtllm"]["local_evidence"], [])

    def test_fp8_hardware_requirement(self):
        ok = ins.inspect_model(show(), {"name": "m"}, None, None, HW_4060)
        old = ins.inspect_model(show(), {"name": "m"}, None, None, HW_OLD)
        self.assertNotIn("fp8_no_soportado_por_hardware", dim(ok, "restricciones"))
        self.assertIn("fp8_no_soportado_por_hardware", dim(old, "restricciones"))
        self.assertTrue(any("no se pudo leer" in f["text"].lower() for f in ins.inspect_model(show(), {"name": "m"}, None, None, {})["trtllm"]["findings"]))

    def test_weight_memory_estimate_is_arithmetic_and_flags_what_does_not_fit(self):
        small = ins.inspect_model(show(parameter_size="1.5B"), {"name": "m"}, None, None, HW_4060)
        big = ins.inspect_model(show(parameter_size="30B"), {"name": "m"}, None, None, HW_4060)
        self.assertTrue(dim(small, "memoria")["fits"]["BF16/FP16"])
        self.assertFalse(dim(big, "memoria")["fits"]["BF16/FP16"])
        self.assertIn("no_cabe_en_bf16", dim(big, "restricciones"))
        self.assertAlmostEqual(dim(big, "memoria")["weights_gib_estimate"]["BF16/FP16"], 55.9, places=1)
        self.assertIsNone(dim(ins.inspect_model(show(parameter_size=None), {"name": "m"}, None, None, HW_4060), "memoria"))

    def test_rules_are_versioned(self):
        rules = ins.load_rules()
        self.assertRegex(rules["rules_version"], r"^\d{4}-\d{2}-\d{2}\.\d+$")
        self.assertEqual(self.info()["trtllm"]["rules_version"], rules["rules_version"])
        self.assertIn("No instala ni convierte", self.info()["trtllm"]["notice"])


if __name__ == "__main__":
    unittest.main()

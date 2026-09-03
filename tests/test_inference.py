from __future__ import annotations

import types
import unittest

from enoki.inference import EnokiPipeline, _LLMBackend


class _FakeBackend:
    def extract(self, texts):
        return [{"text": text, "triples": []} for text in texts]


class EnokiPipelineTest(unittest.TestCase):
    def test_fact_extractor_package_has_no_eager_backend_imports(self):
        import fact_extractor

        self.assertNotIn("ModernOpenIEExtractor", fact_extractor.__dict__)
        self.assertNotIn("EnokiRulesFactExtractor", fact_extractor.__dict__)

    def test_method_aliases_use_public_names(self):
        self.assertEqual(EnokiPipeline("enoki-encoder").method, "encoder")
        self.assertEqual(EnokiPipeline("enoki_llm").method, "llm")
        self.assertEqual(EnokiPipeline("rules").method, "rules")

    def test_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown Enoki method"):
            EnokiPipeline("unknown")

    def test_single_text_always_returns_a_list(self):
        pipeline = EnokiPipeline("encoder")
        pipeline._backend = _FakeBackend()
        self.assertEqual(
            pipeline.extract("Apple acquired Beats."),
            [{"text": "Apple acquired Beats.", "triples": []}],
        )

    def test_rejects_empty_inputs(self):
        pipeline = EnokiPipeline("encoder")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            pipeline.extract([])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            pipeline.extract("  ")


class LLMBackendTest(unittest.TestCase):
    def test_llm_result_uses_common_schema_and_deduplicates(self):
        backend = _LLMBackend(
            model="test-model",
            temperature=0.0,
            prompt="incremental",
            max_retries=0,
            request_timeout=1.0,
            enable_thinking=False,
            max_tokens=50,
        )
        backend._call = lambda **_: types.SimpleNamespace(
            raw='("Apple", "acquired", "Beats")\n'
            '("Apple", "acquired", "Beats")'
        )

        result = backend.extract(["Apple acquired Beats."])

        self.assertEqual(
            result,
            [
                {
                    "text": "Apple acquired Beats.",
                    "triples": [
                        {
                            "subject": "Apple",
                            "predicate": "acquired",
                            "object": "Beats",
                            "confidence": None,
                        }
                    ],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()

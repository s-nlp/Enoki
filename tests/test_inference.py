from __future__ import annotations

import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path

from enoki.inference import EnokiPipeline, _LLMBackend
from enoki.cli import ExtractorMethod, _evaluation_extractor, evaluate_entity, evaluate_sentence, evaluate_span
from model.export import MANIFEST_NAME, export_encoder_model, is_local_encoder_model


class _FakeBackend:
    def extract(self, texts):
        return [{"text": text, "triples": []} for text in texts]


class _TripleBackend:
    def extract(self, texts):
        return [
            {
                "text": text,
                "triples": [
                    {
                        "subject": "Apple",
                        "predicate": "acquired",
                        "object": "Beats Electronics",
                        "confidence": 0.9,
                    },
                    {
                        "subject": "Apple",
                        "predicate": "acquired in",
                        "object": "Beats Electronics 2015",
                        "confidence": 0.9,
                    }
                ],
            }
            for text in texts
        ]


class EnokiPipelineTest(unittest.TestCase):
    def test_legacy_monolith_is_not_part_of_the_package(self):
        package_dir = Path(__file__).resolve().parents[1] / "fact_extractor"
        self.assertFalse((package_dir / "extractor.py").exists())
        self.assertFalse((package_dir / "utils.py").exists())
        self.assertFalse((package_dir / "minie_extractor.py").exists())
        encoder_source = (package_dir / "enoki_encoder_extractor.py").read_text()
        self.assertNotIn("from .extractor", encoder_source)

    def test_fact_extractor_package_has_no_eager_backend_imports(self):
        import fact_extractor

        self.assertNotIn("ModernOpenIEExtractor", fact_extractor.__dict__)
        self.assertNotIn("ModernOpenIEExtractor", fact_extractor._LAZY_EXPORTS)
        self.assertNotIn("EnokiRulesFactExtractor", fact_extractor.__dict__)

    def test_method_aliases_use_public_names(self):
        self.assertEqual(EnokiPipeline().method, "encoder")
        self.assertEqual(EnokiPipeline("enoki-encoder").method, "encoder")
        self.assertEqual(EnokiPipeline("enoki_llm").method, "llm")
        self.assertEqual(EnokiPipeline("rules").method, "rules")

    def test_evaluation_uses_only_current_extractor_names(self):
        self.assertEqual(_evaluation_extractor(ExtractorMethod.enoki_llm), "enoki_llm")
        self.assertNotIn("cycleoie", {method.value for method in ExtractorMethod})
        self.assertNotIn("minie", {method.value for method in ExtractorMethod})
        for command in (evaluate_sentence, evaluate_entity, evaluate_span):
            option = inspect.signature(command).parameters["extractor_method"].default
            self.assertEqual(option.default, ExtractorMethod.enoki_rules)
            self.assertIn("encoder_model", inspect.signature(command).parameters)
            self.assertNotIn("checkpoint", inspect.signature(command).parameters)

    def test_evaluation_uses_live_llm_extraction(self):
        common_source = (Path(__file__).resolve().parents[1] / "evaluation" / "common.py").read_text()
        self.assertIn("EnokiLLMFactExtractor", common_source)
        self.assertNotIn("pre_extracted", common_source)

    def test_training_export_is_a_model_directory(self):
        class _Tokenizer:
            def save_pretrained(self, output_dir):
                Path(output_dir, "tokenizer.json").write_text("{}")

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir, "best.ckpt")
            source.write_bytes(b"checkpoint")
            exported = export_encoder_model(source, _Tokenizer(), Path(temp_dir, "model"))

            self.assertTrue(is_local_encoder_model(exported))
            self.assertTrue((exported / "model.ckpt").is_file())
            self.assertTrue((exported / MANIFEST_NAME).is_file())

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

    def test_detect_returns_fact_triplets_and_hallucination_spans(self):
        import nli

        self.assertNotIn("nli.llm_nli", sys.modules)
        pipeline = EnokiPipeline("rules")
        pipeline._backend = _TripleBackend()
        original = nli.check_nli_batch_fast
        nli.check_nli_batch_fast = lambda *_args, **_kwargs: [
            {"entailment": 0.96, "neutral": 0.03, "contradiction": 0.01},
            {"entailment": 0.03, "neutral": 0.02, "contradiction": 0.95}
        ]
        try:
            result = pipeline.detect(
                context="Apple acquired Beats in 2014.",
                answer="Apple acquired Beats Electronics in 2015.",
            )
        finally:
            nli.check_nli_batch_fast = original

        self.assertEqual(
            result,
            [
                {
                    "span": "Beats Electronics",
                    "start": 15,
                    "end": 32,
                    "fact": {
                        "subject": "Apple",
                        "predicate": "acquired",
                        "object": "Beats Electronics",
                    },
                    "probability": 0.04,
                },
                {
                    "span": "2015",
                    "start": 36,
                    "end": 40,
                    "fact": {
                        "subject": "Apple",
                        "predicate": "acquired in",
                        "object": "2015",
                    },
                    "probability": 0.97,
                }
            ],
        )


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

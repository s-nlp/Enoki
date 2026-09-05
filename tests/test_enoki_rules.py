from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

from fact_extractor.enoki_rules.optimize.runs import RULES_DIR
from fact_extractor.enoki_rules.rule_base import RuleContractError
from fact_extractor.enoki_rules.rule_registry import discover_rules


class RulesInfrastructureTest(unittest.TestCase):
    def test_active_catalogue_has_the_expected_35_rules(self):
        self.assertEqual(len(discover_rules()), 35)

    def test_rule_contract_checks_files_in_the_active_rules_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = (
                Path(temp_dir)
                / "fact_extractor"
                / "enoki_rules"
                / "rules"
                / "wrong_file_name.py"
            )
            source.parent.mkdir(parents=True)
            source.write_text(
                "from fact_extractor.enoki_rules.rule_base import Rule\n"
                "class InvalidRule(Rule):\n"
                "    NAME = 'different_name'\n"
                "    TARGETS = 'test rule'\n"
                "    EXAMPLES = [('Example.', [])]\n"
                "    def apply(self, clause):\n"
                "        return []\n"
            )
            module_name = "enoki_rules_contract_fixture"
            spec = importlib.util.spec_from_file_location(module_name, source)
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                with self.assertRaises(RuleContractError):
                    spec.loader.exec_module(module)  # type: ignore[union-attr]
            finally:
                sys.modules.pop(module_name, None)

    def test_optimizer_uses_the_active_rules_directory(self):
        self.assertEqual(
            RULES_DIR,
            Path(__file__).resolve().parents[1]
            / "fact_extractor"
            / "enoki_rules"
            / "rules",
        )
        self.assertTrue(RULES_DIR.is_dir())


try:
    import spacy
except ModuleNotFoundError:
    spacy = None


@unittest.skipUnless(spacy is not None, "requires the optional rules dependency")
class RulesAdapterTest(unittest.TestCase):
    def test_preserves_a_synthesized_predicate_surface(self):
        from fact_extractor.enoki_rules_extractor import EnokiRulesFactExtractor

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        doc = nlp("Marie Curie, a physicist.")
        triplet = types.SimpleNamespace(
            subject=doc[0:2],
            predicate=doc[3:4],
            argument=types.SimpleNamespace(span=doc[3:5], prep=None),
            predicate_text="is",
            predicate_surface="is",
            negated=False,
            confidence=1.0,
        )
        extractor = types.SimpleNamespace(
            _pipeline=types.SimpleNamespace(extract=lambda text: [triplet])
        )

        groups = EnokiRulesFactExtractor.extract_granular_facts(extractor, doc.text)

        self.assertEqual(groups[0].facts[0].predicate_text, "is")
        self.assertEqual(str(groups[0].facts[0]), "Marie Curie is a physicist")


if __name__ == "__main__":
    unittest.main()

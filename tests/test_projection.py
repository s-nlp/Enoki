"""Regression cases for source positions and incremental attribution."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from enoki import EnokiPipeline
from enoki.projection import project_facts, text_spans
from enoki.encoder_decode import extract_anchored


def triple(text, subject, predicate, obj):
    return dict(subject=subject, predicate=predicate, object=obj, confidence=1.0,
                spans={p: text_spans(text, v) for p, v in
                       [('subject', subject), ('predicate', predicate), ('object', obj)]},
                sentence_start=0)


class ProjectionTests(unittest.TestCase):
    def test_nested_fact_keeps_full_hypothesis_and_only_marks_new_tokens(self):
        text = 'Apple acquired Beats Electronics in 2015.'
        ts = [triple(text, 'Apple', 'acquired', 'Beats Electronics'),
              triple(text, 'Apple', 'acquired in', 'Beats Electronics 2015')]
        self.assertEqual(project_facts(text, ts, [.1, .9], .5)[1], ([[36, 40]], False))
        self.assertEqual(project_facts(text, ts, [.9, .9], .5)[1][1], True)
        # Threshold changes attribution: the base becomes supported.
        self.assertEqual(project_facts(text, ts, [.6, .9], .7)[1], ([[36, 40]], False))

    def test_subject_refinement(self):
        text = 'The French scientist won a prize.'
        ts = [triple(text, 'scientist', 'won', 'a prize'),
              triple(text, 'French scientist', 'won', 'a prize')]
        spans, suppressed = project_facts(text, ts, [.1, .9], .5)[1]
        self.assertEqual([text[s:e] for s, e in spans], ['French'])
        self.assertFalse(suppressed)

    def test_independent_relations_and_mentions_are_not_suppressed(self):
        text = 'Alice visited Paris and Bob visited Paris.'
        a = triple(text, 'Alice', 'visited', 'Paris')
        b = triple(text, 'Bob', 'visited', 'Paris')
        a['spans'].update(predicate=[[6, 13]], object=[[14, 19]])
        b['spans'].update(predicate=[[28, 35]], object=[[36, 41]])
        self.assertEqual(project_facts(text, [a, b], [.9, .9], .5),
                         [([[14, 19]], False), ([[36, 41]], False)])

    def test_ambiguous_generated_object_never_marks_subject_or_predicate(self):
        text = 'Alice visited Paris and Bob visited Paris.'
        self.assertEqual(text_spans(text, 'Paris'), [])
        self.assertEqual(project_facts(text, [triple(text, 'Bob', 'visited', 'Paris')], [.9], .5),
                         [([], False)])

    def test_unicode_offsets_are_original_not_casefold_offsets(self):
        text = 'Straße is in Berlin.'
        self.assertEqual(text_spans(text, 'Berlin'), [[13, 19]])

    def test_discontiguous_projection_does_not_absorb_unselected_words(self):
        text = 'Alice bought red and very expensive cars.'
        t = triple(text, 'Alice', 'bought', 'red cars')
        spans, _ = project_facts(text, [t], [.9], .5)[0]
        self.assertEqual([text[s:e] for s, e in spans], ['red', 'cars'])


class PipelineReportingTests(unittest.TestCase):
    def test_original_sentence_slices_and_global_offsets(self):
        answer = 'Alice visited Paris.\n\nBob  visited Paris.'
        calls = []
        class Backend:
            def extract(self, texts):
                calls.extend(texts)
                return [{'text': text, 'triples': [triple(text, text.split()[0], 'visited', 'Paris')]}
                        for text in texts]
        p = EnokiPipeline()
        p._backend = Backend()
        with patch('nli.check_nli_batch_fast', return_value=[{'neutral': .1}, {'neutral': .9}]):
            report = p.detect(context='Alice visited Paris. Bob stayed home.', answer=answer, return_stats=True)
        self.assertEqual(calls, ['Alice visited Paris.', 'Bob  visited Paris.'])
        self.assertEqual(report['results'][0]['start'], answer.rfind('Paris'))
        self.assertEqual(report['stats']['facts_checked'], 2)
        self.assertEqual(report['stats']['facts_unsupported'], 1)
        self.assertEqual(report['stats']['sentences_total'], 2)

    def test_empty_extraction_is_not_a_successful_check(self):
        p = EnokiPipeline()
        p._backend = SimpleNamespace(extract=lambda texts: [{'text': t, 'triples': []} for t in texts])
        with patch('nli.check_nli_batch_fast') as scorer:
            report = p.detect(context='Evidence.', answer='An answer.', return_stats=True)
        scorer.assert_not_called()
        self.assertEqual(report['results'], [])
        self.assertEqual(report['stats']['status'], 'no_facts')
        self.assertEqual(report['stats']['sentences_without_checked_facts'], 1)

    def test_unlocalized_fact_still_checked_and_inspectable(self):
        p = EnokiPipeline()
        p._backend = SimpleNamespace(extract=lambda texts: [
            {'text': t, 'triples': [triple(t, 'Alice', 'visited', 'Rome')]} for t in texts])
        with patch('nli.check_nli_batch_fast', return_value=[{'contradiction': .9}]):
            report = p.detect(context='Evidence.', answer='Alice visited Paris.', return_all=True, return_stats=True)
        self.assertIsNone(report['results'][0]['span'])
        self.assertEqual(report['stats']['facts_checked'], 1)
        self.assertEqual(report['stats']['facts_unlocalized'], 1)
        self.assertEqual(report['stats']['status'], 'partial')

    def test_full_refinement_sent_to_nli(self):
        text = 'Apple acquired Beats in 2015.'
        p = EnokiPipeline()
        p._backend = SimpleNamespace(extract=lambda texts: [{'text': text, 'triples': [
            triple(text, 'Apple', 'acquired', 'Beats'),
            triple(text, 'Apple', 'acquired in', 'Beats 2015')]}])
        with patch('nli.check_nli_batch_fast', return_value=[{'neutral': .1}, {'neutral': .9}]) as scorer:
            result = p.detect(context='Evidence.', answer=text)
        self.assertEqual(scorer.call_args.args[1][1], 'Apple acquired in Beats 2015')
        self.assertEqual(result[0]['fact']['object'], 'Beats 2015')
        self.assertEqual(result[0]['span'], '2015')


class EncoderLabelsTests(unittest.TestCase):
    def test_decoder_retains_second_identical_mention(self):
        import torch
        text = 'Alice visited Paris and Bob visited Paris.'
        class Encoding(dict):
            def word_ids(self, _):
                return [None] + list(range(11)) + [None]
        def tokenizer(words, **kwargs):
            return Encoding(input_ids=torch.zeros((1, 13), dtype=torch.long),
                            attention_mask=torch.ones((1, 13), dtype=torch.long))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(unused_tokens=['[unused1]', '[unused2]', '[unused3]'], max_length=128)
            def forward(self, **kwargs):
                return SimpleNamespace(predictions=torch.tensor([[[0,0,0,0,1,2,3,0,0,0,0]]]),
                                       confidences=torch.tensor([[.9]]))
        results = extract_anchored([text], Model(), tokenizer, min_confidence=.7, top_k=10)
        t = results[0]['triples'][0]
        self.assertEqual(t['object'], 'Paris')
        self.assertEqual(t['spans']['object'], [[36, 41]])
        model = Model()
        model.config.max_length = 10
        with self.assertRaisesRegex(ValueError, 'No input has been silently truncated'):
            extract_anchored([text], model, tokenizer, min_confidence=.7, top_k=10)

class EvaluationParityTests(unittest.TestCase):
    def test_native_scores_replay_the_same_projection_at_each_threshold(self):
        import spacy
        from fact_extractor.anchored import fact_group
        from nli import score_facts_with_nli
        from evaluation.predictions_io import _apply_incremental_group_threshold
        text = 'Alice bought expensive cars.'
        doc = spacy.blank('en')(text)
        ts = [triple(text, 'Alice', 'bought', 'cars'),
              triple(text, 'Alice', 'bought', 'expensive cars')]
        groups = [fact_group(doc, t) for t in ts]
        scores = score_facts_with_nli(
            context='Evidence.', granular_facts=groups,
            check_nli_batch_fn=lambda *args: [{'neutral': .6}, {'neutral': .9}])
        for threshold in [.5, .7, .95]:
            expected = [s for p, (spans, stop) in zip([.6, .9], project_facts(text, ts, [.6, .9], threshold))
                        if p > threshold and not stop for s in spans]
            self.assertEqual(_apply_incremental_group_threshold(scores, threshold), expected)

    def test_rules_native_span_is_not_replaced_by_first_string_match(self):
        import spacy
        from enoki.inference import _rule_triple
        from fact_extractor.enoki_rules.models import Triplet, Argument
        text = 'Alice visited Paris and Bob visited Paris.'
        doc = spacy.blank('en')(text)
        item = Triplet(subject=doc[4:5], predicate=doc[5:6],
                       argument=Argument(span=doc[6:7], role='location'))
        native = _rule_triple(item)
        self.assertEqual(native['spans']['object'], [[36, 41]])

    def test_span_evaluation_serializes_native_positions_for_replay(self):
        import json
        import spacy
        from fact_extractor.anchored import fact_group
        from evaluation.span import evaluate_span_dataset
        from evaluation.predictions_io import _apply_incremental_group_threshold
        text = 'Alice bought expensive cars.'
        doc = spacy.blank('en')(text)
        ts = [triple(text, 'Alice', 'bought', 'cars'),
              triple(text, 'Alice', 'bought', 'expensive cars')]
        extractor = SimpleNamespace(extract_granular_facts=lambda _: [fact_group(doc, t) for t in ts])
        with patch('evaluation.span.check_nli_batch_fast', return_value=[{'neutral': .1}, {'neutral': .9}]):
            _, predictions, raw = evaluate_span_dataset(
                [{'answer': text, 'context': 'Evidence.', 'labels': [[13, 22]]}],
                extractor, None, 'modernbert', 2048, .5, 'default')
        replay = json.loads(json.dumps(raw))[0]['fact_spans']
        self.assertEqual(predictions, [[[13, 22]]])
        self.assertEqual(_apply_incremental_group_threshold(replay, .5), [[13, 22]])


if __name__ == '__main__':
    unittest.main()

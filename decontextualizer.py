"""
Decontextualizer for resolving coreferences and pronouns in text using fastcoref.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any, Dict


class FastCorefDecontextualizer:
    """Decontextualizer using fastcoref (biu-nlp/f-coref)."""

    def __init__(self):
        # Suppress warnings and verbose logging
        warnings.filterwarnings('ignore')

        # Set environment variables
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['HF_DATASETS_VERBOSITY'] = 'error'
        os.environ['TQDM_DISABLE'] = '1'

        # Suppress verbose logging from all relevant libraries
        for logger_name in ['transformers', 'gliner', 'gliner_spacy', 'datasets',
                           'fastcoref', 'spacy', 'urllib3', 'filelock']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            logging.getLogger(logger_name).propagate = False

        # Suppress progress bars from datasets
        try:
            import datasets
            datasets.logging.set_verbosity_error()
            datasets.disable_progress_bar()
        except:
            pass

        # Disable tqdm only for the fastcoref import, then restore the original
        # class so that Lightning (which subclasses tqdm.tqdm) does not receive
        # a functools.partial instead of a type.
        try:
            import functools
            import tqdm as tqdm_module
            original_tqdm = tqdm_module.tqdm
            original_trange = tqdm_module.trange
            tqdm_module.tqdm = functools.partial(original_tqdm, disable=True)
            tqdm_module.trange = functools.partial(original_trange, disable=True)
        except:
            original_tqdm = original_trange = None

        from fastcoref import FCoref

        if original_tqdm is not None:
            tqdm_module.tqdm = original_tqdm
            tqdm_module.trange = original_trange
        self.model = FCoref(device='cuda')
        # Disable per-call progress bars (fastcoref shows "Inference: 1/1" for each text)
        self.model.enable_progress_bar = False

    # Pronouns substituted in both the rewrite path (sentence-level eval) and
    # the NLI-hypothesis path (entity-/span-level). Relative pronouns
    # (who/whom/whose) are excluded because replacing them destroys the
    # relative-clause connector and breaks OIE parsing.
    _PRONOUNS = {'he', 'she', 'it', 'they', 'him', 'them', 'himself', 'herself', 'itself', 'themselves'}
    # Possessives substituted in the NLI hypothesis only (replacement entries
    # carry hyp_only=True). They are excluded from text rewrites because
    # replacing "His" with "John Smith's" in the OIE input reshapes the NP
    # boundaries; in the hypothesis path there is no OIE step, so the
    # substitution unambiguously helps NLI.
    _POSSESSIVES = {'his', 'her', 'its', 'their'}
    # Definite-NP coref: mentions of the form "the/this/that/these/those X"
    # where X is a common noun (lowercase). fastcoref clusters these with
    # the canonical antecedent, but unlike pronouns they are NOT
    # implicitly resolved by ModernBERT-NLI's attention — the surface form
    # carries no link to the antecedent. Substituting them is novel surface.
    _DEFINITE_NP_RE = re.compile(r'^(the|this|that|these|those)\s+[a-z]', re.IGNORECASE)
    # Reflexives only resolved when sentence-initial (capitalised).
    _SENTENCE_INITIAL_ONLY = {'itself', 'themselves'}
    # Maximum character distance between antecedent end and pronoun start.
    _MAX_COREF_DISTANCE = 500

    @staticmethod
    def _possessive_form(antecedent: str) -> str:
        """Append possessive marker — "'s" or just "'" if antecedent ends in s."""
        if antecedent and antecedent[-1] in ('s', 'S'):
            return antecedent + "'"
        return antecedent + "'s"

    # Regex for parenthetical strip applied during antecedent canonicalization.
    _PAREN_RE = re.compile(r'\s*\([^)]*\)')
    # Em-dash / en-dash / double-hyphen continuations to strip after the dash.
    _DASH_TOKENS = (' — ', ' – ', ' -- ')

    @classmethod
    def _canonicalize_antecedent(cls, text: str) -> str:
        """Trim a coref antecedent span to its canonical name form.

        fastcoref returns mention spans, which on biographical text include
        appositives, parenthetical dates, and em-dash continuations. Splicing
        these into an NLI hypothesis leaks extra claims (dates, titles) and
        dilutes the predicate. We trim:

          "Heinrich Harrer (1912-2006)"                       → "Heinrich Harrer"
          "Julia Faye (born September 24, 1893 – April 6, 1966)" → "Julia Faye"
          "Ronaldo Luís Nazário de Lima, commonly known as Ronaldo" → "Ronaldo Luís Nazário de Lima"
          "Mary Smith, born 1990"                             → "Mary Smith"
        """
        text = cls._PAREN_RE.sub('', text)
        for dash in cls._DASH_TOKENS:
            if dash in text:
                text = text.split(dash, 1)[0]
                break
        comma = text.find(',')
        if comma > 0 and text[:1].isupper():
            after = text[comma + 1:].lstrip()
            if after and after[:1].islower():
                text = text[:comma]
        return text.strip()

    def _collect_replacements(self, document: str, clusters, sent_start: int, sent_end: int):
        """Walk coref clusters and return replacements for pronouns inside [sent_start, sent_end).

        Antecedents may come from anywhere in `document` (including before the target window).
        Returned replacement offsets are relative to sent_start so callers can apply them
        directly to the sentence string.
        """
        replacements = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            live_antecedent_span = cluster[0]
            live_antecedent_text = self._canonicalize_antecedent(
                document[live_antecedent_span[0]:live_antecedent_span[1]]
            )

            for mention_span in cluster[1:]:
                mention_text = document[mention_span[0]:mention_span[1]]
                mention_lower = mention_text.lower()

                in_window = sent_start <= mention_span[0] < sent_end
                distance = mention_span[0] - live_antecedent_span[1]
                within_distance = distance <= self._MAX_COREF_DISTANCE
                antecedent_lower = live_antecedent_text.lower()
                indefinite = antecedent_lower.startswith(('a ', 'an '))

                if mention_lower in self._PRONOUNS:
                    if in_window and within_distance:
                        noisy_mid_sentence = (
                            mention_lower in self._SENTENCE_INITIAL_ONLY
                            and not mention_text[0].isupper()
                        )
                        if not indefinite and not noisy_mid_sentence and live_antecedent_text:
                            replacements.append({
                                'start': mention_span[0] - sent_start,
                                'end': mention_span[1] - sent_start,
                                'replacement': live_antecedent_text,
                                'pronoun_text': mention_text,
                            })
                    # Pronouns don't update the live antecedent.
                elif mention_lower in self._POSSESSIVES:
                    if in_window and within_distance and not indefinite and live_antecedent_text:
                        replacements.append({
                            'start': mention_span[0] - sent_start,
                            'end': mention_span[1] - sent_start,
                            'replacement': self._possessive_form(live_antecedent_text),
                            'pronoun_text': mention_text,
                            'hyp_only': True,
                        })
                    # Possessives don't update the live antecedent.
                elif self._DEFINITE_NP_RE.match(mention_text):
                    if in_window and within_distance and not indefinite and live_antecedent_text:
                        replacements.append({
                            'start': mention_span[0] - sent_start,
                            'end': mention_span[1] - sent_start,
                            'replacement': live_antecedent_text,
                            'pronoun_text': mention_text,
                        })
                    # Definite NPs don't update the live antecedent; they
                    # are less specific than the canonical proper name.
                else:
                    # Non-pronoun, non-definite-NP, non-possessive mention:
                    # a proper-name (re-)introduction that should become the
                    # new live antecedent for downstream pronouns.
                    live_antecedent_span = mention_span
                    live_antecedent_text = self._canonicalize_antecedent(mention_text)
        return replacements

    @staticmethod
    def _apply_replacements(text: str, replacements) -> str:
        # Forward iteration with overlap skipping. Must match the semantics of
        # fact_alignment.build_resolved_to_orig_char_ranges so that the resolved
        # text we hand to OIE is the same text that span normalization rebuilds
        # later — otherwise normalize_fact_spans_to_orig asserts and the sample
        # is dropped. Overlapping replacements (common when definite-NP and
        # pronoun mentions share a span) are skipped in document order.
        parts = []
        cursor = 0
        for rep in sorted(replacements, key=lambda r: (r['start'], r['end'])):
            if rep.get('hyp_only'):
                # Possessive replacements only apply to NLI hypotheses, not to
                # the rewritten text consumed by the OIE extractor.
                continue
            s, e = rep['start'], rep['end']
            if s < cursor:
                if e <= cursor:
                    continue
                s = cursor
            parts.append(text[cursor:s])
            parts.append(rep['replacement'])
            cursor = max(cursor, e)
        parts.append(text[cursor:])
        return ''.join(parts)

    def decontextualize(self, text: str) -> Dict[str, Any]:
        """Resolve coreferences within a single self-contained text."""
        try:
            preds = self.model.predict(texts=[text])
            if preds:
                clusters = preds[0].get_clusters(as_strings=False)
                replacements = self._collect_replacements(text, clusters, 0, len(text))
                return {
                    "resolved_text": self._apply_replacements(text, replacements),
                    "replacements": replacements,
                }
        except Exception as e:
            print(f"Warning: fastcoref failed: {e}")
        return {"resolved_text": text, "replacements": []}

    def decontextualize_in_document(self, document: str, sentence: str) -> Dict[str, Any]:
        """Resolve coreferences in `sentence` using the full `document` as context.

        Runs fastcoref on the whole document so pronouns whose antecedents appear
        in preceding sentences are resolved correctly.  Only the target sentence
        is rewritten; the rest of the document is ignored.
        """
        sent_start = document.find(sentence)
        if sent_start == -1:
            # Sentence not found verbatim — fall back to single-sentence resolution.
            return self.decontextualize(sentence)
        sent_end = sent_start + len(sentence)
        try:
            preds = self.model.predict(texts=[document])
            if preds:
                clusters = preds[0].get_clusters(as_strings=False)
                replacements = self._collect_replacements(document, clusters, sent_start, sent_end)
                return {
                    "resolved_text": self._apply_replacements(sentence, replacements),
                    "replacements": replacements,
                }
        except Exception as e:
            print(f"Warning: fastcoref failed: {e}")
        return {"resolved_text": sentence, "replacements": []}

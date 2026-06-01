"""Fact extraction backed by Stanford OpenIE.

Exposes ``extract_granular_facts(text)`` with the same return type as
``FactExtractor`` (``List[IncrementalFactGroup]``) so it can be dropped into
the existing pipeline (decontextualizer → fact_alignment → NLI).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import spacy
from spacy.tokens import Doc, Span

from .models import Fact, IncrementalFactGroup

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _phrase_words(phrase: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(phrase)]


def _token_word_index(doc: Doc) -> List[Tuple[int, str]]:
    """Return [(token_i, lowered_word), ...] for non-punct/non-space tokens."""
    out: List[Tuple[int, str]] = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        word = tok.text.lower()
        if not word:
            continue
        out.append((tok.i, word))
    return out


class StanfordFactExtractor:
    """Fact extractor that wraps Stanford OpenIE triples.

    Each OpenIE triple ``{subject, relation, object}`` becomes a single
    ``Fact`` inside its own ``IncrementalFactGroup`` (delta = object span).
    Spans are located in a spaCy ``Doc`` so downstream NLI span-marking
    (``start_char``/``end_char``) keeps working.
    """

    def __str__(self):
        return 'Stanford'

    def __init__(
        self,
        nlp=None,
        client=None,
        corenlp_kwargs: Optional[Dict] = None,
        dedup: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            nlp: pre-loaded spaCy model. Defaults to ``en_core_web_trf`` to
                match what the Enoki / NLI pipeline use downstream, so span
                offsets align across extractors.
            client: pre-constructed ``stanza.server.CoreNLPClient``. If
                ``None``, a client is spun up lazily and kept alive for the
                lifetime of this extractor.
            corenlp_kwargs: kwargs forwarded to ``CoreNLPClient(...)`` when
                the client is created lazily (e.g. ``timeout``, ``memory``).
            dedup: drop duplicate ``(subject, relation, object)`` triples
                before constructing facts.
        """
        self.nlp = nlp or spacy.load("en_core_web_trf")
        self._client = client
        self._owns_client = client is None
        self._corenlp_kwargs = corenlp_kwargs or {}
        self.dedup = dedup
        self.verbose = verbose

    def __enter__(self):
        self._ensure_client()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception:
                pass
            self._client = None

    def _ensure_client(self):
        if self._client is None:
            import os
            import stanza
            from stanza.server import CoreNLPClient
            corenlp_home = os.getenv("CORENLP_HOME", os.path.join(str(__import__("pathlib").Path.home()), "stanza_corenlp"))
            if not os.path.exists(corenlp_home):
                logger.info("CoreNLP not found at %s, downloading...", corenlp_home)
                stanza.install_corenlp(dir=corenlp_home)
            kwargs = {"annotators": ["openie"], "timeout": 60000, "memory": "4G", "be_quiet": True}
            kwargs.update(self._corenlp_kwargs)
            self._client = CoreNLPClient(**kwargs)
            self._client.__enter__()
        return self._client

    def _annotate(self, text: str) -> List[Dict]:
        """Call CoreNLP and return a flat list of {subject, relation, object} dicts."""
        client = self._ensure_client()
        ann = client.annotate(text)
        triples = []
        for sentence in ann.sentence:
            for t in sentence.openieTriple:
                triples.append({
                    "subject": t.subject,
                    "relation": t.relation,
                    "object": t.object,
                })
        return triples

    def extract_granular_facts(self, text: str) -> List[IncrementalFactGroup]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        raw_triples = self._annotate(text)

        seen = set()
        facts: List[Fact] = []

        if self.verbose:
            logger.info("sentence  %r", text.strip())

        for tr in raw_triples:
            subj_s = (tr.get("subject") or "").strip()
            rel_s = (tr.get("relation") or "").strip()
            obj_s = (tr.get("object") or "").strip()

            if self.verbose:
                logger.info("  raw triple  (%s | %s | %s)", subj_s, rel_s, obj_s)

            if not subj_s or not rel_s:
                if self.verbose:
                    logger.info("  dropped     (empty subj/rel)")
                continue

            if self.dedup:
                key = (subj_s.lower(), rel_s.lower(), obj_s.lower())
                if key in seen:
                    if self.verbose:
                        logger.info("  dropped     (duplicate)")
                    continue
                seen.add(key)

            subj_span = self._locate_span(doc, subj_s)
            pred_span = self._locate_span(doc, rel_s)
            obj_span = self._locate_span(doc, obj_s) if obj_s else None

            if subj_span is None or pred_span is None:
                if self.verbose:
                    logger.info(
                        "  dropped     subj_span=%s pred_span=%s (span not found)",
                        subj_span, pred_span,
                    )
                continue
            if obj_s and obj_span is None:
                if self.verbose:
                    logger.info("  dropped     obj=%r span not found", obj_s)
                continue

            fact = Fact(
                subject=subj_span,
                predicate=pred_span,
                argument=obj_span,
                prep=None,
            )
            facts.append(fact)
            if self.verbose:
                logger.info("  fact        (%s | %s | %s)", subj_span, pred_span, obj_span)

        return [
            IncrementalFactGroup(
                facts=[f],
                deltas=[f.argument] if f.argument is not None else [],
            )
            for f in facts
        ]


    def _locate_span(self, doc: Doc, phrase: str) -> Optional[Span]:
        """Find the best matching ``Span`` in ``doc`` for ``phrase``.

        Strategy: exact case-insensitive substring first; fall back to a
        token-level subsequence match that ignores punctuation/whitespace
        (OpenIE strips commas, e.g. ``"July 27, 1949"`` → ``"July 27 1949"``).
        """
        if not phrase:
            return None

        lower_text = doc.text.lower()
        needle = phrase.lower()
        idx = lower_text.find(needle)
        if idx >= 0:
            span = doc.char_span(idx, idx + len(phrase), alignment_mode="expand")
            if span is not None:
                return span

        words = _phrase_words(phrase)
        if not words:
            return None

        index = _token_word_index(doc)
        n = len(words)
        for start in range(len(index) - n + 1):
            if all(index[start + k][1] == words[k] for k in range(n)):
                first_tok_i = index[start][0]
                last_tok_i = index[start + n - 1][0]
                return doc[first_tok_i : last_tok_i + 1]

        return None

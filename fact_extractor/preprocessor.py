"""Shared preprocessing for all fact extractors.

Provides two transferable preprocessing steps:

1. ``mask_markdown(text)`` — replace Markdown syntax with spaces while
   preserving every character offset.  Safe to apply before any extractor.

2. ``PreprocessingWrapper`` — wraps any extractor to apply markdown masking
   and, optionally, GLiNER named-entity enhancement.  GLiNER is injected by
   wrapping ``extractor.nlp.__call__`` so that every internal ``nlp(text)``
   call made by the wrapped extractor gets the enriched Doc for free.
   A minimum-length guard skips GLiNER on short phrases so the lemma-fallback
   calls inside ``_locate_span`` are not hit.

What is NOT shared
------------------
Quoted-entity replacement is intentionally not included: it is not a general
preprocessing step and must stay coupled to an extractor that owns its span
alignment policy.
"""

from __future__ import annotations

import re
from typing import List, Optional

_MD_SUBS = [
    # Block-level: headings, horizontal rules, list bullets, numbered lists
    re.compile(r"(?m)^[ \t]*#{1,6}.*$"),
    re.compile(r"(?m)^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$"),
    re.compile(r"(?m)^[ \t]*[-*+][ \t]+"),
    re.compile(r"(?m)^[ \t]*\d+[.)][ \t]+"),
    # Inline: bold / italic / strikethrough markers
    re.compile(r"[*_~]+"),
]

# Minimum text length to trigger GLiNER (avoids firing on short _locate_span
# lemma-fallback phrases like "be" or "estimated to be").
_GLINER_MIN_LEN = 20


def mask_markdown(text: str) -> str:
    """Replace Markdown syntax with spaces, preserving character offsets."""
    def _blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    for pat in _MD_SUBS:
        text = pat.sub(_blank, text)
    return text


class _GlinerNLP:
    """Thin wrapper around a spaCy Language that applies GLiNER after parsing."""

    def __init__(self, nlp, gliner_model: str):
        self._nlp = nlp
        self._gliner_model = gliner_model

    def __call__(self, text: str, **kwargs):
        doc = self._nlp(text, **kwargs)
        if len(text) >= _GLINER_MIN_LEN:
            try:
                from ner_enhancement import enhance_doc_with_gliner
                doc = enhance_doc_with_gliner(
                    doc, text,
                    model_name=self._gliner_model,
                    alignment_mode="expand",
                )
            except Exception:
                pass
        return doc

    def __getattr__(self, name: str):
        return getattr(self._nlp, name)

    # Make the wrapper picklable / hashable alongside the original
    def __repr__(self) -> str:
        return f"GlinerNLP({self._nlp!r}, model={self._gliner_model!r})"


class PreprocessingWrapper:
    """Wraps any fact extractor to add markdown masking and optional GLiNER.

    Usage::

        from fact_extractor import StanfordFactExtractor
        from fact_extractor.preprocessor import PreprocessingWrapper

        ext = PreprocessingWrapper(
            StanfordFactExtractor(nlp=nlp),
            use_gliner=True,
        )
        groups = ext.extract_granular_facts(text)

    All other attributes / methods are forwarded transparently to the wrapped
    extractor, so the wrapper is a drop-in replacement.
    """

    def __init__(
        self,
        extractor,
        use_gliner: bool = False,
        gliner_model: str = "numind/NuNerZero",
    ):
        self._extractor = extractor
        self.use_gliner = use_gliner
        self.gliner_model = gliner_model

        if use_gliner and hasattr(extractor, "nlp"):
            extractor.nlp = _GlinerNLP(extractor.nlp, gliner_model)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def extract_granular_facts(self, text: str):
        return self._extractor.extract_granular_facts(mask_markdown(text))

    def warm_up(self, texts: List[str]) -> None:
        masked = [mask_markdown(t) for t in texts]
        if hasattr(self._extractor, "warm_up"):
            self._extractor.warm_up(masked)

    # ------------------------------------------------------------------
    # Transparent forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self._extractor, name)

    def __repr__(self) -> str:
        return f"PreprocessingWrapper({self._extractor!r}, use_gliner={self.use_gliner})"

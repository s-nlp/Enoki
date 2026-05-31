"""spaCy parser wrapper.

Owns the spaCy ``Language`` instance and the optional GLiNER NER overlay.
Cached per (model, gliner) tuple so the pipeline pays the load cost once.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc


log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _load_nlp(model: str):
    import spacy

    return spacy.load(model)


class Parser:
    """A reusable spaCy parser + NER overlay.

    Construct once per pipeline; call :meth:`parse` per document.
    """

    def __init__(
        self,
        model: str = "en_core_web_trf",
        use_gliner: bool = False,
        gliner_model: str = "numind/NuNerZero",
    ) -> None:
        self.model = model
        self.use_gliner = use_gliner
        self.gliner_model = gliner_model
        self._nlp: Optional["Language"] = None

    @property
    def nlp(self) -> "Language":
        if self._nlp is None:
            self._nlp = _load_nlp(self.model)
        return self._nlp

    def parse(self, text: str) -> "Doc":
        doc = self.nlp(text)
        if self.use_gliner:
            doc = self._apply_gliner_overlay(doc, text)
        return doc

    def _apply_gliner_overlay(self, doc: "Doc", text: str) -> "Doc":
        try:
            # Optional dependency; the import is local so spaCy-only users
            # are not forced to install gliner.
            from ner_enhancement import enhance_doc_with_gliner  # type: ignore

            return enhance_doc_with_gliner(
                doc, text, model_name=self.gliner_model, alignment_mode="expand"
            )
        except ImportError:
            log.warning("GLiNER requested but ner_enhancement is not importable")
            return doc
        except Exception as exc:  # pragma: no cover - GLiNER runtime failure
            log.warning("GLiNER overlay failed: %s", exc)
            return doc


def parse_text(text: str, model: str = "en_core_web_trf") -> "Doc":
    """One-shot parse for ad-hoc/test use."""
    return Parser(model=model).parse(text)

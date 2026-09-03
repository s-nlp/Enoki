"""Live fact extraction through the Enoki LLM backend."""

from __future__ import annotations

from typing import List, Optional

from .models import Fact, IncrementalFactGroup


class EnokiLLMFactExtractor:
    """Adapt :class:`enoki.EnokiPipeline` output to the evaluation interface.

    The LLM produces textual triples.  Evaluation additionally needs spans in
    the answer, so triples whose subject, predicate, or object cannot be
    anchored in the source text are omitted.
    """

    def __init__(
        self,
        nlp,
        model: Optional[str] = None,
        temperature: float = 0.0,
        prompt: str = "incremental",
    ) -> None:
        from enoki.inference import EnokiPipeline

        self.nlp = nlp
        self._pipeline = EnokiPipeline(
            "llm",
            model=model,
            temperature=temperature,
            prompt=prompt,
        )

    def __str__(self) -> str:
        return "EnokiLLM"

    @staticmethod
    def _locate_span(doc, text: str, start: int, end: int):
        phrase = text.strip()
        if not phrase:
            return None
        offset = doc.text[start:end].casefold().find(phrase.casefold())
        if offset < 0:
            return None
        start += offset
        return doc.char_span(start, start + len(phrase), alignment_mode="expand")

    def extract_granular_facts(self, text: str) -> List[IncrementalFactGroup]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        groups: List[IncrementalFactGroup] = []
        seen = set()

        for sentence in doc.sents:
            extracted = self._pipeline.extract(sentence.text)[0]["triples"]
            for triple in extracted:
                subject_text = triple["subject"].strip()
                predicate_text = triple["predicate"].strip()
                object_text = triple["object"].strip()
                key = (subject_text.casefold(), predicate_text.casefold(), object_text.casefold())
                if key in seen:
                    continue
                seen.add(key)

                subject = self._locate_span(doc, subject_text, sentence.start_char, sentence.end_char)
                predicate = self._locate_span(doc, predicate_text, sentence.start_char, sentence.end_char)
                argument = (
                    self._locate_span(doc, object_text, sentence.start_char, sentence.end_char)
                    if object_text
                    else None
                )
                if subject is None or predicate is None or (object_text and argument is None):
                    continue

                fact = Fact(
                    subject=subject,
                    predicate=predicate,
                    argument=argument,
                    predicate_text=predicate_text,
                )
                groups.append(
                    IncrementalFactGroup(
                        facts=[fact],
                        deltas=[argument] if argument is not None else [],
                        confidence=None,
                    )
                )

        return groups

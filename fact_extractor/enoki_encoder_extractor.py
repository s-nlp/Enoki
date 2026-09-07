"""Span-anchored Enoki-Encoder facts for hallucination detection."""

from __future__ import annotations

from typing import List, Optional

from .models import IncrementalFactGroup


class EnokiEncoderFactExtractor:
    """Adapt Enoki-Encoder triples to the span-aware evaluation interface.

    ``model`` accepts the same source as :class:`enoki.EnokiPipeline`: the
    published Hugging Face model ID or a local directory exported by
    ``enoki train encoder``.
    """

    def __init__(
        self,
        nlp,
        model: Optional[str] = None,
        device: str = "auto",
        min_confidence: float = 0.7,
        top_k: int = 10,
    ) -> None:
        from enoki.inference import EnokiPipeline

        self.nlp = nlp
        self._pipeline = EnokiPipeline(
            "encoder",
            model=model,
            device=device,
            min_confidence=min_confidence,
            top_k=top_k,
        )

    def __str__(self) -> str:
        return "EnokiEncoder"

    def extract_granular_facts(self, text: str) -> List[IncrementalFactGroup]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        groups: List[IncrementalFactGroup] = []

        for sentence in doc.sents:
            # spaCy can emit a whitespace-only sentence for inputs containing
            # certain newline/markup layouts. The pipeline deliberately
            # rejects blank inputs, so skip those spans rather than failing
            # extraction for the entire answer.
            if not sentence.text.strip():
                continue
            extracted = self._pipeline.extract(sentence.text)[0]["triples"]
            for triple in extracted:
                from .anchored import fact_group
                # Pipeline offsets are relative to this original sentence slice.
                triple = dict(triple)
                triple["spans"] = {part: [[a + sentence.start_char, b + sentence.start_char]
                                                  for a, b in spans]
                                   for part, spans in triple["spans"].items()}
                triple["sentence_start"] += sentence.start_char
                triple["sentence_end"] += sentence.start_char
                group = fact_group(doc, triple)
                if group is not None:
                    groups.append(group)

        return groups

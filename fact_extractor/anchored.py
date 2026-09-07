"""Adapt native token positions to the legacy Fact container without text search."""
from .models import Fact, IncrementalFactGroup


def fact_group(doc, triple):
    spans = triple["spans"]
    def envelope(part):
        positions = spans.get(part, [])
        if not positions:
            return None
        return doc.char_span(min(a for a, _ in positions), max(b for _, b in positions),
                             alignment_mode="expand")
    subject = envelope("subject")
    if subject is None:
        return None
    argument = envelope("object")
    fact = Fact(subject=subject, predicate=envelope("predicate") or subject,
                argument=argument, predicate_text=triple["predicate"], source_triple=triple)
    return IncrementalFactGroup(facts=[fact], deltas=[argument] if argument is not None else [],
                                confidence=triple.get("confidence"))

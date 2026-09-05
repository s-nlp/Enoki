"""Source-position projection of independently scored, nested facts.

Offsets are half-open Python character offsets, never normalized-text offsets.
Only a supported base fact permits attribution to a newly added token.
"""

from __future__ import annotations

import re
from functools import lru_cache

PARTS = ("subject", "predicate", "object")


def text_spans(text, phrase):
    """Conservative fallback for generated LLM text: reject ambiguous matches."""
    words = re.findall(r"\w+|[^\w\s]", phrase)
    if not words:
        return []
    tokens = list(re.finditer(r"\w+|[^\w\s]", text))
    # Permit non-contiguous words, but never choose arbitrarily among mentions.
    @lru_cache(maxsize=None)
    def visit(word_index, token_index):
        if word_index == len(words):
            return ((),)
        matches = []
        for i in range(token_index, len(tokens)):
            if tokens[i].group().casefold() == words[word_index].casefold():
                for suffix in visit(word_index + 1, i + 1):
                    matches.append(((tokens[i].start(), tokens[i].end()),) + suffix)
                    if len(matches) == 2:
                        return tuple(matches)
        return tuple(matches)
    matches = visit(0, 0)
    return list(map(list, matches[0])) if len(matches) == 1 else []


def merge_spans(text, spans):
    """Merge adjacent selected tokens, without absorbing intervening words."""
    result = []
    for start, end in sorted(set(map(tuple, spans))):
        if result and (start <= result[-1][1] or not text[result[-1][1]:start].strip()):
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _relation(text):
    return re.sub(r"(?:\s+(?:at|by|for|from|in|of|on|to|with))+$", "", text.casefold().strip())


def project_facts(text, triples, probabilities, threshold):
    """Return (source spans, suppressed) per fact; nested failures stop at the base.

    A refinement must preserve anchored subject/object tokens and the relation,
    expanding exactly one argument. Separate mentions never form one chain.
    """
    anchors = [{p: set(map(tuple, t.get("spans", {}).get(p, []))) for p in PARTS}
               for t in triples]
    output = []
    for i, triple in enumerate(triples):
        current = anchors[i]
        default = (current["object"] if triple["object"].strip()
                   else current["predicate"] or current["subject"])
        parents = []
        for j, base in enumerate(triples):
            if i == j or triple.get("sentence_start") != base.get("sentence_start"):
                continue
            previous = anchors[j]
            if not current["subject"] or not previous["subject"]:
                continue
            if _relation(triple["predicate"]) != _relation(base["predicate"]):
                continue
            if not previous["predicate"].issubset(current["predicate"]):
                continue
            for part, fixed in (("object", "subject"), ("subject", "object")):
                if (previous[part] and previous[part] < current[part]
                        and previous[fixed] == current[fixed]):
                    parents.append((j, part))
        # A failing simpler fact prevents blaming just the added modifier.
        suppressed = any(probabilities[j] > threshold for j, _ in parents)
        if parents and not suppressed:
            size = max(len(anchors[j][part]) for j, part in parents)
            closest = [(j, part) for j, part in parents if len(anchors[j][part]) == size]
            deltas = [current[part] - anchors[j][part] for j, part in closest]
            # Multiple equally close decompositions: do not invent a unique cause.
            if all(delta == deltas[0] for delta in deltas):
                default = deltas[0]
        output.append((merge_spans(text, default), suppressed))
    return output

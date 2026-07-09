"""Phrasal verbs from ``data/open_ie/wiktionary/verb_phrases.txt``.

One phrasal verb per line. The file lists base-form multi-word verbs
(``"abide by"``, ``"give up"``, ``"look forward to"``, ...). Rules use this
to recognize multi-word predicates that the dependency parser may not
collapse into a single head.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import FrozenSet, Set


_DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "open_ie"
    / "wiktionary"
    / "verb_phrases.txt"
)


class PhrasalVerbs:
    """A view over the phrasal-verb list."""

    def __init__(self, by_stem: FrozenSet[str], by_first_word: FrozenSet[str]):
        self._by_stem = by_stem
        self._by_first_word = by_first_word

    def __contains__(self, phrase: str) -> bool:
        return phrase.lower() in self._by_stem

    def first_words(self) -> FrozenSet[str]:
        """Return the set of lower-cased first words of any phrasal verb."""
        return self._by_first_word

    def __len__(self) -> int:
        return len(self._by_stem)


@functools.lru_cache(maxsize=1)
def _load() -> PhrasalVerbs:
    by_stem: Set[str] = set()
    by_first: Set[str] = set()
    if _DATA_PATH.exists():
        with _DATA_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                phrase = line.strip()
                if not phrase or phrase.startswith("#"):
                    continue
                lowered = phrase.lower()
                by_stem.add(lowered)
                first = lowered.split(" ", 1)[0]
                if first:
                    by_first.add(first)
    return PhrasalVerbs(frozenset(by_stem), frozenset(by_first))


def get_phrasal_verbs() -> PhrasalVerbs:
    return _load()

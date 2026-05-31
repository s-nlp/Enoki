"""Verb inflections from ``data/open_ie/wiktionary/en_verb_inflections.txt``.

The file is tab-separated::

    stem<TAB>3sg<TAB>presentParticiple<TAB>past<TAB>pastParticiple
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


_DATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "open_ie"
    / "wiktionary"
    / "en_verb_inflections.txt"
)


@dataclass(frozen=True)
class VerbInflections:
    """Forms of a single verb."""

    stem: str
    third_person_singular: str
    present_participle: str
    past: str
    past_participle: str


@functools.lru_cache(maxsize=1)
def _load() -> Dict[str, VerbInflections]:
    if not _DATA_PATH.exists():
        return {}
    inflections: Dict[str, VerbInflections] = {}
    with _DATA_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            stem, third, prog, past, pp = parts
            inflections[stem.lower()] = VerbInflections(
                stem=stem,
                third_person_singular=third,
                present_participle=prog,
                past=past,
                past_participle=pp,
            )
    return inflections


def get_inflections(stem: str) -> Optional[VerbInflections]:
    """Return inflections for ``stem`` (lower-cased lookup) or ``None`` if unknown."""
    return _load().get(stem.lower())

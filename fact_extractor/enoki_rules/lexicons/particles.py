"""Verb particles.

The wiktionary ``extract_english_verb_particles.py`` script in the repo would
produce a per-verb particle list, but its output isn't checked in. Until it
is, we fall back to a small curated set of English particles that the
predicate shaper can attach to a verb head when ``dep_ == "prt"``. This list
is *not* a guess about what the parser will produce — it is only used as a
sanity gate for the optional ``predicate_particle_window`` expansion.
"""

from __future__ import annotations

import functools
from typing import FrozenSet


_CORE_PARTICLES = frozenset(
    {
        "up", "down", "in", "out", "on", "off", "over", "under",
        "away", "back", "around", "along", "through", "across",
        "apart", "together", "ahead", "behind", "aside", "forward",
    }
)


class VerbParticles:
    def __init__(self, particles: FrozenSet[str]):
        self._particles = particles

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._particles

    def __iter__(self):
        return iter(self._particles)

    def __len__(self) -> int:
        return len(self._particles)


@functools.lru_cache(maxsize=1)
def get_particles() -> VerbParticles:
    return VerbParticles(_CORE_PARTICLES)

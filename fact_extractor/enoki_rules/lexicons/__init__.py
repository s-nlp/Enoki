"""Read-only lexical resources used by rules.

These wrap files under ``data/open_ie/wiktionary/`` and a small hard-coded
stoplist. Loaded lazily; the loaders are idempotent and cached per process.
"""

from .inflections import VerbInflections, get_inflections
from .particles import VerbParticles, get_particles
from .phrasal_verbs import PhrasalVerbs, get_phrasal_verbs

__all__ = [
    "VerbInflections",
    "VerbParticles",
    "PhrasalVerbs",
    "get_inflections",
    "get_particles",
    "get_phrasal_verbs",
]

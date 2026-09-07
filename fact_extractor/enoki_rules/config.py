"""Pipeline configuration: rule toggles and shape, filter and optimizer thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class ShapeConfig:
    """Span-shaping policy thresholds."""

    # Strip leading "the"/"a"/"an" from subject and argument spans.
    strip_leading_det: bool = True
    # Include compound modifiers in subject and argument spans.
    include_compound: bool = True
    # Include adjectival modifiers in subject and argument spans.
    include_amod: bool = True
    # Absorb a partitive "of"-PP into a quantifier head ("one of the men").
    include_partitive_of: bool = True
    # Expand a span that overlaps a named entity to the full entity.
    expand_to_full_entity: bool = True
    # Maximum forward distance (tokens) for absorbing a particle or preposition.
    predicate_particle_window: int = 2


@dataclass(frozen=True)
class FilterConfig:
    """Filter toggles and thresholds."""

    # Drop triplets whose subject and argument denote the same entity.
    drop_self_reference: bool = True
    # Token-overlap ratio (of the shorter span) above which subject and
    # argument count as the same entity.
    self_reference_overlap: float = 0.5
    # Drop degenerate arguments (bare percentages, and the two toggles below).
    drop_fragment_args: bool = False
    # Drop arguments that are bare section headers ("History", "Early life").
    drop_section_headers: bool = False
    # Drop bare-year arguments unless the predicate carries temporal context.
    drop_year_only_args: bool = False
    require_predicate: bool = True
    require_subject: bool = True


@dataclass(frozen=True)
class OptimizeConfig:
    """Rule-optimization loop thresholds (see :mod:`.optimize`)."""

    # Weight of predicate coverage in the combined dev score.
    score_lambda: float = 0.25
    # Minimum dev-score gain for a proposal to be accepted.
    accept_delta_s: float = 0.003
    # Largest precision drop a proposal may cause.
    precision_floor_delta: float = -0.0075
    # Score movement below which an iteration counts as a plateau.
    stop_eps: float = 0.006
    # Iterations a rejected cluster waits before being proposed again.
    cooldown_iterations: int = 3
    # Smallest false-negative cluster worth proposing a rule for.
    cluster_size_floor: int = 15
    # Reserved; not read by the loop.
    per_agent_wallclock_sec: int = 30 * 60
    # Total wall-clock budget for one optimization run.
    total_wallclock_sec: int = 24 * 60 * 60
    # Clusters proposed per iteration.
    top_k_clusters_per_iter: int = 5
    # Reserved for gates that are not implemented; not read by the loop.
    lexical_specificity_max_match_rate: float = 0.05
    confidence_smear_conf_threshold: float = 0.95
    confidence_smear_f1_threshold: float = 0.7
    overfit_anti_excludes_examples: bool = True


@dataclass(frozen=True)
class ExtractionConfig:
    """Top-level configuration passed to :class:`~.pipeline.Pipeline`."""

    # Names of rules to run; None means every discovered rule.
    enabled_rules: Optional[FrozenSet[str]] = None
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    spacy_model: str = "en_core_web_trf"
    use_gliner: bool = False
    gliner_model: str = "numind/NuNerZero"

    def with_enabled_rules(self, names: FrozenSet[str]) -> "ExtractionConfig":
        return replace(self, enabled_rules=names)

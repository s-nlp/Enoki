"""Pipeline configuration.

All rule toggles and shape/filter thresholds live here. The optimization loop's
*threshold channel* (PLAN.md §7) tunes the numeric fields; the *rule channel*
flips ``enabled_rules``.

Protected from agent edits (only humans change the schema; agents only flip
values).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class ShapeConfig:
    """Span-shaping policy thresholds."""

    # Strip leading determiners ("the", "a", "an") from subject/object spans.
    strip_leading_det: bool = True
    # Include compound modifiers in subject/object heads.
    include_compound: bool = True
    # Include adjectival modifiers in subject/object heads.
    include_amod: bool = True
    # Absorb a partitive "of"-PP into a quantifier/number subject head
    # ("Some of the methods", "One of the men"): the gold subject span
    # is the whole partitive phrase, not just the quantifier.
    include_partitive_of: bool = True
    # NER-aware boundary expansion: if a span partially overlaps an entity,
    # expand to cover the full entity.
    expand_to_full_entity: bool = True
    # Maximum distance (tokens) the predicate may extend forward to absorb a
    # particle / preposition.
    predicate_particle_window: int = 2


@dataclass(frozen=True)
class FilterConfig:
    """Filter thresholds.

    Each filter has a boolean toggle; numeric fields tune sensitivity.
    """

    # Re-calibrated 2026-05-16 for LSOIE+OpenIE4 via single-toggle
    # ablation (work/ablate_filters.py). OpenIE gold legitimately
    # contains long/fragmentary args, header-like NPs, and year-only
    # time args, so the QA-SRL-era drop_* filters were removing valid
    # predictions. Disabling the three below: ΔS=+0.0006, ΔR=+0.0014,
    # ΔP=-0.0002 (combined). drop_self_reference is KEPT — disabling
    # it cost ΔP=-0.0008 for zero ΔS.
    drop_self_reference: bool = True
    # Threshold on token-overlap ratio between subject and argument before the
    # fact is rejected as self-referential. 0.5 = strictly more than half of
    # the shorter span's tokens overlap.
    self_reference_overlap: float = 0.5
    drop_fragment_args: bool = False
    drop_section_headers: bool = False
    drop_year_only_args: bool = False
    require_predicate: bool = True
    require_subject: bool = True


@dataclass(frozen=True)
class OptimizeConfig:
    """Agent authoring loop knobs (PLAN.md §7, §8)."""

    score_lambda: float = 0.25
    # Re-tuned 2026-05-16 for the LSOIE+OpenIE4 corpus (gold facts
    # ~5.1k vs QA-SRL ~289k; dF1/dTP ≈ 8.4e-5, i.e. ~50 net-correct
    # facts ≈ ΔS 0.004, 1pp precision ≈ 65 facts). Eval is
    # deterministic, so ε is a "worth keeping" bar, not a noise floor.
    #   ε  0.001 → 0.003  : ≈35 net-correct-fact S-gain; filters
    #                       trivial/overfit edits without recreating
    #                       the QA-SRL plateau (real rules move S by
    #                       thousandths here, well above 0.003).
    #   δ −0.005 → -0.0075: allow ≤0.75pp precision give-back — a bit
    #                       looser than QA-SRL since S is now strongly
    #                       recall-driven, still <1pp to protect the
    #                       already-low P≈0.46.
    #   stop_eps 2·ε.
    accept_delta_s: float = 0.003
    precision_floor_delta: float = -0.0075
    stop_eps: float = 0.006
    cooldown_iterations: int = 3
    cluster_size_floor: int = 15
    per_agent_wallclock_sec: int = 30 * 60
    total_wallclock_sec: int = 24 * 60 * 60
    top_k_clusters_per_iter: int = 5
    lexical_specificity_max_match_rate: float = 0.05
    confidence_smear_conf_threshold: float = 0.95
    confidence_smear_f1_threshold: float = 0.7
    overfit_anti_excludes_examples: bool = True


@dataclass(frozen=True)
class ExtractionConfig:
    """Top-level config passed to the pipeline."""

    # Names of rules to run. None means "all auto-discovered rules".
    enabled_rules: Optional[FrozenSet[str]] = None
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    spacy_model: str = "en_core_web_trf"
    use_gliner: bool = False
    gliner_model: str = "numind/NuNerZero"

    def with_enabled_rules(self, names: FrozenSet[str]) -> "ExtractionConfig":
        return replace(self, enabled_rules=names)

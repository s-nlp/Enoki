"""Utility functions for NLI checking."""

import math
import re
import torch
from functools import partial
from typing import List, Dict, Tuple
from spacy.tokens import Span

from nli.base import BaseNLIChecker
from nli.modernbert_nli import ModernBERTEncoderNLI

def hallucination_prob_from_nli(
    nli_score: dict,
    mode: str = "default",
) -> float:
    """
    Calculate hallucination probability from NLI scores.

    Modes:
        - "default": contradiction + neutral
        - "contradiction_only": only contradiction
        - "neutral_only": only neutral

    Args:
        nli_score: Dict with 'entailment', 'neutral', 'contradiction' scores
        mode: Calculation mode (default: "default")

    Returns:
        Hallucination probability (0.0 to 1.0)
    """
    n = float(nli_score.get("neutral", 0))
    c = float(nli_score.get("contradiction", 0))

    if mode == "contradiction_only":
        return c
    elif mode == "neutral_only":
        return n
    else:  # default
        return n + c


def is_context_valid(context: str, min_words: int = 10) -> bool:
    """
    Check if context is valid for NLI scoring.

    Args:
        context: The context/premise text
        min_words: Minimum number of words for valid context

    Returns:
        True if context is valid, False otherwise
    """
    if not context or not context.strip():
        return False

    word_count = len(context.split())
    return word_count >= min_words


# ============================================================================
# GLOBAL CHECKER REGISTRY
# ============================================================================

_CHECKER_REGISTRY: Dict[str, BaseNLIChecker] = {}


def get_nli_checker(method: str = "modernbert", **kwargs) -> BaseNLIChecker:
    """Get or create NLI checker instance (only "modernbert" is supported)."""
    if method not in _CHECKER_REGISTRY:
        if method == "modernbert":
            _CHECKER_REGISTRY[method] = ModernBERTEncoderNLI()
        else:
            raise ValueError(f"Unknown NLI method: {method!r}. Only 'modernbert' is supported.")
    return _CHECKER_REGISTRY[method]


@torch.inference_mode()
def check_nli_batch_fast(
    premise: str,
    hypotheses: list[str],
    *,
    max_length: int = 8000,
    method: str = "modernbert",
    **kwargs
) -> list[dict]:
    """
    Check NLI for a batch of hypotheses given a premise.

    Args:
        premise: The context/premise text
        hypotheses: List of hypothesis strings to check
        max_length: Maximum sequence length for tokenization
        method: NLI method to use (default: "modernbert")
        **kwargs: Additional parameters (e.g., premise_chunk_overlap_sents)

    Returns:
        List of dicts with 'entailment', 'neutral', 'contradiction' probabilities
    """
    checker = get_nli_checker(method)
    return checker.check_batch(premise, hypotheses, max_length=max_length, **kwargs)


def argument_core_span(arg: Span) -> Span | None:
    """
    Make scoring argument shorter by removing leading prepositions and trailing punctuation.

    Args:
        arg: Spacy Span to process

    Returns:
        Cleaned Span or None
    """
    PREP_WORDS = {
        "in","on","at","from","to","as","with","by","for","of","into","onto","over","under","about","around","during","before","after"
    }
    DET_WORDS = {"a","an","the"}

    if arg is None or len(arg) == 0:
        return None

    doc = arg.doc
    lo, hi = arg.start, arg.end

    # trim left punct
    while lo < hi and doc[lo].is_punct:
        lo += 1
    if lo >= hi:
        return arg

    # remove leading preposition if it looks like PP
    t0 = doc[lo]
    t0_low = t0.text.lower()

    looks_like_pp = (
        t0_low in PREP_WORDS
        and (t0.pos_ in {"ADP", "PART"})
        and (t0.dep_ in {"prep", "case"} or t0.head.pos_ in {"VERB", "AUX"})
    )

    if looks_like_pp:
        lo += 1
        # optional: remove "the/a/an" after preposition
        while lo < hi and doc[lo].pos_ == "DET" and doc[lo].text.lower() in DET_WORDS:
            lo += 1

    # trim right punct
    while hi > lo and doc[hi - 1].is_punct:
        hi -= 1

    if hi <= lo:
        return arg  # fallback: don't remove everything

    return doc[lo:hi]


def predicate_core_span(pred: Span) -> Span | None:
    """
    Narrow predicate span to semantic core: keep VERB/AUX + neg/aux + particles, remove ADP/PUNCT.

    Args:
        pred: Spacy Span to process

    Returns:
        Core predicate Span or None
    """
    if pred is None:
        return None

    keep = []
    for t in pred:
        if t.is_punct:
            continue
        if t.pos_ in {"VERB", "AUX"}:
            keep.append(t)
        elif t.dep_ in {"aux", "auxpass", "neg"}:
            keep.append(t)
        elif t.pos_ == "PART" and t.text.lower() == "to":
            keep.append(t)

    if not keep:
        return None

    lo = min(t.i for t in keep)
    hi = max(t.i for t in keep)
    return pred.doc[lo : hi + 1]


def score_facts_with_nli(
    *,
    context: str,
    granular_facts: list,
    check_nli_batch_fn,
    chunk_size: int | None = None,
    incremental_stop_threshold: float = 0.5,
    hall_prob_mode: str = "default",
    pronoun_map: list | None = None,
) -> List[Dict]:
    """
    Score facts using NLI and attribute spans.

    Args:
        context: The premise/context text
        granular_facts: List of Fact objects OR IncrementalFactGroup objects
        check_nli_batch_fn: Function to check NLI (e.g., check_nli_batch_fast)
        chunk_size: Optional batch size for NLI checking
        incremental_stop_threshold: Threshold for stopping incremental group checking.
            If a fact has hall_prob > threshold or contradiction > threshold, all
            subsequent facts in the same incremental group will have hall_prob set to 0.
            Default: 0.5

    Returns:
        List of dicts with fact info, spans, and NLI scores
    """
    # Handle IncrementalFactGroup objects
    # Check if first item has 'facts' attribute (IncrementalFactGroup)
    if granular_facts and hasattr(granular_facts[0], 'facts'):
        # IncrementalFactGroup format - flatten to facts
        # Track group membership and deltas for each fact
        all_facts = []
        fact_to_group = {}  # maps fact index to (group_index, fact_index_in_group)
        fact_to_delta = {}  # maps fact index to delta span
        fact_to_conf = {}   # maps global_fact_idx → group.confidence (extractor score)

        fact_to_clause_type = {}  # global_fact_idx → clause_type str or None

        for group_idx, group in enumerate(granular_facts):
            for fact_idx, fact in enumerate(group.facts):
                global_fact_idx = len(all_facts)
                all_facts.append(fact)
                fact_to_group[global_fact_idx] = (group_idx, fact_idx)
                fact_to_clause_type[global_fact_idx] = getattr(group, "clause_type", None)
                conf = getattr(group, "confidence", None)
                if conf is not None:
                    fact_to_conf[global_fact_idx] = float(conf)

                # Store delta for this fact
                if group.deltas and fact_idx < len(group.deltas):
                    fact_to_delta[global_fact_idx] = group.deltas[fact_idx]

        facts_to_check = all_facts
    else:
        # Legacy format: plain list of Fact objects
        facts_to_check = granular_facts
        fact_to_group = None
        fact_to_delta = None
        fact_to_conf = {}

    # Collect unique hypotheses
    hyps: List[str] = []
    hyp2idx: Dict[str, int] = {}

    # Items store span attributions with reference to hypothesis
    items: List[Dict] = []

    # Build a lookup: pronoun_text (lowercase) -> antecedent. The pronoun_map
    # may list the same surface form multiple times (one entry per mention);
    # keep the earliest (most local) antecedent for each pronoun, then sort
    # by descending pronoun length so "themselves" is matched before "them".
    _pronoun_lookup: list[tuple[str, str, re.Pattern]] = []
    if pronoun_map:
        seen: dict[str, str] = {}
        for r in sorted(pronoun_map, key=lambda x: x['start']):
            pt = r.get('pronoun_text', '')
            ant = r.get('replacement', '')
            if pt and ant and pt.lower() not in seen:
                seen[pt.lower()] = ant
        for pron_lower, antecedent in sorted(seen.items(), key=lambda kv: -len(kv[0])):
            pat = re.compile(r'\b' + re.escape(pron_lower) + r'\b', re.IGNORECASE)
            _pronoun_lookup.append((pron_lower, antecedent, pat))

    for fact_idx, fact in enumerate(facts_to_check):
        # Use Fact.__str__() which returns natural sentence format
        hyp = str(fact)

        # Substitute pronouns with their antecedent anywhere in the hypothesis
        # (subject, object, oblique) so NLI sees a specific named entity rather
        # than an ambiguous pronoun. Spans are unaffected because OIE ran on
        # the original (unmodified) text. Word boundaries (\b) prevent matching
        # inside other words ("the" vs "they", "it" vs "with").
        if _pronoun_lookup:
            for _pron_lower, antecedent, pat in _pronoun_lookup:
                hyp = pat.sub(antecedent, hyp)

        if hyp not in hyp2idx:
            hyp2idx[hyp] = len(hyps)
            hyps.append(hyp)

        # Determine which span to use for hallucination marking
        # For incremental facts, use delta instead of full argument
        arg_span_for_marking = None
        if fact_to_delta is not None and fact_idx in fact_to_delta:
            # Use delta as the span to mark
            arg_span_for_marking = fact_to_delta[fact_idx]
        else:
            # Use full argument (legacy behavior)
            arg_span_for_marking = getattr(fact, "argument", None)

        # Argument span (if exists)
        if arg_span_for_marking is not None:
            arg_core = argument_core_span(arg_span_for_marking) or arg_span_for_marking
            s = int(getattr(arg_core, "start_char", -1))
            e = int(getattr(arg_core, "end_char", -1))
            if s >= 0 and e > s and arg_core.text.strip():
                item = {
                    "fact": hyp,
                    "span_kind": "argument",
                    "span_start": s,
                    "span_end": e,
                    "span_text": arg_core.text,
                    "source_text": arg_core.doc.text,
                    "fact_idx": fact_idx,
                    "group_info": fact_to_group.get(fact_idx) if fact_to_group else None,
                    "clause_type": fact_to_clause_type.get(fact_idx) if fact_to_clause_type else None,
                }
                if fact_idx in fact_to_conf:
                    item["triple_conf"] = fact_to_conf[fact_idx]
                items.append(item)

        # Predicate span (always)
        pred = getattr(fact, "predicate", None)
        if pred is not None:
            pred_core = predicate_core_span(pred)
            if pred_core is not None and pred_core.text.strip():
                s = int(getattr(pred_core, "start_char", -1))
                e = int(getattr(pred_core, "end_char", -1))
                if s >= 0 and e > s:
                    item = {
                        "fact": hyp,
                        "span_kind": "predicate",
                        "span_start": s,
                        "span_end": e,
                        "span_text": pred_core.text,
                        "source_text": pred_core.doc.text,
                        "fact_idx": fact_idx,
                        "group_info": fact_to_group.get(fact_idx) if fact_to_group else None,
                        "clause_type": fact_to_clause_type.get(fact_idx) if fact_to_clause_type else None,
                    }
                    if fact_idx in fact_to_conf:
                        item["triple_conf"] = fact_to_conf[fact_idx]
                    items.append(item)

    if not hyps:
        return []

    def _run_batch(hyps_batch: List[str]) -> List[Dict[str, float]]:
        return check_nli_batch_fn(context, hyps_batch)

    nli_scores_all: List[Dict[str, float]] = []
    if chunk_size is None or chunk_size >= len(hyps):
        nli_scores_all = _run_batch(hyps)
    else:
        for i in range(0, len(hyps), chunk_size):
            nli_scores_all.extend(_run_batch(hyps[i:i+chunk_size]))

    assert len(nli_scores_all) == len(hyps)

    hyp2nli = {h: nli_scores_all[i] for i, h in enumerate(hyps)}

    # Add NLI scores and hallucination probability to each span
    for it in items:
        nli = hyp2nli[it["fact"]]
        it.update(nli)
        it["hall_prob"] = hallucination_prob_from_nli(nli, mode=hall_prob_mode)

    # Apply incremental group stopping logic
    # If we have incremental groups, stop checking after first incorrect fact in each group
    if fact_to_group is not None:
        # Group items by group_idx and fact_idx_in_group
        group_items: Dict[int, Dict[int, List[Dict]]] = {}

        for it in items:
            group_info = it.get("group_info")
            if group_info is None:
                continue

            group_idx, fact_idx_in_group = group_info

            if group_idx not in group_items:
                group_items[group_idx] = {}
            if fact_idx_in_group not in group_items[group_idx]:
                group_items[group_idx][fact_idx_in_group] = []

            group_items[group_idx][fact_idx_in_group].append(it)

        # For each group, check facts in order and stop after first incorrect one
        HALL_PROB_THRESHOLD = incremental_stop_threshold
        CONTRADICTION_THRESHOLD = incremental_stop_threshold

        for group_idx, facts_dict in group_items.items():
            # Sort facts by fact_idx_in_group (order of increment)
            sorted_fact_indices = sorted(facts_dict.keys())

            # Find first incorrect fact
            first_incorrect_idx = None
            for fact_idx_in_group in sorted_fact_indices:
                fact_items = facts_dict[fact_idx_in_group]

                # Check if any span of this fact is incorrect
                is_incorrect = False
                for it in fact_items:
                    hall_prob = it.get("hall_prob", 0.0)
                    contradiction = it.get("contradiction", 0.0)

                    if hall_prob > HALL_PROB_THRESHOLD or contradiction > CONTRADICTION_THRESHOLD:
                        is_incorrect = True
                        break

                if is_incorrect:
                    first_incorrect_idx = fact_idx_in_group
                    break

            # If we found an incorrect fact, propagate its score to all subsequent facts.
            # Later chain steps build on a hallucinated base, so they are at least as
            # hallucinated as the first bad step.
            if first_incorrect_idx is not None:
                trigger_items = facts_dict[first_incorrect_idx]
                trigger_score = max((it.get("hall_prob", 0.0) for it in trigger_items), default=0.0)
                for fact_idx_in_group in sorted_fact_indices:
                    if fact_idx_in_group > first_incorrect_idx:
                        for it in facts_dict[fact_idx_in_group]:
                            it["hall_prob"] = trigger_score
                            it["_stopped_by_incremental_logic"] = True

    return items


def _is_incremental_continuation(fact_a: List[str], fact_b: List[str]) -> bool:
    """True if fact_b is an incremental extension of fact_a.

    Pattern A: same (subj, pred), arg_b starts or ends with arg_a.
    Pattern B: same subj, pred_b starts with pred_a and contains arg_a
               (pred extension absorbing the previous arg as a modifier).
    """
    if not fact_a or not fact_b:
        return False
    subj_a = fact_a[0].strip().lower()
    subj_b = fact_b[0].strip().lower()
    if subj_a != subj_b:
        return False
    pred_a = fact_a[1].strip().lower() if len(fact_a) > 1 else ""
    arg_a  = fact_a[2].strip().lower() if len(fact_a) > 2 else ""
    pred_b = fact_b[1].strip().lower() if len(fact_b) > 1 else ""
    arg_b  = fact_b[2].strip().lower() if len(fact_b) > 2 else ""

    if not arg_a:
        return False

    # Pattern A: same predicate, arg_b is an extension of arg_a
    if pred_a == pred_b and (arg_b.startswith(arg_a) or arg_b.endswith(arg_a)):
        return True

    # Pattern B: pred_b extends pred_a and has absorbed arg_a
    if pred_b.startswith(pred_a) and arg_a in pred_b:
        return True

    return False


def group_incremental_triplets(
    triplets: List[List[str]],
    spans: List[List[int]],
) -> List[List[Tuple[List[str], List[int]]]]:
    """Group flat incremental triplets into chains.

    Consecutive triplets that extend each other (same subject plus argument-
    or predicate-extension) are placed in the same group.  Each group is a
    list of (triplet, span) pairs in original order.
    """
    if not triplets:
        return []
    groups: List[List[Tuple[List[str], List[int]]]] = []
    current: List[Tuple[List[str], List[int]]] = [(triplets[0], spans[0])]
    for i in range(1, len(triplets)):
        prev_triplet, _ = current[-1]
        if _is_incremental_continuation(prev_triplet, triplets[i]):
            current.append((triplets[i], spans[i]))
        else:
            groups.append(current)
            current = [(triplets[i], spans[i])]
    groups.append(current)
    return groups


def score_preextracted_with_nli(
    *,
    context: str,
    triplet_span_pairs: List[Tuple[List[str], List[int]]],
    check_nli_batch_fn,
    hall_prob_mode: str = "default",
    answer: str = "",
    incremental: bool = False,
) -> List[Dict]:
    """
    Score pre-extracted triplets with NLI and return span-annotated results.

    Args:
        context: Premise text for NLI.
        triplet_span_pairs: List of ([subject, predicate, object], [start, end]).
            Spans are character offsets in the original answer text.
        check_nli_batch_fn: Callable(context, hypotheses) → list of NLI score dicts.
        hall_prob_mode: Hallucination probability aggregation mode.
        answer: Original answer text used to locate the hal span (last triple element).
        incremental: If True, detect incremental chains among consecutive triplets and
            tag each item with group_info=(group_id, position_in_group).  Spans are
            taken directly from the JSONL file (the delta span) rather than a text
            search.  Callers can then apply early-stopping per group at threshold time.

    Returns:
        List of dicts with keys orig_span_start, orig_span_end, hall_prob,
        entailment, neutral, contradiction, fact, span_kind='argument'.
        When incremental=True, group_info is (group_id, position_in_group).
    """
    if not triplet_span_pairs:
        return []

    # Build (triplet, span, group_id, group_pos) list
    if incremental:
        triplets = [t for t, _ in triplet_span_pairs]
        spans    = [s for _, s in triplet_span_pairs]
        groups   = group_incremental_triplets(triplets, spans)
        flat: List[Tuple[List[str], List[int], int, int]] = []
        for gid, group in enumerate(groups):
            for gpos, (triplet, span) in enumerate(group):
                flat.append((triplet, span, gid, gpos))
    else:
        flat = [(t, s, None, None) for t, s in triplet_span_pairs]

    hypotheses: List[str] = []
    for triplet, _, _, _ in flat:
        parts = [p.strip() for p in triplet if p and p.strip()]
        hypotheses.append(" ".join(parts))

    # Deduplicate while preserving insertion order
    seen: Dict[str, int] = {}
    unique_hyps: List[str] = []
    for hyp in hypotheses:
        if hyp not in seen:
            seen[hyp] = len(unique_hyps)
            unique_hyps.append(hyp)

    nli_results = check_nli_batch_fn(context, unique_hyps)
    hyp2nli: Dict[str, Dict] = dict(zip(unique_hyps, nli_results))

    items = []
    for idx, ((triplet, span, gid, gpos), hyp) in enumerate(zip(flat, hypotheses)):
        nli = hyp2nli[hyp]

        if incremental:
            # Use JSONL span directly — it already points to the delta (arg2)
            span_start, span_end = span[0], span[1]
        else:
            # Hal span = last element of the triple located in the answer text.
            # Fall back to the JSONL span if the text cannot be found.
            hal_span_text = triplet[-1].strip() if triplet else ""
            if answer and hal_span_text:
                pos = answer.find(hal_span_text)
                if pos != -1:
                    span_start, span_end = pos, pos + len(hal_span_text)
                else:
                    span_start, span_end = span[0], span[1]
            else:
                span_start, span_end = span[0], span[1]

        items.append({
            "fact": hyp,
            "span_kind": "argument",
            "orig_span_start": span_start,
            "orig_span_end": span_end,
            "fact_idx": idx,
            "group_info": (gid, gpos) if gid is not None else None,
            "clause_type": None,
            **nli,
            "hall_prob": hallucination_prob_from_nli(nli, mode=hall_prob_mode),
        })

    return items

"""Utility functions for NLI checking."""

import math
import torch
from functools import partial
from typing import List, Dict
from spacy.tokens import Span

from nli.base import BaseNLIChecker
from nli.modernbert_nli import ModernBERTEncoderNLI
from nli.alignscore_nli import AlignScoreNLI

try:
    from nli.llm_nli import QwenNLI_06B, QwenNLI_4B, QwenNLI_8B, HAS_VLLM
except ImportError:
    HAS_VLLM = False
    QwenNLI_06B = None
    QwenNLI_4B = None
    QwenNLI_8B = None

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
    """
    Get or create NLI checker instance.

    Args:
        method: NLI method name (default: "modernbert")
            - "modernbert": ModernBERT encoder model
            - "alignscore": AlignScore NLI model
            - "qwen_06b": Qwen 0.6B LLM
            - "qwen_4b": Qwen 4B LLM
            - "qwen_8b": Qwen 8B LLM

    Returns:
        BaseNLIChecker instance
    """
    if method not in _CHECKER_REGISTRY:
        if method == "modernbert":
            _CHECKER_REGISTRY[method] = ModernBERTEncoderNLI()
        elif method == "alignscore":
            _CHECKER_REGISTRY[method] = AlignScoreNLI(**kwargs)
        elif method == "qwen_06b":
            _CHECKER_REGISTRY[method] = QwenNLI_06B(**kwargs)
        elif method == "qwen_4b":
            _CHECKER_REGISTRY[method] = QwenNLI_4B(**kwargs)
        elif method == "qwen_8b":
            _CHECKER_REGISTRY[method] = QwenNLI_8B(**kwargs)
        else:
            raise ValueError(f"Unknown NLI method: {method}")

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

        for group_idx, group in enumerate(granular_facts):
            for fact_idx, fact in enumerate(group.facts):
                global_fact_idx = len(all_facts)
                all_facts.append(fact)
                fact_to_group[global_fact_idx] = (group_idx, fact_idx)

                # Store delta for this fact
                if group.deltas and fact_idx < len(group.deltas):
                    fact_to_delta[global_fact_idx] = group.deltas[fact_idx]

        facts_to_check = all_facts
    else:
        # Legacy format: plain list of Fact objects
        facts_to_check = granular_facts
        fact_to_group = None
        fact_to_delta = None

    # Collect unique hypotheses
    hyps: List[str] = []
    hyp2idx: Dict[str, int] = {}

    # Items store span attributions with reference to hypothesis
    items: List[Dict] = []

    for fact_idx, fact in enumerate(facts_to_check):
        # Use Fact.__str__() which returns natural sentence format
        hyp = str(fact)

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
                items.append({
                    "fact": hyp,
                    "span_kind": "argument",
                    "span_start": s,
                    "span_end": e,
                    "span_text": arg_core.text,
                    "source_text": arg_core.doc.text,
                    "fact_idx": fact_idx,
                    "group_info": fact_to_group.get(fact_idx) if fact_to_group else None,
                })

        # Predicate span (always)
        pred = getattr(fact, "predicate", None)
        if pred is not None:
            pred_core = predicate_core_span(pred)
            if pred_core is not None and pred_core.text.strip():
                s = int(getattr(pred_core, "start_char", -1))
                e = int(getattr(pred_core, "end_char", -1))
                if s >= 0 and e > s:
                    items.append({
                        "fact": hyp,
                        "span_kind": "predicate",
                        "span_start": s,
                        "span_end": e,
                        "span_text": pred_core.text,
                        "source_text": pred_core.doc.text,
                        "fact_idx": fact_idx,
                        "group_info": fact_to_group.get(fact_idx) if fact_to_group else None,
                    })

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

            # If we found an incorrect fact, set hall_prob = 0 for all subsequent facts in group
            if first_incorrect_idx is not None:
                for fact_idx_in_group in sorted_fact_indices:
                    if fact_idx_in_group > first_incorrect_idx:
                        # This fact comes after the first incorrect one - set hall_prob = 0
                        for it in facts_dict[fact_idx_in_group]:
                            it["hall_prob"] = 0.0
                            it["_stopped_by_incremental_logic"] = True

    return items

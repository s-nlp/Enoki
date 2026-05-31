"""Adapters: LSOIE and OpenIE4 gold -> :class:`GoldTriplet`.

Replaces the QA-SRL dev corpus. Both sources are n-ary OpenIE tuples
(one predicate + labelled argument spans); we flatten each tuple into
SPO ``GoldTriplet``s the existing token-overlap scorer already
understands:

- subject  = A0 (LSOIE) / ARG1 (OpenIE4)   — required
- predicate = P  (LSOIE) / REL  (OpenIE4)   — required
- primary argument = A1 (LSOIE) / ARG2 (OpenIE4), role="object"
- extra args = A2/A3 (LSOIE), TIME/LOC (OpenIE4) -> one extra
  GoldTriplet each, sharing subject+predicate, role tagged accordingly.

``role``/``prep``/``is_passive``/``is_negated`` are not consulted by
``metrics.is_match``; they are filled with safe defaults.
``predicate_verb_index`` *is* used (coverage), so we set it to the
predicate head index.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..preprocess.ptb import normalize_ptb_tokens, ptb_detokenize
from .qasrl_to_spo import GoldTriplet, QASRLConversionResult

_UNUSED = {"[unused1]", "[unused2]", "[unused3]"}


def _first_span(spans: Dict[str, List[List[int]]], role: str) -> Optional[Tuple[int, int]]:
    runs = spans.get(role)
    if not runs:
        return None
    s, e = runs[0]
    return (s, e)


def _triplets_for_extraction(
    sentence_id: str,
    tokens: Tuple[str, ...],
    subject_span: Optional[Tuple[int, int]],
    predicate_text: str,
    predicate_index: int,
    arg_runs: List[Tuple[str, Optional[Tuple[int, int]]]],
) -> List[GoldTriplet]:
    """Flatten one OpenIE extraction into SPO GoldTriplets.

    Returns ``[]`` if the extraction lacks a subject or predicate.
    """
    if subject_span is None or not predicate_text.strip():
        return []
    if not arg_runs:
        arg_runs = [("other", None)]  # subject+predicate only
    out: List[GoldTriplet] = []
    for role, span in arg_runs:
        out.append(
            GoldTriplet(
                sentence_id=sentence_id,
                tokens=tokens,
                subject_span=subject_span,
                predicate_text=predicate_text,
                predicate_verb_index=predicate_index,
                argument_span=span,
                role=role,
                prep=None,
                is_passive=False,
                is_negated=False,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# LSOIE                                                                        #
# --------------------------------------------------------------------------- #

_LSOIE_EXTRA = {"A2": "other", "A3": "other"}


def _lsoie_extraction_triplets(rec: dict) -> List[GoldTriplet]:
    tokens: List[str] = normalize_ptb_tokens(rec["tokens"])
    spans: Dict[str, List[List[int]]] = rec["spans"]
    subj = _first_span(spans, "A0")
    p_span = _first_span(spans, "P")
    if p_span is not None:
        predicate_text = " ".join(tokens[p_span[0]:p_span[1]])
        predicate_index = rec.get("head_pred_id", p_span[0])
    else:
        predicate_text = rec.get("pred", "")
        predicate_index = rec.get("head_pred_id", 0)
    arg_runs: List[Tuple[str, Optional[Tuple[int, int]]]] = []
    a1 = _first_span(spans, "A1")
    if a1 is not None:
        arg_runs.append(("object", a1))
    for role_key, role_name in _LSOIE_EXTRA.items():
        sp = _first_span(spans, role_key)
        if sp is not None:
            arg_runs.append((role_name, sp))
    return _triplets_for_extraction(
        rec["sentence_id"], tuple(tokens), subj, predicate_text,
        predicate_index, arg_runs,
    )


def iter_lsoie(
    path: Path, max_per_split: Optional[int] = None
) -> Iterable[QASRLConversionResult]:
    """Group consecutive extractions sharing the same sentence into one
    result (LSOIE orders all run_ids of a sentence contiguously)."""
    n = 0
    cur_key: Optional[str] = None
    cur_text = ""
    cur_id = ""
    cur_triplets: List[GoldTriplet] = []

    def emit() -> Optional[QASRLConversionResult]:
        if cur_key is None:
            return None
        if not cur_triplets:
            return QASRLConversionResult(
                sentence_id=cur_id, sentence_text=cur_text,
                dropped=True, drop_reason="no usable extraction",
            )
        return QASRLConversionResult(
            sentence_id=cur_id, sentence_text=cur_text,
            triplets=list(cur_triplets),
        )

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            norm = normalize_ptb_tokens(rec["tokens"])
            key = " ".join(norm)
            if key != cur_key:
                if max_per_split is not None and n >= max_per_split:
                    res = emit()
                    if res is not None:
                        yield res
                    return
                res = emit()
                if res is not None:
                    yield res
                    n += 1
                cur_key = key
                cur_text = ptb_detokenize(norm)
                cur_id = f"{rec['sentence_id']}#g{n}"
                cur_triplets = []
            cur_triplets.extend(_lsoie_extraction_triplets(rec))
        if not (max_per_split is not None and n >= max_per_split):
            res = emit()
            if res is not None:
                yield res


# --------------------------------------------------------------------------- #
# OpenIE4 (token / tag-line text format)                                       #
# --------------------------------------------------------------------------- #

_OIE4_EXTRA_ROLE = {"TIME": "time", "LOC": "location"}


def _contiguous_runs(labels: List[str], target: str) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, lab in enumerate(labels + ["__END__"]):
        if lab == target and start is None:
            start = i
        elif lab != target and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _convert_openie4_block(
    sent_id: str, tokens: List[str], tag_lines: List[List[str]]
) -> QASRLConversionResult:
    tokens = normalize_ptb_tokens(tokens)
    text = ptb_detokenize(tokens)
    tok = tuple(tokens)
    triplets: List[GoldTriplet] = []
    for k, labels in enumerate(tag_lines):
        labels = labels[: len(tokens)]
        subj_runs = _contiguous_runs(labels, "ARG1")
        rel_idx = [i for i, l in enumerate(labels) if l == "REL"]
        if not subj_runs or not rel_idx:
            continue
        predicate_text = " ".join(tokens[i] for i in rel_idx)
        arg_runs: List[Tuple[str, Optional[Tuple[int, int]]]] = []
        for r in _contiguous_runs(labels, "ARG2"):
            arg_runs.append(("object", r))
        for tag, role in _OIE4_EXTRA_ROLE.items():
            for r in _contiguous_runs(labels, tag):
                arg_runs.append((role, r))
        triplets.extend(
            _triplets_for_extraction(
                f"{sent_id}#{k}", tok, subj_runs[0],
                predicate_text, rel_idx[0], arg_runs,
            )
        )
    if not triplets:
        return QASRLConversionResult(
            sentence_id=sent_id, sentence_text=text,
            dropped=True, drop_reason="no usable extraction",
        )
    return QASRLConversionResult(
        sentence_id=sent_id, sentence_text=text, triplets=triplets
    )


def iter_openie4(
    path: Path, max_per_split: Optional[int] = None
) -> Iterable[QASRLConversionResult]:
    """Stream OpenIE4: a sentence line then N tag lines, repeating.

    Sentence lines end with the ``[unused1] [unused2] [unused3]``
    placeholders; tag lines are pure label vocab. We strip the three
    placeholders (and their always-NONE label columns) before building.
    """
    sent_count = 0
    cur_tokens: Optional[List[str]] = None
    cur_tags: List[List[str]] = []

    def flush() -> Optional[QASRLConversionResult]:
        nonlocal cur_tokens, cur_tags
        if cur_tokens is None:
            return None
        res = _convert_openie4_block(
            f"openie4-{sent_count}", cur_tokens, cur_tags
        )
        cur_tokens, cur_tags = None, []
        return res

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            toks = line.split(" ")
            is_sentence = toks[-3:] == ["[unused1]", "[unused2]", "[unused3]"]
            if is_sentence:
                res = flush()
                if res is not None:
                    yield res
                sent_count += 1
                if max_per_split is not None and sent_count > max_per_split:
                    cur_tokens = None
                    break
                cur_tokens = [t for t in toks if t not in _UNUSED]
                cur_tags = []
            elif cur_tokens is not None:
                cur_tags.append(toks)
        res = flush()
        if res is not None:
            yield res

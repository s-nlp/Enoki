"""Deterministic conversion of QA-SRL Bank 2.0 records to gold SPO triplets.

A QA-SRL ``Sentence`` carries, per verb, a set of ``QuestionLabel`` entries.
Each question has a wh-word, grammatical slots, and one or more answer
``spans`` (token-index intervals). The conversion fits these into our flat
SPO scheme as follows:

* The **subject** of every triplet for a verb is the answer span(s) of any
  ``wh in {who}`` question on that verb. If multiple ``who``-questions or
  spans exist, each becomes a separate gold subject candidate.
* The **predicate** is the surface verb form rebuilt from
  ``verbInflectedForms`` and the question slot grammar
  (``tense`` + ``isPerfect`` + ``isProgressive`` + ``isNegated`` + ``isPassive``
  + the slot's ``prep`` if any). The predicate string mirrors the rules
  pipeline's "verb head + particle + attached prep" convention.
* The **argument** comes from the answer span of a non-subject wh-question:

  - ``what`` (or the ``obj``/``obj2`` slot answer) → role ``object``;
  - ``when`` → ``time``;
  - ``where`` → ``location``;
  - ``how`` → ``manner``;
  - ``why`` → ``cause`` (loose mapping; the agent loop can refine).

Sentences whose conversion yields zero ``who``-answers — i.e. no confident
subject for any verb — are dropped from the calibration set and logged.
``convert_sentence`` returns a :class:`QASRLConversionResult` so the runner
can record the drop reason.

For ``qanom``, the same predicate-slot grammar applies; the predicate token
is the nominalized noun rather than a verb. The conversion is identical
because we render the surface form from ``verbInflectedForms`` if present
or fall back to the slot ``verb`` field text.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ARG_WHS_TO_ROLE = {
    "what": "object",
    "when": "time",
    "where": "location",
    "how": "manner",
    "why": "cause",
    "which": "object",
}


def _question_is_subject_seeking(slots: Dict[str, str]) -> bool:
    """Return True if this question's wh-answer fills the subject role.

    Per QA-SRL slot grammar:
    - "Who posted something?" : wh=who, subj=_, obj=something → subj-seeking
    - "What ended?"           : wh=what, subj=_, obj=_         → subj-seeking
      (for non-human subjects)
    - "What did someone post?": wh=what, subj=someone, obj=_   → NOT subj-seeking

    Heuristic: the wh fills the subject slot iff the question's ``subj`` slot
    is ``_`` (empty/placeholder). The ``who`` wh-word always satisfies this;
    ``what``/``which`` also satisfy it when no separate human placeholder is
    present in subj.
    """
    if slots.get("subj", "_") == "_" and slots.get("wh") in {"who", "what", "which"}:
        return True
    return False


@dataclass(frozen=True)
class GoldTriplet:
    """A flat gold triplet derived from one QA-SRL question pair.

    Spans are stored as ``(start, end)`` token-index intervals (the QA-SRL
    convention: end is exclusive). The runner aligns them with spaCy token
    indices via the shared sentenceTokens list when comparing to predicted
    triplets.
    """

    sentence_id: str
    tokens: Tuple[str, ...]
    subject_span: Tuple[int, int]
    predicate_text: str
    predicate_verb_index: int
    argument_span: Optional[Tuple[int, int]]
    role: str
    prep: Optional[str]
    is_passive: bool
    is_negated: bool

    @property
    def subject_text(self) -> str:
        return " ".join(self.tokens[self.subject_span[0]:self.subject_span[1]])

    @property
    def argument_text(self) -> Optional[str]:
        if self.argument_span is None:
            return None
        return " ".join(self.tokens[self.argument_span[0]:self.argument_span[1]])


@dataclass
class QASRLConversionResult:
    """Result of converting one QA-SRL sentence."""

    sentence_id: str
    sentence_text: str
    triplets: List[GoldTriplet] = field(default_factory=list)
    dropped: bool = False
    drop_reason: Optional[str] = None


def _surface_predicate(
    sentence_tokens: Tuple[str, ...],
    verb_idx: int,
    question_slots: Dict[str, str],
    is_passive: bool,
    is_perfect: bool,
    is_progressive: bool,
    is_negated: bool,
) -> str:
    """Synthesize a gold predicate text faithful to the sentence's surface.

    Strategy:
    - Anchor on ``sentence_tokens[verb_idx]`` — the literal verb form as it
      appears in the sentence, so we don't have to re-inflect.
    - For passive / perfect / progressive constructions, prepend the
      immediately-preceding aux/copula tokens (looking back up to 3 tokens
      until we leave the contiguous AUX run). This captures "was signed",
      "has been chosen", etc.
    - Append the slot's ``prep`` if any. Negation is not included in the
      predicate text — the GoldTriplet's ``is_negated`` flag carries it.
    """
    parts: List[str] = []
    if is_passive or is_perfect or is_progressive:
        # Walk backwards collecting plausible auxiliaries.
        i = verb_idx - 1
        aux: List[str] = []
        while i >= 0 and verb_idx - i <= 3:
            tok = sentence_tokens[i]
            lower = tok.lower()
            if lower in _AUX_FORMS:
                aux.append(tok)
                i -= 1
                continue
            break
        parts.extend(reversed(aux))
    if 0 <= verb_idx < len(sentence_tokens):
        parts.append(sentence_tokens[verb_idx])

    prep = question_slots.get("prep", "_")
    if prep and prep != "_":
        parts.append(prep)
    return " ".join(parts).strip()


_AUX_FORMS = {
    "am", "are", "is", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
}


def _spans_of_judgment(answer_judgments) -> Iterable[Tuple[int, int]]:
    """Flatten valid-answer spans across judgments."""
    seen = set()
    for j in answer_judgments:
        if not j.get("isValid"):
            continue
        spans = j.get("spans") or []
        for span in spans:
            key = (span[0], span[1])
            if key in seen:
                continue
            seen.add(key)
            yield key


def convert_sentence(sentence_json: Dict) -> QASRLConversionResult:
    """Convert one QA-SRL ``Sentence`` JSON object into gold triplets."""
    sentence_id = sentence_json["sentenceId"]
    tokens = tuple(sentence_json["sentenceTokens"])
    text = " ".join(tokens)
    result = QASRLConversionResult(sentence_id=sentence_id, sentence_text=text)

    verb_entries = sentence_json.get("verbEntries") or sentence_json.get(
        "nominalEntries", {}
    )
    if not verb_entries:
        result.dropped = True
        result.drop_reason = "no verb entries"
        return result

    found_any = False
    for verb_idx_str, verb_entry in verb_entries.items():
        verb_idx = int(verb_idx_str)
        question_labels = verb_entry.get("questionLabels", {})

        # Gather subject candidates from any subject-seeking question.
        subject_spans: List[Tuple[int, int]] = []
        subject_qkeys = set()
        for qkey, qlabel in question_labels.items():
            slots = qlabel["questionSlots"]
            if _question_is_subject_seeking(slots):
                spans = list(_spans_of_judgment(qlabel["answerJudgments"]))
                if spans:
                    subject_qkeys.add(qkey)
                    subject_spans.extend(spans)

        if not subject_spans:
            continue

        for qkey, qlabel in question_labels.items():
            slots = qlabel["questionSlots"]
            wh = slots.get("wh")
            # Skip the subject-seeking questions themselves (they don't
            # describe an argument).
            if qkey in subject_qkeys:
                continue
            role = ARG_WHS_TO_ROLE.get(wh, "other")
            predicate_text = _surface_predicate(
                tokens,
                verb_idx,
                slots,
                is_passive=qlabel.get("isPassive", False),
                is_perfect=qlabel.get("isPerfect", False),
                is_progressive=qlabel.get("isProgressive", False),
                is_negated=qlabel.get("isNegated", False),
            )
            prep = slots.get("prep")
            prep = prep if prep and prep != "_" else None
            arg_spans = list(_spans_of_judgment(qlabel["answerJudgments"]))
            if not arg_spans:
                arg_spans = [None]  # type: ignore[list-item]

            for subj_span in subject_spans:
                for arg_span in arg_spans:
                    result.triplets.append(
                        GoldTriplet(
                            sentence_id=sentence_id,
                            tokens=tokens,
                            subject_span=subj_span,
                            predicate_text=predicate_text,
                            predicate_verb_index=verb_idx,
                            argument_span=arg_span,
                            role=role,
                            prep=prep,
                            is_passive=qlabel.get("isPassive", False),
                            is_negated=qlabel.get("isNegated", False),
                        )
                    )
                    found_any = True

    if not found_any:
        result.dropped = True
        result.drop_reason = "no usable answer spans"
    return result


def iter_jsonl_gz(path: Path) -> Iterator[Dict]:
    """Stream one JSON object per line from a gzipped JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

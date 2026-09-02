"""
ragtruth_utils.py  –  Shared helpers for loading the RAGTruth dataset.

RAGTruth (wandb/RAGTruth-processed) structure per row:
  id                          – unique row id
  query                       – the question / instruction given to the LLM
  context                     – source passages (plain text)
  output                      – LLM response to evaluate
  task_type                   – 'QA' | 'Summary' | 'Data2txt'
  quality                     – 'good' | 'truncated' | 'incorrect_refusal'
  model                       – which LLM generated the response
  temperature                 – sampling temperature (str)
  hallucination_labels        – JSON string: list of character-span dicts, or '[]'
  hallucination_labels_processed – dict with counts per category

Each hallucination label span:
  {
    "start": int,         # char offset in 'output'
    "end":   int,         # char offset (exclusive)
    "text":  str,         # the hallucinated text
    "label_type": str,    # "Evident Conflict" | "Evident Baseless Info" |
                          #  "Subtle Conflict"  | "Subtle Baseless Info"
    "meta":  str,
    "implicit_true": bool,
    "due_to_null":   bool,
  }

Gold label mapping (sentence-level, derived by span overlap):
  Any sentence overlapping at least one hallucination span -> gold_supported = False
  Sentences with no overlapping span                       -> gold_supported = True
  Rows with quality != 'good' are skipped (gold_supported = None)

The iterator yields one flat dict per sentence, closely mirroring the
anah_utils.iter_anah_sentences interface so all runners can use the same
plumbing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sentence splitting (reuses NLTK when available, same as anah_utils)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
except Exception:
    _nltk_sent_tokenize = None


def split_into_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if _nltk_sent_tokenize is not None:
        try:
            return [s.strip() for s in _nltk_sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Span → character-range helpers
# ---------------------------------------------------------------------------

def _char_ranges_for_sentences(
    full_text: str, sentences: List[str]
) -> List[Tuple[int, int]]:
    """Return (start, end) char offsets for each sentence within full_text.

    Uses a simple greedy left-to-right scan.  If a sentence cannot be found
    (edge case due to normalisation) the range is set to (-1, -1).
    """
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for sent in sentences:
        idx = full_text.find(sent, cursor)
        if idx == -1:
            # Fallback: try stripping and searching
            stripped = sent.strip()
            idx = full_text.find(stripped, cursor)
        if idx == -1:
            ranges.append((-1, -1))
        else:
            end = idx + len(sent)
            ranges.append((idx, end))
            cursor = end
    return ranges


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_hallucination_labels(raw: Any) -> List[Dict[str, Any]]:
    """Parse the hallucination_labels field (JSON string or list) into a list."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s == "[]":
            return []
        try:
            return json.loads(s)
        except Exception:
            return []
    return []


def sentence_gold_supported(
    sent_start: int,
    sent_end: int,
    hal_spans: List[Dict[str, Any]],
) -> bool:
    """Return False if any hallucination span overlaps this sentence."""
    if sent_start == -1:
        # Could not locate sentence – treat conservatively as supported
        return True
    for span in hal_spans:
        sp_start = span.get("start", -1)
        sp_end = span.get("end", -1)
        if sp_start < 0 or sp_end < 0:
            continue
        if _spans_overlap(sent_start, sent_end, sp_start, sp_end):
            return False
    return True


# ---------------------------------------------------------------------------
# Main iterator – yields one flat dict per sentence
# ---------------------------------------------------------------------------

def iter_ragtruth_sentences(
    rows: List[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Iterate over RAGTruth rows yielding one dict per sentence.

    Each yielded dict mirrors the anah_utils.iter_anah_sentences interface:

        example_index     – row index in the input list
        row_id            – original dataset 'id' field
        task_type         – 'QA' | 'Summary' | 'Data2txt'
        model             – LLM that generated the response
        question          – the prompt / query
        context           – source passages (used as 'reference' / evidence)
        output            – full LLM response
        sentence          – this sentence's text
        sentence_index    – position within the response
        answer_sentences  – all sentences of the response (for context window)
        hallucination_type – label_type of overlapping span, or 'No Hallucination'
        gold_supported    – True / False  (None rows are pre-filtered by caller)
    """
    for ex_i, row in enumerate(rows):
        output = (row.get("output") or "").strip()
        if not output:
            continue

        question = (row.get("query") or "").strip()
        context = (row.get("context") or "").strip()
        task_type = row.get("task_type", "")
        model = row.get("model", "")
        row_id = row.get("id", str(ex_i))

        hal_spans = parse_hallucination_labels(row.get("hallucination_labels"))
        sentences = split_into_sentences(output)
        if not sentences:
            continue

        char_ranges = _char_ranges_for_sentences(output, sentences)

        for sent_i, (sentence, (s_start, s_end)) in enumerate(
            zip(sentences, char_ranges)
        ):
            if not sentence.strip():
                continue

            gold = sentence_gold_supported(s_start, s_end, hal_spans)

            # Determine hallucination_type label for metadata
            if gold:
                hall_type = "No Hallucination"
            else:
                # Pick the first overlapping span's label_type
                hall_type = "Hallucination"
                for span in hal_spans:
                    sp_s = span.get("start", -1)
                    sp_e = span.get("end", -1)
                    if sp_s >= 0 and sp_e >= 0 and _spans_overlap(s_start, s_end, sp_s, sp_e):
                        hall_type = span.get("label_type", "Hallucination")
                        break

            yield {
                "example_index": ex_i,
                "row_id": row_id,
                "task_type": task_type,
                "model": model,
                "question": question,
                # context = source passages (used as evidence / reference)
                "context": context,
                # aliases for cross-runner compatibility
                "reference": context,
                "ann_reference": context,
                "output": output,
                "sentence": sentence,
                "sentence_index": sent_i,
                "answer_sentences": sentences,
                "answer_text": output,
                "hallucination_type": hall_type,
                "gold_supported": gold,
            }


# ---------------------------------------------------------------------------
# Loader (used by runners that accept a pre-sampled JSONL)
# ---------------------------------------------------------------------------

def load_ragtruth_rows(
    sample_file: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Load a pre-sampled RAGTruth JSONL and return (sentence_rows, n_examples).

    Each line in the JSONL is already a flat sentence row produced by
    iter_ragtruth_sentences / sample_ragtruth_250.py.
    """
    rows: List[Dict[str, Any]] = []
    with open(sample_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_examples = len({r.get("example_index", i) for i, r in enumerate(rows)})
    print(
        f"RAGTruth: loaded {len(rows)} sentence rows "
        f"({n_examples} source responses) from '{sample_file}'",
        flush=True,
    )
    return rows, n_examples

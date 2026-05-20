"""
anah_utils.py  –  Shared helpers for loading the ANAH dataset.

ANAH (opencompass/anah) structure per row:
  name                      – topic name
  documents                 – list[str]  reference documents
  selected_questions        – list[str]  (one question per row)
  GPT3.5_answers_D          – list[str]  GPT-3.5 answer(s)
  InternLM_answers          – list[str]  InternLM answer(s)
  human_GPT3.5_answers_D_ann– list[list[str]]  sentence-level annotations for GPT-3.5
  human_InternLM_answers_ann– list[list[str]]  sentence-level annotations for InternLM
  language                  – str

Each annotation string has the format:
  <Hallucination> None|Contradictory|Unverifiable
  <Reference> ...text... [<SEP> ...more text...]
  [<Correction> ...]          (only when hallucination present)

  OR for sentences with no verifiable claim:
  <No Fact>
  <Reference> None
  <Correction> None

Gold label mapping:
  "No Hallucination"           -> gold_supported = True
  "Contradictory Hallucination"
  "Unverifiable Hallucination" -> gold_supported = False
  "No Fact"                    -> gold_supported = None  (skip)
"""

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

_NO_FACT_RE = re.compile(r"^<No Fact>", re.IGNORECASE)
_HALL_RE = re.compile(r"<Hallucination>\s*(.*?)(?:\n|$)")
_REF_RE = re.compile(r"<Reference>\s*(.*?)(?=\n<|$)", re.DOTALL)


def parse_anah_annotation(ann: str) -> Tuple[str, str]:
    """Parse one ANAH annotation string.

    Returns (hallucination_type, reference_text).
    hallucination_type is one of:
        'No Hallucination', 'Contradictory Hallucination',
        'Unverifiable Hallucination', 'No Fact'
    """
    ann = str(ann or "").strip()

    if _NO_FACT_RE.match(ann):
        return "No Fact", ""

    hall_m = _HALL_RE.search(ann)
    hall_val = hall_m.group(1).strip() if hall_m else ""

    ref_m = _REF_RE.search(ann)
    ref_text = ref_m.group(1).strip() if ref_m else ""
    # Flatten <SEP> markers into a single string
    ref_text = re.sub(r"\s*<SEP>\s*", " | ", ref_text).strip()

    if hall_val in ("None", ""):
        hall_type = "No Hallucination"
    elif "Contradictory" in hall_val:
        hall_type = "Contradictory Hallucination"
    elif "Unverifiable" in hall_val:
        hall_type = "Unverifiable Hallucination"
    else:
        hall_type = hall_val  # fallback: keep raw value

    return hall_type, ref_text


def gold_from_hallucination_type(hall_type: str) -> Optional[bool]:
    """Map hallucination_type string -> gold_supported bool (or None to skip)."""
    if hall_type == "No Hallucination":
        return True
    if hall_type == "No Fact":
        return None  # no verifiable claim – skip
    return False  # Contradictory / Unverifiable


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
except Exception:  # pragma: no cover - optional dependency
    _nltk_sent_tokenize = None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preferring NLTK to match the paper setup."""
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
# Main iterator – yields one flat dict per annotated sentence
# ---------------------------------------------------------------------------

def iter_anah_sentences(
    ds: Any,
    model_key: str = "GPT3.5_answers_D",
    ann_key: str = "human_GPT3.5_answers_D_ann",
    max_examples: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Iterate over ANAH dataset yielding one dict per annotated sentence.

    Args:
        ds:           HuggingFace dataset object (split already selected).
        model_key:    Which model's answers to use
                      ('GPT3.5_answers_D' or 'InternLM_answers').
        ann_key:      Corresponding annotation key.
        max_examples: If > 0, stop after this many *examples* (not sentences).

    Yields dicts with keys:
        example_index, model_key, question, reference (full docs joined),
        sentence, hallucination_type, gold_supported,
        ann_reference (the specific reference fragment from annotation)
    """
    for ex_i, ex in enumerate(ds):
        if max_examples and ex_i >= max_examples:
            break

        question = (ex.get("selected_questions") or [""])[0]
        question = str(question).strip()

        # Full reference = all documents joined
        docs = ex.get("documents") or []
        full_reference = "\n\n".join(str(d) for d in docs if str(d).strip())

        answers = ex.get(model_key) or []
        anns_outer = ex.get(ann_key) or []

        for ans_idx, (answer, anns) in enumerate(zip(answers, anns_outer)):
            answer = str(answer or "").strip()
            if not answer:
                continue

            # Split the full answer once and keep it so downstream methods can
            # recreate Claimify's preceding/following-sentence context.
            sentences = split_into_sentences(answer)

            # anns is a list[str] – one annotation per sentence
            if not isinstance(anns, list):
                anns = [anns]

            for sent_i, (sentence, ann_str) in enumerate(zip(sentences, anns)):
                hall_type, ann_ref = parse_anah_annotation(ann_str)
                gold_supported = gold_from_hallucination_type(hall_type)

                yield {
                    "example_index": ex_i,
                    "answer_index": ans_idx,
                    "sentence_index": sent_i,
                    "model_key": model_key,
                    "question": question,
                    "answer_text": answer,
                    "answer_sentences": sentences,
                    # full document pool for this topic (use as reference)
                    "reference": full_reference,
                    # the specific reference fragment cited in the annotation
                    "ann_reference": ann_ref,
                    "sentence": sentence,
                    "hallucination_type": hall_type,
                    "gold_supported": gold_supported,
                }


def load_anah_sentences(
    anah_split: str = "train",
    model_key: str = "GPT3.5_answers_D",
    ann_key: str = "human_GPT3.5_answers_D_ann",
    max_examples: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Load ANAH and return (flat_sentence_rows, n_examples).

    Skips 'No Fact' sentences (gold_supported is None).
    Returns all other sentences with gold_supported set.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("opencompass/anah", split=anah_split)
    n_examples = len(ds)

    rows = []
    skipped_no_fact = 0
    for row in iter_anah_sentences(ds, model_key=model_key, ann_key=ann_key,
                                    max_examples=max_examples):
        if row["gold_supported"] is None:
            skipped_no_fact += 1
            continue
        rows.append(row)

    print(
        f"ANAH ({anah_split}): {n_examples} examples → "
        f"{len(rows)} evaluable sentences "
        f"(skipped {skipped_no_fact} 'No Fact')",
        flush=True,
    )
    return rows, n_examples

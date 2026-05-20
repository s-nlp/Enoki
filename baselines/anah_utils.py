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
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+|[A-Za-z])[\.)]?\s*$")
_INITIALISM_RE = re.compile(r"(?:\b[A-Z]\.){2,}$")
_COMMON_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|Mt|vs|etc)\.$", re.IGNORECASE
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)

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


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def _looks_like_fragment(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if _LIST_MARKER_RE.fullmatch(s):
        return True
    if len(s) <= 4 and _token_count(s) <= 1:
        return True
    if s.endswith(":"):
        return True
    return False


def _should_merge_with_next(cur: str, nxt: str) -> bool:
    cur = (cur or "").strip()
    nxt = (nxt or "").strip()
    if not cur or not nxt:
        return False
    if _looks_like_fragment(cur):
        return True
    if _INITIALISM_RE.search(cur) or _COMMON_ABBREV_RE.search(cur):
        return True
    if cur.endswith((" -", "(", "/", ",")):
        return True
    if _token_count(cur) <= 2 and nxt[:1].islower():
        return True
    return False


def _align_sentences_to_annotations(text: str, target_count: int) -> List[str]:
    """
    Split `text` into sentence-like units and try to keep them aligned with the
    annotation count used by ANAH.

    The main failure mode we want to avoid is over-splitting around list markers
    (e.g. `1.`) and abbreviations/initialisms (e.g. `U.S.`, `P. G. T.`), which
    creates fragment rows like `4.` or `The U.S.`.
    """
    sentences = split_into_sentences(text)
    if target_count <= 0 or len(sentences) <= 1 or len(sentences) == target_count:
        return sentences

    merged = list(sentences)
    changed = True
    while len(merged) > target_count and changed:
        changed = False
        i = 0
        out: List[str] = []
        while i < len(merged):
            cur = merged[i]
            nxt = merged[i + 1] if i + 1 < len(merged) else ""
            if i + 1 < len(merged) and _should_merge_with_next(cur, nxt):
                out.append(f"{cur.rstrip()} {nxt.lstrip()}".strip())
                i += 2
                changed = True
                continue
            out.append(cur)
            i += 1
        merged = out

    while len(merged) > target_count and len(merged) >= 2:
        # Greedy fallback: merge the shortest fragment into its left neighbor.
        idx = min(range(1, len(merged)), key=lambda i: len(merged[i].strip()))
        merged[idx - 1] = f"{merged[idx - 1].rstrip()} {merged[idx].lstrip()}".strip()
        del merged[idx]

    return [s.strip() for s in merged if s.strip()]


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

            # anns is a list[str] – one annotation per sentence
            if not isinstance(anns, list):
                anns = [anns]

            # Split the full answer once and keep it so downstream methods can
            # recreate Claimify's preceding/following-sentence context.
            sentences = _align_sentences_to_annotations(answer, len(anns))

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

"""
Row-level OpenIE triplet extractor for factextractor-style backends.

Output contract: one JSON object per source example:
  {
    "id": "...",
    "source_text": "...",   # exact text sent through sentence splitting; spans are relative to it
    "answer": "...",        # full original answer/response when available
    "context": "...",       # original reference/retrieval context when available
    "triplets": [[subj, pred, obj], ...],
    "spans": [[start, end], ...],
    "extract_time_s": 0.0,
    "extract_total_tokens": 0,
    "extract_flops": 0.0,
    "extract_tflops_throughput": 0.0
  }

OpenAI credentials are read from environment variables (or a .env file):
  OPENAI_API_KEY   (or LLM_PROXY_API_KEY / API_KEY)
  OPENAI_BASE_URL  (or LLM_PROXY_BASE_URL) — optional

Datasets:
  ragtruth / ragtruth-test: HF wandb/RAGTruth-processed, split=test by default.
  ragtruth-sentence: local JSONL with one sentence per row.
  psiloqa / psiloqa-test: HF s-nlp/PsiloQA, split=test.
  halluentity: HF samuelyeh/HalluEntity, split=train by default.
  factcheckbench: local factcheck-GPT-benchmark JSONL.
  bench: local JSONL with one sentence per row.
  anah: local anah.jsonl.
  mushroom: local JSON/JSONL file.
"""

from __future__ import annotations

import ast
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# -----------------------------
# System prompts
# -----------------------------

SYSTEM_PROMPT_ORIGINAL = """Following the guidelines, extract information from the given sentence.

Guidelines:
(1) Each triple should represent two entities, concepts, or events connected by an explicit relation.
For example, from the sentence "Michael Jordan, who is a former basketball player, was born in Brooklyn.",
the valid triples are:
("Michael Jordan", "is", "former basketball player")
("Michael Jordan", "was born in", "Brooklyn")

(2) If an extraction is expressed in passive voice, extract the passive relation. If the active equivalent is also directly supported by the sentence, extract it as well.
For example, from the sentence "The ball was kicked by John.", extract:
("The ball", "was kicked by", "John")
("John", "kicked", "The ball")

(3) If the sentence contains attribution of another piece of information, extract the attributed claim and the attribution relation separately.
For example, from the sentence "Conspiracy theorists say that Barack Obama was born in Kenya.", extract:
("Barack Obama", "was born in", "Kenya")
("Conspiracy theorists", "say that", "Barack Obama was born in Kenya")

(4) Do not extract incomplete clauses, i.e. triples that lack crucial information.
For example, from the sentence "He was honored by the river being named after him.", do not extract:
("He", "was honored by", "the river")
Instead, extract:
("He", "was honored by the river being named after", "him")
("the river", "being named after", "him")

(5) Do not use conjunctive phrases as a subject or object. Split them into separate triples.
For example, from the sentence "Michael Jordan and Scottie Pippen played for Chicago Bulls.", extract:
("Michael Jordan", "played for", "Chicago Bulls")
("Scottie Pippen", "played for", "Chicago Bulls")
Do not extract:
("Michael Jordan and Scottie Pippen", "played for", "Chicago Bulls")

(6) Extract only explicit information. Do not infer facts that are not stated in the sentence.
Prefer words and phrases that appear in the original sentence.

Do not miss any explicit fact.

Output format:
- Write only the extracted triples.
- Write one triple per line.
- Each triple must have exactly this format:
("subject", "relation", "object")
- Do not output explanations, comments, bullet points, JSON, Markdown, or any text outside the triples.
"""

SYSTEM_PROMPT_INCREMENTAL = """Following the guidelines, extract information from the given sentence.

Guidelines:
(1) Each triple should represent two entities, concepts, or events connected by an explicit relation.
For example, from the sentence "Michael Jordan, who is a player, was born in Brooklyn.",
the valid triples are:
("Michael Jordan", "is", "player")
("Michael Jordan", "was born in", "Brooklyn")

(2) Incremental argument spans are useful when the argument contains modifiers that cannot stand alone in the same relation without the head word.

For example, from the sentence "Michael Jordan is a former basketball player.",
extract:
("Michael Jordan", "is", "player")
("Michael Jordan", "is", "basketball player")
("Michael Jordan", "is", "former basketball player")

Do not extract:
("Michael Jordan", "is", "basketball")
("Michael Jordan", "is", "former")
("Michael Jordan", "is", "former basketball")

(3) Composite arguments may be split into meaningful components for granular verification.

Extract the full composite argument only when it adds information beyond its individual components, such as tying components together or preserving a necessary grouping between components.

Do not apply this rule to ordinary noun phrases where modifiers depend on a head word. In such cases, use incremental argument spans as described in (2).

For example, from the sentence "Michael Jordan was born in Brooklyn, USA.",
extract:
("Michael Jordan", "was born in", "Brooklyn")
("Michael Jordan", "was born in", "USA")

For example, from the sentence "She joined on March 15, 2021.",
extract:
("She", "joined on", "March")
("She", "joined on", "15")
("She", "joined on", "2021")

Do not extract:
("She", "joined on", "March 15, 2021")

(4) Incremental expansion can add modifiers either before or after the core argument span.
For example, from the sentence "April is the fourth month of the calendar year",
extract:
("April", "is", "month")
("April", "is", "month of the year")
("April", "is", "month of the calendar year")
("April", "is", "fourth month of the calendar year")

(5) If an extraction is expressed in passive voice, extract the passive relation. If the active equivalent is also directly supported by the sentence, extract it as well.
For example, from the sentence "The ball was kicked by John.", extract:
("The ball", "was kicked by", "John")
("John", "kicked", "The ball")

(6) If the sentence contains attribution of another piece of information, extract the attributed claim and the attribution relation separately.
For example, from the sentence "Conspiracy theorists say that Barack Obama was born in Kenya.", extract:
("Barack Obama", "was born in", "Kenya")
("Conspiracy theorists", "say that", "Barack Obama was born in Kenya")

(7) Do not extract incomplete clauses, i.e. triples that lack crucial information.
For example, from the sentence "He was honored by the river being named after him.", do not extract:
("He", "was honored by", "the river")
Instead, extract:
("He", "was honored by the river being named after", "him")
("the river", "being named after", "him")

(8) Do not use conjunctive phrases as a subject or object. Split them into separate triples.
For example, from the sentence "Michael Jordan and Scottie Pippen played for Chicago Bulls.", extract:
("Michael Jordan", "played for", "Chicago Bulls")
("Scottie Pippen", "played for", "Chicago Bulls")
Do not extract:
("Michael Jordan and Scottie Pippen", "played for", "Chicago Bulls")

(9) Extract only explicit information. Do not infer facts that are not stated in the sentence.
Prefer words and phrases that appear in the original sentence.

Do not miss any explicit fact.

Output format:
- Write only the extracted triples.
- Write one triple per line.
- Each triple must have exactly this format:
("subject", "relation", "object")
- Do not output explanations, comments, bullet points, JSON, Markdown, or any text outside the triples."""

USER_PROMPT_TEMPLATE = """### Input:
{sentence}

### KG:"""


# -----------------------------
# Text and sentence / argument spans
# -----------------------------

@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


DOUBLE_QUOTE_CHARS = {'"', '“', '”', '„', '‟', '«', '»'}
SINGLE_QUOTE_CHARS = {"'", '‘', '’', '‚', '‛', '`', '´'}
DASH_CHARS = {'-', '‐', '‑', '‒', '–', '—', '−'}
SPACY_LANG_ALIASES = {
    "all": "en",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "pt-br": "pt",
}


def make_flexible_literal_pattern(value: str) -> str:
    parts: List[str] = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch.isspace():
            while i < len(value) and value[i].isspace():
                i += 1
            parts.append(r"\s+")
            continue
        if ch in DOUBLE_QUOTE_CHARS:
            parts.append(r'["“”„‟«»]')
        elif ch in SINGLE_QUOTE_CHARS:
            parts.append(r"['‘’‚‛`´]")
        elif ch in DASH_CHARS:
            parts.append(r"[-‐‑‒–—−]")
        else:
            parts.append(re.escape(ch))
        i += 1
    return "".join(parts)


def text_variants(value: str) -> List[str]:
    value = value.strip().replace("\xa0", " ")
    variants = [value, normalize_space(value), value.replace("'", '"'), value.replace('"', "'")]
    result: List[str] = []
    seen = set()
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def find_text_span_in_window(
    full_text: str,
    needle: str,
    *,
    window_start: int,
    window_end: int,
) -> Optional[Tuple[int, int]]:
    window_start = max(0, window_start)
    window_end = min(len(full_text), window_end)
    if window_start >= window_end or not needle.strip():
        return None

    window = full_text[window_start:window_end]
    for cand in text_variants(needle):
        pos = window.find(cand)
        if pos != -1:
            return trim_span(full_text, window_start + pos, window_start + pos + len(cand))

    pattern = make_flexible_literal_pattern(normalize_space(needle.replace("\xa0", " ")))
    if not pattern:
        return None

    for flags in (re.DOTALL, re.DOTALL | re.IGNORECASE):
        m = re.search(pattern, window, flags=flags)
        if m:
            return trim_span(full_text, window_start + m.start(), window_start + m.end())
    return None


def find_argument_span(
    full_text: str,
    argument_text: str,
    *,
    sentence_start: int,
    sentence_end: int,
) -> Optional[Tuple[int, int]]:
    span = find_text_span_in_window(
        full_text,
        argument_text,
        window_start=sentence_start,
        window_end=sentence_end,
    )
    if span is not None:
        return span
    return find_text_span_in_window(full_text, argument_text, window_start=0, window_end=len(full_text))


COMMON_ABBREVIATIONS = {
    "a", "adj", "adm", "adv", "al", "approx", "apr", "aug", "ave", "bros",
    "capt", "cf", "co", "col", "corp", "dec", "dept", "dr", "e.g", "ed",
    "eds", "eq", "esp", "etc", "ex", "feb", "fig", "gen", "gov", "i.e",
    "inc", "jan", "jr", "jul", "jun", "ltd", "mar", "mr", "mrs", "ms",
    "mt", "no", "nov", "oct", "op", "pl", "prof", "repr", "rev", "sec",
    "sen", "sep", "sept", "sr", "st", "u.k", "u.s", "u.s.a", "vs", "vol",
}


def _previous_token_before(text: str, i: int) -> str:
    j = i
    while j > 0 and text[j - 1].isspace():
        j -= 1
    k = j
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] in {".", "-"}):
        k -= 1
    return text[k:j]


def _next_nonspace_index(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    return i


def _is_period_sentence_boundary(text: str, i: int, after_closing: int) -> bool:
    prev_ch = text[i - 1] if i > 0 else ""
    next_ch = text[i + 1] if i + 1 < len(text) else ""

    if prev_ch.isdigit() and next_ch.isdigit():
        return False

    token = _previous_token_before(text, i).strip(". ").lower()
    if token in COMMON_ABBREVIATIONS:
        return False

    raw_token = _previous_token_before(text, i).strip()
    if len(raw_token.strip(".")) == 1 and raw_token.strip(".").isupper():
        return False

    j = _next_nonspace_index(text, after_closing)
    if j < len(text) and text[j].islower():
        return False

    return True


def split_sentences_fast_with_spans(text: str) -> List[SentenceSpan]:
    if not text or not text.strip():
        return []

    closing = set('"\'”»)]}')
    spans: List[Tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch not in ".!?…":
            i += 1
            continue

        end = i + 1
        while end < n and text[end] in closing:
            end += 1

        if end < n and not text[end].isspace():
            i += 1
            continue

        if ch == "." and not _is_period_sentence_boundary(text, i, end):
            i += 1
            continue

        next_i = _next_nonspace_index(text, end)
        if next_i >= n or text[next_i].isupper() or text[next_i].isdigit() or text[next_i] in '"“‘«([':
            sent_start, sent_end = trim_span(text, start, end)
            if sent_start < sent_end:
                spans.append((sent_start, sent_end))
            start = next_i
            i = next_i
            continue

        i += 1

    sent_start, sent_end = trim_span(text, start, n)
    if sent_start < sent_end:
        spans.append((sent_start, sent_end))

    return [
        SentenceSpan(text=normalize_space(text[start:end].replace("\xa0", " ")), start=start, end=end)
        for start, end in spans
        if normalize_space(text[start:end])
    ]


def split_sentences_with_spans(text: str) -> List[SentenceSpan]:
    if not text or not text.strip():
        return []
    try:
        return split_sentences_fast_with_spans(text)
    except Exception:
        return split_sentences_regex_fallback_with_spans(text)


def split_sentences_regex_fallback_with_spans(text: str) -> List[SentenceSpan]:
    split_pat = re.compile(r'([.!?…]["\'”»)\]]*)\s+(?=[A-Z0-9"“‘«([])', flags=re.MULTILINE)
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in split_pat.finditer(text):
        end = match.end(1)
        sent_start, sent_end = trim_span(text, start, end)
        if sent_start < sent_end:
            spans.append((sent_start, sent_end))
        start = match.end()

    sent_start, sent_end = trim_span(text, start, len(text))
    if sent_start < sent_end:
        spans.append((sent_start, sent_end))

    return [
        SentenceSpan(text=normalize_space(text[start:end].replace("\xa0", " ")), start=start, end=end)
        for start, end in spans
        if normalize_space(text[start:end])
    ]


# -----------------------------
# Dataset loading
# -----------------------------

TEXT_FIELD_CANDIDATES: Dict[str, Sequence[str]] = {
    "ragtruth": ("output", "model_output", "response", "answer", "text"),
    "ragtruth_sentence": ("sentence", "text", "claim", "output", "model_output", "response", "answer"),
    "mushroom": ("model_output_text", "model_output", "output", "answer", "response", "text"),
    "psiloqa": ("llm_answer", "hypothesis", "model_output", "model_output_text", "answer", "text"),
    "halluentity": ("response", "answer", "model_output", "output", "text"),
    "factcheckbench": ("decontext", "text", "response"),
    "anah": ("sentence", "text", "answer", "response"),
    "bench": ("sentence", "text", "answer", "response"),
}

ANSWER_FIELD_CANDIDATES: Dict[str, Sequence[str]] = {
    "ragtruth": ("output", "model_output", "response", "answer", "text"),
    "ragtruth_sentence": ("answer", "response", "output", "model_output", "model_output_text", "text", "sentence"),
    "mushroom": ("model_output_text", "model_output", "output", "answer", "response", "text"),
    "psiloqa": ("llm_answer", "hypothesis", "model_output", "model_output_text", "answer", "text"),
    "halluentity": ("response", "answer", "model_output", "output", "text"),
    "factcheckbench": ("response", "answer", "output", "text"),
    "anah": ("answer", "response", "sentence", "text"),
    "bench": ("answer", "response", "sentence", "text"),
}

CONTEXT_FIELD_CANDIDATES: Dict[str, Sequence[str]] = {
    "ragtruth": (
        "context", "source", "source_text", "reference", "references", "evidence",
        "retrieved_context", "passages", "documents", "source_info", "input",
    ),
    "ragtruth_sentence": (
        "context", "source", "source_text", "reference", "references", "evidence",
        "retrieved_context", "passages", "documents", "source_info", "input",
    ),
    "mushroom": (
        "context", "model_input", "input", "source", "source_text", "reference",
        "references", "evidence", "passages", "documents",
    ),
    "psiloqa": (
        "context", "source_text", "source", "reference", "references", "evidence",
        "paragraph", "passage", "wiki_context", "document",
    ),
    "halluentity": (
        "context", "source", "source_text", "reference", "references", "evidence",
        "article", "passage", "document",
    ),
    "factcheckbench": ("prompt", "context", "source", "reference", "references", "evidence"),
    "anah": ("context", "source", "reference", "references", "evidence", "passage"),
    "bench": ("context", "source", "reference", "references", "evidence", "passage"),
}


def canonical_dataset_name(name: str) -> str:
    if name in {"ragtruth", "ragtruth-test", "ragtruth-train"}:
        return "ragtruth"
    if name in {"ragtruth-sentence", "ragtruth_sentence", "ragtruth-sentence-level", "ragtruth_sentence_level"}:
        return "ragtruth_sentence"
    if name in {"psiloqa", "psiloqa-test", "psiloqa-train"}:
        return "psiloqa"
    if name == "mushroom":
        return "mushroom"
    if name in {"halluentity", "halluentity-train", "halluentity-test"}:
        return "halluentity"
    if name in {"factcheckbench", "factcheck-gpt-benchmark", "factcheckgpt", "factcheck-gpt"}:
        return "factcheckbench"
    if name in {"anah", "anah-jsonl"}:
        return "anah"
    if name in {"bench", "bench-jsonl", "bench-sentences"}:
        return "bench"
    raise ValueError(f"Unknown dataset: {name}")


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not all(isinstance(x, dict) for x in data):
            raise ValueError("JSON list must contain objects")
        return data

    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "items"):
            value = data.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value

    raise ValueError("Expected JSONL objects, JSON list[object], or JSON object with data/rows/examples/items")


def first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def first_present_value(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        if isinstance(value, (list, tuple, dict)):
            if value:
                return value
            continue
        return value
    return None


def extract_answer_and_context(row: Dict[str, Any], dataset: str, text: str) -> Tuple[Any, Any]:
    answer = first_present_value(row, ANSWER_FIELD_CANDIDATES.get(dataset, ()))
    context = first_present_value(row, CONTEXT_FIELD_CANDIDATES.get(dataset, ()))
    return (answer if answer is not None else text), context


def make_source_id(row: Dict[str, Any], row_index: int) -> str:
    for key in ("id", "sample_id", "example_id", "uid", "uuid"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(row_index)


def short_hash(*parts: Any, n: int = 10) -> str:
    h = hashlib.sha1()
    for part in parts:
        if part is None:
            part = ""
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:n]


def sort_factcheckbench_sentence_key(item: Tuple[str, Any]) -> Tuple[int, str]:
    key, _ = item
    m = re.search(r"(\d+)$", str(key))
    if m:
        return (int(m.group(1)), str(key))
    return (10**9, str(key))


def iter_factcheckbench_decontexts(args: Any) -> Iterator[Dict[str, Any]]:
    if not args.input_path:
        raise ValueError("--input-path is required for --dataset factcheckbench")

    rows = load_json_or_jsonl(Path(args.input_path))
    target_field = getattr(args, "factcheckbench_text_field", "decontext") or "decontext"
    fallback_to_text = getattr(args, "factcheckbench_fallback_to_text", False)

    for row_index, row in enumerate(rows):
        sentences = row.get("sentences")
        if not isinstance(sentences, dict):
            continue

        prompt = row.get("prompt", "")
        for sentence_key, sentence_obj in sorted(sentences.items(), key=sort_factcheckbench_sentence_key):
            if not isinstance(sentence_obj, dict):
                continue

            value = sentence_obj.get(target_field)
            if (not isinstance(value, str) or not value.strip()) and fallback_to_text:
                value = sentence_obj.get("text")
            if not isinstance(value, str) or not value.strip():
                continue

            source_id = (
                f"factcheckbench:{row_index}:{sentence_key}:"
                f"{short_hash(prompt, sentence_key, value)}"
            )
            answer, context = extract_answer_and_context(row, "factcheckbench", value)
            yield {
                "dataset": "factcheckbench",
                "row_index": row_index,
                "source_id": source_id,
                "text": value,
                "answer": answer,
                "context": context,
            }


def make_sentence_source_id(row: Dict[str, Any], row_index: int, *, prefix: str, text: str) -> str:
    for key in ("sentence_id", "sent_id", "claim_id"):
        value = row.get(key)
        if value is not None:
            return str(value)

    parent_id = None
    for key in ("id", "sample_id", "example_id", "uid", "uuid"):
        value = row.get(key)
        if value is not None:
            parent_id = str(value)
            break

    if parent_id:
        for key in ("sentence_index", "sent_index", "sentence_idx", "sent_idx"):
            if row.get(key) is not None:
                return f"{parent_id}:sent{row[key]}"
        return parent_id

    return f"{prefix}:{row_index}:{short_hash(text)}"


def iter_ragtruth_local_rows(args: Any) -> Iterator[Dict[str, Any]]:
    if not args.input_path:
        raise ValueError("--input-path is required for local ragtruth loading")

    candidates = TEXT_FIELD_CANDIDATES["ragtruth"]
    if args.text_field:
        candidates = (args.text_field,)

    rows = load_json_or_jsonl(Path(args.input_path))
    task_type_filter = (args.ragtruth_task_type or "all").lower()

    for row_index, row in enumerate(rows):
        if task_type_filter != "all" and row.get("task_type") is not None:
            if str(row.get("task_type")) != args.ragtruth_task_type:
                continue

        text = first_present(row, candidates)
        if not text:
            continue

        answer, context = extract_answer_and_context(row, "ragtruth", text)
        yield {
            "dataset": "ragtruth",
            "row_index": row_index,
            "source_id": make_source_id(row, row_index),
            "text": text,
            "answer": answer,
            "context": context,
            "raw_row": dict(row),
        }


def iter_ragtruth_sentence_rows(args: Any) -> Iterator[Dict[str, Any]]:
    if not args.input_path:
        raise ValueError("--input-path is required for --dataset ragtruth-sentence")

    candidates = (getattr(args, "ragtruth_sentence_text_field", "sentence") or "sentence",)
    if args.text_field:
        candidates = (args.text_field,)

    rows = load_json_or_jsonl(Path(args.input_path))
    task_type_filter = (args.ragtruth_task_type or "all").lower()

    for row_index, row in enumerate(rows):
        if task_type_filter != "all" and row.get("task_type") is not None:
            if str(row.get("task_type")) != args.ragtruth_task_type:
                continue

        text = first_present(row, candidates)
        if not text:
            continue

        answer, context = extract_answer_and_context(row, "ragtruth_sentence", text)
        yield {
            "dataset": "ragtruth_sentence",
            "row_index": row_index,
            "source_id": make_sentence_source_id(row, row_index, prefix="ragtruth-sentence", text=text),
            "text": text,
            "answer": answer,
            "context": context,
            "raw_row": dict(row),
        }


def iter_bench_rows(args: Any) -> Iterator[Dict[str, Any]]:
    if not args.input_path:
        raise ValueError("--input-path is required for --dataset bench")

    candidates = (getattr(args, "bench_text_field", "sentence") or "sentence",)
    if args.text_field:
        candidates = (args.text_field,)

    rows = load_json_or_jsonl(Path(args.input_path))
    for row_index, row in enumerate(rows):
        text = first_present(row, candidates)
        if not text:
            continue
        source_id = row.get("id")
        if source_id is None:
            source_id = f"bench:{row_index}"
        answer, context = extract_answer_and_context(row, "bench", text)
        yield {
            "dataset": "bench",
            "row_index": row_index,
            "source_id": str(source_id),
            "text": text,
            "answer": answer,
            "context": context,
            "raw_row": dict(row),
        }


def iter_anah_rows(args: Any) -> Iterator[Dict[str, Any]]:
    if not args.input_path:
        raise ValueError("--input-path is required for --dataset anah")

    candidates = (getattr(args, "anah_text_field", "sentence") or "sentence",)
    if args.text_field:
        candidates = (args.text_field,)

    rows = load_json_or_jsonl(Path(args.input_path))
    for row_index, row in enumerate(rows):
        text = first_present(row, candidates)
        if not text:
            continue
        source_id = row.get("id")
        if source_id is None:
            source_id = f"anah:{row_index}"
        answer, context = extract_answer_and_context(row, "anah", text)
        yield {
            "dataset": "anah",
            "row_index": row_index,
            "source_id": str(source_id),
            "text": text,
            "answer": answer,
            "context": context,
            "raw_row": dict(row),
        }


def iter_dataset_rows(args: Any) -> Iterator[Dict[str, Any]]:
    dataset = canonical_dataset_name(args.dataset)
    candidates = TEXT_FIELD_CANDIDATES[dataset]
    if args.text_field:
        candidates = (args.text_field,)

    if dataset == "ragtruth":
        if args.input_path:
            yield from iter_ragtruth_local_rows(args)
            return

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install datasets: pip install datasets") from e

        ds = load_dataset("wandb/RAGTruth-processed", split="test", cache_dir=args.cache_dir)
        if args.ragtruth_task_type.lower() != "all":
            task_type = args.ragtruth_task_type
            ds = ds.filter(lambda x: x.get("task_type") == task_type)

        for row_index, row in enumerate(ds):
            row = dict(row)
            text = first_present(row, candidates)
            if text:
                answer, context = extract_answer_and_context(row, dataset, text)
                yield {
                    "dataset": dataset,
                    "row_index": row_index,
                    "source_id": make_source_id(row, row_index),
                    "text": text,
                    "answer": answer,
                    "context": context,
                }

    elif dataset == "ragtruth_sentence":
        yield from iter_ragtruth_sentence_rows(args)

    elif dataset == "psiloqa":
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install datasets: pip install datasets") from e

        ds = load_dataset("s-nlp/PsiloQA", split="test", cache_dir=args.cache_dir)
        if args.lang and args.lang.lower() != "all":
            lang = args.lang
            ds = ds.filter(lambda x: x.get("lang") == lang)

        for row_index, row in enumerate(ds):
            row = dict(row)
            text = first_present(row, candidates)
            if text:
                answer, context = extract_answer_and_context(row, dataset, text)
                yield {
                    "dataset": dataset,
                    "row_index": row_index,
                    "source_id": make_source_id(row, row_index),
                    "text": text,
                    "answer": answer,
                    "context": context,
                }

    elif dataset == "halluentity":
        try:
            from datasets import DatasetDict, load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install datasets: pip install datasets") from e

        split = getattr(args, "halluentity_split", "train") or "train"
        try:
            loaded = load_dataset("samuelyeh/HalluEntity", split=split, cache_dir=args.cache_dir)
        except Exception as e:
            try:
                loaded_dict = load_dataset("samuelyeh/HalluEntity", cache_dir=args.cache_dir)
            except Exception:
                raise e
            if isinstance(loaded_dict, DatasetDict):
                if split in loaded_dict:
                    loaded = loaded_dict[split]
                else:
                    first_split = next(iter(loaded_dict.keys()))
                    print(
                        f"[warn] HalluEntity split {split!r} not found; using {first_split!r}.",
                        file=sys.stderr,
                    )
                    loaded = loaded_dict[first_split]
            else:
                loaded = loaded_dict

        for row_index, row in enumerate(loaded):
            row = dict(row)
            text = first_present(row, candidates)
            if text:
                answer, context = extract_answer_and_context(row, dataset, text)
                yield {
                    "dataset": dataset,
                    "row_index": row_index,
                    "source_id": make_source_id(row, row_index),
                    "text": text,
                    "answer": answer,
                    "context": context,
                }

    elif dataset == "factcheckbench":
        yield from iter_factcheckbench_decontexts(args)

    elif dataset == "bench":
        yield from iter_bench_rows(args)

    elif dataset == "anah":
        yield from iter_anah_rows(args)

    elif dataset == "mushroom":
        if not args.input_path:
            raise ValueError("--input-path is required for --dataset mushroom")

        rows = load_json_or_jsonl(Path(args.input_path))
        for row_index, row in enumerate(rows):
            text = first_present(row, candidates)
            if text:
                answer, context = extract_answer_and_context(row, dataset, text)
                yield {
                    "dataset": dataset,
                    "row_index": row_index,
                    "source_id": make_source_id(row, row_index),
                    "text": text,
                    "answer": answer,
                    "context": context,
                }

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


# -----------------------------
# OpenAI-compatible client
# -----------------------------

_thread_local = threading.local()


@dataclass(frozen=True)
class ModelCallResult:
    raw: str
    extract_time_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    flops: float
    tflops_throughput: float


def get_usage_token_count(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    value = None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def make_client() -> Any:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install openai: pip install openai") from e

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_PROXY_API_KEY")
        or os.getenv("API_KEY")
    )
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_PROXY_BASE_URL")
    )

    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY (or LLM_PROXY_API_KEY) in your environment or .env file"
        )

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_thread_client() -> Any:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = make_client()
        _thread_local.client = client
    return client


def call_model_once(
    client: Any,
    *,
    model: str,
    sentence: str,
    system_prompt: str,
    temperature: float,
    request_timeout: float,
    model_params: float,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
) -> ModelCallResult:
    prompt = USER_PROMPT_TEMPLATE.format(sentence=sentence)
    t0 = time.perf_counter()
    # Default OFF here; pass enable_thinking=True only for an explicit thinking-mode.
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "timeout": request_timeout,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**request_kwargs)
    extract_time_s = time.perf_counter() - t0

    usage = getattr(resp, "usage", None)
    prompt_tokens = get_usage_token_count(usage, "prompt_tokens")
    completion_tokens = get_usage_token_count(usage, "completion_tokens")
    total_tokens = get_usage_token_count(usage, "total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    flops = 2.0 * float(model_params) * float(total_tokens) if total_tokens > 0 else 0.0
    tflops_throughput = flops / (extract_time_s * 1e12) if extract_time_s > 0 and flops > 0 else 0.0

    return ModelCallResult(
        raw=resp.choices[0].message.content or "",
        extract_time_s=extract_time_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        flops=flops,
        tflops_throughput=tflops_throughput,
    )


def call_model_with_retries(
    *,
    model: str,
    sentence: str,
    system_prompt: str,
    temperature: float,
    max_retries: int,
    request_timeout: float,
    model_params: float,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
) -> ModelCallResult:
    for attempt in range(max_retries + 1):
        try:
            client = get_thread_client()
            return call_model_once(
                client,
                model=model,
                sentence=sentence,
                system_prompt=system_prompt,
                temperature=temperature,
                request_timeout=request_timeout,
                model_params=model_params,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
            )
        except Exception as e:
            if attempt >= max_retries:
                raise

            sleep_s = min(90.0, (2 ** attempt) + random.random())
            print(
                f"[retry] attempt={attempt + 1}/{max_retries} "
                f"sleep={sleep_s:.1f}s err={type(e).__name__}: {e}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    raise RuntimeError("unreachable")


# -----------------------------
# Parsing model output
# -----------------------------

def clean_model_output(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:text|python|json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    text = re.sub(r"^###\s*KG\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^KG\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def is_abstain_output(cleaned: str) -> bool:
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return len(lines) == 1 and re.fullmatch(r"abstain[.!]?", lines[0], flags=re.IGNORECASE) is not None


def parse_tuple_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = line.strip()
    if not line or line.lower() == "abstain":
        return None

    try:
        value = ast.literal_eval(line)
        if isinstance(value, tuple) and len(value) in {2, 3}:
            if not all(isinstance(x, str) for x in value):
                return None
            values = [x.strip() for x in value]
            if len(value) == 2 and values[0] and values[1]:
                return (values[0], values[1], "")
            if len(value) == 3 and values[0] and values[1]:
                return (values[0], values[1], values[2])
    except Exception:
        pass

    if line.startswith("(") and line.endswith(")"):
        fields = re.findall(r"(['\"])(.*?)(?<!\\)\1", line)
        values = [value.strip() for _, value in fields]
        if len(values) == 2 and values[0] and values[1]:
            return (values[0], values[1], "")
        if len(values) >= 3 and values[0] and values[1]:
            return (values[0], values[1], values[2])

    return None


def parse_model_output(raw: str) -> Tuple[List[Tuple[str, str, str]], bool, Optional[str]]:
    cleaned = clean_model_output(raw)
    if not cleaned:
        return [], False, "empty_output"

    if is_abstain_output(cleaned):
        return [], True, None

    triples: List[Tuple[str, str, str]] = []
    unparsed_lines: List[str] = []

    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("### input") or line.lower().startswith("### kg"):
            continue
        parsed = parse_tuple_line(line)
        if parsed is None:
            unparsed_lines.append(line)
        else:
            triples.append(parsed)

    triples = dedupe_tuple_triples(triples)
    if triples:
        parse_error = None if not unparsed_lines else f"ignored_{len(unparsed_lines)}_unparsed_lines"
        return triples, False, parse_error

    return [], False, "no_parsable_triples"


def dedupe_tuple_triples(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    seen = set()
    result: List[Tuple[str, str, str]] = []
    for s, p, o in triples:
        key = (normalize_space(s).lower(), normalize_space(p).lower(), normalize_space(o).lower())
        if key in seen:
            continue
        seen.add(key)
        result.append((s, p, o))
    return result


# -----------------------------
# Row processing and output
# -----------------------------

def triple_key(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    return (
        normalize_space(triple[0]).lower(),
        normalize_space(triple[1]).lower(),
        normalize_space(triple[2]).lower(),
    )


def process_row(row: Dict[str, Any], args: Any) -> Dict[str, Any]:
    precomputed_sentences = row.get("sentences")
    if precomputed_sentences is not None:
        sentences = precomputed_sentences
    else:
        sentences = split_sentences_with_spans(row["text"])

    triplets: List[List[str]] = []
    spans: List[List[int]] = []
    seen = set()

    extract_time_s = 0.0
    extract_prompt_tokens = 0
    extract_completion_tokens = 0
    extract_total_tokens = 0
    extract_flops = 0.0
    sentence_metrics: List[Dict[str, Any]] = []

    for sentence_index, sentence in enumerate(sentences):
        try:
            model_result = call_model_with_retries(
                model=args.model,
                sentence=sentence.text,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_retries=args.max_retries,
                request_timeout=args.request_timeout,
                model_params=args.model_params,
                enable_thinking=getattr(args, "enable_thinking", False),
                max_tokens=getattr(args, "max_tokens", None),
            )

            extract_time_s += model_result.extract_time_s
            extract_prompt_tokens += model_result.prompt_tokens
            extract_completion_tokens += model_result.completion_tokens
            extract_total_tokens += model_result.total_tokens
            extract_flops += model_result.flops

            if args.save_sentence_metrics:
                sentence_metrics.append(
                    {
                        "sentence_index": sentence_index,
                        "sentence_start": sentence.start,
                        "sentence_end": sentence.end,
                        "extract_time_s": model_result.extract_time_s,
                        "extract_prompt_tokens": model_result.prompt_tokens,
                        "extract_completion_tokens": model_result.completion_tokens,
                        "extract_total_tokens": model_result.total_tokens,
                        "extract_flops": model_result.flops,
                        "extract_tflops_throughput": model_result.tflops_throughput,
                    }
                )

            parsed_triples, abstained, parse_error = parse_model_output(model_result.raw)
            if args.verbose and (parse_error or abstained):
                print(
                    f"[parse] id={row['source_id']} sent={sentence_index} "
                    f"triples={len(parsed_triples)} abstained={abstained} parse_error={parse_error} "
                    f"extract_time_s={model_result.extract_time_s:.4f} tokens={model_result.total_tokens}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"[error] id={row['source_id']} sent={sentence_index} "
                f"span=({sentence.start},{sentence.end}) err={type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue

        for triple in parsed_triples:
            span_target = triple[2] if triple[2].strip() else triple[1]
            arg_span = find_argument_span(
                row["text"],
                span_target,
                sentence_start=sentence.start,
                sentence_end=sentence.end,
            )
            if arg_span is None:
                if args.verbose:
                    print(
                        f"[span-miss] id={row['source_id']} sent={sentence_index} "
                        f"target={span_target!r} triple={triple!r}",
                        file=sys.stderr,
                    )
                if not args.keep_unlocalized:
                    continue
                arg_span = (-1, -1)

            key = triple_key(triple)
            if key in seen:
                continue
            seen.add(key)

            triplets.append([triple[0], triple[1], triple[2]])
            spans.append([arg_span[0], arg_span[1]])

    metrics = {
        "extract_sentence_count": len(sentences),
        "extract_time_s": extract_time_s,
        "extract_prompt_tokens": extract_prompt_tokens,
        "extract_completion_tokens": extract_completion_tokens,
        "extract_total_tokens": extract_total_tokens,
        "extract_flops": extract_flops,
        "extract_tflops_throughput": (
            extract_flops / (extract_time_s * 1e12)
            if extract_time_s > 0 and extract_flops > 0
            else 0.0
        ),
    }
    if args.save_sentence_metrics:
        metrics["extract_sentence_metrics"] = sentence_metrics

    return build_output_row(row, triplets, spans, metrics=metrics)


def build_output_row(
    row: Dict[str, Any],
    triplets: List[List[str]],
    spans: List[List[int]],
    *,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = {
        "id": row["source_id"],
        "source_text": row.get("text", ""),
        "answer": row.get("answer", row.get("text", "")),
        "context": row.get("context"),
        "triplets": triplets,
        "spans": spans,
    }
    if metrics:
        out.update(metrics)

    raw_row = row.get("raw_row")
    if isinstance(raw_row, dict):
        merged = dict(raw_row)
        merged.update(out)
        return merged

    return out


def load_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()

    done = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                row_id = obj.get("id")
                if row_id is not None:
                    done.add(str(row_id))
            except Exception:
                print(f"[warn] Cannot parse existing output line {line_no}; ignoring it.", file=sys.stderr)
    return done


def append_jsonl(output_path: Path, obj: Dict[str, Any], fsync: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def iter_rows_for_processing(args: Any, done_ids: set) -> Iterator[Dict[str, Any]]:
    rows_seen = 0
    skipped_existing = 0

    for row in iter_dataset_rows(args):
        if args.limit is not None and rows_seen >= args.limit:
            break
        rows_seen += 1

        source_id = str(row["source_id"])
        if source_id in done_ids:
            skipped_existing += 1
            continue

        yield row

    print(f"[prepared] rows_seen={rows_seen}, skipped_existing_rows={skipped_existing}", file=sys.stderr)


def run_sequential(args: Any, output_path: Path, done_ids: set) -> None:
    rows_written = 0
    for row in iter_rows_for_processing(args, done_ids):
        out = process_row(row, args)
        append_jsonl(output_path, out, fsync=not args.no_fsync)
        done_ids.add(str(out["id"]))
        rows_written += 1
        print(
            f"[ok] row={row['row_index']} id={out['id']} triplets={len(out['triplets'])} "
            f"extract_time_s={out.get('extract_time_s', 0.0):.4f} "
            f"tokens={out.get('extract_total_tokens', 0)}",
            flush=True,
        )

    print(f"Done. rows_written={rows_written}, output={output_path}")


def run_parallel(args: Any, output_path: Path, done_ids: set) -> None:
    workers = max(1, args.workers)
    max_in_flight = args.max_in_flight or workers * 4
    if max_in_flight < workers:
        max_in_flight = workers

    rows_written = 0
    row_iter = iter_rows_for_processing(args, done_ids)
    futures: Dict[cf.Future, Dict[str, Any]] = {}

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        def submit_until_full() -> None:
            while len(futures) < max_in_flight:
                try:
                    row = next(row_iter)
                except StopIteration:
                    return
                fut = executor.submit(process_row, row, args)
                futures[fut] = row

        submit_until_full()
        while futures:
            done, _ = cf.wait(futures.keys(), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                row = futures.pop(fut)
                try:
                    out = fut.result()
                except Exception as e:
                    print(
                        f"[error] row={row['row_index']} id={row['source_id']} "
                        f"err={type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    out = build_output_row(row, [], [])

                append_jsonl(output_path, out, fsync=not args.no_fsync)
                done_ids.add(str(out["id"]))
                rows_written += 1
                print(
                    f"[ok] row={row['row_index']} id={out['id']} triplets={len(out['triplets'])} "
                    f"extract_time_s={out.get('extract_time_s', 0.0):.4f} "
                    f"tokens={out.get('extract_total_tokens', 0)}",
                    flush=True,
                )

            submit_until_full()

    print(f"Done. rows_written={rows_written}, output={output_path}")


# -----------------------------
# Public entry point for CLI
# -----------------------------

class _Args:
    """Simple namespace for passing parameters through the extraction pipeline."""
    pass


def run_extraction(
    *,
    dataset: str,
    output: str,
    input_path: Optional[str] = None,
    model: str = "gpt-oss-120b",
    prompt: str = "incremental",
    temperature: float = 0.0,
    workers: int = 1,
    max_in_flight: Optional[int] = None,
    lang: str = "en",
    text_field: Optional[str] = None,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    max_retries: int = 8,
    request_timeout: float = 120.0,
    model_params: float = 120e9,
    save_sentence_metrics: bool = False,
    no_resume: bool = False,
    no_fsync: bool = False,
    ragtruth_task_type: str = "QA",
    ragtruth_sentence_text_field: str = "sentence",
    halluentity_split: str = "train",
    factcheckbench_text_field: str = "decontext",
    factcheckbench_fallback_to_text: bool = False,
    bench_text_field: str = "sentence",
    anah_text_field: str = "sentence",
    verbose: bool = False,
    keep_unlocalized: bool = False,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
) -> None:
    """Run triplet extraction and write results to *output* JSONL."""
    if prompt == "incremental":
        system_prompt = SYSTEM_PROMPT_INCREMENTAL
    elif prompt == "original":
        system_prompt = SYSTEM_PROMPT_ORIGINAL
    else:
        raise ValueError(f"Unknown prompt variant: {prompt!r}; choose 'incremental' or 'original'")

    args = _Args()
    args.dataset = dataset
    args.input_path = input_path
    args.model = model
    args.system_prompt = system_prompt
    args.temperature = temperature
    args.workers = workers
    args.max_in_flight = max_in_flight
    args.lang = lang

    args.text_field = text_field
    args.cache_dir = cache_dir
    args.limit = limit
    args.max_retries = max_retries
    args.request_timeout = request_timeout
    args.model_params = model_params
    args.save_sentence_metrics = save_sentence_metrics
    args.no_resume = no_resume
    args.no_fsync = no_fsync
    args.ragtruth_task_type = ragtruth_task_type
    args.ragtruth_sentence_text_field = ragtruth_sentence_text_field
    args.halluentity_split = halluentity_split
    args.factcheckbench_text_field = factcheckbench_text_field
    args.factcheckbench_fallback_to_text = factcheckbench_fallback_to_text
    args.bench_text_field = bench_text_field
    args.anah_text_field = anah_text_field
    args.verbose = verbose
    args.keep_unlocalized = keep_unlocalized
    args.enable_thinking = enable_thinking
    args.max_tokens = max_tokens

    if workers < 1:
        raise ValueError("workers must be >= 1")

    output_path = Path(output)
    done_ids = set() if no_resume else load_done_ids(output_path)

    if workers == 1:
        run_sequential(args, output_path, done_ids)
    else:
        run_parallel(args, output_path, done_ids)


# -----------------------------
# Standalone CLI entry point
# -----------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset",
        required=True,
        choices=[
            "ragtruth", "ragtruth-test", "ragtruth-train",
            "ragtruth-sentence", "ragtruth_sentence", "ragtruth-sentence-level", "ragtruth_sentence_level",
            "mushroom",
            "psiloqa", "psiloqa-test", "psiloqa-train",
            "halluentity", "halluentity-train", "halluentity-test",
            "factcheckbench", "factcheck-gpt-benchmark", "factcheckgpt", "factcheck-gpt",
            "bench", "bench-jsonl", "bench-sentences",
            "anah", "anah-jsonl",
        ],
    )
    p.add_argument("--input-path", default=None)
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument(
        "--prompt",
        default="incremental",
        choices=["incremental", "original"],
        help="Prompt variant: 'incremental' (cycleoie_with_incrementality, default) or 'original' (cycleoie_original)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-in-flight", type=int, default=None)
    p.add_argument("--lang", default="en")
    p.add_argument("--text-field", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--model-params", type=float, default=120e9)
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Let hybrid-thinking models (e.g. Qwen3.6-35B-A3B) emit chain-of-thought "
             "before the KG output. OFF by default: for the latency rebuttal this avoids "
             "an uncontrolled reasoning-token confound and matches clean_model_output's "
             "parsing, which does not strip a 'Thinking Process:' preamble.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=None,
        help="Cap generated tokens per extraction call (recommended when thinking is "
             "enabled, or as a safety net regardless — one call per sentence adds up fast).",
    )
    p.add_argument("--save-sentence-metrics", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-fsync", action="store_true")
    p.add_argument("--ragtruth-task-type", default="QA")
    p.add_argument("--ragtruth-sentence-text-field", default="sentence")
    p.add_argument("--halluentity-split", default="train")
    p.add_argument("--factcheckbench-text-field", default="decontext", choices=["decontext", "text", "revised_decontext"])
    p.add_argument("--factcheckbench-fallback-to-text", action="store_true")
    p.add_argument("--bench-text-field", default="sentence")
    p.add_argument("--anah-text-field", default="sentence")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--keep-unlocalized", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_extraction(
        dataset=args.dataset,
        output=args.output,
        input_path=args.input_path,
        model=args.model,
        prompt=args.prompt,
        temperature=args.temperature,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        lang=args.lang,

        text_field=args.text_field,
        cache_dir=args.cache_dir,
        limit=args.limit,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        model_params=args.model_params,
        save_sentence_metrics=args.save_sentence_metrics,
        no_resume=args.no_resume,
        no_fsync=args.no_fsync,
        ragtruth_task_type=args.ragtruth_task_type,
        ragtruth_sentence_text_field=args.ragtruth_sentence_text_field,
        halluentity_split=args.halluentity_split,
        factcheckbench_text_field=args.factcheckbench_text_field,
        factcheckbench_fallback_to_text=args.factcheckbench_fallback_to_text,
        bench_text_field=args.bench_text_field,
        anah_text_field=args.anah_text_field,
        verbose=args.verbose,
        keep_unlocalized=args.keep_unlocalized,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

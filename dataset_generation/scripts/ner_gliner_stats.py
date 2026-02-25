#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER stats collection using NuNER_Zero (GLiNER) with batch inference.

Reads an input JSONL with questions/contexts (from qa_gen/filtered outputs),
runs NER on context, question, and optional answer, and writes flattened
prediction records for analysis.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import torch
from gliner import GLiNER


GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "date",
    "year",
    "time",
    "work",
    "title",
    "concept",
    "event",
    "product",
    "quantity",
    "money",
    "percent",
    "number",
    "language",
    "nationality",
    "name",
]

GLINER_LABELS = [l.lower() for l in GLINER_LABELS]


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def merge_entities(entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    if not entities:
        return []
    merged: List[Dict[str, Any]] = []
    current = entities[0]
    for nxt in entities[1:]:
        if nxt["label"] == current["label"] and nxt["start"] <= current["end"] + 1:
            current["text"] = text[current["start"] : nxt["end"]].strip()
            current["end"] = nxt["end"]
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def chunk_text(text: str, chunk_chars: int, stride_chars: int) -> List[Tuple[int, str]]:
    if chunk_chars <= 0:
        return [(0, text)]
    if stride_chars <= 0:
        stride_chars = chunk_chars
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append((start, text[start:end]))
        if end == n:
            break
        start += stride_chars
    return chunks


def infer_batch_with_sliding(
    model: GLiNER,
    texts: List[str],
    labels: List[str],
    *,
    min_score: float,
    chunk_chars: int,
    stride_chars: int,
    batch_size: int = 16,
) -> List[List[Dict[str, Any]]]:
    """Batch inference with sliding window chunking."""
    if not texts:
        return []

    # Build chunks for all texts
    # chunk_info: list of (text_idx, offset, chunk_text)
    chunk_info: List[Tuple[int, int, str]] = []
    for text_idx, text in enumerate(texts):
        if not text:
            continue
        for offset, chunk in chunk_text(text, chunk_chars, stride_chars):
            chunk_info.append((text_idx, offset, chunk))

    if not chunk_info:
        return [[] for _ in texts]

    # Process chunks in batches
    all_chunk_ents: List[List[Dict[str, Any]]] = []
    for i in tqdm(range(0, len(chunk_info), batch_size)):
        batch = chunk_info[i : i + batch_size]
        batch_texts = [c[2] for c in batch]
        batch_results = model.batch_predict_entities(
            batch_texts, labels, threshold=min_score
        )
        all_chunk_ents.extend(batch_results)

    # Reassemble entities per original text
    text_entities: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(texts))}

    for (text_idx, offset, _chunk), ents in zip(chunk_info, all_chunk_ents):
        text = texts[text_idx]
        for e in ents:
            if e.get("score", 0) < min_score:
                continue
            text_entities[text_idx].append(
                {
                    "start": e["start"] + offset,
                    "end": e["end"] + offset,
                    "text": text[e["start"] + offset : e["end"] + offset],
                    "label": e["label"],
                    "score": e.get("score"),
                }
            )

    # Sort and merge per text
    results = []
    for text_idx, text in enumerate(texts):
        ents = text_entities[text_idx]
        ents = sorted(ents, key=lambda x: (x["start"], x["end"]))
        results.append(merge_entities(ents, text))

    return results


def pick_questions(rec: Dict[str, Any]) -> List[str]:
    qs = rec.get("questions")
    if isinstance(qs, list):
        return [str(q) for q in qs if isinstance(q, str) and q.strip()]
    for k in ("questions_final", "questions_parsed"):
        qs2 = rec.get(k)
        if isinstance(qs2, list):
            return [str(q) for q in qs2 if isinstance(q, str) and q.strip()]
    q = rec.get("question") or rec.get("prompt")
    return [q] if isinstance(q, str) and q.strip() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL (questions/contexts).")
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSONL base path (will add .contexts/.questions/.answers suffixes if split).",
    )
    ap.add_argument(
        "--max-records", type=int, default=0, help="Max records to process (0 = all)."
    )
    ap.add_argument("--min-score", type=float, default=0.0, help="Min entity score.")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument(
        "--chunk-chars",
        type=int,
        default=0,
        help="Max chars per chunk (0 = no chunking).",
    )
    ap.add_argument(
        "--chunk-stride",
        type=int,
        default=0,
        help="Stride in chars for chunking (0 = chunk size).",
    )
    ap.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for inference."
    )
    ap.add_argument(
        "--extract",
        type=str,
        default="contexts+questions+answers",
        help="List of what to extract: contexts,questions,answers. Use + or , as separator (default: all)",
    )
    ap.add_argument(
        "--split-output",
        action="store_true",
        help="Save separate files for contexts/questions/answers.",
    )
    args = ap.parse_args()

    # Parse extract flag (supports both , and + as separators for sbatch compatibility)
    import re
    extract_set = set(x.strip().lower() for x in re.split(r'[,+]', args.extract))
    do_contexts = "contexts" in extract_set
    do_questions = "questions" in extract_set
    do_answers = "answers" in extract_set
    print(f"Extract flag: '{args.extract}' -> {extract_set}", file=sys.stderr)
    print(f"do_contexts={do_contexts}, do_questions={do_questions}, do_answers={do_answers}", file=sys.stderr)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model numind/NuNerZero on {device} ...", file=sys.stderr)
    model = GLiNER.from_pretrained("numind/NuNerZero", device=device)

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all records
    print(f"Loading records from {in_path} ...", file=sys.stderr)
    records = list(read_jsonl(in_path))
    if args.max_records > 0:
        records = records[: args.max_records]
    print(f"Loaded {len(records)} records", file=sys.stderr)

    # Collect all texts
    contexts: List[str] = []
    questions: List[str] = []
    answers: List[str] = []
    rec_question_map: List[Tuple[int, int]] = []  # (rec_idx, q_idx within that rec)

    for rec_idx, rec in enumerate(records):
        ctx = (
            rec.get("context_text")
            or rec.get("context")
            or (rec.get("source") or {}).get("context_text")
            or ""
        )
        contexts.append(ctx)

        qs = pick_questions(rec)
        ans = (
            rec.get("answer")
            or rec.get("response")
            or (rec.get("source") or {}).get("answer")
            or ""
        )

        for q_idx, q in enumerate(qs):
            questions.append(q)
            answers.append(ans)
            rec_question_map.append((rec_idx, q_idx))

    print(
        f"Texts: {len(contexts)} contexts, {len(questions)} questions, {len(answers)} answers",
        file=sys.stderr,
    )

    # Batch NER (conditional)
    ctx_ents_all: List[List[Dict[str, Any]]] = []
    q_ents_all: List[List[Dict[str, Any]]] = []
    ans_ents_all: List[List[Dict[str, Any]]] = []

    if do_contexts:
        print("Running NER on contexts...", file=sys.stderr)
        ctx_ents_all = infer_batch_with_sliding(
            model,
            contexts,
            GLINER_LABELS,
            min_score=args.min_score,
            chunk_chars=args.chunk_chars,
            stride_chars=args.chunk_stride,
            batch_size=args.batch_size,
        )
    else:
        ctx_ents_all = [[] for _ in contexts]

    if do_questions:
        print("Running NER on questions...", file=sys.stderr)
        q_ents_all = infer_batch_with_sliding(
            model,
            questions,
            GLINER_LABELS,
            min_score=args.min_score,
            chunk_chars=args.chunk_chars,
            stride_chars=args.chunk_stride,
            batch_size=args.batch_size,
        )
    else:
        q_ents_all = [[] for _ in questions]

    if do_answers:
        print("Running NER on answers...", file=sys.stderr)
        ans_ents_all = infer_batch_with_sliding(
            model,
            answers,
            GLINER_LABELS,
            min_score=args.min_score,
            chunk_chars=args.chunk_chars,
            stride_chars=args.chunk_stride,
            batch_size=args.batch_size,
        )
    else:
        ans_ents_all = [[] for _ in answers]

    # Write results
    print("Writing results...", file=sys.stderr)
    label_counter: Counter = Counter()

    # Prepare output paths
    out_base = out_path.stem
    out_dir = out_path.parent
    out_suffix = out_path.suffix or ".jsonl"

    if args.split_output:
        # Split into separate files
        ctx_path = out_dir / f"{out_base}.contexts{out_suffix}" if do_contexts else None
        q_path = out_dir / f"{out_base}.questions{out_suffix}" if do_questions else None
        ans_path = out_dir / f"{out_base}.answers{out_suffix}" if do_answers else None

        # Write contexts (one per record)
        if ctx_path and do_contexts:
            with ctx_path.open("w") as f_ctx:
                for rec_idx, rec in enumerate(records):
                    ctx_ents = ctx_ents_all[rec_idx]
                    for e in ctx_ents:
                        label_counter[e["label"]] += 1
                    out_rec = {
                        "context_id": rec.get("context_id"),
                        "title": rec.get("title"),
                        "context": contexts[rec_idx],
                        "context_entities": ctx_ents,
                        "ner_model": "numind/NuNerZero",
                        "labels": GLINER_LABELS,
                    }
                    f_ctx.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            print(f"  Wrote {len(records)} context rows to {ctx_path}", file=sys.stderr)

        # Write questions (one per question)
        if q_path and do_questions:
            with q_path.open("w") as f_q:
                for i, (rec_idx, _) in enumerate(rec_question_map):
                    rec = records[rec_idx]
                    q_ents = q_ents_all[i]
                    for e in q_ents:
                        label_counter[e["label"]] += 1
                    out_rec = {
                        "context_id": rec.get("context_id"),
                        "title": rec.get("title"),
                        "question": questions[i],
                        "question_entities": q_ents,
                        "ner_model": "numind/NuNerZero",
                        "labels": GLINER_LABELS,
                    }
                    f_q.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            print(
                f"  Wrote {len(rec_question_map)} question rows to {q_path}",
                file=sys.stderr,
            )

        # Write answers (one per question/answer pair)
        if ans_path and do_answers:
            with ans_path.open("w") as f_ans:
                for i, (rec_idx, _) in enumerate(rec_question_map):
                    rec = records[rec_idx]
                    ans_ents = ans_ents_all[i]
                    for e in ans_ents:
                        label_counter[e["label"]] += 1
                    out_rec = {
                        "context_id": rec.get("context_id"),
                        "title": rec.get("title"),
                        "question": questions[i],
                        "answer": answers[i],
                        "answer_entities": ans_ents,
                        "ner_model": "numind/NuNerZero",
                        "labels": GLINER_LABELS,
                    }
                    f_ans.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            print(
                f"  Wrote {len(rec_question_map)} answer rows to {ans_path}",
                file=sys.stderr,
            )

    else:
        # Combined output (original behavior)
        with out_path.open("w") as f_out:
            for i, (rec_idx, _) in enumerate(rec_question_map):
                rec = records[rec_idx]
                ctx_ents = ctx_ents_all[rec_idx] if do_contexts else []
                q_ents = q_ents_all[i] if do_questions else []
                ans_ents = ans_ents_all[i] if do_answers else []

                for e in ctx_ents + q_ents + ans_ents:
                    label_counter[e["label"]] += 1

                out_rec = {
                    "context_id": rec.get("context_id"),
                    "title": rec.get("title"),
                    "question": questions[i],
                    "answer": answers[i],
                    "context_entities": ctx_ents,
                    "question_entities": q_ents,
                    "answer_entities": ans_ents,
                    "ner_model": "numind/NuNerZero",
                    "labels": GLINER_LABELS,
                }
                f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        print(
            f"Done: wrote {len(rec_question_map)} rows to {out_path}", file=sys.stderr
        )

    print(f"Label counts: {dict(label_counter)}", file=sys.stderr)


if __name__ == "__main__":
    main()

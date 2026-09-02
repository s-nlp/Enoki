#!/usr/bin/env python3
"""
Dataset loaders for hallucination detection evaluation.

Supports:
- Factcheck-Bench
- HalluEntity
- PsiloQA (span-level)
- Mushroom (span-level)
- RAGTruth (span-level)
- ANAH (sentence-level inside response; opencompass/anah on HF)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple




def load_factcheckbench_dataset(data_dir: str = ".") -> Tuple[List[Dict], str]:
    """
    Load Factcheck-Bench dataset.

    Args:
        data_dir: Base directory containing factcheckbench/ subdirectory

    Returns:
        Tuple of (samples, context_field_name)
    """
    import json

    data_path = Path(data_dir) / "factcheckbench" / "factcheck-GPT-benchmark.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Factcheck-Bench not found at {data_path}")

    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    samples = []
    for item_idx, item in enumerate(data):
        if item.get('response_factuality') == 'NA':
            continue

        for sent_key, sent_data in item.get('sentences', {}).items():
            if sent_data.get('sentence_factuality_label') == 'NA':
                continue

            # Combine human and auto evidence as context
            context_parts = sent_data.get('human_evidence', [])
            for auto_list in sent_data.get('auto_evidence', []):
                context_parts.extend(auto_list)
            context = "\n".join(context_parts)

            samples.append({
                'id': f'factcheckbench:{item_idx}:{sent_key}',
                'text': sent_data.get('decontext', ''),
                'context': context,
                'label': sent_data['sentence_factuality_label'],  # 1 = factual, 0 = not factual
            })

    return samples, 'sentence'


def load_halluentity_dataset(data_dir: str = ".", use_wikipedia_contexts: bool = False) -> Tuple[List[Dict], str]:
    """
    Load HalluEntity dataset.

    Args:
        data_dir: Base directory containing halluentity/ subdirectory
        use_wikipedia_contexts: If True, require contexts CSV file. If False, use empty context (for testing only).

    Returns:
        Tuple of (samples, context_field_name)

    Note:
        HalluEntity requires Wikipedia contexts for proper evaluation.
        Without contexts, NLI scores will be meaningless.
    """
    from datasets import load_dataset

    ds = load_dataset("samuelyeh/HalluEntity", split='train')

    # Load Wikipedia contexts
    contexts_path = Path(data_dir) / "halluentity" / "halluentity_contexts.csv"
    contexts_dict = {}

    if contexts_path.exists():
        import pandas as pd
        contexts_df = pd.read_csv(contexts_path)
        # Use 'text' column for contexts
        for idx in range(len(contexts_df)):
            contexts_dict[idx] = contexts_df.iloc[idx]['text'] if 'text' in contexts_df.columns else ''
        print(f"Loaded {len(contexts_dict)} Wikipedia contexts from {contexts_path}")
    elif use_wikipedia_contexts:
        raise FileNotFoundError(
            f"Wikipedia contexts required but not found at {contexts_path}\n"
            f"HalluEntity evaluation requires Wikipedia articles as ground truth."
        )
    else:
        print(f"WARNING: No Wikipedia contexts found at {contexts_path}")
        print(f"         Using empty contexts (for testing only).")
        print(f"         NLI scores will be meaningless without real contexts.")

    samples = []
    for idx, row in enumerate(ds):
        # Get context (Wikipedia article or empty)
        context = contexts_dict.get(idx, '')

        samples.append({
            'id': str(idx),
            'text': row['response'],
            'context': context,
            'entities': row['entity'],
            'entity_labels': row['entity_label'],  # True = factual, False = hallucination
            'prompt': row['prompt'],
        })

    return samples, 'entity'


# ============================================================================
# SPAN-LEVEL DATASETS
# ============================================================================

def load_psiloqa_dataset(split: str = 'test') -> List[Dict]:
    """
    Load PsiloQA dataset (span-level).

    Args:
        split: HuggingFace split name ('test' or 'train')

    Returns:
        List of dicts with 'question', 'context', 'answer', 'labels' (spans)
    """
    from datasets import load_dataset
    ds = load_dataset("s-nlp/PsiloQA", split=split)
    ds = ds.filter(lambda x: x['lang'] == 'en')

    data = []
    for row in ds:
        data.append({
            "id": row["id"],
            "question": row.get("question", ""),
            "context": row["wiki_passage"],
            "answer": row["llm_answer"],
            "labels": row["labels"]
        })
    return data


def load_mushroom_dataset(data_dir: str = ".") -> List[Dict]:
    """
    Load Mushroom dataset (span-level).

    Args:
        data_dir: Base directory containing mushroom/ subdirectory

    Returns:
        List of dicts with 'question', 'context', 'answer', 'labels' (spans)
    """
    ds_path = Path(data_dir) / "mushroom" / "mushroom.en-tst.v1.extra.with_context.jsonl"
    if not ds_path.exists():
        raise FileNotFoundError(f"Mushroom dataset not found at {ds_path}")

    data = []
    with open(ds_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            data.append({
                "id": row["id"],
                "question": row.get("model_input", ""),
                "context": row["context"],
                "answer": row["model_output_text"],
                "labels": row["hard_labels"]
            })
    return data


def load_ragtruth_dataset(split: str = 'test') -> List[Dict]:
    """
    Load RAGTruth dataset (span-level).

    Args:
        split: HuggingFace split name ('test' or 'train')

    Returns:
        List of dicts with 'question', 'context', 'answer', 'labels' (spans)
    """
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split=split)
    ds = ds.filter(lambda x: x['task_type'] == 'QA')

    data = []
    for row in ds:
        labels = json.loads(row['hallucination_labels'])
        data.append({
            "id": str(row["id"]),
            # wandb/RAGTruth-processed stores the question/instruction under
            # 'query' (see baselines/ragtruth_utils.py) — surface it as
            # 'question' so downstream prompts (e.g. singlestep baseline)
            # actually get it instead of silently seeing "".
            "question": row.get("query", ""),
            "context": row["context"],
            "answer": row["output"],
            "labels": [[s['start'], s['end']] for s in labels]
        })
    return data


def _format_ragtruth_context(source_info, task_type: str) -> str:
    """Format RAGTruth source_info into a plain-text context string."""
    if task_type == "Summary":
        return source_info if isinstance(source_info, str) else str(source_info)
    if task_type == "QA":
        question = source_info.get("question", "")
        passages = source_info.get("passages", "")
        return f"Question: {question}\n\n{passages}"
    if task_type == "Data2txt":
        import json as _json
        return _json.dumps(source_info, ensure_ascii=False)
    return str(source_info)


def load_ragtruth_response_dataset(
    ragtruth_dir: str = "RAGTruth",
    task_types: Tuple[str, ...] = ("QA",),
    filter_quality: bool = True,
) -> List[Dict]:
    """Load RAGTruth test responses as response-level samples from the local clone.

    Each sample is one full LLM response; label is whether the response contains
    any hallucination span. This matches the RAGTruth baseline evaluation metric.

    Label convention matches FELM: ``label=1`` = factual, ``label=0`` = hallucinated.
    The evaluator inverts internally (``gold = 1 - label``).

    ``id`` is the RAGTruth response id, so ``pre_extracted_refchecker`` facts
    (900 QA responses in ``ragtruth-refchecker-v1-triples.jsonl``) are looked
    up directly by id.

    Args:
        ragtruth_dir: Path to the cloned RAGTruth repository (contains dataset/).
        task_types:   Subset of task types to include.
        filter_quality: If True, drop ``incorrect_refusal`` responses.
    """
    ragtruth_path = Path(ragtruth_dir)
    response_path = ragtruth_path / "dataset" / "response.jsonl"
    source_path   = ragtruth_path / "dataset" / "source_info.jsonl"

    if not response_path.exists():
        raise FileNotFoundError(f"RAGTruth response file not found: {response_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"RAGTruth source_info file not found: {source_path}")

    sources: Dict[str, Dict] = {}
    with open(source_path, encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            sources[s["source_id"]] = s

    samples: List[Dict] = []
    with open(response_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] != "test":
                continue
            if filter_quality and r.get("quality") == "incorrect_refusal":
                continue

            src = sources.get(r["source_id"], {})
            if src.get("task_type") not in task_types:
                continue

            context = _format_ragtruth_context(src.get("source_info", ""), src["task_type"])
            is_halu = any(
                not lbl.get("implicit_true", False)
                for lbl in r.get("labels", [])
            )

            samples.append({
                "id": r["id"],
                "text": r["response"],
                "context": context,
                "label": 0 if is_halu else 1,   # FELM convention: 1=factual
                "task_type": src["task_type"],
            })

    return samples


def load_much_dataset(data_dir: str = ".") -> List[Dict]:
    """
    Load MUCH dataset (span-level).

    Args:
        data_dir: Base directory containing much/ subdirectory

    Returns:
        List of dicts with 'context', 'answer', 'labels' (spans)
    """
    ds_path = Path(data_dir) / "much" / "much_span_level.jsonl"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"MUCH dataset not found at {ds_path}\n"
            f"Run scripts/convert_much_to_span_level.py to create it."
        )

    data = []
    with open(ds_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            data.append({
                "context": row["context"],
                "answer": row["answer"],
                "labels": row["labels"]
            })
    return data


# ============================================================================
# ANAH — opencompass/anah on HuggingFace
# ============================================================================

# Canonical label buckets derived from annotation-string tags.
ANAH_LABEL_NO_FACT = "no_fact"         # sentence has no factual claim
ANAH_LABEL_OK = "ok"                   # <Hallucination> None (factually supported)
ANAH_LABEL_CONTRADICTORY = "contradictory"  # contradicts reference
ANAH_LABEL_UNVERIFIABLE = "unverifiable"    # not in reference


_ANAH_HEAD_RE = re.compile(r"^\s*<(No Fact|Hallucination)>\s*([^\n<]*)", re.IGNORECASE)


def parse_anah_annotation(ann: str) -> Dict:
    """
    Parse one ANAH annotation string.

    Returns a dict with:
      - label: one of {ok, no_fact, contradictory, unverifiable, unknown}
      - reference: supporting/contradicting reference text (or None)
      - correction: suggested correction (or None)
      - raw_tag: the token right after <Hallucination> (or 'No Fact')
    """
    out = {"label": "unknown", "reference": None, "correction": None, "raw_tag": None}
    if not ann:
        return out

    head = _ANAH_HEAD_RE.search(ann)
    if head:
        head_type = head.group(1).strip()
        head_val = (head.group(2) or "").strip()
        out["raw_tag"] = head_val or head_type
        if head_type.lower() == "no fact":
            out["label"] = ANAH_LABEL_NO_FACT
        else:
            # <Hallucination> {None|Contradictory|Unverifiable}
            val = head_val.lower()
            if val in ("none", ""):
                out["label"] = ANAH_LABEL_OK
            elif "contradict" in val:
                out["label"] = ANAH_LABEL_CONTRADICTORY
            elif "unverif" in val:
                out["label"] = ANAH_LABEL_UNVERIFIABLE

    m_ref = re.search(r"<Reference>\s*(.*?)(?=(?:<Correction>|$))", ann, re.DOTALL)
    if m_ref:
        ref = m_ref.group(1).strip()
        out["reference"] = None if ref.lower() in ("", "none") else ref

    m_cor = re.search(r"<Correction>\s*(.*)", ann, re.DOTALL)
    if m_cor:
        cor = m_cor.group(1).strip()
        out["correction"] = None if cor.lower() in ("", "none") else cor

    return out


# Sentence segmenter for answers. Simple and deterministic so it aligns
# with the number of annotation entries the dataset provides; we fall
# back to the raw count mismatch case by trimming to min(len()).
_ANAH_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(])")


def split_anah_sentences(text: str) -> List[str]:
    """Split an ANAH response into sentences (approximate)."""
    if not text or not text.strip():
        return []
    parts = _ANAH_SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_anah_dataset(
    *,
    language: str = "en",
    sources: Tuple[str, ...] = ("gpt35", "internlm"),
    cache_dir: str = None,
    keep_no_fact: bool = False,
) -> List[Dict]:
    """
    Load the opencompass/anah dataset.

    Each row in the original dataset is a topic with multiple Q/A pairs; each
    answer is annotated at the sentence level. We flatten to one record per
    (topic, question, answer_source, sentence_index).

    Args:
        language: keep rows with ``row['language'] == language``.
        sources: subset of {"gpt35", "internlm"} to include.
        cache_dir: passed to ``datasets.load_dataset``.
        keep_no_fact: if False (default), drop sentences labelled <No Fact>
            since they carry no verifiable claim.

    Returns:
        List of dicts with keys:
            - context: concatenated reference documents
            - question: the selected_question
            - answer: the full response text (useful for span tasks)
            - sentence: one sentence of the answer
            - sentence_idx: index into the response
            - label: 0 (factual) or 1 (hallucinated)  — see below
            - raw_label: canonical tag (ok / contradictory / unverifiable / no_fact)
            - source: "gpt35" or "internlm"
            - topic: row['name']
            - reference: annotator-provided supporting passage (may be None)
            - correction: annotator-provided correction (may be None)

        label mapping for the binary field:
            ok            -> 0
            contradictory -> 1
            unverifiable  -> 1
            no_fact       -> 0 (only included when keep_no_fact=True)
    """
    from datasets import load_dataset

    ds = load_dataset("opencompass/anah", cache_dir=cache_dir)
    if "train" in ds:
        ds = ds["train"]
    else:  # defensive
        ds = list(ds.values())[0]

    source_map = {
        "gpt35": ("GPT3.5_answers_D", "human_GPT3.5_answers_D_ann"),
        "internlm": ("InternLM_answers", "human_InternLM_answers_ann"),
    }
    for src in sources:
        if src not in source_map:
            raise ValueError(f"unknown anah source: {src!r}")

    samples: List[Dict] = []
    for row_idx, row in enumerate(ds):
        if language and row.get("language") != language:
            continue
        context = "\n\n".join(row.get("documents") or [])
        questions = row.get("selected_questions") or []
        topic = row.get("name") or ""

        for src in sources:
            ans_field, ann_field = source_map[src]
            answers = row.get(ans_field) or []
            anns = row.get(ann_field) or []
            for q_idx, (answer, ann_list) in enumerate(zip(answers, anns)):
                question = questions[q_idx] if q_idx < len(questions) else ""
                sentences = split_anah_sentences(answer)
                # Align sentences with annotation list length. ANAH's native
                # sentence count may differ from our split — trim to min to
                # keep a 1:1 mapping.
                n = min(len(sentences), len(ann_list))
                for s_idx in range(n):
                    parsed = parse_anah_annotation(ann_list[s_idx])
                    lbl = parsed["label"]
                    if lbl == "no_fact" and not keep_no_fact:
                        continue
                    if lbl == "unknown":
                        continue
                    binary = 1 if lbl in (ANAH_LABEL_CONTRADICTORY, ANAH_LABEL_UNVERIFIABLE) else 0
                    samples.append({
                        "id": f"anah_{row_idx}_{q_idx}_{src}_{s_idx}",
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "sentence": sentences[s_idx],
                        "sentence_idx": s_idx,
                        "label": binary,
                        "raw_label": lbl,
                        "source": src,
                        "topic": topic,
                        "reference": parsed["reference"],
                        "correction": parsed["correction"],
                    })

    return samples


def load_anah_local(
    data_dir: str = ".",
    *,
    sources: Tuple[str, ...] = ("gpt35", "internlm"),
    keep_no_fact: bool = False,
) -> List[Dict]:
    """Load ANAH sentence-level samples from a local JSONL file.

    Reads ``data/anah/anah.jsonl`` produced by ``scripts/prepare_anah.py``
    and returns records in the same shape as :func:`load_anah_dataset`.

    Args:
        data_dir: Root directory containing the ``anah/anah.jsonl`` file.
        sources: Subset of {"gpt35", "internlm"} to include.
        keep_no_fact: If False, drop sentences whose hal_type is "no_fact".
    """
    path = Path(data_dir) / "anah" / "anah.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"ANAH local file not found: {path}\n"
            "Run `python3 scripts/prepare_anah.py` first."
        )

    samples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("source") not in sources:
                continue
            if r.get("hal_type") == ANAH_LABEL_NO_FACT and not keep_no_fact:
                continue
            samples.append({
                "id":           r.get("id", ""),
                "context":      r["context"],
                "question":     r.get("question", ""),
                "answer":       r["answer"],
                "sentence":     r["sentence"],
                "sentence_idx": r["sentence_idx"],
                "label":        r["hallucination_label"],
                "raw_label":    r.get("hal_type", ""),
                "source":       r.get("source", ""),
                "topic":        r.get("topic", ""),
                "reference":    None,
                "correction":   None,
            })
    return samples


def load_sampled_jsonl(path: str) -> List[Dict]:
    """Load a pre-sampled sentence-level JSONL (anah_sampled / ragtruth_sampled format).

    Expected fields per record: ``sentence``, ``gold_supported`` (True=factual),
    and either ``context`` (ragtruth) or ``reference`` (anah) for the grounding text.

    Returns samples in evaluate_sentence_level shape:
        ``{"text": ..., "context": ..., "label": int}``
    where ``label=1`` means factual (same convention as FELM/FCB).

    The ``id`` field is set to match PreExtractedFactExtractor's indexing:
    - ANAH:     ``anah_{example_index}_{answer_index}_{src}_{sentence_index}``
    - RAGTruth: ``{row_id}_{sentence_index}``
    """
    _MODEL_KEY_TO_SRC = {
        "GPT3.5_answers_D": "gpt35",
        "InternLM_answers": "internlm",
    }
    records: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            context = r.get("context") or r.get("reference") or ""
            sample: Dict = {
                "text":    r["sentence"],
                "context": context,
                "label":   int(bool(r["gold_supported"])),
                "answer":  r.get("answer_text", ""),
            }
            if "model_key" in r and "example_index" in r:
                src = _MODEL_KEY_TO_SRC.get(r["model_key"], r["model_key"])
                sample["id"] = (
                    f"anah_{r['example_index']}"
                    f"_{r['answer_index']}"
                    f"_{src}"
                    f"_{r['sentence_index']}"
                )
            elif "row_id" in r and "sentence_index" in r:
                sample["id"] = f"{r['row_id']}_{r['sentence_index']}"
            records.append(sample)
    return records


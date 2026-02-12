#!/usr/bin/env python3
"""
Dataset loaders for hallucination detection evaluation.

Supports:
- FELM (wk, science, writing_rec subsets)
- Factcheck-Bench
- HalluEntity
- PsiloQA (span-level)
- Mushroom (span-level)
- RAGTruth (span-level)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_felm_dataset(subset: str, data_dir: str = ".") -> Tuple[List[Dict], str]:
    """
    Load FELM dataset.

    Args:
        subset: One of 'wk', 'science', 'writing_rec'
        data_dir: Base directory containing felm/ subdirectory

    Returns:
        Tuple of (samples, context_field_name)
    """
    from datasets import load_from_disk

    ds_path = Path(data_dir) / "felm" / subset
    if not ds_path.exists():
        raise FileNotFoundError(f"FELM dataset not found at {ds_path}")

    ds = load_from_disk(str(ds_path))

    # Filter samples with valid context
    ds = ds.filter(lambda x: x['ref_text'] and 'Ошибка' not in x['ref_text'])

    samples = []
    for row in ds['test']:
        for segment, label in zip(row['segmented_response'], row['labels']):
            samples.append({
                'text': segment,
                'context': row['ref_text'],
                'label': label,  # 1 = factual, 0 = hallucination
                'index': row['index'],
            })

    return samples, 'sentence'


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
    for item in data:
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

def load_psiloqa_dataset() -> List[Dict]:
    """
    Load PsiloQA dataset (span-level).

    Returns:
        List of dicts with 'context', 'answer', 'labels' (spans)
    """
    from datasets import load_dataset
    ds = load_dataset("s-nlp/PsiloQA", split='test')
    ds = ds.filter(lambda x: x['lang'] == 'en')

    data = []
    for row in ds:
        data.append({
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
        List of dicts with 'context', 'answer', 'labels' (spans)
    """
    ds_path = Path(data_dir) / "mushroom" / "mushroom.en-tst.v1.extra.with_context.jsonl"
    if not ds_path.exists():
        raise FileNotFoundError(f"Mushroom dataset not found at {ds_path}")

    data = []
    with open(ds_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            data.append({
                "context": row["context"],
                "answer": row["model_output_text"],
                "labels": row["hard_labels"]
            })
    return data


def load_ragtruth_dataset() -> List[Dict]:
    """
    Load RAGTruth dataset (span-level).

    Returns:
        List of dicts with 'context', 'answer', 'labels' (spans)
    """
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split='test')
    ds = ds.filter(lambda x: x['task_type'] == 'QA')

    data = []
    for row in ds:
        labels = json.loads(row['hallucination_labels'])
        data.append({
            "context": row["context"],
            "answer": row["output"],
            "labels": [[s['start'], s['end']] for s in labels]
        })
    return data


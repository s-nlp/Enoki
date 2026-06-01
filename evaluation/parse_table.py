"""
Per-sentence dependency parse + extracted facts debug table.

Produces a CSV with one row per sentence:
  sample_id | coref | sentence | dep_parse | facts

dep_parse  — compact text representation of the spaCy dependency tree:
             "John/PROPN -nsubj-> visited | visited/VERB -ROOT-> visited | ..."
facts      — incremental fact groups extracted from that sentence:
             each group on a separate line, deltas shown as [Δ: 'token']
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


# =============================================================================
# Dep-parse formatting
# =============================================================================

def format_dep_tree(sent) -> str:
    """Compact linear representation of a spaCy sentence dependency tree."""
    parts = []
    for tok in sent:
        if tok.is_space:
            continue
        head = tok.head.text if tok.head != tok else "ROOT"
        parts.append(f"{tok.text}/{tok.pos_} -{tok.dep_}-> {head}")
    return " | ".join(parts)


# =============================================================================
# Main builder
# =============================================================================

def build_parse_table(
    data: List[Dict],
    extractor,
    decontextualizer,
    coref: bool,
    output_file: Path,
) -> None:
    """
    Iterate over span-dataset samples, run NLP, and write a per-sentence table.

    The spaCy doc is obtained from the first extracted Fact (no extra nlp call).
    Falls back to extractor.nlp(text) when no facts are found.
    """
    rows: List[Dict] = []

    for sample_idx, row in enumerate(tqdm(data, desc="Building parse table", file=__import__("sys").stderr)):
        answer = row["answer"]

        # Decontextualize if enabled
        if decontextualizer is not None:
            try:
                dc = decontextualizer.decontextualize(answer)
                text = dc["resolved_text"]
            except Exception:
                text = answer
        else:
            text = answer

        # Extract incremental fact groups
        try:
            groups = extractor.extract_granular_facts(text)
        except Exception:
            groups = []

        # Obtain spaCy doc from the first Fact (avoids double parse)
        doc = None
        for g in groups:
            if g.facts:
                doc = g.facts[0].subject.doc
                break
        if doc is None:
            try:
                doc = extractor.nlp(text)
            except Exception:
                continue

        # Map sentence start_char → list of formatted fact strings
        sent_facts: Dict[int, List[str]] = {}
        for group in groups:
            if not group.facts:
                continue
            sent_start = group.facts[0].subject.sent.start_char
            if sent_start not in sent_facts:
                sent_facts[sent_start] = []
            # Show all incremental steps with their delta
            for fact, delta in zip(group.facts, group.deltas):
                sent_facts[sent_start].append(f"{fact}  [Δ: '{delta.text}']")

        # One CSV row per sentence
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            rows.append({
                "sample_id": sample_idx,
                "coref": "yes" if coref else "no",
                "sentence": sent_text,
                "dep_parse": format_dep_tree(sent),
                "facts": "\n".join(sent_facts.get(sent.start_char, [])),
            })

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sample_id", "coref", "sentence", "dep_parse", "facts"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved parse table ({len(rows)} rows) to {output_file}")

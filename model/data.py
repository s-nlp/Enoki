"""Shared utilities for IGL OpenIE data loading."""

from __future__ import annotations

import torch

LABELS = ["NONE", "ARG1", "REL", "ARG2", "LOC_TMP", "TYPE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# Appended to every sentence so the model can express "is X" / "is X of" / "is X from"
# copular relations (rel_case 1/2/3).
UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]


def collate(batch: list[dict], pad_id: int, max_depth: int) -> dict:
    max_seq = max(len(ex["input_ids"]) for ex in batch)
    max_words = max(ex["num_words"] for ex in batch)

    input_ids_out, attn_out, ws_out, lm_out = [], [], [], []

    for ex in batch:
        pad = max_seq - len(ex["input_ids"])
        input_ids_out.append(ex["input_ids"] + [pad_id] * pad)
        attn_out.append(ex["attention_mask"] + [0] * pad)
        ws_out.append(ex["word_starts"] + [0] * (max_words - ex["num_words"]))

        mat = [row + [-100] * (max_words - len(row)) for row in ex["label_matrix"]]
        while len(mat) < max_depth:
            mat.append([-100] * max_words)
        lm_out.append(mat[:max_depth])

    return {
        "input_ids":      torch.tensor(input_ids_out, dtype=torch.long),
        "attention_mask": torch.tensor(attn_out,      dtype=torch.long),
        "word_starts":    torch.tensor(ws_out,         dtype=torch.long),
        "label_matrix":   torch.tensor(lm_out,         dtype=torch.long),
        "num_words":      torch.tensor([ex["num_words"]    for ex in batch], dtype=torch.long),
        "n_real_words":   torch.tensor([ex["n_real_words"] for ex in batch], dtype=torch.long),
        "sentences":      [ex["sentence"] for ex in batch],
    }

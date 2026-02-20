#!/usr/bin/env python3
import json
import argparse
import re
import string
from collections import defaultdict
from pathlib import Path
from datasets import load_from_disk

# --- Logic Ported Exactly from Script 2 + Binary Fix ---

def is_garbage_binary(text: str) -> bool:
    """
    Checks if a string is actually binary garbage.
    If more than 20% of characters are non-printable, it's not text.
    """
    if not text:
        return True
    printable = set(string.printable)
    non_printable_count = sum(1 for char in text if char not in printable)
    return (non_printable_count / len(text)) > 0.2

def clean_text(text) -> str:
    """Matches Script 2 cleaning logic but adds a shield against binary blobs."""
    if text is None:
        return ""
    
    # Handle Bytes (Common in FELM ref_text)
    if isinstance(text, bytes):
        try:
            # Try to decode, ignore errors to prevent crashes
            text = text.decode("utf-8", errors="ignore")
        except:
            return ""

    # Basic Script 2 Cleaning: whitespace normalization
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Final Guard: If the result is still binary mojibake, kill it
    if is_garbage_binary(text):
        return ""
        
    return text

def to_gold_supported_felm(x) -> bool:
    """Exact label mapping from Script 2."""
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    
    s = str(x).strip().lower()
    if s in {"supported", "support", "entailed", "entails", "true", "yes", "1"}:
        return True
    if s in {"not_supported", "not supported", "unsupported", "contradicted", "false", "no", "0"}:
        return False
    return False

# --- Processing Logic ---

def felm_to_factcheckbench(felm_dir: str, output_path: str):
    dataset = load_from_disk(felm_dir)
    # Handle different HF dataset structures
    if hasattr(dataset, "keys") and "test" in dataset:
        ds = dataset["test"]
    elif hasattr(dataset, "keys") and "train" in dataset:
        ds = dataset["train"]
    else:
        ds = dataset

    docs = defaultdict(dict)
    
    for row in ds:
        # Clean segmented response
        raw_sents = row.get("segmented_response", [])
        raw_labels = row.get("labels", [])
        sents = [clean_text(s) for s in raw_sents]
        
        # Clean context (auto_evidence) - This is where the garbage usually is
        context = clean_text(row.get("ref_text", ""))
        auto_evidence = [context] if context else []

        doc_idx = row.get("index", "unk")
        doc_id = f"felm_{doc_idx}"

        for i, (sent, label) in enumerate(zip(sents, raw_labels)):
            if not sent: 
                continue 
            
            sent_id = f"sent_{i}"
            docs[doc_id][sent_id] = {
                "text": sent,
                "sentence_factuality_label": to_gold_supported_felm(label),
                "human_evidence": [],
                "auto_evidence": auto_evidence,
            }

    # Write to JSONL
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        for doc_id, sentences in docs.items():
            f.write(json.dumps({
                "doc_id": doc_id,
                "sentences": sentences
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--felm_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    felm_to_factcheckbench(args.felm_dir, args.output)
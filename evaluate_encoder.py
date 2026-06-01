#!/usr/bin/env python3
"""Evaluate a trained IGL model on CaRB, OIE16-C, and WiRe57-C.

Requires the benchmark submodules from modern_openie (CaRB, WiRe57, openie6).
By default the script looks for them at ../modern_openie relative to this file.
Pass --modern-openie-dir to override.

Usage:
  python evaluate_encoder.py --checkpoint checkpoints/best.ckpt
  python evaluate_encoder.py --checkpoint checkpoints/best.ckpt --split dev
  python evaluate_encoder.py --checkpoint checkpoints/best.ckpt --out predictions.tsv
  python evaluate_encoder.py --predictions predictions.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile

import nltk
import torch
from transformers import AutoTokenizer

from model.model import IGLModel
from model.data import UNUSED_TOKENS
from model.predict import extract

# ── Benchmark paths (default: ../modern_openie relative to this file) ────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODERN_OPENIE = os.path.join(_HERE, "..", "modern_openie")


def _resolve_paths(modern_openie_dir: str):
    oie6_carb_dir = os.path.join(modern_openie_dir, "openie6", "carb")
    carb_dir      = os.path.join(modern_openie_dir, "CaRB")
    wire57_gold   = os.path.join(modern_openie_dir, "WiRe57", "data",
                                 "WiRe57_343-manual-oie.json")
    return oie6_carb_dir, carb_dir, wire57_gold


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_benchmark_modules(oie6_carb_dir: str, carb_dir: str):
    sys.path.insert(0, oie6_carb_dir)
    carb_mod  = _load_module("oie6_carb",    os.path.join(oie6_carb_dir, "carb.py"))
    oie16_mod = _load_module("oie6_oie16",   os.path.join(oie6_carb_dir, "oie16.py"))
    match_mod = _load_module("oie6_matcher", os.path.join(oie6_carb_dir, "matcher.py"))
    sys.path.pop(0)

    sys.path.insert(0, carb_dir)
    from oie_readers.tabReader import TabReader
    from oie_readers.benchmarkGoldReader import BenchmarkGoldReader
    sys.path.pop(0)

    return carb_mod.Benchmark, oie16_mod.Benchmark, match_mod.Matcher, TabReader, BenchmarkGoldReader


# ── WiRe57-C helpers ──────────────────────────────────────────────────────────

def _load_wire57_gold(gold_path: str) -> tuple[list[str], dict]:
    data = json.load(open(gold_path, encoding="utf-8"))
    gold_by_sent: dict[str, list] = {}
    sentences: list[str] = []
    for group in data.values():
        for item in group:
            sent = item["sent"]
            if sent not in gold_by_sent:
                gold_by_sent[sent] = []
                sentences.append(sent)
            for t in item.get("tuples", []):
                gold_by_sent[sent].append({
                    "arg1": {"words": t["arg1"]["words"]},
                    "rel":  {"words": [w for w in t["rel"]["words"] if w != "inf"]},
                    "arg2": {"words": t["arg2"]["words"]},
                })
    return sentences, gold_by_sent


def _wire57_tuple_match(pred: dict, gold: dict):
    precision = [0, 0]
    recall    = [0, 0]
    for part in ("arg1", "rel", "arg2"):
        pred_words = pred[part].split() if isinstance(pred[part], str) else pred[part]
        gold_words = gold[part]["words"]
        if not pred_words:
            if gold_words:
                return False
            continue
        matching = sum(1 for w in pred_words if w in gold_words)
        if matching == 0:
            return False
        precision[0] += matching
        precision[1] += len(pred_words)
        recall[0]    += matching
        recall[1]    += len(gold_words)
    if not precision[1] or not recall[1]:
        return False
    return [precision[0] / precision[1], recall[0] / recall[1]]


def _wire57_f1(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _wire57_sentence_score(gold_tuples: list, pred_tuples: list) -> dict:
    scores = [[_wire57_tuple_match(p, g) for p in pred_tuples] for g in gold_tuples]
    matches: list[tuple[int, int]] = []
    while True:
        best_f1, best_i, best_j = 0.0, None, None
        for i, row in enumerate(scores):
            if i in {m[0] for m in matches}:
                continue
            for j, s in enumerate(row):
                if j in {m[1] for m in matches}:
                    continue
                if s and _wire57_f1(*s) > best_f1:
                    best_f1, best_i, best_j = _wire57_f1(*s), i, j
        if best_f1 == 0.0:
            break
        matches.append((best_i, best_j))
    prec_scores = [scores[i][j][0] for i, j in matches]
    rec_scores  = [scores[i][j][1] for i, j in matches]
    n_pred = len(pred_tuples) if pred_tuples else 1
    return {
        "precision": [sum(prec_scores), n_pred],
        "recall":    [sum(rec_scores),  len(gold_tuples)],
    }


def run_wire57(gold_path: str, results_by_sent: dict) -> None:
    sentences, gold_by_sent = _load_wire57_gold(gold_path)

    prec_num = prec_denom = rec_num = rec_denom = 0
    for sent in sentences:
        gold_tuples = gold_by_sent[sent]
        pred_triples = results_by_sent.get(sent, [])
        pred_tuples = [{"arg1": a1, "rel": rel, "arg2": a2}
                       for _, a1, rel, a2 in pred_triples]
        s = _wire57_sentence_score(gold_tuples, pred_tuples)
        prec_num   += s["precision"][0]
        prec_denom += s["precision"][1]
        rec_num    += s["recall"][0]
        rec_denom  += s["recall"][1]

    prec = prec_num / prec_denom if prec_denom else 0.0
    rec  = rec_num  / rec_denom  if rec_denom  else 0.0
    f1   = _wire57_f1(prec, rec)

    print(f"\n{'─'*40}")
    print(f"  WiRe57-C   word-overlap tuple match")
    print(f"{'─'*40}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ({len(sentences)} sentences, {sum(len(v) for v in gold_by_sent.values())} gold tuples)")
    print(f"{'─'*40}")


def run_oie16c_benchmark(OIE16Benchmark, Matcher, TabReader, BenchmarkGoldReader,
                          carb_gold: str, sentences: list,
                          results_by_sent: dict, out_prefix: str) -> None:
    pred_file = f"{out_prefix}_OIE16C.tsv"
    pr_file   = pred_file.replace(".tsv", "_pr.tsv")

    with open(pred_file, "w") as f:
        for sent in sentences:
            for conf, arg1, rel, arg2 in results_by_sent.get(sent, []):
                parts = [sent, f"{conf:.4f}", rel, arg1]
                if arg2:
                    parts.append(arg2)
                f.write("\t".join(parts) + "\n")

    bgr = BenchmarkGoldReader()
    bgr.read(carb_gold)

    bench = OIE16Benchmark.__new__(OIE16Benchmark)
    bench.gold = bgr.oie

    tr = TabReader()
    tr.read(pred_file)

    auc_score, optimal = bench.compare(
        predicted=tr.oie,
        matchingFunc=Matcher.lexicalMatch,
        output_fn=pr_file,
    )
    precision, recall, f1, _ = optimal

    print(f"\n{'─'*40}")
    print(f"  OIE16-C   lexical tuple match (CaRB gold)")
    print(f"{'─'*40}")
    print(f"  AUC:       {auc_score:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"{'─'*40}")
    print(f"  PR curve → {pr_file}")


def run_benchmark(Benchmark, Matcher, TabReader,
                  label: str, sentences: list, gold_file: str,
                  results_by_sent: dict, out_prefix: str) -> None:
    pred_file = f"{out_prefix}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.tsv"
    pr_file   = pred_file.replace(".tsv", "_pr.tsv")

    with open(pred_file, "w") as f:
        for sent in sentences:
            for conf, arg1, rel, arg2 in results_by_sent.get(sent, []):
                parts = [sent, f"{conf:.4f}", rel, arg1]
                if arg2:
                    parts.append(arg2)
                f.write("\t".join(parts) + "\n")

    bench = Benchmark(gold_file)
    tr = TabReader()
    tr.read(pred_file)

    auc_score, (precision, recall, f1, _), _ = bench.compare(
        predicted=tr.oie,
        matchingFunc=Matcher.binary_linient_tuple_match,
        output_fn=pr_file,
    )

    print(f"\n{'─'*40}")
    print(f"  {label}   binary linient tuple match")
    print(f"{'─'*40}")
    print(f"  AUC:       {auc_score:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"{'─'*40}")
    print(f"  PR curve → {pr_file}")


def merge_incremental(results_by_sent: dict) -> dict:
    """Drop incremental sub-facts: for each (arg1, rel) group, remove any triple
    whose arg2 is a case-insensitive substring of a longer arg2 in the same group.
    """
    merged = {}
    total_before = total_after = 0
    for sent, triples in results_by_sent.items():
        total_before += len(triples)
        groups: dict[tuple, list] = {}
        for t in triples:
            key = (t[1].lower(), t[2].lower())
            groups.setdefault(key, []).append(t)

        kept = []
        for group in groups.values():
            arg2s = [t[3].lower() for t in group]
            for i, t in enumerate(group):
                if not any(i != j and arg2s[i] in arg2s[j] for j in range(len(group))):
                    kept.append(t)
        total_after += len(kept)
        merged[sent] = kept

    print(f"  merge-incremental: {total_before:,} → {total_after:,} triples "
          f"({total_before - total_after:,} sub-facts removed)")
    return merged


def load_predictions_jsonl(path: str) -> dict:
    """Load a predictions JSONL into results_by_sent.

    Each line: {"sentence": "...", "triplets": [[arg1, rel, arg2], ...], ...}
    Returns {sentence: [(conf, arg1, rel, arg2), ...]} with conf=1.0.
    """
    results: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sent = rec["sentence"]
            triples = []
            for t in rec.get("triplets", []):
                if len(t) >= 3:
                    arg1, rel, arg2 = t[0], t[1], t[2]
                elif len(t) == 2:
                    arg1, rel, arg2 = t[0], t[1], ""
                else:
                    continue
                triples.append((1.0, arg1, rel, arg2))
            results[sent] = triples
    return results


def run_evaluation(
    checkpoint: str | None = None,
    predictions: str | None = None,
    split: str = "test",
    batch_size: int = 32,
    top_k: int = 10,
    out: str | None = None,
    merge_incremental_flag: bool = False,
    modern_openie_dir: str | None = None,
    wire57_gold: str | None = None,
) -> None:
    if predictions is None and checkpoint is None:
        raise ValueError("Provide --checkpoint or --predictions")

    nltk.download("punkt_tab", quiet=True)

    mopenie = modern_openie_dir or _DEFAULT_MODERN_OPENIE
    oie6_carb_dir, carb_dir, wire57_gold_default = _resolve_paths(mopenie)
    if wire57_gold is None:
        wire57_gold = wire57_gold_default

    if not os.path.isdir(oie6_carb_dir):
        print(f"WARNING: openie6/carb not found at {oie6_carb_dir}. "
              f"Pass --modern-openie-dir or set up the modern_openie submodules.")
        return
    if not os.path.isdir(carb_dir):
        print(f"WARNING: CaRB not found at {carb_dir}.")
        return

    Benchmark, OIE16Benchmark, Matcher, TabReader, BenchmarkGoldReader = \
        _load_benchmark_modules(oie6_carb_dir, carb_dir)

    carb_input = os.path.join(carb_dir, "data", f"{split}.txt")
    carb_gold  = os.path.join(carb_dir, "data", "gold", f"{split}.tsv")
    carb_sents = [l.strip() for l in open(carb_input) if l.strip()]
    benchmarks = [(f"CaRB {split}", carb_sents, carb_gold)]

    wire57_sents: list[str] = []
    if os.path.exists(wire57_gold):
        wire57_sents, _ = _load_wire57_gold(wire57_gold)
    else:
        print(f"(WiRe57-C skipped — gold file not found at {wire57_gold})")

    if predictions:
        print(f"Loading predictions from {predictions} ...")
        results_by_sent = load_predictions_jsonl(predictions)
        print(f"  {len(results_by_sent)} sentences loaded")
        if merge_incremental_flag:
            results_by_sent = merge_incremental(results_by_sent)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IGLModel.load_from_checkpoint(checkpoint, map_location=device)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": UNUSED_TOKENS})

        all_sentences = list(dict.fromkeys(
            s for _, sents, _ in benchmarks for s in sents
        ) | dict.fromkeys(wire57_sents))
        print(f"Running inference on {len(all_sentences)} unique sentences...")
        results = extract(all_sentences, model, tokenizer, top_k, batch_size, device)
        results_by_sent = {sent: triples for sent, triples in results}

        if merge_incremental_flag:
            results_by_sent = merge_incremental(results_by_sent)

    out_prefix = out or tempfile.mktemp(suffix="")
    for label, sents, gold_file in benchmarks:
        run_benchmark(Benchmark, Matcher, TabReader,
                      label, sents, gold_file, results_by_sent, out_prefix)

    run_oie16c_benchmark(OIE16Benchmark, Matcher, TabReader, BenchmarkGoldReader,
                          carb_gold, carb_sents, results_by_sent, out_prefix)

    if wire57_sents:
        run_wire57(wire57_gold, results_by_sent)

    if out is None:
        import glob
        for f in glob.glob(f"{out_prefix}_*.tsv"):
            if not f.endswith("_pr.tsv"):
                os.remove(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate IGL OpenIE on CaRB, OIE16-C, and WiRe57-C")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to .ckpt file (required unless --predictions is given)")
    p.add_argument("--predictions",  default=None,
                   help="JSONL predictions file (skips model inference)")
    p.add_argument("--split",        default="test", choices=["dev", "test"])
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--top-k",        type=int, default=10)
    p.add_argument("--out",          default=None,
                   help="Prefix for output prediction files (optional)")
    p.add_argument("--merge-incremental", action="store_true")
    p.add_argument("--modern-openie-dir", default=None,
                   help=f"Path to modern_openie clone (default: {_DEFAULT_MODERN_OPENIE})")
    p.add_argument("--wire57-gold",  default=None,
                   help="Override path to WiRe57 JSON gold file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        checkpoint=args.checkpoint,
        predictions=args.predictions,
        split=args.split,
        batch_size=args.batch_size,
        top_k=args.top_k,
        out=args.out,
        merge_incremental_flag=args.merge_incremental,
        modern_openie_dir=args.modern_openie_dir,
        wire57_gold=args.wire57_gold,
    )

"""Extract open IE triples with a trained IGL model."""

from __future__ import annotations

import sys

import torch
import nltk
from transformers import AutoTokenizer

from model.model import IGLModel, LABEL2ID
from model.data import UNUSED_TOKENS


def _word_starts(word_ids: list, max_words: int) -> list[int]:
    ws: list[int] = []
    seen: set[int] = set()
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in seen:
            ws.append(pos)
            seen.add(wid)
    return ws + [0] * (max_words - len(ws))


def _tokenize_words(sentence: str) -> list[str]:
    """Match NLTK's word tokenization without downloading punkt data."""
    return nltk.word_tokenize(sentence, preserve_line=True)


@torch.inference_mode()
def extract(
    sentences: list[str],
    model: IGLModel,
    tokenizer: AutoTokenizer,
    top_k: int,
    batch_size: int,
    device: torch.device,
    min_conf: float = 0.0,
    max_length: int = 128,
) -> list[tuple[str, list[tuple]]]:
    """Return list of (sentence, triples) where triples = [(conf, arg1, rel, arg2), ...]."""
    model.eval()
    results: list[tuple] = []

    n_unused = len(UNUSED_TOKENS)

    for i in range(0, len(sentences), batch_size):
        sents = sentences[i : i + batch_size]

        words_batch = [_tokenize_words(s) + UNUSED_TOKENS for s in sents]
        enc = tokenizer(
            words_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        max_w = max(
            len({wid for wid in enc.word_ids(b) if wid is not None})
            for b in range(len(sents))
        )
        ws_batch = [_word_starts(enc.word_ids(b), max_w) for b in range(len(sents))]
        nw_batch = [len({wid for wid in enc.word_ids(b) if wid is not None})
                    for b in range(len(sents))]

        preds, confs = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            torch.tensor(ws_batch, dtype=torch.long, device=device),
            num_words=torch.tensor(nw_batch, dtype=torch.long, device=device),
        )

        for b in range(len(sents)):
            words = _tokenize_words(sents[b])
            nw = nw_batch[b]
            n_real = nw - n_unused

            triples: list[tuple] = []
            for d in range(preds.shape[1]):
                row = preds[b, d, :nw].tolist()
                conf = confs[b, d].item()

                arg1, rel, arg2 = [], [], []
                for w, lid in zip(words, row[:n_real]):
                    if lid == LABEL2ID["ARG1"]:
                        arg1.append(w)
                    elif lid == LABEL2ID["REL"]:
                        rel.append(w)
                    elif lid in (LABEL2ID["ARG2"], LABEL2ID["LOC_TMP"], LABEL2ID["TYPE"]):
                        arg2.append(w)

                # Check [unused1/2/3] positions for implicit "is X" relation prefix
                rel_case = 0
                for k, lid in enumerate(row[n_real:n_real + n_unused]):
                    if lid == LABEL2ID["REL"]:
                        rel_case = k + 1
                        break

                def _dedup_words(words: list) -> list:
                    seen: set = set()
                    out: list = []
                    for w in words:
                        if w not in seen:
                            seen.add(w)
                            out.append(w)
                    return out

                arg1 = _dedup_words(arg1)
                rel  = _dedup_words(rel)
                arg2 = _dedup_words(arg2)

                rel_str = " ".join(rel)
                if rel_case == 1:
                    rel_str = "is " + rel_str
                elif rel_case == 2:
                    rel_str = "is " + rel_str + " of"
                elif rel_case == 3:
                    rel_str = "is " + rel_str + " from"

                if arg1 and rel_str and conf >= min_conf:
                    triples.append((conf, " ".join(arg1), rel_str, " ".join(arg2)))

            seen_keys: set[tuple] = set()
            unique: list[tuple] = []
            for t in triples:
                key = (t[1], t[2], t[3])
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique.append(t)

            unique.sort(reverse=True)
            results.append((sents[b], unique[:top_k]))

    return results


def main() -> None:
    import argparse
    p = argparse.ArgumentParser("IGL OpenIE inference")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--input",      default=None,  help="Input file (default: stdin)")
    p.add_argument("--output",     default=None,  help="Output file (default: stdout)")
    p.add_argument("--top-k",      type=int,   default=5)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--min-conf",   type=float, default=0.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IGLModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model.hparams.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": UNUSED_TOKENS})

    src = open(args.input) if args.input else sys.stdin
    sentences = [line.strip() for line in src if line.strip()]
    if args.input:
        src.close()

    results = extract(sentences, model, tokenizer, args.top_k, args.batch_size, device,
                      min_conf=args.min_conf)

    dst = open(args.output, "w") if args.output else sys.stdout
    for sent, triples in results:
        dst.write(f"{sent}\n")
        for conf, a1, rel, a2 in triples:
            dst.write(f"  {conf:.3f}: ({a1}; {rel}; {a2})\n")
        dst.write("\n")
    if args.output:
        dst.close()


if __name__ == "__main__":
    main()

"""Decode IGL labels without discarding their original token positions."""

from __future__ import annotations


def extract_anchored(texts, model, tokenizer, *, min_confidence, top_k, local=False):
    import torch
    from nltk.tokenize import NLTKWordTokenizer
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    word_tokenizer = NLTKWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    unused = ["[unused1]", "[unused2]", "[unused3]"] if local else list(model.config.unused_tokens)
    limit = 128 if local else model.config.max_length
    device = next(model.parameters()).device
    results = []
    for text in texts:
        words = word_tokenizer.tokenize(text)
        offsets = list(word_tokenizer.span_tokenize(text))
        if len(words) != len(offsets):
            raise ValueError("Encoder word tokenization could not preserve source offsets")
        encoded = tokenizer(words + unused, is_split_into_words=True,
                            truncation=False, return_tensors="pt")
        length = encoded["input_ids"].shape[1]
        if length > limit:
            raise ValueError(
                f"One sentence needs {length} encoder tokens; the model limit is {limit}. "
                "Split this sentence into shorter sentences before extraction. "
                "No input has been silently truncated."
            )
        starts = []
        previous = None
        for pos, word_id in enumerate(encoded.word_ids(0)):
            if word_id is not None and word_id != previous:
                starts.append(pos)
            previous = word_id
        with torch.inference_mode():
            output = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                word_starts=torch.tensor([starts], device=device),
                num_words=torch.tensor([len(words) + len(unused)], device=device),
            )
        predictions, confidences = output if local else (output.predictions, output.confidences)
        triples = []
        for row, score in zip(predictions[0].tolist(), confidences[0].tolist()):
            if score < min_confidence:
                continue
            indices = {
                "subject": [i for i, label in enumerate(row[:len(words)]) if label == 1],
                "predicate": [i for i, label in enumerate(row[:len(words)]) if label == 2],
                "object": [i for i, label in enumerate(row[:len(words)]) if label in (3, 4, 5)],
            }
            triple = {part: detokenizer.detokenize([words[i] for i in ids])
                      for part, ids in indices.items()}
            for k, label in enumerate(row[len(words):len(words) + len(unused)]):
                if label == 2:
                    triple["predicate"] = ("is " + triple["predicate"] +
                                           {0: "", 1: " of", 2: " from"}.get(k, "")).strip()
                    break
            if not triple["subject"] or not triple["predicate"]:
                continue
            triple["confidence"] = score
            triple["spans"] = {part: [list(offsets[i]) for i in ids]
                               for part, ids in indices.items()}
            triples.append(triple)
        unique = []
        seen = set()
        for triple in sorted(triples, key=lambda t: t["confidence"], reverse=True):
            key = tuple((part, triple[part], tuple(map(tuple, triple["spans"][part])))
                        for part in ("subject", "predicate", "object"))
            if key not in seen:
                seen.add(key)
                unique.append(triple)
        results.append({"text": text, "triples": unique[:top_k]})
    return results

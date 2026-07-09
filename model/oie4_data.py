"""DataModule for OpenIE-4 silver label files (Zenodo record 4094228).

Download:
    pip install zenodo-get
    zenodo_get 4094228
    tar xzf data.tar.gz   # produces openie4_labels/train_labels, openie4_labels/dev_labels

Label file format (one sentence block per blank-separated group):
    The company was founded in 1990 . [unused1] [unused2] [unused3]
    NONE ARG1 ARG1 REL ARG2 ARG2 NONE NONE NONE NONE
    ARG1 ARG1 NONE REL NONE NONE NONE NONE NONE NONE

    Next sentence [unused1] [unused2] [unused3]
    ...

Labels per word position (including [unused1/2/3]):
    NONE=0  ARG1=1  REL=2  ARG2=3  LOC/TIME=4  TYPE=5  ARGS=3
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import lightning as L

from model.data import UNUSED_TOKENS, collate

_LABEL_DICT = {
    "NONE": 0, "ARG1": 1, "REL": 2, "ARG2": 3,
    "LOC": 4, "TIME": 4, "TYPE": 5, "ARGS": 3,
}
_N_UNUSED    = len(UNUSED_TOKENS)
_MAX_REAL_WORDS = 100   # sentences longer than this are skipped (matches original)


def parse_label_file(fp: str, allow_negatives: bool = False) -> list[dict]:
    """Parse an OIE4 label file into sentence dicts compatible with LSOIEDataset.

    When allow_negatives=True, sentences with no extraction rows (written by
    make_stage3_data.py --include-negatives) are kept with extractions=[].
    """
    sentences: list[dict] = []
    current_words: list[str] | None = None
    current_extractions: list[list[int]] = []
    new_example = True

    with open(fp, encoding="utf-8") as f:
        lines = f.readlines()

    def _flush():
        nonlocal current_words, current_extractions
        if current_words is not None and (current_extractions or allow_negatives):
            sentences.append({"words": current_words, "extractions": current_extractions})
        current_words = None
        current_extractions = []

    for raw in lines:
        line = raw.rstrip("\n").strip()

        if line == "":
            new_example = True
            _flush()
            continue

        if "[unused" in line or new_example:
            _flush()
            new_example = False
            words = line.split()
            n_real = len(words) - _N_UNUSED
            if n_real < 1 or n_real > _MAX_REAL_WORDS:
                current_words = None
            else:
                current_words = words
        else:
            if current_words is None:
                continue
            parts = line.split()
            labels = [_LABEL_DICT.get(p, 0) for p in parts]
            n = len(current_words)
            labels = (labels + [0] * n)[:n]          # trim/pad to word count
            # must have ARG1 + REL; drop 1-token REL with no ARG2 (silver noise)
            if 1 in labels and 2 in labels and (labels.count(2) > 1 or 3 in labels):
                current_extractions.append(labels)

    _flush()
    return sentences


class OIE4Dataset(Dataset):
    """Dataset for OpenIE-4 label files.

    Words already contain [unused1/2/3]; we do NOT re-append them.
    The label matrix includes labels for those positions (may be TYPE/REL).
    """

    def __init__(
        self,
        sentences: list[dict],
        tokenizer: AutoTokenizer,
        max_depth: int = 5,
        max_length: int = 128,
    ):
        self.sentences   = sentences
        self.tokenizer   = tokenizer
        self.max_depth   = max_depth
        self.max_length  = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        ex    = self.sentences[idx]
        words = ex["words"]                            # includes [unused1/2/3]
        extractions = ex["extractions"]

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        word_ids       = enc.word_ids()

        word_starts: list[int] = []
        seen: set[int] = set()
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                word_starts.append(pos)
                seen.add(wid)

        num_words = len(word_starts)
        n_unused_present = min(_N_UNUSED, num_words)
        n_real = num_words - n_unused_present

        if extractions:
            mat: list[list[int]] = [
                (row + [0] * num_words)[:num_words]
                for row in extractions[: self.max_depth]
            ]
        else:
            # Negative example: teach the model to predict nothing (all-O at depth 0)
            mat = [[0] * num_words]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "word_starts":    word_starts,
            "label_matrix":   mat,
            "num_words":      num_words,
            "n_real_words":   n_real,
            "sentence":       " ".join(words[:n_real]),
        }


class OIE4DataModule(L.LightningDataModule):
    """Loads OpenIE-4 silver label files for training.

    If dev_fp is None or identical to train_fp, the training file is
    automatically split 95% train / 5% dev using a fixed seed.
    """

    def __init__(
        self,
        tokenizer_name: str,
        train_fp: str,
        dev_fp: str | None = None,
        dev_split: float = 0.05,
        batch_size: int = 24,
        max_depth: int = 5,
        max_length: int = 128,
        num_workers: int = 4,
        allow_negatives: bool = False,
    ):
        super().__init__()
        self.tokenizer_name  = tokenizer_name
        self.train_fp        = train_fp
        self.dev_fp          = dev_fp
        self.dev_split       = dev_split
        self.batch_size      = batch_size
        self.max_depth       = max_depth
        self.max_length      = max_length
        self.num_workers     = num_workers
        self.allow_negatives = allow_negatives
        self.tokenizer: AutoTokenizer | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": UNUSED_TOKENS})

        print(f"Parsing {self.train_fp} …")
        all_sents = parse_label_file(self.train_fp, allow_negatives=self.allow_negatives)
        print(f"  {len(all_sents):,} sentences total")

        if self.dev_fp and self.dev_fp != self.train_fp:
            print(f"Parsing {self.dev_fp} …")
            train_sents = all_sents
            dev_sents   = parse_label_file(self.dev_fp, allow_negatives=self.allow_negatives)
            print(f"  {len(dev_sents):,} dev sentences")
        else:
            import random
            rng = random.Random(42)
            rng.shuffle(all_sents)
            n_dev = max(1, int(len(all_sents) * self.dev_split))
            dev_sents, train_sents = all_sents[:n_dev], all_sents[n_dev:]
            print(f"  auto-split → {len(train_sents):,} train / {len(dev_sents):,} dev")

        self._train = OIE4Dataset(train_sents, self.tokenizer, self.max_depth, self.max_length)
        self._val   = OIE4Dataset(dev_sents,   self.tokenizer, self.max_depth, self.max_length)

    def _loader(self, ds: OIE4Dataset, shuffle: bool) -> DataLoader:
        fn = partial(collate, pad_id=self.tokenizer.pad_token_id, max_depth=self.max_depth)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=fn,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

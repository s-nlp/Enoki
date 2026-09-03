#!/usr/bin/env python3
"""Build the annotated EnokiQA dev and test splits for Hugging Face."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("hf_enokiqa")
SPLITS = {
    "dev": ("enokiqa_dev.jsonl", 1_995),
    "test": ("enokiqa_test.jsonl", 1_995),
}
FIELDS = [
    "id",
    "title",
    "question",
    "answer",
    "context",
    "n_sentences",
    "n_triplets",
    "sentences",
]
SENTENCE_FIELDS = ["sentence_index", "sentence", "triples"]
TRIPLE_FIELDS = [
    "triplet",
    "span",
    "hypothesis",
    "entailment",
    "neutral",
    "contradiction",
    "hall_prob",
]

TRIPLE_TYPE = pa.struct(
    [
        pa.field("triplet", pa.list_(pa.string(), 3), nullable=False),
        pa.field("span", pa.list_(pa.int32()), nullable=False),
        pa.field("hypothesis", pa.string(), nullable=False),
        pa.field("entailment", pa.float64(), nullable=False),
        pa.field("neutral", pa.float64(), nullable=False),
        pa.field("contradiction", pa.float64(), nullable=False),
        pa.field("hall_prob", pa.float64(), nullable=False),
    ]
)
SENTENCE_TYPE = pa.struct(
    [
        pa.field("sentence_index", pa.int32(), nullable=False),
        pa.field("sentence", pa.string(), nullable=False),
        pa.field("triples", pa.list_(TRIPLE_TYPE), nullable=False),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("answer", pa.string(), nullable=False),
        pa.field("context", pa.string(), nullable=False),
        pa.field("n_sentences", pa.int32(), nullable=False),
        pa.field("n_triplets", pa.int32(), nullable=False),
        pa.field("sentences", pa.list_(SENTENCE_TYPE), nullable=False),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def validate_record(record: dict, source: Path, line_number: int) -> None:
    location = f"{source.name}:{line_number}"
    if list(record) != FIELDS:
        raise ValueError(f"{location}: unexpected top-level fields")
    if record["n_sentences"] != len(record["sentences"]):
        raise ValueError(f"{location}: n_sentences mismatch")

    triple_count = 0
    for sentence_index, sentence in enumerate(record["sentences"]):
        if list(sentence) != SENTENCE_FIELDS:
            raise ValueError(f"{location}: unexpected sentence fields")
        if sentence["sentence_index"] != sentence_index:
            raise ValueError(f"{location}: non-sequential sentence_index")
        for triple in sentence["triples"]:
            if list(triple) != TRIPLE_FIELDS:
                raise ValueError(f"{location}: unexpected triple fields")
            if len(triple["triplet"]) != 3:
                raise ValueError(f"{location}: triplet must contain three strings")
            if len(triple["span"]) not in (0, 2):
                raise ValueError(f"{location}: span must be empty or [start, end]")
            probability_sum = (
                triple["entailment"]
                + triple["neutral"]
                + triple["contradiction"]
            )
            if abs(probability_sum - 1.0) > 1e-6:
                raise ValueError(f"{location}: NLI probabilities do not sum to one")
            expected_hall_prob = triple["neutral"] + triple["contradiction"]
            if abs(triple["hall_prob"] - expected_hall_prob) > 1e-6:
                raise ValueError(f"{location}: hall_prob mismatch")
            triple_count += 1

    if record["n_triplets"] != triple_count:
        raise ValueError(f"{location}: n_triplets mismatch")


def build_split(
    source: Path,
    destination: Path,
    expected_rows: int,
    batch_size: int,
) -> int:
    temporary = destination.with_name(f".{destination.name}.tmp")
    records: list[dict] = []
    row_count = 0
    writer = pq.ParquetWriter(
        temporary,
        SCHEMA,
        compression="zstd",
        compression_level=6,
        write_statistics=True,
    )
    completed = False
    try:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                validate_record(record, source, line_number)
                records.append(record)
                if len(records) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
                    row_count += len(records)
                    records.clear()
        if records:
            writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
            row_count += len(records)
        completed = True
    finally:
        writer.close()
        if not completed:
            temporary.unlink(missing_ok=True)

    if row_count != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Expected {expected_rows:,} rows in {source.name}, wrote {row_count:,}"
        )
    temporary.replace(destination)
    return row_count


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # The Hub release intentionally contains annotations only.
    (data_dir / "train-00000-of-00001.parquet").unlink(missing_ok=True)
    (data_dir / "validation-00000-of-00001.parquet").unlink(missing_ok=True)

    total_rows = 0
    for split, (filename, expected_rows) in SPLITS.items():
        source = source_dir / filename
        if not source.is_file():
            raise SystemExit(f"Source JSONL not found: {source}")
        destination = data_dir / f"{split}-00000-of-00001.parquet"
        rows = build_split(source, destination, expected_rows, args.batch_size)
        total_rows += rows
        print(f"Built {rows:,} rows: {destination}")

    release_dir = Path(__file__).resolve().parent
    dataset_card = (release_dir / "README.template").read_text(encoding="utf-8")
    (output_root / "README.md").write_text(
        dataset_card.replace(
            "{{BANNER_PATH}}",
            "./enoki-banner.png",
        ),
        encoding="utf-8",
    )
    shutil.copy2(
        REPO_ROOT / "assets" / "enoki-openie-banner.png",
        output_root / "enoki-banner.png",
    )
    print(f"Built {total_rows:,} annotated rows in {output_root}")


if __name__ == "__main__":
    main()

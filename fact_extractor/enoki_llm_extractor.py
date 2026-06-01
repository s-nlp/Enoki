"""Pre-extracted fact extractor: serves facts from a JSONL file, no runtime extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PreExtractedFactExtractor:
    """
    Pseudo-extractor that serves pre-extracted facts from a JSONL file.

    No NLP extraction is performed at runtime; facts are read from a
    pre-computed file keyed by sample ID.  NLI scoring still runs at
    evaluation time.

    Expected JSONL format (one record per line):
        {"id": "tst-en-1",
         "triplets": [["subj", "pred", "obj"], ...],
         "spans": [[start, end], ...]}

    Each triplet and its corresponding span share the same list index.
    Spans are character offsets in the original answer text.
    """

    def __init__(self, facts_file: str | Path):
        self._records: Dict[str, Dict] = {}
        self._prefix_records: Dict[str, Dict] = {}
        self._load(facts_file)

    def __str__(self) -> str:
        return "PreExtracted (RefChecker)"

    # Maps HF answer-key → short source tag used in dataset sample IDs
    _MODEL_KEY_TO_SRC = {
        "GPT3.5_answers_D": "gpt35",
        "InternLM_answers": "internlm",
    }

    @staticmethod
    def _normalise_record(record: Dict) -> Dict:
        """Return a normalised record with 'id', 'triplets', 'spans' keys.

        Handles two formats:
        • Standard:  {"id": ..., "triplets": [[s,p,o], ...], "spans": [[a,b], ...]}
        • CycleOIE sentence-level train format:
              {"key": ..., "source_id": ..., "sentence_index": ...,
               "sentence": ..., "triples": [{"subject":s,"predicate":p,"object":o}, ...]}
        """
        if "triplets" in record:
            extra: Dict = {}
            if "spans" not in record:
                # Sentence-level files have triplets but no spans; derive from sentence text.
                sentence = record.get("sentence", "")
                spans = []
                for triplet in record["triplets"]:
                    obj = triplet[-1].strip() if triplet else ""
                    pos = sentence.find(obj) if obj else -1
                    spans.append([pos, pos + len(obj)] if pos >= 0 else [0, 0])
                extra["spans"] = spans
            if "id" not in record:
                # ANAH sentence-level format: build a stable ID from positional metadata.
                if "model_key" in record and "example_index" in record:
                    src = PreExtractedFactExtractor._MODEL_KEY_TO_SRC.get(
                        record["model_key"], record["model_key"]
                    )
                    extra["id"] = (
                        f"anah_{record['example_index']}"
                        f"_{record['answer_index']}"
                        f"_{src}"
                        f"_{record['sentence_index']}"
                    )
                elif "row_id" in record and "sentence_index" in record:
                    extra["id"] = f"{record['row_id']}_{record['sentence_index']}"
                else:
                    extra["id"] = ""
            if extra:
                return {**record, **extra}
            return record  # already standard

        # CycleOIE train format
        raw_triples = record.get("triples") or []
        triplets, spans = [], []
        sentence = record.get("sentence", "")
        for t in raw_triples:
            if isinstance(t, dict):
                triplet = [t.get("subject", ""), t.get("predicate", ""), t.get("object", "")]
            else:
                triplet = list(t)
            triplets.append(triplet)
            # Approximate span: locate the object text within the sentence.
            obj = triplet[-1].strip() if triplet else ""
            pos = sentence.find(obj) if obj else -1
            spans.append([pos, pos + len(obj)] if pos >= 0 else [0, 0])

        return {
            "id": record.get("id") or record.get("key") or "",
            "source_id": record.get("source_id"),
            "sentence_index": record.get("sentence_index"),
            "triplets": triplets,
            "spans": spans,
        }

    def _load(self, facts_file: str | Path) -> None:
        # Accumulate per-response facts before storing (for sentence-level files
        # where multiple records share the same row_id / source_id).
        response_level: Dict[str, Dict] = {}

        with open(facts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = self._normalise_record(json.loads(line))
                entry = {"triplets": record["triplets"], "spans": record["spans"]}
                self._records[record["id"]] = entry
                # Also index by dataset:item_idx:sent_key (without trailing hash)
                prefix = ":".join(record["id"].split(":")[:3])
                self._prefix_records[prefix] = entry
                # Sampled ANAH files store positional metadata; register under the
                # full dataset-style ID (anah_{row}_{q}_{src}_{sent}) so that
                # has_sample() and get_facts_by_id() work for filtering/lookup.
                if "example_index" in record and "model_key" in record:
                    src = self._MODEL_KEY_TO_SRC.get(record["model_key"])
                    if src is not None:
                        dataset_id = (
                            f"anah_{record['example_index']}"
                            f"_{record['answer_index']}"
                            f"_{src}"
                            f"_{record['sentence_index']}"
                        )
                        self._records[dataset_id] = entry
                # Sentence-level files (RAGTruth sampled, CycleOIE train): merge
                # all sentences for the same response under the response-level ID
                # (row_id or source_id) for response-level lookup.
                # Also index per-sentence by "{row_id}_{sentence_index}" so that
                # sentence-level evaluation can look up facts for individual sentences.
                agg_key = (
                    str(record["row_id"]) if "row_id" in record
                    else record.get("source_id")
                )
                if agg_key:
                    if agg_key not in response_level:
                        response_level[agg_key] = {"triplets": [], "spans": []}
                    response_level[agg_key]["triplets"].extend(record["triplets"])
                    response_level[agg_key]["spans"].extend(record["spans"])
                    if "sentence_index" in record:
                        sent_key = f"{agg_key}_{record['sentence_index']}"
                        self._records[sent_key] = entry

        self._records.update(response_level)

    def has_sample(self, sample_id: str) -> bool:
        return sample_id in self._records

    def get_facts_by_id(
        self, sample_id: str
    ) -> Optional[List[Tuple[List[str], List[int]]]]:
        """Return list of (triplet, span) pairs for *sample_id*, or None if absent.

        Falls back to a 3-part prefix lookup (dataset:item_idx:sent_key) when
        the full ID (with trailing hash) is not present.
        """
        record = self._records.get(sample_id) or self._prefix_records.get(sample_id)
        if record is None:
            return None
        return list(zip(record["triplets"], record["spans"]))

    def extract_granular_facts(self, text: str) -> list:
        """Stub — use get_facts_by_id for actual lookup."""
        return []


class StandalonePreExtractedFactExtractor:
    """
    Pseudo-extractor for the sentence-split standalone format.

    The standalone JSONL has one record *per sentence* with IDs of the form
    ``{source_id}::sent{NNNN}``.  There is no ``spans`` field; character
    offsets are derived at load time by searching for each arg2 text within
    the sentence boundaries stored in the record.

    Expected JSONL format (one record per line):
        {
          "id": "11904::sent0000",
          "source_id": "11904",
          "sentence_start": 0, "sentence_end": 184,
          "answer": "<full answer text>",
          "context": "<context text or null>",
          "triplets": [["subj", "pred", "obj"], ...]
        }

    Records are grouped by ``source_id`` so that lookups by the dataset's
    sample ID (e.g. ``"11904"``) return all triplets across all sentences.
    """

    def __init__(self, facts_file: str | Path):
        self._records: Dict[str, List[Tuple[List[str], List[int]]]] = {}
        self._load(facts_file)

    def __str__(self) -> str:
        return "StandalonePreExtracted"

    def _load(self, facts_file: str | Path) -> None:
        from collections import defaultdict
        grouped: Dict[str, List[Tuple[List[str], List[int]]]] = defaultdict(list)

        with open(facts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                source_id = record.get("source_id") or record["id"].split("::")[0]
                triplets = record.get("triplets") or []
                answer = record.get("answer", "")
                sent_start = record.get("sentence_start", 0)
                sent_end = record.get("sentence_end", len(answer))

                for triplet in triplets:
                    span = self._locate_span(triplet, answer, sent_start, sent_end)
                    grouped[source_id].append((triplet, span))

        self._records = dict(grouped)

    @staticmethod
    def _locate_span(
        triplet: List[str],
        answer: str,
        sent_start: int,
        sent_end: int,
    ) -> List[int]:
        """Find arg2 text inside [sent_start, sent_end], fall back to wider search."""
        arg2 = triplet[-1].strip() if triplet else ""
        if not arg2:
            return [sent_start, sent_start]
        # Prefer a match within the sentence window
        window = answer[sent_start:sent_end]
        pos = window.find(arg2)
        if pos != -1:
            return [sent_start + pos, sent_start + pos + len(arg2)]
        # Wider search across the full answer
        pos = answer.find(arg2)
        if pos != -1:
            return [pos, pos + len(arg2)]
        return [sent_start, sent_start]

    def has_sample(self, sample_id: str) -> bool:
        return sample_id in self._records

    def get_facts_by_id(
        self, sample_id: str
    ) -> Optional[List[Tuple[List[str], List[int]]]]:
        """Return list of (triplet, span) pairs for *sample_id*, or None if absent."""
        return self._records.get(sample_id) or None

    def extract_granular_facts(self, text: str) -> list:
        """Stub — use get_facts_by_id for actual lookup."""
        return []

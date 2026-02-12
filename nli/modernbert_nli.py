"""ModernBERT-based NLI checker."""

import torch
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional, Tuple

from nli.base import BaseNLIChecker


class ModernBERTEncoderNLI(BaseNLIChecker):
    """
    NLI checker using ModernBERT encoder model.
    Based on tasksource/ModernBERT-large-nli.
    """

    MODEL_NAME = "tasksource/ModernBERT-large-nli"

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize ModernBERT NLI checker.

        Args:
            model_name: HuggingFace model name (default: tasksource/ModernBERT-large-nli)
            device: Device to use (default: auto-detect cuda/cpu)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._label_idxs = None

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self.tokenizer is not None and self.model is not None:
            return

        print(f"Loading ModernBERT NLI model on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Use float16 only on CUDA
        if self.device == "cuda":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
            ).to(self.device).eval()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
            ).to(self.device).eval()

        print(f"ModernBERT NLI model loaded successfully")

    def _get_label_idxs(self) -> tuple[int, int, int]:
        """Get indices for entailment, neutral, and contradiction labels."""
        if self._label_idxs is not None:
            return self._label_idxs

        if self.model is None:
            self._load_model()

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

        def find(substr: str) -> int:
            for i, lbl in id2label.items():
                if substr in lbl:
                    return i
            raise ValueError(f"Can't find label containing {substr!r} in {id2label}")

        self._label_idxs = (find("entail"), find("neutral"), find("contra"))
        return self._label_idxs

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for chunking long premises.

        Prefer spaCy sentencizer when available; fall back to a simple regex splitter.
        """
        text = (text or "").strip()
        if not text:
            return []

        try:
            import spacy  # type: ignore

            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            return sents if sents else [text]
        except Exception:
            import re

            parts = re.split(r"(?<=[.!?])\s+", text)
            parts = [p.strip() for p in parts if p.strip()]
            return parts if parts else [text]

    def _pair_len(self, premise: str, hypothesis: str) -> int:
        enc = self.tokenizer(
            premise,
            hypothesis,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(enc["input_ids"])

    def _chunk_premise_by_sentences(
        self,
        premise: str,
        *,
        hypotheses: List[str],
        max_length: int,
        overlap_sents: int = 1,
    ) -> List[str]:
        """
        Chunk premise into N parts (sentence-based) so that each (premise_chunk, hypothesis)
        fits into max_length without truncation.
        """
        if not premise.strip():
            return [premise]

        # Use the longest hypothesis (by token length) as worst-case.
        if hypotheses:
            hyp_lens = [
                (h, len(self.tokenizer(h, add_special_tokens=False, truncation=False)["input_ids"]))
                for h in hypotheses
            ]
            longest_hyp = max(hyp_lens, key=lambda x: x[1])[0]
        else:
            longest_hyp = ""

        # Fast-path: already fits.
        if self._pair_len(premise, longest_hyp) <= max_length:
            return [premise]

        sents = self._split_into_sentences(premise)
        if not sents:
            return [premise]

        chunks: List[str] = []
        cur: List[str] = []

        def flush():
            nonlocal cur
            if cur:
                chunks.append(" ".join(cur).strip())
                if overlap_sents > 0:
                    cur = cur[-overlap_sents:]
                else:
                    cur = []

        for sent in sents:
            candidate = (" ".join(cur + [sent])).strip() if cur else sent.strip()
            if not candidate:
                continue

            if self._pair_len(candidate, longest_hyp) <= max_length:
                cur.append(sent.strip())
                continue

            # Candidate too long: flush current and try sentence alone.
            flush()

            if self._pair_len(sent, longest_hyp) <= max_length:
                cur.append(sent.strip())
                continue

            # Single sentence doesn't fit: last-resort split by tokens (still deterministic).
            tok_ids = self.tokenizer(
                sent,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            empty_pair = self._pair_len("", longest_hyp)
            budget = max(1, max_length - empty_pair)

            for i in range(0, len(tok_ids), budget):
                sub_ids = tok_ids[i : i + budget]
                sub_text = self.tokenizer.decode(sub_ids, skip_special_tokens=True).strip()
                if sub_text:
                    chunks.append(sub_text)

        flush()
        return [c for c in chunks if c]

    @torch.inference_mode()
    def check_batch(
        self,
        premise: str,
        hypotheses: List[str],
        *,
        max_length: int = 2048,
        premise_chunk_overlap_sents: int = 1,
        debug_chunk_log_path: Optional[str] = None,
        debug_entail_threshold: float = 0.9,
        debug_max_records: int = 50,
    ) -> List[Dict[str, float]]:
        """
        Check NLI for a batch of hypotheses given a premise.

        Args:
            premise: The context/premise text
            hypotheses: List of hypothesis strings to check
            max_length: Maximum sequence length for tokenization

        Returns:
            List of dicts with 'entailment', 'neutral', 'contradiction' probabilities
        """
        if self.tokenizer is None or self.model is None:
            self._load_model()

        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]

        hypotheses = [str(h) for h in hypotheses]
        if not hypotheses:
            return []

        idx_ent, idx_neu, idx_con = self._get_label_idxs()

        premise_chunks = self._chunk_premise_by_sentences(
            premise,
            hypotheses=hypotheses,
            max_length=max_length,
            overlap_sents=premise_chunk_overlap_sents,
        )

        def _run_once(prem: str) -> List[Dict[str, float]]:
            premises = [prem] * len(hypotheses)
            enc = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=False,  # ensured by chunking
                return_tensors="pt",
            )

            # Safety: if chunking missed something, fall back to truncation of premise (last resort).
            if enc["input_ids"].shape[1] > max_length:
                enc = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation="only_first",
                    max_length=max_length,
                    return_tensors="pt",
                )

            enc = {k: v.to(self.model.device, non_blocking=True) for k, v in enc.items()}

            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
            )
            with autocast_ctx:
                logits = self.model(**enc).logits

            probs = logits.softmax(-1).float().cpu()
            return [
                {
                    "entailment": float(p[idx_ent]),
                    "neutral": float(p[idx_neu]),
                    "contradiction": float(p[idx_con]),
                }
                for p in probs
            ]

        if len(premise_chunks) == 1:
            return _run_once(premise_chunks[0])

        # Aggregate: per hypothesis pick chunk with MIN hall_prob := contradiction + neutral.
        best: List[Optional[Dict[str, float]]] = [None] * len(hypotheses)
        best_key: List[Tuple[float, float]] = [(1e9, 1e9)] * len(hypotheses)
        best_chunk_idx: List[int] = [-1] * len(hypotheses)

        for ci, chunk in enumerate(premise_chunks):
            scores = _run_once(chunk)
            for i, s in enumerate(scores):
                hall_prob = s["contradiction"] + s["neutral"]
                key = (hall_prob, s["contradiction"])
                if key < best_key[i]:
                    best_key[i] = key
                    best[i] = s
                    best_chunk_idx[i] = ci

        # Optional debug logging: if best entailment comes from non-first chunk, write JSONL.
        if debug_chunk_log_path:
            try:
                import json

                written = 0
                with open(debug_chunk_log_path, "a", encoding="utf-8") as f:
                    for i, s in enumerate(best):
                        if s is None:
                            continue
                        if best_chunk_idx[i] <= 0:
                            continue
                        if s["entailment"] < debug_entail_threshold:
                            continue
                        rec = {
                            "hypothesis": hypotheses[i],
                            "best_chunk_idx": best_chunk_idx[i],
                            "scores": s,
                            "max_length": max_length,
                            "premise_num_chunks": len(premise_chunks),
                            "premise_first_chunk_preview": premise_chunks[0][:400],
                            "premise_best_chunk_preview": premise_chunks[best_chunk_idx[i]][:400],
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                        if written >= debug_max_records:
                            break
            except Exception:
                pass

        return [
            b if b is not None else {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
            for b in best
        ]

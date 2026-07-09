from __future__ import annotations
"""Fact extraction backed by trained Modern OpenIE / IGL model."""

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
import torch
import spacy
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer

from typing import Set
from .models import Fact, IncrementalFactGroup

# Import IGLModel from the sibling modern_openie package so both projects
# share exactly one definition.
_MODERN_OIE_ROOT = Path(__file__).resolve().parents[2] / "modern_openie"
if str(_MODERN_OIE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODERN_OIE_ROOT))

from src.model import IGLModel, LABEL2ID, ID2LABEL, NUM_LABELS  # noqa: E402

# Appended to every sentence so the model can express "is X" / "is X of" / "is X from"
# copular relations (rel_case 1/2/3).
UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]



logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)

_PRED_AUX_DEPS: Set[str] = {"aux", "auxpass", "neg"}
_PRED_HEAD_POS: Set[str] = {"VERB", "AUX", "ADJ"}
_PRED_XCOMP_POS: Set[str] = {"VERB", "AUX", "ADJ"}


def _expand_predicate(doc: Doc, pred_s: Span) -> Span:
    """Expand a verb span to the full verbal complex (aux, neg, xcomp, phrasal prep)."""
    root = pred_s.root
    indices: Set[int] = set(range(pred_s.start, pred_s.end))

    if root.dep_ in {"aux", "auxpass"} and root.head.pos_ in _PRED_HEAD_POS:
        root = root.head
        indices.add(root.i)

    for child in root.children:
        if child.dep_ in _PRED_AUX_DEPS:
            indices.add(child.i)

    for child in root.children:
        if child.dep_ == "xcomp" and child.pos_ in _PRED_XCOMP_POS:
            indices.add(child.i)
            for gc in child.children:
                if gc.dep_ in _PRED_AUX_DEPS or gc.dep_ in {"prep", "prt", "advmod"}:
                    indices.add(gc.i)

    if not indices:
        return pred_s

    cur_end = max(indices) + 1
    for child in root.children:
        if child.dep_ == "advcl" and child.tag_ == "VBG":
            advcl_left = min(t.i for t in child.subtree)
            if advcl_left == cur_end:
                indices.add(child.i)
                for gc in child.children:
                    if gc.dep_ == "prep":
                        indices.add(gc.i)
                        break

    cur_max = max(indices)
    for child in root.children:
        if child.dep_ in {"prep", "prt", "agent"} and child.i == cur_max + 1:
            indices.add(child.i)
            cur_max = child.i
            for gc in child.children:
                if gc.dep_ in {"prep", "prt"} and gc.i == cur_max + 1:
                    indices.add(gc.i)
                    cur_max = gc.i
                    break

    return doc[min(indices): max(indices) + 1]


def _group_igl_triples(facts: List[Fact], helper) -> List[IncrementalFactGroup]:
    """Group IGL triples into incremental chains without NP expansion.

    The IGL model already outputs incremental triples, so we skip NP expansion
    and group the raw triples directly via the helper's grouping logic.
    """
    return helper._create_incremental_groups(facts)


def _phrase_words(phrase: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(phrase)]


def _token_word_index(doc: Doc) -> List[Tuple[int, str]]:
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        out.append((tok.i, tok.text.lower()))
    return out


def _word_starts(word_ids: list, max_words: int) -> list[int]:
    ws = []
    seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in seen:
            ws.append(pos)
            seen.add(wid)
    return ws + [0] * (max_words - len(ws))


class ModernOpenIEExtractor:
    """Fact extractor backed by trained IGL/OpenIE6-style model."""

    def __str__(self):
        from pathlib import Path
        return f"ModernOpenIE_{Path(self.checkpoint).stem}"

    def __init__(
        self,
        checkpoint: str,
        nlp=None,
        device: Optional[str] = None,
        top_k: int = 10,
        max_length: int = 128,
        dedup: bool = True,
        incremental: bool = True,
        verbose: bool = False,
    ):
        self.checkpoint = checkpoint
        self.nlp = nlp or spacy.load("en_core_web_trf")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = IGLModel.load_from_checkpoint(checkpoint, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.hparams.model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": UNUSED_TOKENS})

        self.top_k = top_k
        self.max_length = max_length
        self.dedup = dedup
        self.incremental = incremental
        self.verbose = verbose
        self._helper = None

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    def extract_granular_facts(self, text: str) -> List[IncrementalFactGroup]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        facts: List[Fact] = []
        fact_confs: List[float] = []   # parallel to facts; carries IGL depth confidence
        seen = set()

        for sent_span in doc.sents:
            sent_text = sent_span.text.strip()
            if not sent_text:
                continue

            triples = self._extract_triples([sent_text])[0][1]

            if self.verbose:
                logger.info("sentence %r", sent_text)

            for conf, subj_s, rel_s, obj_s in triples:
                subj_s = subj_s.strip()
                rel_s = rel_s.strip()
                obj_s = obj_s.strip()

                if self.verbose:
                    logger.info("  raw triple %.3f (%s | %s | %s)", conf, subj_s, rel_s, obj_s)

                if not subj_s or not rel_s:
                    continue

                if self.dedup:
                    key = (
                        sent_span.start_char,
                        sent_span.end_char,
                        subj_s.lower(),
                        rel_s.lower(),
                        obj_s.lower(),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                subj_span = self._locate_span(sent_span, subj_s)
                pred_span = self._locate_span(sent_span, rel_s)
                obj_span = self._locate_span(sent_span, obj_s) if obj_s else None

                if subj_span is None:
                    if self.verbose:
                        logger.info("  dropped: subject span not found: %r", subj_s)
                    continue

                if pred_span is None:
                    pred_span = self._locate_predicate_fallback(sent_span, rel_s)

                if pred_span is None:
                    if self.verbose:
                        logger.info("  dropped: predicate span not found: %r", rel_s)
                    continue

                if obj_s and obj_span is None:
                    if self.verbose:
                        logger.info("  dropped: object span not found: %r", obj_s)
                    continue

                pred_span = _expand_predicate(doc, pred_span)

                facts.append(Fact(
                    subject=subj_span,
                    predicate=pred_span,
                    argument=obj_span,
                    prep=None,
                ))
                fact_confs.append(conf)

        if self.incremental:
            groups = _group_igl_triples(facts, self._get_helper())
            conf_map = {
                (
                    f.subject.start, f.subject.end,
                    f.predicate.start, f.predicate.end,
                    f.argument.start if f.argument else -1,
                    f.argument.end if f.argument else -1,
                ): c
                for f, c in zip(facts, fact_confs)
            }
            for g in groups:
                head = g.facts[-1]
                key = (
                    head.subject.start, head.subject.end,
                    head.predicate.start, head.predicate.end,
                    head.argument.start if head.argument else -1,
                    head.argument.end if head.argument else -1,
                )
                g.confidence = conf_map.get(key, 0.0)
            return groups

        return [
            IncrementalFactGroup(
                facts=[f],
                deltas=[f.argument] if f.argument is not None else [],
                confidence=c,
            )
            for f, c in zip(facts, fact_confs)
        ]

    @torch.inference_mode()
    def _extract_triples(
        self,
        sentences: List[str],
    ) -> List[Tuple[str, List[Tuple[float, str, str, str]]]]:
        results = []
        n_unused = len(UNUSED_TOKENS)

        words_batch = [nltk.word_tokenize(s) + UNUSED_TOKENS for s in sentences]

        enc = self.tokenizer(
            words_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        max_w = max(
            len({wid for wid in enc.word_ids(b) if wid is not None})
            for b in range(len(sentences))
        )

        ws_batch = [_word_starts(enc.word_ids(b), max_w) for b in range(len(sentences))]
        nw_batch = [
            len({wid for wid in enc.word_ids(b) if wid is not None})
            for b in range(len(sentences))
        ]

        preds, confs = self.model(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
            torch.tensor(ws_batch, dtype=torch.long, device=self.device),
            num_words=torch.tensor(nw_batch, dtype=torch.long, device=self.device),
        )

        for b, sent in enumerate(sentences):
            words = nltk.word_tokenize(sent)
            nw = nw_batch[b]
            n_real = max(0, nw - n_unused)

            triples = []

            for d in range(preds.shape[1]):
                row = preds[b, d, :nw].tolist()
                conf = float(confs[b, d].item())

                arg1, rel, arg2 = [], [], []

                for w, lid in zip(words, row[:n_real]):
                    if lid == LABEL2ID["ARG1"]:
                        arg1.append(w)
                    elif lid == LABEL2ID["REL"]:
                        rel.append(w)
                    elif lid in (
                        LABEL2ID["ARG2"],
                        LABEL2ID["LOC_TMP"],
                        LABEL2ID["TYPE"],
                    ):
                        arg2.append(w)

                rel_case = 0
                for k, lid in enumerate(row[n_real:n_real + n_unused]):
                    if lid == LABEL2ID["REL"]:
                        rel_case = k + 1
                        break

                rel_str = " ".join(rel)
                if rel_case == 1:
                    rel_str = "is " + rel_str
                elif rel_case == 2:
                    rel_str = "is " + rel_str + " of"
                elif rel_case == 3:
                    rel_str = "is " + rel_str + " from"

                if arg1 and rel_str:
                    triples.append((
                        conf,
                        " ".join(arg1),
                        rel_str,
                        " ".join(arg2),
                    ))

            seen = set()
            unique = []
            for t in triples:
                key = (t[1].lower(), t[2].lower(), t[3].lower())
                if key not in seen:
                    seen.add(key)
                    unique.append(t)

            unique.sort(key=lambda x: x[0], reverse=True)
            results.append((sent, unique[:self.top_k]))

        return results


    def _get_helper(self):
        if self._helper is None:
            from .extractor import FactExtractor
            self._helper = FactExtractor(
                nlp=self.nlp,
                use_gliner=False,
                use_improvements=False,
            )
        return self._helper

    def _locate_predicate_fallback(self, doc: Doc, relation: str) -> Optional[Span]:
        rel = relation.strip()

        for prefix in ("is ",):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]

        for suffix in (" of", " from"):
            if rel.endswith(suffix):
                rel = rel[: -len(suffix)]

        rel = rel.strip()
        if not rel:
            return None

        return self._locate_span(doc, rel)

    def _locate_span(self, doc: Doc, phrase: str) -> Optional[Span]:
        if not phrase:
            return None

        lower_text = doc.text.lower()
        needle = phrase.lower()
        idx = lower_text.find(needle)
        if idx >= 0:
            span = doc.char_span(idx, idx + len(phrase), alignment_mode="expand")
            if span is not None:
                return span

        words = _phrase_words(phrase)
        if not words:
            return None

        index = _token_word_index(doc)
        n = len(words)

        for start in range(len(index) - n + 1):
            if all(index[start + k][1] == words[k] for k in range(n)):
                first_tok_i = index[start][0]
                last_tok_i = index[start + n - 1][0]
                return doc[first_tok_i:last_tok_i + 1]

        return None



"""Fact extraction backed by MinIE (via miniepy JNI wrapper).

Same interface as :class:`StanfordFactExtractor`: ``extract_granular_facts(text)``
returns ``List[IncrementalFactGroup]`` with spaCy ``Span`` objects so the
downstream NLI pipeline works unchanged.

Setup:
    $ # miniepy jar built once via `mvn package assembly:single
    $ #     -DdescriptorId=jar-with-dependencies` in the miniepy repo.
    $ export JAVA_HOME=/path/to/jdk8
    $ export CLASSPATH=/path/to/minie-0.0.1-SNAPSHOT.jar

MinIE vs Stanford OpenIE: tighter, minimized triples (e.g. removes overly
specific modifiers, decomposes appositives), but still Java-backed via JNI.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import spacy
from spacy.tokens import Doc, Span

from .models import Fact, IncrementalFactGroup


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_QUANT_RE = re.compile(r"\bQUANT_[A-Z]+_\d+\b\s*")  # MinIE quantity placeholders (S, R, O, L, …)
_QUANT_COUNT_RE = re.compile(r"\bQUANT_[A-Z]+_\d+\b")  # same without trailing \s*
_DISCOURSE_PREFIX_RE = re.compile(r"^(yes|no|well|sure|okay|right|absolutely|exactly)[,!.]?\s+", re.IGNORECASE)
_NEG_RE = re.compile(r"\b(not|never|n't)\b", re.IGNORECASE)  # negation words MinIE strips from relations
_VERB_POS = {"AUX", "VERB", "PART"}
_AUX_DEPS = {"aux", "auxpass", "neg"}
_DET_SKIP = frozenset({"the", "a", "an"})

# Default jar + JDK paths (can be overridden via env vars before import).
_DEFAULT_JAR = "/qwarium/home/e.rykov/miniepy/target/minie-0.0.1-SNAPSHOT.jar"
_DEFAULT_JDK = "/qwarium/home/e.rykov/opt/jdk8u482-b08"
_DEFAULT_MINIEPY_PY = "/qwarium/home/e.rykov/miniepy/src/main/python"


def _phrase_words(phrase: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(phrase)]


def _quant_map_from_java(java_prop) -> Dict[str, str]:
    """Build {QUANT_X_N: original_text} by reading quantity annotations off the Java object.

    MinIE stores quantities per triple position: 0=subject, 1=relation, 2=object.
    Falls back to empty dict on any JVM error so callers degrade gracefully.
    """
    quant_map: Dict[str, str] = {}
    try:
        for pos in range(3):
            quants = java_prop.getQuantities(pos)
            for i in range(quants.size()):
                q = quants.get(i)
                q_id = str(q.getId())      # e.g. "R_1", "O_1"
                words = q.getQuantityWords()
                text = " ".join(str(words.get(j).word()) for j in range(words.size()))
                if q_id and text:
                    quant_map[f"QUANT_{q_id}"] = text
    except Exception:
        pass
    return quant_map


def _neg_words_from_java(java_prop) -> str:
    """Return the negation words MinIE stripped from the relation (e.g. 'not'), or ''."""
    try:
        neg_words = java_prop.getPolarity().getNegativeWords()
        return " ".join(str(neg_words.get(i).word()) for i in range(neg_words.size()))
    except Exception:
        return ""


def _restore_annotations(s: str, quant_map: Dict[str, str]) -> str:
    for placeholder, original in quant_map.items():
        s = s.replace(placeholder, original)
    return s


def _expand_quant_nums(doc: Doc, span: Span, phrase: str) -> Span:
    """Expand span to absorb numeric tokens displaced by QUANT placeholders.

    MinIE replaces numbers with QUANT_X_N; after we strip those and match the
    remaining text, the actual number tokens ("six", "15") are left adjacent
    but outside the span.  We check both ends: left if the placeholder was
    phrase-initial, right if phrase-final, so "QUANT_O_1 districts" →
    "six districts" and "districts QUANT_O_1" → "districts six".
    """
    quant_count = len(_QUANT_COUNT_RE.findall(phrase))
    if quant_count == 0:
        return span

    start, end = span.start, span.end

    # Expand left while adjacent token is numeric and we still have budget.
    absorbed = 0
    while absorbed < quant_count and start > 0:
        tok = doc[start - 1]
        if tok.like_num or tok.pos_ == "NUM":
            start -= 1
            absorbed += 1
        else:
            break

    # Expand right for any remaining placeholders not resolved on the left.
    while absorbed < quant_count and end < len(doc):
        tok = doc[end]
        if tok.like_num or tok.pos_ == "NUM":
            end += 1
            absorbed += 1
        else:
            break

    return doc[start:end]


def _token_word_index(doc: Doc) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        w = tok.text.lower()
        if w:
            out.append((tok.i, w))
    return out


class MinIEFactExtractor:
    """Fact extractor that wraps MinIE (via pyjnius → MinIE Java classes).

    Each MinIE proposition ``(subject; relation; object)`` becomes a single
    ``Fact`` in its own ``IncrementalFactGroup`` (delta = object span). Subject
    / relation / object strings are located back in a spaCy ``Doc`` so that
    ``Span.start_char`` / ``end_char`` propagate through NLI span-marking.


    Subclasses ``MinIEFactExtractorSafe``, ``MinIEFactExtractorComplete``,
    ``MinIEFactExtractorAggressive``, and ``MinIEFactExtractorDictionary``
    fix the mode and can be selected by name in the CLI / evaluation harness.
    """

    def __str__(self):
        return f'Minie[{self.mode}]'

    def __init__(
        self,
        nlp=None,
        jar_path: Optional[str] = None,
        java_home: Optional[str] = None,
        miniepy_python_path: Optional[str] = None,
        mode: str = "SAFE",
        dedup: bool = True,
        verbose: bool = False,
        max_workers: int = 1,
    ):
        """
        Args:
            nlp: pre-loaded spaCy model (defaults to ``en_core_web_trf`` so
                parsed spans match what Enoki / NLI pipeline use downstream).
            jar_path: path to miniepy fat jar (``-jar-with-dependencies``).
            java_home: ``JAVA_HOME`` (must be JDK8, not JDK17 — MinIE targets 1.8).
            miniepy_python_path: directory containing ``miniepy/__init__.py``
                (the Python bindings from the miniepy repo).
            mode: MinIE mode — one of ``SAFE``, ``AGGRESSIVE``, ``DICTIONARY``,
                ``COMPLETE``.
            dedup: drop duplicate ``(subject, relation, object)`` propositions.
            verbose: emit DEBUG-level log lines showing raw MinIE triples and
                the facts they resolve to (or why they were dropped).
            max_workers: number of threads for parallel per-sentence MinIE calls.
                Default 1 (serial). Requires pyjnius >= 1.4 for safe JVM thread
                attachment. Set to os.cpu_count() or a fixed value like 4.
        """
        self.nlp = nlp or spacy.load("en_core_web_trf")
        self.mode = mode
        self.dedup = dedup
        self.verbose = verbose
        self.max_workers = max_workers
        self._jar_path = jar_path or os.environ.get("MINIE_JAR", _DEFAULT_JAR)
        self._java_home = java_home or os.environ.get("JAVA_HOME") or _DEFAULT_JDK
        self._miniepy_python_path = (
            miniepy_python_path
            or os.environ.get("MINIEPY_PY_PATH", _DEFAULT_MINIEPY_PY)
        )
        self._minie = None
        # pyjnius JVM is not safe for concurrent calls from different Python
        # threads.  Acquiring this lock serialises cross-sample parallel
        # extraction (extraction_workers > 1) so that JVM calls never overlap.
        self._extraction_lock = threading.Lock()

    def __enter__(self):
        self._ensure_minie()
        return self

    def __exit__(self, exc_type, exc, tb):
        # pyjnius attaches a single JVM to the process; nothing to release.
        return None

    def close(self) -> None:
        return None

    def _ensure_minie(self):
        if self._minie is not None:
            return self._minie
        # MinIE targets Java 1.8 and uses javax.xml.bind (removed in Java 11+).
        # Force JAVA_HOME to a JDK8 before pyjnius boots the JVM.
        os.environ["JAVA_HOME"] = self._java_home
        os.environ["CLASSPATH"] = self._jar_path
        import sys
        if self._miniepy_python_path not in sys.path:
            sys.path.insert(0, self._miniepy_python_path)
        if "jnius" in sys.modules:
            raise RuntimeError(
                "pyjnius was already imported before MinIEFactExtractor could "
                "set JAVA_HOME to JDK8. Set JAVA_HOME=" + self._java_home +
                " before starting the process, or construct MinIEFactExtractor "
                "before importing jnius elsewhere."
            )
        from miniepy import MinIE as _MinIE  # type: ignore
        self._minie = _MinIE()
        return self._minie

    def _call_minie(self, sent_text: str):
        """Call MinIE for one sentence and return its propositions.

        Runs in a thread when max_workers > 1.  pyjnius >= 1.4 attaches new
        threads to the JVM automatically; on older versions this may deadlock.
        """
        return self._minie.get_propositions(sent_text, mode=self.mode)

    def extract_granular_facts(self, text: str) -> List[IncrementalFactGroup]:
        if not text or not text.strip():
            return []

        self._ensure_minie()
        doc = self.nlp(text)

        # Collect cleaned sentence texts, preserving order.
        sent_texts: List[str] = []
        for sent in doc.sents:
            st = sent.text.strip()
            if not st:
                continue
            st = _DISCOURSE_PREFIX_RE.sub("", st)
            if st:
                sent_texts.append(st)

        if not sent_texts:
            return []

        # Dispatch MinIE calls — parallel when max_workers > 1.
        if self.max_workers > 1 and len(sent_texts) > 1:
            sent_props: List = [None] * len(sent_texts)
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sent_texts))) as pool:
                future_to_idx = {pool.submit(self._call_minie, st): i
                                 for i, st in enumerate(sent_texts)}
                for future in as_completed(future_to_idx):
                    i = future_to_idx[future]
                    sent_props[i] = future.result()  # re-raises on exception
        else:
            sent_props = [self._call_minie(st) for st in sent_texts]

        facts: List[Fact] = []
        seen = set()

        for sent_text, props in zip(sent_texts, sent_props):
            if self.verbose:
                logger.info("sentence  %r", sent_text)
            if props is None:
                continue

            for prop in props:
                triple = prop.triple
                if len(triple) < 2:
                    continue
                subj_s = (triple[0] or "").strip()
                rel_s = (triple[1] or "").strip()
                obj_s = (triple[2] or "").strip() if len(triple) >= 3 else ""

                if self.verbose:
                    logger.info("  raw triple  (%s | %s | %s)", subj_s, rel_s, obj_s)

                # Restore quantities and negation stripped by MinIE
                quant_map = _quant_map_from_java(prop.java_obj)
                subj_s = _restore_annotations(subj_s, quant_map)
                rel_s = _restore_annotations(rel_s, quant_map)
                obj_s = _restore_annotations(obj_s, quant_map)
                if _QUANT_COUNT_RE.search(subj_s + rel_s + obj_s):
                    logger.warning("QUANT placeholders not restored (Java API unavailable): %s", triple)
                neg = _neg_words_from_java(prop.java_obj)
                # Stage 2: text-level negation restoration — fallback for when
                # the Java polarity API is unavailable (miniepy wrapping issue).
                # If the source sentence contains a negation word that MinIE
                # stripped but _neg_words_from_java couldn't recover, re-inject it.
                if not neg:
                    m = _NEG_RE.search(sent_text)
                    if m and not _NEG_RE.search(rel_s):
                        neg = m.group(0)
                if neg and neg.lower() not in rel_s.lower():
                    rel_s = neg + " " + rel_s

                if self.verbose and (subj_s, rel_s, obj_s) != (
                    (triple[0] or "").strip(),
                    (triple[1] or "").strip(),
                    (triple[2] or "").strip() if len(triple) >= 3 else "",
                ):
                    logger.info("  restored    (%s | %s | %s)", subj_s, rel_s, obj_s)

                if not subj_s or not rel_s:
                    if self.verbose:
                        logger.info("  dropped     (empty subj/rel after restore)")
                    continue

                if self.dedup:
                    key = (subj_s.lower(), rel_s.lower(), obj_s.lower())
                    if key in seen:
                        if self.verbose:
                            logger.info("  dropped     (duplicate)")
                        continue
                    seen.add(key)

                subj_span = self._locate_span(doc, subj_s)
                pred_span = self._locate_pred_span(doc, rel_s)
                obj_span = self._locate_span(doc, obj_s) if obj_s else None

                if subj_span is None or pred_span is None:
                    if self.verbose:
                        logger.info(
                            "  dropped     subj_span=%s pred_span=%s (span not found)",
                            subj_span, pred_span,
                        )
                    continue
                if obj_s and obj_span is None:
                    if self.verbose:
                        logger.info("  dropped     obj=%r span not found", obj_s)
                    continue

                facts.append(Fact(
                    subject=subj_span,
                    predicate=pred_span,
                    argument=obj_span,
                    prep=None,
                ))
                if self.verbose:
                    logger.info(
                        "  fact        (%s | %s | %s)",
                        subj_span, pred_span, obj_span,
                    )

        return [
            IncrementalFactGroup(
                facts=[f],
                deltas=[f.argument if f.argument is not None else f.predicate],
            )
            for f in facts
        ]


    def _locate_pred_span(self, doc: Doc, relation: str) -> Optional[Span]:
        """Locate predicate span with fallbacks for relations that include
        non-verbal adjuncts or unresolved QUANT placeholders.

        1. Exact / lemma match.
        2. Strip residual QUANT_X_N placeholders (defensive — Java restoration
           may fail), retry exact match.
        3. Filter relation to verb-chain tokens (AUX/VERB/PART) only, using the
           QUANT-stripped string so stray placeholders don't corrupt POS tagging.
           If that finds a short span, extend via dep tree to the full verb phrase.
        """
        # Stage 1: normal match
        span = self._locate_span(doc, relation)
        if span is not None:
            return span

        # Stage 2: strip residual QUANT_X_N (Java restoration may have failed)
        clean = _QUANT_RE.sub("", relation).strip()
        if clean and clean != relation:
            span = self._locate_span(doc, clean)
            if span is not None:
                return span
        else:
            clean = relation

        # Stage 3: filter to verb-chain tokens (use QUANT-stripped string so
        # "QUANT_O_1" tokens don't confuse POS tagger or pollute verb_words)
        rel_doc = self.nlp(clean)
        verb_words = [t.text for t in rel_doc
                      if t.pos_ in _VERB_POS and t.lower_ != "not"
                      and not t.is_punct and not t.is_space]
        if not verb_words:
            return None

        span = self._locate_span(doc, " ".join(verb_words))
        if span is None:
            return None

        # If the match is very short (1-2 tokens), extend to the full verb phrase
        # by walking up spaCy's dep tree (e.g. bare "be" → "is estimated to be").
        if len(span) <= 2:
            root = span.root
            indices = {root.i}
            for child in root.children:
                if child.dep_ in _AUX_DEPS:
                    indices.add(child.i)
            head = root.head
            if head.i != root.i and head.pos_ in _VERB_POS:
                indices.add(head.i)
                for child in head.children:
                    if child.dep_ in _AUX_DEPS or (
                        child.dep_ == "xcomp" and child.i <= root.i
                    ):
                        indices.add(child.i)
            if len(indices) > len(span):
                span = doc[min(indices) : max(indices) + 1]

        return span

    def _locate_span(self, doc: Doc, phrase: str) -> Optional[Span]:
        """Exact substring first; fall back to token-subsequence ignoring
        punctuation (MinIE strips commas and inflects — e.g. ``"be designed by"``
        → original ``"is designed by"``). Lemma fallback catches inflection."""
        if not phrase:
            return None

        lower_text = doc.text.lower()
        needle = phrase.lower()
        idx = lower_text.find(needle)
        if idx >= 0:
            span = doc.char_span(idx, idx + len(phrase), alignment_mode="expand")
            if span is not None:
                return span

        # Strip QUANT placeholders (QUANT_O_N, QUANT_S_N, etc.) before
        # word-based matching — these never appear in the original text.
        clean_phrase = _QUANT_RE.sub("", phrase).strip()
        had_quant = clean_phrase != phrase
        if had_quant and clean_phrase:
            idx = lower_text.find(clean_phrase.lower())
            if idx >= 0:
                span = doc.char_span(idx, idx + len(clean_phrase), alignment_mode="expand")
                if span is not None:
                    return _expand_quant_nums(doc, span, phrase)
        else:
            clean_phrase = phrase

        words = _phrase_words(clean_phrase)
        if not words:
            return None

        index_surface = _token_word_index(doc)
        n = len(words)

        # Strict consecutive match.
        for start in range(len(index_surface) - n + 1):
            if all(index_surface[start + k][1] == words[k] for k in range(n)):
                first_tok_i = index_surface[start][0]
                last_tok_i = index_surface[start + n - 1][0]
                span = doc[first_tok_i : last_tok_i + 1]
                return _expand_quant_nums(doc, span, phrase) if had_quant else span

        # Fuzzy: MinIE drops determiners ("the"/"a"/"an") from multi-word NPs.
        # Allow those tokens to be skipped in the doc token stream.
        span = self._locate_span_skip_dets(doc, words, index_surface)
        if span is not None:
            return _expand_quant_nums(doc, span, phrase) if had_quant else span

        # Lemma fallback — MinIE lemmatizes verbs ("was" -> "be", "is" ~ lemma
        # "be"), so match phrase lemmas against doc-token lemmas.
        phrase_lemmas = [t.lemma_.lower() for t in self.nlp(clean_phrase) if not t.is_space and not t.is_punct]
        if len(phrase_lemmas) != n:
            phrase_lemmas = words  # fallback if lemmatizer yields different count

        index_lemma = [
            (tok.i, (tok.lemma_ or tok.text).lower())
            for tok in doc if not tok.is_space and not tok.is_punct
        ]
        for start in range(len(index_lemma) - n + 1):
            ok = True
            for k in range(n):
                surface = doc[index_lemma[start + k][0]].text.lower()
                lemma = index_lemma[start + k][1]
                if (words[k] != surface and words[k] != lemma
                        and phrase_lemmas[k] != surface and phrase_lemmas[k] != lemma):
                    ok = False
                    break
            if ok:
                first_tok_i = index_lemma[start][0]
                last_tok_i = index_lemma[start + n - 1][0]
                span = doc[first_tok_i : last_tok_i + 1]
                return _expand_quant_nums(doc, span, phrase) if had_quant else span

        return None

    @staticmethod
    def _locate_span_skip_dets(
        doc: Doc,
        words: List[str],
        index_surface: List[Tuple[int, str]],
    ) -> Optional[Span]:
        """Token-subsequence match that allows _DET_SKIP tokens in the doc to
        be skipped between consecutive phrase words (handles MinIE dropping
        "the"/"a"/"an" from extracted NP strings)."""
        n = len(words)
        m = len(index_surface)
        for start in range(m):
            if index_surface[start][1] != words[0]:
                continue
            matched = [start]
            pi = 1   # phrase word index
            di = start + 1  # doc token index
            while pi < n and di < m:
                tok_w = index_surface[di][1]
                if tok_w == words[pi]:
                    matched.append(di)
                    pi += 1
                    di += 1
                elif tok_w in _DET_SKIP:
                    di += 1  # skip determiner in doc
                else:
                    break
            if pi == n:
                first_tok_i = index_surface[matched[0]][0]
                last_tok_i = index_surface[matched[-1]][0]
                return doc[first_tok_i : last_tok_i + 1]
        return None


class MinIEFactExtractorSafe(MinIEFactExtractor):
    """MinIE in SAFE mode (default — minimal relation trimming)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("mode", "SAFE")
        super().__init__(**kwargs)


class MinIEFactExtractorComplete(MinIEFactExtractor):
    """MinIE in COMPLETE mode (no minimization — full Stanford-like triples)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("mode", "COMPLETE")
        super().__init__(**kwargs)


class MinIEFactExtractorAggressive(MinIEFactExtractor):
    """MinIE in AGGRESSIVE mode (maximum minimization)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("mode", "AGGRESSIVE")
        super().__init__(**kwargs)


class MinIEFactExtractorDictionary(MinIEFactExtractor):
    """MinIE in DICTIONARY mode (dictionary-guided minimization)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("mode", "DICTIONARY")
        super().__init__(**kwargs)

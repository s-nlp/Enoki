"""
Post-processing layer for fact_extractor.

Two things live here:

1. `preprocess_input(text)` — run *before* `extract_granular_facts`. Strips
   RAG attribution preambles, rescues imperatives (adds synthetic "You "
   subject) and existential "there is/are" constructions.

2. `filter_fact_groups(groups)` — run *after* extraction. Drops degenerate
   IncrementalFactGroups produced by dependency-parser artifacts: deltas that
   are pure function words, deltas circular with their predicate, evaluative
   adjectives, and meta-commentary / RAG-safety boilerplate.

The rules are derived from the FP analysis in `ragtruth_fp_sample.md` and the
zero-triple analysis in `ragtruth_error_analysis.md`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Input preprocessing
# ---------------------------------------------------------------------------

# RAG attribution preambles that contaminate dep parses. We strip them and
# record the character shift so span offsets can be restored by the caller.
_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*\(?\s*Passages?\s+\d+[\):,.]?\s*", re.IGNORECASE),
    re.compile(r"^\s*Based\s+on\s+(the\s+)?(given|provided)\s+passages?[,:]?\s*", re.IGNORECASE),
    re.compile(r"^\s*According\s+to\s+(the\s+)?(given|provided)\s+passages?[,:]?\s*", re.IGNORECASE),
    re.compile(r"^\s*From\s+(the\s+)?(given|provided)\s+passages?[,:]?\s*", re.IGNORECASE),
    re.compile(r"^\s*(Sure|Certainly|Of course|Absolutely)[,!.]?\s*", re.IGNORECASE),
    re.compile(r"^\s*Unfortunately[,.]?\s*", re.IGNORECASE),
    re.compile(r"^\s*(In summary|To summarize|Overall)[,:]?\s*", re.IGNORECASE),
    re.compile(r"^\s*Note[:\-]\s*", re.IGNORECASE),
]

# "there is/are X" → "X exists". Handles the 59 RAGTruth existential
# sentences that currently produce `there | ... | X` malformed triples.
_EXISTENTIAL = re.compile(
    r"\bthere\s+(?:(?:does\s+not\s+|do\s+not\s+|doesn't\s+|don't\s+)?"
    r"(?:seem\s+to\s+)?(?:appear\s+to\s+)?)(is|are|was|were)\b",
    re.IGNORECASE,
)

# Imperative rescue: a sentence starting with a bare verb (lemma of VB form)
# gets a synthetic "You " subject so spaCy assigns nsubj. Very conservative:
# only applies to short initial imperatives that look like how-to steps.
_IMPERATIVE_STARTERS = {
    "scroll", "click", "tap", "press", "select", "choose", "open", "close",
    "go", "navigate", "check", "verify", "ensure", "make", "follow",
    "add", "remove", "delete", "update", "install", "uninstall", "download",
    "upload", "run", "execute", "launch", "start", "stop", "restart",
    "set", "unset", "configure", "change", "enable", "disable", "turn",
    "tap", "hold", "swipe", "drag", "drop", "move", "copy", "paste",
    "enter", "type", "input", "fill", "submit", "save", "load", "import",
    "export", "search", "find", "look", "use", "try", "take", "put",
    "place", "leave", "keep", "hold", "wait", "pause", "resume", "continue",
    "repeat", "avoid", "remember", "note", "consider", "read", "write",
    "heat", "preheat", "cook", "bake", "boil", "simmer", "stir", "mix",
    "combine", "season", "serve", "pour", "cut", "chop", "slice", "dice",
    "apply", "spread", "wash", "rinse", "dry", "clean",
}


@dataclass
class PreprocessResult:
    """Result of :func:`preprocess_input`."""

    text: str  # possibly rewritten text, safe to feed to extractor
    char_shifts: List[Tuple[int, int]]  # [(orig_pos, new_pos - orig_pos), ...]
    notes: List[str]  # human-readable log of what was rewritten


def _split_sentences(text: str) -> List[Tuple[str, int]]:
    """Simple sentence split that preserves char offsets in original text."""
    sents = []
    start = 0
    for match in re.finditer(r"([.!?])(?:\s+|\Z)", text):
        end = match.end()
        sent = text[start:end]
        if sent.strip():
            sents.append((sent, start))
        start = end
    if start < len(text):
        rest = text[start:]
        if rest.strip():
            sents.append((rest, start))
    return sents


def preprocess_input(text: str) -> PreprocessResult:
    """
    Rewrite ``text`` to be friendlier to the extractor.

    The returned text is safe to feed into ``extract_granular_facts``. If you
    need to map spans back to the original text, use ``char_shifts``.
    """
    if not text or not text.strip():
        return PreprocessResult(text=text, char_shifts=[], notes=[])

    notes: List[str] = []
    out_parts: List[str] = []

    for sent, _start in _split_sentences(text):
        stripped = sent

        # 1. Strip RAG attribution preambles.
        for pat in _PREAMBLE_PATTERNS:
            new = pat.sub("", stripped, count=1)
            if new != stripped:
                notes.append(f"stripped preamble matching {pat.pattern!r}")
                stripped = new
                break

        if not stripped.strip():
            continue

        # 2. Existential rescue: "there is X" → "X is present"
        m = _EXISTENTIAL.search(stripped)
        if m and m.start() < 4:  # must be near the start of sentence
            # naive but effective: replace "there is" / "there are" with a
            # placeholder subject; lets the extractor pick the real subject.
            stripped = _EXISTENTIAL.sub(
                lambda mm: "it " + mm.group(1),
                stripped,
                count=1,
            )
            notes.append("rewrote existential 'there is/are' → 'it is/are'")

        # 3. Imperative rescue: prepend "You " if sentence starts with a verb
        # that we recognize as an imperative starter.
        first_word = stripped.lstrip().split(maxsplit=1)[0] if stripped.strip() else ""
        first_lemma = re.sub(r"[^\w]", "", first_word).lower()
        if first_lemma in _IMPERATIVE_STARTERS:
            leading_ws = stripped[: len(stripped) - len(stripped.lstrip())]
            stripped = f"{leading_ws}You {stripped.lstrip()}"
            notes.append(f"rescued imperative starting with {first_lemma!r}")

        out_parts.append(stripped)

    new_text = " ".join(p.strip() for p in out_parts if p.strip())
    if not new_text:
        new_text = text  # fallback: don't return empty

    return PreprocessResult(text=new_text, char_shifts=[], notes=notes)


# ---------------------------------------------------------------------------
# 2. Post-extraction filters
# ---------------------------------------------------------------------------

# Single-word deltas that carry no propositional content.
_EVALUATIVE_DELTAS = {
    "numerous", "many", "several", "various", "different", "multiple",
    "important", "interesting", "key", "crucial", "significant", "notable",
    "major", "minor", "main", "primary", "secondary", "essential",
    "useful", "helpful", "valuable", "relevant", "appropriate",
    "common", "typical", "standard", "usual", "normal",
    "certain", "specific", "particular", "general",
    "good", "bad", "great", "best", "worst", "better", "worse",
    "new", "old", "modern", "traditional",
    "big", "small", "large", "huge", "tiny",
    "role", "thing", "aspect", "factor", "issue", "point", "way",
}

# Function-word POS tags. A delta whose every token is function-word is noise.
_FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "DET", "PART", "PUNCT", "SPACE"}

# Meta-commentary / LLM safety-disclaimer subjects or arguments.
_META_SUBSTRINGS = (
    "the passages",
    "the passage",
    "the provided passages",
    "the given passages",
    "the context",
    "the question",
    "additional context",
    "more context",
    "more information",
    "a healthcare professional",
    "a medical professional",
    "a doctor",
    "a licensed professional",
    "a qualified professional",
    "professional advice",
    "these steps",
    "the following steps",
    "the following",
    "the above",
)


# ---------------------------------------------------------------------------
# Step 1 – sentence-level boilerplate
# ---------------------------------------------------------------------------

# Sentences that are pure RAG meta-commentary and carry no extractable facts.
# Matched against the full sentence text (case-insensitive).
_BOILERPLATE_SENT_RE = re.compile(
    r"based\s+on\s+(the\s+|this\s+|these\s+)?(provided|given)\b"
    r"|unable\s+to\s+(answer|provide|address|determine|find)\b"
    r"|cannot\s+(answer|provide|address|determine|find)\b"
    r"|can[' ]?t\s+(answer|provide|address|determine|find)\b"
    r"|(the\s+)?(passages?|context|document|text)\s+(do(es)?(\s+not)?|don[' ]?t|doesn[' ]?t)\s+"
    r"(provide|contain|include|mention|specify|state|say|discuss)\b"
    r"|here\s+is\s+(my\s+|the\s+|an?\s+)?(answer|response|summary)\b"
    r"|according\s+to\s+(the\s+|this\s+|a\s+|given\s+|provided\s+)?(passage|context|document|text|information)\b"
    r"|(i\s+)?(am\s+)?(not|unable)\s+(able\s+)?to\s+(answer|provide|address)\b"
    r"|no\s+(specific\s+|relevant\s+)?(information|data|details?|mention)\s+"
    r"(is\s+|are\s+)?(provided|given|available|found|present)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Step 2 – discourse / meta words inside fact spans
# ---------------------------------------------------------------------------

# Word-boundary match so "answering" or "passage-way" don't trigger.
_DISCOURSE_WORD_RE = re.compile(
    r"\b(passage|passages|answer|answers|question|questions|"
    r"provided|given|information|example|examples|context|"
    r"document|documents|aforementioned|aforemention)\b",
    re.IGNORECASE,
)

# "here" only triggers as a standalone subject/argument (not in "here is X").
_HERE_STANDALONE_RE = re.compile(r"^\s*here\s*$", re.IGNORECASE)


def _span_text(span) -> str:
    if span is None:
        return ""
    return getattr(span, "text", str(span))


def _is_boilerplate_sentence(sent_text: str) -> Optional[str]:
    """Return reason string if the sentence is RAG boilerplate, else None."""
    m = _BOILERPLATE_SENT_RE.search(sent_text)
    if m:
        return f"boilerplate sentence: {m.group(0)!r}"
    return None


def _is_discourse_fact(fact) -> Optional[str]:
    """Return reason string if any span of the fact contains a discourse word."""
    subj_text = _span_text(getattr(fact, "subject", None))
    rel_text  = _span_text(getattr(fact, "predicate", None))
    obj_text  = _span_text(getattr(fact, "argument", None))

    for role, text in (("subject", subj_text), ("relation", rel_text), ("object", obj_text)):
        if not text:
            continue
        m = _DISCOURSE_WORD_RE.search(text)
        if m:
            return f"discourse word {m.group(0)!r} in {role}"
        if role in ("subject", "object") and _HERE_STANDALONE_RE.match(text):
            return f"bare 'here' as {role}"

    return None


_WH_TAGS: frozenset = frozenset({"WRB", "WP", "WDT", "WP$"})  # where/when/how/who/which/whose

_MODAL_LEMMAS: frozenset = frozenset({
    "can", "could", "may", "might", "shall", "should", "will", "would",
    "must", "need", "dare", "ought", "be", "have", "do",
})

# Discourse connectives / bare stance adverbs that appear as spurious arguments
# (e.g. "can be separated however", "compounds mix just").
_DISCOURSE_CONNECTIVES: frozenset = frozenset({
    "therefore", "however", "thus", "hence", "moreover", "furthermore",
    "consequently", "nevertheless", "nonetheless", "otherwise", "meanwhile",
    "additionally", "subsequently", "accordingly", "likewise", "instead",
    "still", "yet", "also", "too", "though", "although",
    # bare stance / hedge adverbs that carry no propositional content alone
    "just", "only", "even", "simply", "merely", "largely", "mainly",
    "primarily", "generally", "typically", "usually", "commonly",
    "often", "always", "never", "rarely", "sometimes", "again",
})


def _is_stopword_only_subject(fact) -> Optional[str]:
    """Return reason string if the subject span consists entirely of stopwords."""
    subj = getattr(fact, "subject", None)
    if subj is None:
        return None
    toks = [t for t in _span_tokens(subj)
            if not getattr(t, "is_space", False) and not getattr(t, "is_punct", False)]
    if not toks:
        return None
    if all(getattr(t, "is_stop", False) for t in toks):
        words = [t.text for t in toks]
        return f"stopword-only subject: {words!r}"
    return None


def _is_bare_modal_no_arg(fact) -> Optional[str]:
    """Return reason string if the predicate is only modals/auxiliaries and
    argument is None — e.g. 'Surgical incision sites can also'."""
    if getattr(fact, "argument", None) is not None:
        return None
    pred = getattr(fact, "predicate", None)
    if pred is None:
        return None
    toks = [t for t in _span_tokens(pred)
            if not getattr(t, "is_space", False) and not getattr(t, "is_punct", False)]
    if not toks:
        return None
    if all(
        (getattr(t, "lemma_", "") or t.text).lower() in _MODAL_LEMMAS
        or t.text.lower() in _DISCOURSE_CONNECTIVES
        or getattr(t, "pos_", "") in {"AUX", "ADV", "PART"}
        for t in toks
    ):
        return f"bare-modal predicate with no argument: {[t.text for t in toks]!r}"
    return None


def _is_garbage_argument(fact) -> Optional[str]:
    """Return reason string if the argument is a bare WH-word, function-word
    sequence, or verb leaked from a relative/interrogative clause.

    Catches patterns like:
      - argument = "where" / "when" / "how much" / "in which"
      - argument = "have" / "earn" (verb as argument — claucy xcomp leak)
    """
    arg = getattr(fact, "argument", None)
    if arg is None:
        return None
    toks = [t for t in _span_tokens(arg)
            if not getattr(t, "is_space", False) and not getattr(t, "is_punct", False)]
    if not toks:
        return None

    # WH-word in any position → relative/interrogative fragment
    if any(getattr(t, "tag_", "") in _WH_TAGS for t in toks):
        return f"WH-word argument: {[t.text for t in toks]!r}"

    # All tokens are function-words → syntactic fragment, no content
    if all(getattr(t, "pos_", "") in _FUNCTION_POS for t in toks):
        return f"function-word-only argument: {[t.text for t in toks]!r}"

    # Argument is a bare verb (claucy sometimes leaks the embedded verb as arg)
    if len(toks) == 1 and getattr(toks[0], "pos_", "") == "VERB":
        return f"bare-verb argument: {toks[0].text!r}"

    # Bare discourse connective (therefore, however, thus, …)
    if all(t.text.lower() in _DISCOURSE_CONNECTIVES for t in toks):
        return f"discourse-connective argument: {[t.text for t in toks]!r}"

    return None


def _sentence_of(fact) -> str:
    """Best-effort: get the sentence text containing the fact's subject span."""
    subj = getattr(fact, "subject", None)
    if subj is None:
        return ""
    try:
        return subj.sent.text
    except Exception:
        return ""


_NORMALIZE_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_span(span) -> str:
    """Lowercase, strip punctuation and extra whitespace for comparison."""
    text = _span_text(span)
    return _NORMALIZE_RE.sub("", text).lower().split()


def _is_tautological_fact(fact) -> Optional[str]:
    """Return reason string if subject and object normalize to the same tokens."""
    obj = getattr(fact, "argument", None)
    if obj is None:
        return None
    subj_norm = _normalize_span(getattr(fact, "subject", None))
    obj_norm = _normalize_span(obj)
    if subj_norm and subj_norm == obj_norm:
        return f"tautological: subject == object ({_span_text(getattr(fact, 'subject', None))!r})"
    return None


def _span_tokens(span) -> list:
    """Safely iterate spaCy tokens inside a Span or Token-like object."""
    if span is None:
        return []
    if hasattr(span, "__iter__"):
        try:
            return list(span)
        except TypeError:
            pass
    return [span]


def _is_degenerate_delta(delta, predicate) -> Optional[str]:
    """Return a reason string if the delta is degenerate, else None."""
    toks = [t for t in _span_tokens(delta) if not getattr(t, "is_space", False)]
    if not toks:
        return "empty delta"

    # 1. All function words
    if all(getattr(t, "pos_", "") in _FUNCTION_POS for t in toks):
        return f"function-word delta: {[t.text for t in toks]!r}"

    # 2. Single-token evaluative / generic word
    if len(toks) == 1:
        lemma = (getattr(toks[0], "lemma_", "") or toks[0].text).lower()
        if lemma in _EVALUATIVE_DELTAS:
            return f"evaluative delta: {toks[0].text!r}"

    # 3. Circular: delta lemma already in predicate
    pred_lemmas = {
        (getattr(t, "lemma_", "") or t.text).lower()
        for t in _span_tokens(predicate)
    }
    content_lemmas = {
        (getattr(t, "lemma_", "") or t.text).lower()
        for t in toks
        if getattr(t, "pos_", "") not in _FUNCTION_POS
    }
    if content_lemmas and content_lemmas.issubset(pred_lemmas):
        return f"circular delta vs predicate: {content_lemmas}"

    return None


def _is_meta_fact(fact) -> Optional[str]:
    """Return a reason string if a fact is LLM meta-commentary."""
    def _text(span):
        if span is None:
            return ""
        return getattr(span, "text", str(span)).lower()

    subj = _text(getattr(fact, "subject", None))
    arg = _text(getattr(fact, "argument", None))

    for frag in _META_SUBSTRINGS:
        if frag in subj:
            return f"meta subject: {frag!r}"
        if frag in arg:
            return f"meta argument: {frag!r}"

    # Bare "there" subject (existential parse leaked through)
    if subj.strip() in {"there", "it"} and len(arg.split()) < 2:
        return "degenerate existential subject"

    return None


@dataclass
class FilterStats:
    """Accounting of what the filters removed."""

    kept_groups: int = 0
    dropped_groups: int = 0
    dropped_facts: int = 0
    reasons: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def filter_fact_groups(groups, *, verbose: bool = False):
    """
    Filter an extractor output (list of ``IncrementalFactGroup``).

    Steps applied per fact:
      1. Sentence-level boilerplate (RAG meta-commentary, refusals).
      2a. Discourse words in subject/relation/object spans.
      2b. Stopword-only subject.
      3. Meta-commentary substrings (legacy).
      4. Degenerate / circular deltas (legacy).

    Pass ``verbose=True`` or set logging level to DEBUG to see drop reasons.
    Returns ``(filtered_groups, stats)``.
    """
    if verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    stats = FilterStats()
    kept = []

    for group in groups:
        # Build a parallel list of facts/deltas, dropping bad rows.
        kept_facts = []
        kept_deltas = []
        deltas = getattr(group, "deltas", []) or []
        for idx, fact in enumerate(group.facts):
            delta = deltas[idx] if idx < len(deltas) else None

            # Step 1: sentence-level boilerplate
            sent_text = _sentence_of(fact)
            boilerplate_reason = _is_boilerplate_sentence(sent_text) if sent_text else None
            if boilerplate_reason:
                stats.dropped_facts += 1
                stats.reasons.append(boilerplate_reason)
                logger.debug("  [drop] %s — %s", fact, boilerplate_reason)
                continue

            # Step 2a: discourse / meta words in any span
            discourse_reason = _is_discourse_fact(fact)
            if discourse_reason:
                stats.dropped_facts += 1
                stats.reasons.append(discourse_reason)
                logger.debug("  [drop] %s — %s", fact, discourse_reason)
                continue

            # Step 2b: tautological fact (subject == object)
            tautology_reason = _is_tautological_fact(fact)
            if tautology_reason:
                stats.dropped_facts += 1
                stats.reasons.append(tautology_reason)
                logger.debug("  [drop] %s — %s", fact, tautology_reason)
                continue

            # Step 2c: stopword-only subject
            stopword_reason = _is_stopword_only_subject(fact)
            if stopword_reason:
                stats.dropped_facts += 1
                stats.reasons.append(stopword_reason)
                logger.debug("  [drop] %s — %s", fact, stopword_reason)
                continue

            # Step 2d: bare modal predicate with no argument
            bare_modal_reason = _is_bare_modal_no_arg(fact)
            if bare_modal_reason:
                stats.dropped_facts += 1
                stats.reasons.append(bare_modal_reason)
                logger.debug("  [drop] %s — %s", fact, bare_modal_reason)
                continue

            # Step 2e: garbage argument (WH-fragment, bare verb, function-words)
            garbage_arg_reason = _is_garbage_argument(fact)
            if garbage_arg_reason:
                stats.dropped_facts += 1
                stats.reasons.append(garbage_arg_reason)
                logger.debug("  [drop] %s — %s", fact, garbage_arg_reason)
                continue

            # Existing filters: meta-commentary substrings and degenerate deltas
            meta_reason = _is_meta_fact(fact)
            if meta_reason:
                stats.dropped_facts += 1
                stats.reasons.append(meta_reason)
                logger.debug("  [drop] %s — %s", fact, meta_reason)
                continue

            if delta is not None:
                deg_reason = _is_degenerate_delta(delta, getattr(fact, "predicate", None))
                if deg_reason:
                    stats.dropped_facts += 1
                    stats.reasons.append(deg_reason)
                    logger.debug("  [drop] %s — %s", fact, deg_reason)
                    continue

            kept_facts.append(fact)
            kept_deltas.append(delta)

        if kept_facts:
            # Preserve original group type by mutating a shallow copy.
            group.facts = kept_facts
            group.deltas = [d for d in kept_deltas if d is not None]
            kept.append(group)
            stats.kept_groups += 1
        else:
            stats.dropped_groups += 1

    return kept, stats


# ---------------------------------------------------------------------------
# 3. Score-level post-processing: backward entailment suppression
# ---------------------------------------------------------------------------

def backward_suppress_scores(
    scored_items: list,
    *,
    entailment_threshold: float = 0.2,
) -> list:
    """
    Fix partial-delta false positives from the incremental decomposition.

    In the current pipeline, a fact group like:

        He | was born on | June                (hp=0.99)   <-- FP
        He | was born on | June 6              (hp=0.00)
        He | was born on | June 6, 1948        (hp=0.00)   <-- fully entailed

    produces an FP on "June" because the NLI sees only the partial delta in
    isolation. The existing ``incremental_stop_threshold`` only masks
    *forward* within a group; it never reaches back.

    This function walks each group in order. If any later fact in the group
    has ``hall_prob < entailment_threshold`` (i.e. the larger, more-specific
    claim is entailed by the context), every *earlier* fact in the same
    group is zeroed. Rationale: intermediate deltas are stepping stones to
    the full claim; if the full claim is supported, the stepping stones are
    scoring artifacts, not hallucinations.

    Operates in place and returns the same list for chaining.
    """
    # Group items by group_info[0] (the group index) in the order they appear.
    # Items without group_info are left alone (predicate rows, legacy facts).
    groups: dict[int, list[int]] = {}
    for i, item in enumerate(scored_items):
        gi = item.get("group_info")
        if gi is None:
            continue
        group_idx = gi[0]
        groups.setdefault(group_idx, []).append(i)

    for _group_idx, indices in groups.items():
        # Sort by fact_idx_within_group so "later" truly means later.
        indices.sort(key=lambda i: scored_items[i].get("group_info", (0, 0))[1])

        # Find the latest index whose fact is "well entailed".
        last_entailed = -1
        for pos, i in enumerate(indices):
            if scored_items[i].get("hall_prob", 1.0) < entailment_threshold:
                last_entailed = pos

        if last_entailed > 0:
            for pos in range(last_entailed):
                scored_items[indices[pos]]["hall_prob_raw"] = scored_items[indices[pos]].get("hall_prob")
                scored_items[indices[pos]]["hall_prob"] = 0.0
                scored_items[indices[pos]]["suppressed_backward"] = True

    return scored_items

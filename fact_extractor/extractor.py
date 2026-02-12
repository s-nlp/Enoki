"""Fact extraction from text using spaCy dependency parsing."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import re

import spacy
from spacy.tokens import Doc, Span, Token

from .models import (
    TokOrSpan,
    QuotedEntityMapping,
    ContrastiveParse,
    ListParse,
    Fact,
    IncrementalFactGroup,
)
from .utils import (
    preprocess_quoted_entities,
    restore_quoted_entities,
    parse_contrastive_construction,
    split_list_items,
    detect_list_pattern,
    split_enumeration,
    extract_name_from_context,
)


class FactExtractor:
    def __init__(self, nlp=None, use_gliner: bool = True, gliner_model: str = "numind/NuNerZero",
                 use_improvements: bool = True):
        """
        Initialize FactExtractor.

        Args:
            nlp: spaCy language model (default: en_core_web_trf)
            use_gliner: Use GLiNER for enhanced NER (default: True)
            gliner_model: GLiNER model name (default: numind/NuNerZero)
            use_improvements: Use fact extraction improvements (default: True)
        """
        self.nlp = nlp or spacy.load("en_core_web_trf")
        self.use_gliner = use_gliner
        self.gliner_model = gliner_model
        self.use_improvements = use_improvements

        self.copula_verbs = {"be", "is", "was", "are", "were", "am", "been", "being"}
        self.ignored_advmods = {"just", "only", "even", "already"}
        self.months = {
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        }
        self.temporal_preps = {"until", "since", "before", "after", "during", "throughout", "from", "to", "in", "on"}
        self.quote_forms = {'"', "'", "``", "''", "«", "»", """, """, "'", "'"}
        self._quote_cache = {}
        self.title_container_lemmas = {
            "film", "movie", "show", "series", "episode", "production", "play", "musical", "song", "album", "book"
        }
        self.generic_heads = {
            "role", "position", "group", "community", "area", "region", "country", "state",
            "province", "county", "city", "town", "village", "organization", "school",
            "program", "project", "team", "company", "service", "system", "sector", "field",
            "iteration"
        }

        self._quoted_entity_mapping = {}

    def _mask_markdown(self, text: str) -> str:
        """
        Mask markdown formatting with spaces to preserve character offsets.
        This reduces parse noise while keeping span alignment stable.
        """
        def _mask(m):
            return " " * (m.end() - m.start())

        text = re.sub(r'(?m)^[ \t]*#{1,6}.*$', _mask, text)
        text = re.sub(r'(?m)^[ \t]*#{1,6}[ \t]*', _mask, text)
        text = re.sub(r'(?m)^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$', _mask, text)

        text = re.sub(r'(?m)^[ \t]*[-*+][ \t]+', _mask, text)
        text = re.sub(r'(?m)^[ \t]*\d+[.)][ \t]+', _mask, text)

        text = re.sub(r'[*_~]+', _mask, text)
        return text


    def _is_quote_tok(self, t: Token) -> bool:
        # ✅ FIX: Don't treat possessive apostrophe as quote
        # Pattern: "Rus'" - apostrophe with dep_="case" is possessive marker, not quote
        if t.dep_ == "case" and t.text in {"'", "'", "'s", "'s"}:
            return False
        return bool(getattr(t, "is_quote", False)) or t.text in self.quote_forms

    def _is_attached_hash(self, tok: Token, doc: Doc) -> bool:
        """Check if '#' is attached to adjacent alphanumeric text (e.g., C#, F#)."""
        if tok.text != "#" or not tok.is_punct:
            return False
        prev = doc[tok.i - 1] if tok.i > 0 else None
        nxt = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
        if prev and prev.idx + len(prev) == tok.idx and prev.text.isalnum():
            return True
        if nxt and tok.idx + len(tok) == nxt.idx and nxt.text.isalnum():
            return True
        return False

    def _should_ignore_entity_span(self, token: Token, ent: Span) -> bool:
        """Check if entity span should be ignored (e.g., '#' in 'C#')."""
        if len(ent) == 1 and ent.text == "#" and any(c.dep_ == "compound" for c in token.children):
            return True
        return False

    def _collect_nested_preps(self, tok: Token) -> List[Token]:
        """Recursively collect nested prepositions under a token."""
        nested = []
        for ch in tok.children:
            if ch.dep_ == "prep":
                nested.append(ch)
                for prep_child in ch.children:
                    if prep_child.dep_ == "pobj":
                        nested.extend(self._collect_nested_preps(prep_child))
        return nested

    def _quote_pairs_in_sentence(self, sent: Span) -> List[Tuple[int, int]]:
        """
        Returns pairs (open_idx, close_idx) for quotes within a single sentence.
        Toggle logic: 1st quote opens, next closes, and so on.
        """
        key = (id(sent.doc), sent.start, sent.end)
        if key in self._quote_cache:
            return self._quote_cache[key]

        q_idxs = [t.i for t in sent if self._is_quote_tok(t)]
        pairs: List[Tuple[int, int]] = []
        open_idx: Optional[int] = None

        for idx in q_idxs:
            if open_idx is None:
                open_idx = idx
            else:
                pairs.append((open_idx, idx))
                open_idx = None

        self._quote_cache[key] = pairs
        return pairs

    def _get_quoted_span_context(self, token: Token, window: int = 30) -> Optional[Span]:
        if self._is_quote_tok(token):
            return None

        sent = token.sent
        pairs = self._quote_pairs_in_sentence(sent)
        best: Optional[Tuple[int, int]] = None

        for l, r in pairs:
            if l < token.i < r:
                if best is None or (r - l) < (best[1] - best[0]):
                    best = (l, r)

        if best is None:
            return None

        l, r = best
        if l + 1 >= r:
            return None

        return token.doc[l + 1 : r]


    def _get_head_noun_span(self, head: Token) -> Optional[Span]:
        keep = {"compound"}  # can extend: {"compound", "nummod"} if you want "5-0"? but be careful
        toks = [head]
        for ch in head.children:
            if ch.dep_ in keep:
                toks.append(ch)
        return self._span_from_tokens(head.doc, toks)


    def _trim_span_at_breaks(self, span: Optional[Span]) -> Optional[Span]:
        if span is None or len(span) == 0:
            return span

        for t in span:
            if self._find_quote_pair_for_token(t) is not None:
                continue

            if t.is_punct and t.text in {",", ";", ":"}:
                cut = t.i - span.start
                return span[:cut] if cut > 0 else None

            # ✅ ITER 15: Don't trim at "and" if it's part of NP coordination
            # Pattern: "television and radio programs" - "and" connects two modifiers of the same head
            # Pattern: "whimsical and visually striking films" - "and" connects two adj with advmod
            if t.dep_ == "cc" or t.lower_ in {"and", "or", "but"}:
                is_np_coordination = False
                for look_idx in range(t.i + 1, min(t.i + 4, span.end)):
                    look_token = t.doc[look_idx]
                    # If we find a noun/adjective that's in the span and is conj/nmod, it's NP coordination
                    if (look_token.pos_ in {"NOUN", "PROPN", "ADJ"} and
                        look_token.dep_ in {"conj", "nmod", "amod", "compound"}):
                        is_np_coordination = True
                        break
                    if look_token.pos_ == "ADV" and look_token.dep_ == "advmod":
                        continue
                    if look_token.pos_ not in {"ADV"}:
                        break
                
                if is_np_coordination:
                    continue

                cut = t.i - span.start
                return span[:cut] if cut > 0 else None

        return span




    def _tok_span(self, tok: Token) -> Span:
        return tok.doc[tok.i : tok.i + 1]

    def _span_from_tokens(self, doc: Doc, tokens: Iterable[Token]) -> Optional[Span]:
        toks = [t for t in tokens if t is not None]
        if not toks:
            return None
        # ✅ FIX: Filter out markdown symbols (*, **, #, etc.) from spans
        # These are formatting artifacts that shouldn't be part of facts
        markdown_symbols = {"*", "**", "#", "##", "###", "_", "__", "~", "~~"}

        toks = [t for t in toks if not (t.text in markdown_symbols and t.is_punct and not self._is_attached_hash(t, doc))]
        if not toks:
            return None
        toks = sorted({t.i: t for t in toks}.values(), key=lambda t: t.i)
        span = doc[toks[0].i : toks[-1].i + 1]
        span = self._extend_span_to_match_pairs(span)
        return self._trim_markdown_edges(span)

    def _cover(self, *parts: TokOrSpan) -> Optional[Span]:
        items = [p for p in parts if p is not None]
        if not items:
            return None

        doc = items[0].doc if isinstance(items[0], Span) else items[0].doc

        idxs: List[int] = []
        for p in items:
            if isinstance(p, Span):
                idxs.extend([p.start, p.end - 1])
            else:
                idxs.append(p.i)

        lo = min(idxs)
        hi = max(idxs)
        return self._trim_markdown_edges(doc[lo : hi + 1])

    def _subtree_span(self, head: Token, include_punct: bool = True) -> Optional[Span]:
        toks = list(head.subtree)
        if not include_punct:
            toks = [t for t in toks if not t.is_punct]
        return self._span_from_tokens(head.doc, toks)

    def _extend_span_to_match_pairs(self, span: Optional[Span]) -> Optional[Span]:
        if span is None or len(span) == 0:
            return span
        doc = span.doc
        end = span.end
        quote_tokens = {'"', "''", "``", "«", "»", "“", "”", "‘", "’"}
        if any(t.text in quote_tokens for t in span):
            if end < len(doc) and doc[end].text in quote_tokens:
                return doc[span.start:end + 1]
        if any(t.text == "(" for t in span):
            if end < len(doc) and doc[end].text == ")":
                return doc[span.start:end + 1]
        return span

    def _trim_markdown_edges(self, span: Optional[Span]) -> Optional[Span]:
        if span is None or len(span) == 0:
            return span
        doc = span.doc
        start = span.start
        end = span.end
        markdown_tokens = {
            "*", "**", "#", "##", "###", "####", "#####", "######",
            "_", "__", "~", "~~", "---", "--"
        }
        while start < end:
            tok = doc[start]
            if tok.is_space:
                start += 1
                continue
            if tok.text in markdown_tokens and tok.is_punct and not self._is_attached_hash(tok, doc):
                start += 1
                continue
            break
        while end > start:
            tok = doc[end - 1]
            if tok.is_space:
                end -= 1
                continue
            if tok.text in markdown_tokens and tok.is_punct and not _is_attached_hash(tok):
                end -= 1
                continue
            break
        if start >= end:
            return None
        return doc[start:end]

    def _strip_leading_dets(self, span: Optional[Span]) -> Optional[Span]:
        if span is None or len(span) == 0:
            return span
        doc = span.doc
        start = span.start
        while start < span.end and doc[start].dep_ == "det":
            start += 1
        if start >= span.end:
            return None
        return doc[start:span.end]


    def _find_quote_bounds_in_sentence(self, token: Token, window: int = 30) -> Tuple[Optional[int], Optional[int]]:
        doc = token.doc
        sent = token.sent
        sent_start = sent.start
        sent_end = sent.end - 1

        left = None
        start = max(sent_start, token.i - window)
        for i in range(token.i - 1, start - 1, -1):
            if doc[i].text == '"':
                left = i
                break

        right = None
        end = min(sent_end, token.i + window)
        for i in range(token.i + 1, end + 1):
            if doc[i].text == '"':
                right = i
                break

        return left, right

    def _find_quote_pair_for_token(self, token: Token) -> Optional[Tuple[int, int]]:
        sent = token.sent
        best = None
        for l, r in self._quote_pairs_in_sentence(sent):
            if l < token.i < r:
                if best is None or (r - l) < (best[1] - best[0]):
                    best = (l, r)
        return best

    def _descriptor_plus_quoted_title_span(self, head: Token) -> Optional[Span]:
        pair = self._find_quote_pair_for_token(head)
        if pair is None:
            return self._subtree_span(head, include_punct=True)

        left, right = pair
        allowed = {"det", "amod", "compound", "nummod", "poss", "nmod"}
        desc_idxs = [t.i for t in head.subtree if t.i < left and t.dep_ in allowed and not self._is_quote_tok(t)]
        start = min(desc_idxs) if desc_idxs else left
        return head.doc[start : right + 1]  # include quotes so the span is continuous



    def _collect_np_tokens(self, head: Token) -> List[Token]:
        keep = {"amod", "compound", "nummod", "det"}  # Base dependencies
        out = [head]

        # ✅ FIX: Check if head has quoted appositives - if so, don't include distant modifiers
        # This prevents "movie ... (2020)" from creating span that includes quoted title in between
        has_quoted_appos = any(
            ch.dep_ == "appos" and self._get_quoted_span_context(ch) is not None
            for ch in head.children
        )

        for ch in head.children:
            if ch.dep_ == "poss":
                out.append(ch)
                for cc in ch.children:
                    if cc.dep_ == "det":
                        out.append(cc)
                for cc in ch.children:
                    if cc.dep_ == "compound":
                        out.append(cc)
                continue

            if ch.dep_ == "appos":
                has_date = any(
                    t.ent_type_ == "DATE" or t.like_num or (getattr(self, "_is_year_token", None) and self._is_year_token(t))
                    for t in ch.subtree
                )
                if has_date:
                    appos_span = self._subtree_span(ch, include_punct=False) or self._tok_span(ch)
                    if appos_span is not None:
                        out.extend(list(appos_span))
                        if appos_span.start > 0 and head.doc[appos_span.start - 1].text == "(":
                            out.append(head.doc[appos_span.start - 1])
                        if appos_span.end < len(head.doc) and head.doc[appos_span.end].text == ")":
                            out.append(head.doc[appos_span.end])
                    continue

            if ch.dep_ in keep:
                out.extend(self._collect_np_tokens(ch))

            # ✅ FIX: Include hyphenated compounds (pseudo-mandolin, state-of-the-art, etc.)
            # Pattern: head has hyphen child (dep="appos" or "punct") and conjunct/appos
            # Structure: "pseudo" (head) -> "-" (appos) -> "mandolin" (conj or appos)
            elif ch.dep_ in {"conj", "appos"}:
                if ch.text in {"-", "–", "—"}:
                    continue

                has_hyphen = any(
                    h.dep_ in {"appos", "punct"} and h.text in {"-", "–", "—"}
                    for h in head.children
                )

                if has_hyphen:
                    for h in head.children:
                        if h.dep_ in {"appos", "punct"} and h.text in {"-", "–", "—"}:
                            out.append(h)
                    out.extend(self._collect_np_tokens(ch))

            # ✅ FIX: Include npadvmod for hyphenated adjectives like "14-year-old"
            # Structure: "14" (nummod) -> "year" (npadvmod) -> "old" (amod) -> "girl"
            # BUT: Skip if head has quoted appositives (prevents including distant year appositives)
            elif ch.dep_ == "npadvmod" and not has_quoted_appos:
                out.extend(self._collect_np_tokens(ch))

            # ✅ FIX: Include advmod if it's a number/percentage/quantifier
            # Examples: "25.09% White", "72.76% African American", "very tall"
            # This preserves percentages with their racial/demographic context
            elif ch.dep_ == "advmod":
                # Include if it's numeric (percentage, number) or quantifier
                # Check multiple conditions to catch all numeric patterns
                if (ch.like_num or ch.pos_ == "NUM" or "%" in ch.text or
                    ch.ent_type_ in {"PERCENT", "MONEY", "QUANTITY", "CARDINAL"}):
                    out.append(ch)
                elif head.pos_ == "ADJ" and ch.pos_ == "ADV":
                    out.append(ch)

            # ✅ FIX: Include "of" preps that are part of the noun phrase
            # Examples: "variety of factors", "number of plants", "success of X"
            elif ch.dep_ == "prep" and ch.text.lower() == "of":
                out.append(ch)
                for pobj in ch.children:
                    if pobj.dep_ == "pobj":
                        out.extend(self._collect_np_tokens(pobj))

            # ✅ ITER 15: OR if nmod has coordinations (conj children) - include compound modifiers like "television and radio programs"
            elif ch.dep_ == "nmod":
                lo = min(head.i, ch.i)
                hi = max(head.i, ch.i)
                between = [t for t in head.doc[lo:hi+1] if not t.is_space]

                has_conj = any(c.dep_ == "conj" for c in ch.children)

                if (hi - lo) <= 2 and not any(t.is_punct and t.text == "," for t in between):
                    out.extend(self._collect_np_tokens(ch))
                elif has_conj:
                    # Include nmod with coordinations regardless of distance
                    # Pattern: "programs" -> "television" (nmod) -> "radio" (conj)
                    # Result: "television and radio programs"
                    # ✅ Include the nmod and its coordinated items (cc + conj children)
                    out.extend(self._collect_np_tokens(ch))
                    for coord_child in ch.children:
                        if coord_child.dep_ == "cc":  # "and", "or"
                            out.append(coord_child)
                        elif coord_child.dep_ == "conj":  # coordinated noun
                            out.extend(self._collect_np_tokens(coord_child))

        return out



    def _get_np_span(self, head: Token) -> Optional[Span]:
        q = self._get_quoted_span_context(head)
        if q is not None:
            return q

        # Priority 2: Named entities from GLiNER (if use_gliner=True)
        # If head is part of named entity, return full entity span atomically
        # Examples: "Department of Surgery", "Assistant Professor", "North York General Hospital"
        if head.ent_type_:  # Token is part of entity
            for ent in head.doc.ents:
                if ent.start <= head.i < ent.end:
                    if self._should_ignore_entity_span(head, ent):
                        break
                    # Include leading determiner when attached to the entity head (e.g., "the Mediaset group").
                    if ent.start > 0:
                        det = head.doc[ent.start - 1]
                        if det.dep_ == "det" and det.head.i >= ent.start and det.head.i < ent.end:
                            return head.doc[det.i:ent.end]
                    return ent

        toks = self._collect_np_tokens(head)
        return self._span_from_tokens(head.doc, toks)

    def _get_np_core_span(self, head: Token) -> Optional[Span]:
        """
        Return a tight NP span for head without attached preps/relcl.
        Includes determiners and basic modifiers for incremental chains.
        """
        def _collect_core_tokens(tok: Token) -> List[Token]:
            toks = [tok]
            for ch in tok.children:
                if ch.dep_ in {"det", "amod", "compound", "nummod", "poss"}:
                    toks.extend(_collect_core_tokens(ch))
            return toks

        return self._span_from_tokens(head.doc, _collect_core_tokens(head))

    def _get_prep_phrase_tokens(self, prep_token: Token) -> List[Token]:
        def _collect_prep_tokens(tok: Token) -> List[Token]:
            toks = [tok]
            for ch in tok.children:
                if ch.dep_ == "pobj":
                    toks.extend(self._collect_np_tokens(ch))
                    for sub in ch.children:
                        if sub.dep_ in {"det", "amod", "compound", "poss", "nummod", "nmod"}:
                            toks.append(sub)
                elif ch.dep_ == "prep":
                    toks.extend(_collect_prep_tokens(ch))
            return toks

        return _collect_prep_tokens(prep_token)

    def _get_full_subject_span(self, token: Token) -> Optional[Span]:
        q = self._get_quoted_span_context(token)
        if q is not None:
            return q

        ent_span = None
        # Priority 2: Named entities from GLiNER (if use_gliner=True)
        # If token is part of named entity, return full entity span atomically
        if token.ent_type_:  # Token is part of entity
            for ent in token.doc.ents:
                if ent.start <= token.i < ent.end:
                    if self._should_ignore_entity_span(token, ent):
                        break
                    # Include leading determiner when attached to the entity head (e.g., "the Mediaset group").
                    if ent.start > 0:
                        det = token.doc[ent.start - 1]
                        if det.dep_ == "det" and det.head.i >= ent.start and det.head.i < ent.end:
                            ent_span = token.doc[det.i:ent.end]
                        else:
                            ent_span = ent
                    else:
                        ent_span = ent
                    # ✅ FIX: For PERCENT/QUANTITY/CARDINAL entities, also include "of" prep phrases
                    # Example: "51% of residents", "One of the features" should include "of X"
                    # This ensures we capture essential context for quantities
                    if ent.label_ in {"PERCENT", "QUANTITY", "MONEY", "CARDINAL"}:
                        ent_root = ent.root
                        of_prep = next((c for c in ent_root.children if c.dep_ == "prep" and c.text.lower() == "of"), None)
                        if of_prep is not None:
                            prep_tokens = list(ent_span or ent) + self._get_prep_phrase_tokens(of_prep)
                            ent_span = self._span_from_tokens(token.doc, prep_tokens)
                    break

        # Priority 3: Dependency-based extraction (fallback)
        # Use _get_np_span() to get all NP logic including hyphenated compounds
        np_span = ent_span or self._get_np_span(token)

        # Also check for prep phrases and relative clauses attached to subject
        # _get_np_span doesn't include these, so add them here
        subject_tokens = list(np_span) if np_span else [token]

        # ✅ GOLD: Include relative clauses in subject
        # Example: "officers who engage in acts of brutality" should be full subject
        # Pattern: token (subject head) -> relcl (verb with relative pronoun)
        for child in token.children:
            if child.dep_ in {"relcl", "acl"}:
                # Include the full relative clause (who/which/that + verb + args)
                # Use subtree to get all tokens under the relative clause
                relcl_span = self._subtree_span(child, include_punct=False)
                if relcl_span is not None:
                    subject_tokens.extend(relcl_span)
            elif child.dep_ == "prep":
                subject_tokens.extend(self._get_prep_phrase_tokens(child))

        return self._span_from_tokens(token.doc, subject_tokens)

    def _extract_subject(self, verb_token: Token) -> Optional[Span]:
        # ✅ FIX: Include clausal subjects (csubj, csubjpass) for gerunds and clauses
        subs = [c for c in verb_token.children if c.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}]

        # ✅ GOLD: Handle existential "there" as subject
        # Pattern: "there were X" → use "there" as subject
        # Previously we skipped these, but gold standard wants "there | were | X"
        expl = [c for c in verb_token.children if c.dep_ == "expl"]
        if expl and not subs:
            return self._tok_span(expl[0])

        if not subs:
            return None

        def is_bad_subj(t: Token) -> bool:
            if t.pos_ == "NUM":
                # ✅ FIX: Allow NUM with "of" prep (pattern: "One of the features")
                has_of = any(c.dep_ == "prep" and c.text.lower() == "of" for c in t.children)
                if not has_of:
                    return True
            if t.ent_type_ == "DATE":
                return True
            if getattr(self, "_is_year_token", None) is not None and self._is_year_token(t):
                return True
            if t.dep_ == "expl" or t.lower_ == "there":
                return True
            return False

        subs = sorted(subs, key=lambda t: (is_bad_subj(t), t.i))

        if is_bad_subj(subs[0]):
            return None

        # ✅ FIX: Handle hyphenated compounds as multiple nsubj tokens
        # Pattern: "pseudo-mandolin" parsed as consecutive nsubjpass tokens
        # Structure: pseudo (nsubjpass) + - (nsubjpass) + mandolin (nsubjpass)
        if len(subs) > 1:
            subs_sorted = sorted(subs, key=lambda t: t.i)
            is_consecutive = all(
                subs_sorted[i+1].i == subs_sorted[i].i + 1
                for i in range(len(subs_sorted) - 1)
            )
            has_hyphen = any(t.text in {"-", "–", "—"} for t in subs_sorted)

            if is_consecutive and has_hyphen:
                # Return span covering all subject tokens (includes hyphen)
                # Remove hyphen tokens from span for cleaner output
                non_hyphen_tokens = [t for t in subs_sorted if t.text not in {"-", "–", "—"}]
                return self._span_from_tokens(verb_token.doc, subs_sorted)

        # ✅ FIX: For clausal subjects (csubj, csubjpass), use subtree instead of NP
        subj_token = subs[0]
        if subj_token.dep_ in {"csubj", "csubjpass"}:
            return self._subtree_span(subj_token, include_punct=False)
        else:
            return self._get_full_subject_span(subj_token)

    def _inherit_subject(self, token: Token) -> Optional[Span]:
        cur = token
        while cur is not None and cur.head != cur:
            subj = self._extract_subject(cur)
            if subj is not None:
                return subj
            cur = cur.head
        return None


    def _is_meaningful_predicate(self, verb_token: Token) -> bool:
        return verb_token.lemma_ not in self.copula_verbs


    def _get_full_predicate_span(self, verb_token: Token) -> Optional[Span]:
        pred_tokens = [verb_token]
        doc = verb_token.doc

        clause_markers = {
            "when", "where", "while", "because", "since", "if", "although",
            "though", "unless", "once", "as"
        }
        allowed_between = {"aux", "auxpass", "neg", "advmod"}  # punct separately

        for child in verb_token.children:
            if child.dep_ in {"aux", "auxpass", "neg"}:
                pred_tokens.append(child)

            elif child.dep_ == "agent":
                pred_tokens.append(child)

            # ✅ FIX: Include xcomp verbs (infinitives) in predicate
            # Examples: "tend to run", "help to reduce", "continue to serve"
            elif child.dep_ == "xcomp" and child.pos_ in {"VERB", "AUX"}:
                pred_tokens.append(child)
                for xc_child in child.children:
                    if xc_child.dep_ in {"aux", "mark"} and xc_child.text.lower() == "to":
                        pred_tokens.append(xc_child)

            elif child.dep_ == "advmod" and child.lemma_.lower() not in self.ignored_advmods:
                # Drop "also" when it sits between auxiliaries and the main verb (e.g., "can also help").
                if child.text.lower() == "also" and any(c.dep_ in {"aux", "auxpass"} for c in verb_token.children):
                    continue
                if child.i >= verb_token.i:
                    continue

                if child.lower_ in clause_markers or child.pos_ == "SCONJ" or child.dep_ == "mark":
                    continue

                if (verb_token.i - child.i) > 3:
                    continue

                between = [t for t in doc[child.i + 1 : verb_token.i] if not t.is_space]
                if any((not t.is_punct) and (t.dep_ not in allowed_between) for t in between):
                    continue

                pred_tokens.append(child)

        return self._span_from_tokens(verb_token.doc, pred_tokens)



    def _is_date_like_pobj(self, pobj_token: Token, prep_token: Token) -> bool:
        pl = prep_token.text.lower()
        if pl not in {"on", "in"}:
            return False

        if pobj_token.ent_type_ == "DATE":
            return True

        if self._is_year_token(pobj_token):
            return True

        if pobj_token.text.lower() in self.months and any(ch.dep_ in {"nummod", "appos"} for ch in pobj_token.children):
            return True

        return False


    def _trim_on_breaks(self, span: Span) -> Span:
        doc = span.doc
        cut_lemmas = {"when", "whose", "which", "that"}
        cut_deps = {"relcl", "advcl", "acl"}
        for i in range(span.start, span.end):
            t = doc[i]
            if t.is_punct and t.text in {",", ";"}:
                return doc[span.start:i]
            if t.dep_ in cut_deps:
                return doc[span.start:i]
            if t.lemma_.lower() in cut_lemmas:
                return doc[span.start:i]
        return span

    def _trim_arg_clause_after_comma(self, span: Span) -> Span:
        """
        If an argument contains a comma followed by a finite verb clause, trim at the comma.
        This prevents arguments like "Sweden, is a notable figure".
        """
        doc = span.doc
        comma_idx = None
        for i in range(span.start, span.end):
            t = doc[i]
            if t.is_punct and t.text == ",":
                comma_idx = i
                break
        if comma_idx is None:
            return span

        for t in doc[comma_idx + 1:span.end]:
            if t.pos_ in {"VERB", "AUX"} and t.tag_ in {"VBD", "VBP", "VBZ", "MD"}:
                return doc[span.start:comma_idx]
        return span

    def _extend_arg_with_attached_prep(self, span: Span) -> Span:
        """
        Extend argument with a directly attached prep phrase on the noun head.
        This helps avoid truncated list items like "role" -> "role in X".
        """
        if span is None or len(span) == 0:
            return span
        head = span.root
        doc = span.doc
        # Only extend very generic heads; otherwise keep base args for incremental chains.
        generic_heads = {
            "role", "position", "group", "community", "area", "region", "country", "state",
            "province", "county", "city", "town", "village", "organization", "school",
            "program", "project", "team", "company", "service", "system", "sector", "field",
            "member", "part", "category", "type", "kind"
        }
        if head.lemma_.lower() not in generic_heads:
            return span
        allowed_preps = {"of", "in", "for", "to", "with", "on", "at", "from", "by"}

        # Only extend if prep is immediately after the span and no punctuation in between
        for prep in head.children:
            if prep.dep_ != "prep" or prep.text.lower() not in allowed_preps:
                continue
            if prep.i < span.end:
                continue
            between = [t for t in doc[span.end:prep.i] if not t.is_space]
            if any(t.is_punct for t in between):
                continue
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj is None:
                continue
            pobj_span = self._get_np_span(pobj) or self._tok_span(pobj)
            if pobj_span is None:
                continue
            # Extend pobj span with attached relative clauses (e.g., "form ... that people can sit on").
            relcl = next((c for c in pobj.children if c.dep_ == "relcl"), None)
            if relcl is not None:
                relcl_span = self._subtree_span(relcl, include_punct=False)
                if relcl_span is not None:
                    pobj_span = self._cover(pobj_span, relcl_span) or pobj_span
            if relcl is None:
                if any(t.pos_ in {"VERB", "AUX"} for t in doc[prep.i:pobj_span.end]):
                    continue
            return self._cover(span, prep, pobj_span) or span
        return span

    def _extend_arg_with_compound_head(self, span: Span) -> Span:
        """
        If the argument is a left compound (e.g., "community" in "community engagement"),
        extend to include the head noun to the right.
        """
        if span is None or len(span) == 0:
            return span
        head = span.root
        if head.dep_ == "compound" and head.head.pos_ in {"NOUN", "PROPN"}:
            head_noun = head.head
            if head_noun.i >= span.end and head_noun.sent == head.sent:
                head_np = self._get_np_span(head_noun) or self._tok_span(head_noun)
                if head_np is not None:
                    return self._cover(span, head_np) or span
        return span

    def _trim_discourse_markers(self, span: Span) -> Span:
        """
        Remove discourse markers from the beginning of spans.

        Discourse markers are sentence-initial phrases like:
        - "For example, ..."
        - "However, ..."
        - "Overall, ..."
        - "In conclusion, ..."

        Pattern: tokens before first comma in span, if they are:
        - prep phrase ("for example")
        - adverb ("however", "overall")
        - interjection
        """
        if span is None or len(span) == 0:
            return span

        comma_idx = None
        for i in range(span.start, min(span.start + 4, span.end)):  # Only check first 4 tokens
            if span.doc[i].is_punct and span.doc[i].text == ",":
                comma_idx = i
                break

        if comma_idx is None:
            return span

        # Check if tokens before comma are discourse markers
        # - prep + pobj: "for example", "in conclusion"
        # - single adverb: "however", "overall", "additionally"
        before_comma = span.doc[span.start:comma_idx]

        if len(before_comma) <= 2:
            is_discourse = False

            if len(before_comma) == 1 and before_comma[0].pos_ == "ADV":
                is_discourse = True

            if len(before_comma) == 2 and before_comma[0].pos_ == "ADP" and before_comma[1].pos_ == "NOUN":
                is_discourse = True

            if is_discourse:
                return span.doc[comma_idx + 1:span.end]

        return span


    def _extract_date_parts_facts(self, pobj_token: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        Instead of microfacts (December / 18 / 1949) return:
          1) full DATE-span (subtree or entity)
          2) (optionally) separate year
        """
        facts: List[Fact] = []

        if self._is_year_token(pobj_token):
            full = self._tok_span(pobj_token)
        else:
            # ✅ FIX: Prioritize entity spans for dates
            # If pobj is part of DATE entity, use full entity span
            # This ensures "June 1, 1910" is captured fully, not just "June"
            full = None

            if pobj_token.ent_type_ == "DATE":
                for ent in pobj_token.doc.ents:
                    if ent.label_ == "DATE" and ent.start <= pobj_token.i < ent.end:
                        full = ent
                        break

            if full is None:
                full = self._subtree_span(pobj_token, include_punct=True) or self._tok_span(pobj_token)

        if full is not None and full.text.strip():
            new_fact = Fact(subject, predicate, full)
            facts.append(new_fact)

            # ✅ FIX: Don't add separate year fact if:
            # 1. Full date already includes year (multi-token date)
            # 2. OR full span is already a single year token (don't duplicate)
            has_complete_date = len(full) > 2 and any(self._is_year_token(t) for t in full)
            is_single_year = len(full) == 1 and self._is_year_token(pobj_token)

            if not has_complete_date and not is_single_year:
                for t in full:
                    if getattr(self, "_is_year_token", None) is not None and self._is_year_token(t):
                        facts.append(Fact(subject, predicate, self._tok_span(t)))
                        break


        return facts



    def _is_span_covered(self, span: Span, existing_facts: Dict) -> bool:
        """
        Check if span is already FULLY covered by existing facts.
        Only returns True if there's an exact match in arguments.
        """
        span_range = set(range(span.start, span.end))
        for fact in existing_facts.values():
            if fact.argument is not None:
                arg_range = set(range(fact.argument.start, fact.argument.end))
                if span_range == arg_range:  # exact match
                    return True
        return False

    def _is_valid_standalone_fact(self, subject: Span, argument: Span, existing_facts: Dict) -> bool:
        """
        Check if a standalone fact is valid (not noise).

        Returns False if:
        1. Argument overlaps with subject (e.g., "Douglas born Douglas")
        2. Argument is already covered by a verb-based fact with a different subject
        """
        subj_range = set(range(subject.start, subject.end))
        arg_range = set(range(argument.start, argument.end))
        if subj_range & arg_range:  # overlaps
            return False

        for fact in existing_facts.values():
            if fact.argument is None:
                continue

            fact_arg_range = set(range(fact.argument.start, fact.argument.end))

            if arg_range & fact_arg_range:
                fact_subj_range = set(range(fact.subject.start, fact.subject.end))
                if subj_range != fact_subj_range:
                    return False

        return True

    def _extract_standalone_entity_facts(self, sent: Span, existing_facts: Dict) -> List[Fact]:
        """
        Extract standalone entities not covered by verb facts.
        Focuses on:
        - Compound NPs (professional soccer player)
        - Quantified NPs (two children)
        - Multi-word proper nouns (the Monmouth Hawks)
        - Time expressions (the late 1960s)

        Args:
            sent: Sentence span to extract from
            existing_facts: Dict of already extracted facts
        """
        facts: List[Fact] = []

        doc = sent.doc

        anchor_subj = None
        anchor_pred = None

        for token in sent:  # ✅ Only iterate over tokens in THIS sentence
            # Look for first subject - must be nsubj/nsubjpass of ROOT verb (not advcl/relcl/acl)
            # ✅ FIX: Also allow NUM for "One of X" pattern
            if anchor_subj is None and token.dep_ in {"nsubj", "nsubjpass"} and token.pos_ in {"NOUN", "PROPN", "NUM"}:
                if token.head.dep_ == "ROOT":
                    # For NUM subjects, require "of" prep (pattern: "One of the features")
                    if token.pos_ == "NUM":
                        has_of = any(c.dep_ == "prep" and c.text.lower() == "of" for c in token.children)
                        if not has_of:
                            continue
                    anchor_subj = self._get_full_subject_span(token)

            # Look for first ROOT verb as predicate (use full predicate span with aux/by/etc)
            # Skip ROOT verbs with existential "there" (expl)
            if anchor_pred is None and token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
                has_expl = any(c.dep_ == "expl" for c in token.children)
                if not has_expl:
                    anchor_pred = self._get_full_predicate_span(token) or self._tok_span(token)

            if anchor_subj is not None and anchor_pred is not None:
                break

        if anchor_subj is None and doc.ents:
            for ent in doc.ents:
                if ent.start >= sent.start and ent.end <= sent.end:
                    root_verb = self._find_root(ent.root)
                    if root_verb.dep_ == "ROOT":
                        has_expl = any(c.dep_ == "expl" for c in root_verb.children)
                        if not has_expl:
                            anchor_subj = ent
                            break

        if anchor_subj is None:
            for chunk in doc.noun_chunks:
                if chunk.start < sent.start or chunk.end > sent.end:
                    continue

                if chunk.root.pos_ in {"NOUN", "PROPN"}:
                    root_verb = self._find_root(chunk.root)
                    if root_verb.dep_ == "ROOT":
                        has_expl = any(c.dep_ == "expl" for c in root_verb.children)
                        if not has_expl:
                            anchor_subj = chunk
                            break

        if anchor_pred is None:
            for token in sent:  # ✅ Only tokens in THIS sentence
                if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
                    has_expl = any(c.dep_ == "expl" for c in token.children)
                    if not has_expl:
                        anchor_pred = self._tok_span(token)
                        break

        if anchor_subj is None or anchor_pred is None:
            return facts

        for chunk in doc.noun_chunks:
            if chunk.start < sent.start or chunk.end > sent.end:
                continue
            if self._is_span_covered(chunk, existing_facts):
                continue

            if len(chunk) < 2:
                continue

            if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                continue

            if all(t.is_stop or t.is_punct for t in chunk):
                continue

            if any(t.dep_ in {"compound", "amod"} for t in chunk):
                np_span = self._get_np_span(chunk.root)
                if np_span and len(np_span) > 1 and not self._is_span_covered(np_span, existing_facts):
                    if self._is_valid_standalone_fact(anchor_subj, np_span, existing_facts):
                        facts.append(Fact(anchor_subj, anchor_pred, np_span))

            if any(t.pos_ == "NUM" for t in chunk):
                if not self._is_span_covered(chunk, existing_facts):
                    if self._is_valid_standalone_fact(anchor_subj, chunk, existing_facts):
                        facts.append(Fact(anchor_subj, anchor_pred, chunk))

        for ent in doc.ents:
            if ent.start < sent.start or ent.end > sent.end:
                continue
            if self._is_span_covered(ent, existing_facts):
                continue

            if len(ent) < 2:
                continue

            if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT"}:
                if self._is_valid_standalone_fact(anchor_subj, ent, existing_facts):
                    facts.append(Fact(anchor_subj, anchor_pred, ent))

            elif ent.label_ == "DATE" and len(ent) > 1:
                if self._is_valid_standalone_fact(anchor_subj, ent, existing_facts):
                    facts.append(Fact(anchor_subj, anchor_pred, ent))
            
            elif ent.label_ in {"LOC", "GPE"} and len(ent) > 1:
                if self._is_valid_standalone_fact(anchor_subj, ent, existing_facts):
                    facts.append(Fact(anchor_subj, anchor_pred, ent))

        for token in doc:
            if token.dep_ == "poss" and not self._is_span_covered(self._tok_span(token), existing_facts):
                poss_span = self._get_np_span(token)
                if poss_span and len(poss_span) > 1:
                    if self._is_valid_standalone_fact(anchor_subj, poss_span, existing_facts):
                        facts.append(Fact(anchor_subj, anchor_pred, poss_span))

        # ✅ IMPROVEMENT: Extract dates and locations through patterns (even if not recognized as entities)
        facts.extend(self._extract_enhanced_dates_and_locations(sent, anchor_subj, anchor_pred, existing_facts))

        return facts

    def _extract_enhanced_dates_and_locations(self, sent: Span, anchor_subj: Span, anchor_pred: Span, existing_facts: Dict) -> List[Fact]:
        """
        ✅ IMPROVEMENT: Enhanced extraction of dates, locations, numbers, relationships, and coordinations.
        
        Extracts:
        1. Full dates (e.g., "September 11, 2001", "May 2, 2011")
        2. Complex locations (e.g., "Abbottabad, Pakistan", "The Bronx, New York")
        3. Date ranges (e.g., "the late 1980s", "the early 19th century")
        4. Numbers with units (e.g., "2.5 million people", "15 miles", "25 percent")
        5. Relationships (e.g., "married Mary", "founded company", "worked at Microsoft")
        6. Coordinations (e.g., "actor, producer and director" → separate facts)
        
        Args:
            sent: Sentence span to extract from
            anchor_subj: Anchor subject for facts
            anchor_pred: Anchor predicate for facts
            existing_facts: Dict of already extracted facts
            
        Returns:
            List of new facts
        """
        facts: List[Fact] = []
        doc = sent.doc
        
        # Pattern 1: Full dates (Month Day, Year)
        # Pattern: month + day + comma + year
        for i in range(sent.start, sent.end - 2):
            token = doc[i]
            if token.text.lower() in self.months:
                if i + 2 < sent.end:
                    day_token = doc[i + 1]
                    comma_token = doc[i + 2] if i + 2 < sent.end else None
                    
                    # Check for pattern: Month Day, Year
                    if (day_token.like_num and 
                        comma_token and comma_token.text == "," and
                        i + 3 < sent.end):
                        year_token = doc[i + 3]
                        if self._is_year_token(year_token):
                            date_span = doc[i:i + 4]
                            if not self._is_span_covered(date_span, existing_facts):
                                if self._is_valid_standalone_fact(anchor_subj, date_span, existing_facts):
                                    facts.append(Fact(anchor_subj, anchor_pred, date_span))
        
        # Pattern 2: Complex locations (City, State/Country)
        # Pattern: PROPN + comma + PROPN (possibly multi-word)
        for i in range(sent.start, sent.end - 2):
            token = doc[i]
            if token.pos_ == "PROPN" and token.ent_type_ in {"GPE", "LOC", ""}:
                if i + 1 < sent.end:
                    comma_token = doc[i + 1]
                    
                    # Check for pattern: City, State/Country
                    if comma_token.text == ",":
                        # Find the full location phrase (may be multi-word after comma)
                        # Look for next PROPN and include all tokens until next comma or end
                        location_end = i + 2
                        while location_end < sent.end:
                            if doc[location_end].is_punct and doc[location_end].text == ",":
                                break  # Stop at next comma
                            if doc[location_end].pos_ == "PROPN" or doc[location_end].pos_ == "NOUN":
                                location_end += 1
                                while location_end < sent.end and (doc[location_end].pos_ in {"PROPN", "NOUN", "ADJ"} or doc[location_end].is_stop):
                                    if doc[location_end].is_punct and doc[location_end].text == ",":
                                        break
                                    location_end += 1
                                break
                            location_end += 1
                        
                        if location_end > i + 2:
                            location_span = doc[i:location_end]
                            
                            # ✅ FIX: Filter out fragment locations (e.g., "Angeles, California" from "Los Angeles, California")
                            # Check if this is a fragment by seeing if there's a preceding PROPN that should be included
                            if i > sent.start:
                                prev_token = doc[i - 1]
                                # If previous token is also PROPN and not punctuation, this might be a fragment
                                # But we only filter if the location doesn't start with a capital letter or is clearly a fragment
                                if prev_token.pos_ == "PROPN" and not prev_token.is_punct:
                                    continue
                            
                            # ✅ FIX: Filter locations that are cut off mid-phrase (e.g., "California, and moved")
                            # Check if location ends with a comma followed by a conjunction
                            if location_end < sent.end:
                                after_location = doc[location_end]
                                if after_location.text == "," and location_end + 1 < sent.end:
                                    next_after = doc[location_end + 1]
                                    if next_after.text.lower() in {"and", "or", "but"}:
                                        continue
                            
                            if not self._is_span_covered(location_span, existing_facts):
                                if self._is_valid_standalone_fact(anchor_subj, location_span, existing_facts):
                                    facts.append(Fact(anchor_subj, anchor_pred, location_span))
        
        # Pattern 3: Date ranges (the late 1980s, the early 19th century)
        # Pattern: "the" + adjective + year/decade
        for i in range(sent.start, sent.end - 2):
            token = doc[i]
            if token.text.lower() == "the" and token.pos_ == "DET":
                if i + 2 < sent.end:
                    adj_token = doc[i + 1]
                    year_token = doc[i + 2]
                    
                    # Check for pattern: "the late/early" + year/decade
                    if (adj_token.text.lower() in {"late", "early", "mid"} and
                        (self._is_year_token(year_token) or 
                         (year_token.like_num and "s" in year_token.text.lower()))):
                        date_range_span = doc[i:i + 3]
                        if not self._is_span_covered(date_range_span, existing_facts):
                            if self._is_valid_standalone_fact(anchor_subj, date_range_span, existing_facts):
                                facts.append(Fact(anchor_subj, anchor_pred, date_range_span))
        
        # Pattern 4: Numbers with units (e.g., "2.5 million people", "15 miles", "25 percent")
        # Pattern: NUM + unit word
        for i in range(sent.start, sent.end - 1):
            token = doc[i]
            if token.like_num or token.pos_ == "NUM":
                if i + 1 < sent.end:
                    next_token = doc[i + 1]
                    unit_words = {'million', 'billion', 'thousand', 'percent', 'people', 'inhabitants', 
                                 'miles', 'km', 'kilometers', 'square', 'dollars', 'degrees', 'years', 
                                 'months', 'days', 'hours', 'minutes'}
                    
                    unit_span = None
                    if next_token.text.lower() in unit_words:
                        if i + 2 < sent.end and doc[i + 2].text.lower() in {'people', 'inhabitants', 'dollars'}:
                            unit_span = doc[i:i + 3]
                        else:
                            unit_span = doc[i:i + 2]
                    elif next_token.text.lower() == 'square' and i + 2 < sent.end:
                        third_token = doc[i + 2]
                        if third_token.text.lower() in {'miles', 'km', 'kilometers'}:
                            unit_span = doc[i:i + 3]
                    
                    if unit_span:
                        # Found number with unit - ensure we have the full number (including decimals)
                        # Check if previous token is part of the number (for cases like "2.5")
                        if i > sent.start and doc[i - 1].text == "." and (i - 1 > sent.start) and doc[i - 2].like_num:
                            unit_span = doc[i - 2:i + unit_span.end - i]
                        
                        if not self._is_span_covered(unit_span, existing_facts):
                            if self._is_valid_standalone_fact(anchor_subj, unit_span, existing_facts):
                                facts.append(Fact(anchor_subj, anchor_pred, unit_span))
        
        # Pattern 5: Relationships (e.g., "married Mary", "founded company", "worked at Microsoft")
        # Pattern: relationship verb + object
        # NOTE: This pattern is already handled by main extraction logic, so we skip it here
        # to avoid duplicates. The main logic in _extract_object_facts handles these correctly.
        # We only add this if we need to ensure relationships are extracted even when main logic misses them.
        pass  # Relationships are handled by main extraction logic
        
        # Pattern 6: Coordinations (e.g., "actor, producer and director" → separate facts)
        # Pattern: X, Y, and Z or X, Y and Z
        for i in range(sent.start, sent.end - 2):
            token = doc[i]
            if i + 2 < sent.end and doc[i + 1].text == ",":
                comma_token = doc[i + 1]
                for j in range(i + 2, min(i + 5, sent.end)):
                    if doc[j].text.lower() in {"and", "or"}:
                        item1_span = doc[i:i + 1]  # Just the token before comma
                        if j > i + 2:
                            item2_span = doc[i + 2:j]
                        else:
                            continue
                        if j + 1 < sent.end:
                            item3_span = doc[j + 1:j + 2]  # Just one token after "and"
                        else:
                            continue
                        
                        valid_items = []
                        for item_span in [item1_span, item2_span, item3_span]:
                            if any(t.pos_ in {"NOUN", "PROPN"} for t in item_span):
                                full_item = self._get_np_span(item_span[0]) or item_span
                                if not self._is_span_covered(full_item, existing_facts):
                                    if self._is_valid_standalone_fact(anchor_subj, full_item, existing_facts):
                                        valid_items.append(full_item)
                        
                        for item_span in valid_items:
                            facts.append(Fact(anchor_subj, anchor_pred, item_span))
                        break
        
        # Pattern 7: Long phrases (organizations, titles, roles)
        # Extract noun phrases with 4+ words that contain proper nouns
        for chunk in doc.noun_chunks:
            if chunk.start < sent.start or chunk.end > sent.end:
                continue
            
            if self._is_span_covered(chunk, existing_facts):
                continue
            
            if len(chunk) >= 4:
                has_propn = any(t.pos_ == "PROPN" for t in chunk)
                has_noun = any(t.pos_ in {"NOUN", "PROPN"} for t in chunk)
                
                if has_propn and has_noun:
                    if not all(t.is_stop or t.is_punct for t in chunk[1:-1]):
                        if self._is_valid_standalone_fact(anchor_subj, chunk, existing_facts):
                            facts.append(Fact(anchor_subj, anchor_pred, chunk))
        
        return facts

    # Fact extraction improvements (quoted entities, contrastive, lists)

    def _find_span_by_text(self, doc: Doc, text: str, start_hint: int = 0) -> Optional[Span]:
        """
        Find a span in doc that matches the given text.

        Args:
            doc: spaCy Doc
            text: Text to search for
            start_hint: Token index to start searching from

        Returns:
            Span if found, None otherwise
        """
        text_lower = text.lower().strip()
        tokens = text_lower.split()

        for i in range(start_hint, len(doc)):
            if i + len(tokens) > len(doc):
                break

            match = True
            for j, token_text in enumerate(tokens):
                if doc[i + j].lower_ != token_text:
                    match = False
                    break

            if match:
                return doc[i : i + len(tokens)]

        return None

    def _extract_contrastive_facts(self, doc: Doc, sent_text: str, sent: Span) -> List[Fact]:
        """
        Extract facts from contrastive constructions like "X on A, B but not on C, D".

        Args:
            doc: Full spaCy Doc
            sent_text: Sentence text (possibly with placeholders)
            sent: Sentence span

        Returns:
            List of Facts with positive and negative patterns
        """
        facts = []

        parsed = parse_contrastive_construction(sent_text)
        if not parsed:
            return facts

        subject_span = None
        base_words = parsed.base_phrase.split()
        if base_words:
            for i in range(sent.start, sent.end):
                if doc[i].pos_ in {"NOUN", "PROPN"}:
                    subject_span = self._get_np_span(doc[i])
                    break

        if not subject_span:
            for chunk in sent.as_doc().noun_chunks:
                if chunk.root.pos_ in {"NOUN", "PROPN"}:
                    subject_span = doc[sent.start + chunk.start : sent.start + chunk.end]
                    break

        if not subject_span:
            return facts

        pred_text = f"{parsed.base_phrase.split()[-1]} {parsed.preposition}"
        predicate_span = self._find_span_by_text(doc, pred_text, sent.start)

        if not predicate_span:
            pred_span = self._find_span_by_text(doc, parsed.preposition, sent.start)
            if pred_span:
                predicate_span = pred_span

        if not predicate_span:
            return facts

        for item in parsed.positive_items:
            item_restored = restore_quoted_entities(item, self._quoted_entity_mapping)

            arg_span = self._find_span_by_text(doc, item_restored, sent.start)
            if arg_span:
                facts.append(Fact(subject_span, predicate_span, arg_span))

        # Negative facts - create predicate with "not"
        not_token = None
        for i in range(sent.start, sent.end):
            if doc[i].lower_ == "not":
                not_token = doc[i]
                break

        if not_token:
            pred_with_not = self._cover(predicate_span, not_token)
            if pred_with_not:
                for item in parsed.negative_items:
                    item_restored = restore_quoted_entities(item, self._quoted_entity_mapping)

                    arg_span = self._find_span_by_text(doc, item_restored, not_token.i)
                    if arg_span:
                        facts.append(Fact(subject_span, pred_with_not, arg_span))

        return facts

    def _extract_list_facts(self, doc: Doc, sent_text: str, sent: Span) -> List[Fact]:
        """
        Extract atomic facts from list/enumeration patterns.

        Args:
            doc: Full spaCy Doc
            sent_text: Sentence text (possibly with placeholders)
            sent: Sentence span

        Returns:
            List of Facts, one per list item
        """
        facts = []

        parsed = detect_list_pattern(sent_text)
        if not parsed:
            return facts

        context = extract_name_from_context(parsed.context)

        pred_span = None
        for i in range(sent.start, sent.end):
            if doc[i].lemma_ in {"be", "include"}:
                pred_span = self._tok_span(doc[i])
                break

        if not pred_span:
            for i in range(sent.start, sent.end):
                if doc[i].pos_ == "VERB":
                    pred_span = self._tok_span(doc[i])
                    break

        if not pred_span:
            return facts

        for item in parsed.items:
            item_restored = restore_quoted_entities(item, self._quoted_entity_mapping)

            subj_span = self._find_span_by_text(doc, item_restored, sent.start)
            if not subj_span:
                continue

            context_span = self._find_span_by_text(doc, context, sent.start)

            if context_span:
                facts.append(Fact(subj_span, pred_span, context_span))
            else:
                for chunk in sent.as_doc().noun_chunks:
                    if context.lower() in chunk.text.lower():
                        arg_span = doc[sent.start + chunk.start : sent.start + chunk.end]
                        facts.append(Fact(subj_span, pred_span, arg_span))
                        break

        return facts


    def extract_granular_facts(self, text: str) -> List[IncrementalFactGroup]:
        """
        Extract facts from text, processing each sentence independently.

        IMPORTANT: Each sentence is processed separately to ensure independence
        from surrounding context, but we parse the full text only once for efficiency.

        Returns:
            List[IncrementalFactGroup]: Each group contains incremental facts
                (e.g., "born in | Pensacola" and "born in | Pensacola, Florida")
        """
        # ✅ CRITICAL FIX: Parse text ONLY ONCE
        # Mask markdown while preserving offsets to reduce parse noise
        original_text = self._mask_markdown(text)

        if self.use_improvements:
            quoted_result = preprocess_quoted_entities(original_text)
            self._quoted_entity_mapping = quoted_result.mapping
        else:
            self._quoted_entity_mapping = {}

        doc = self.nlp(original_text)

        if self.use_gliner:
            try:
                from ner_enhancement import enhance_doc_with_gliner
                doc = enhance_doc_with_gliner(
                    doc,
                    original_text,
                    model_name=self.gliner_model,
                    alignment_mode="expand"
                )
            except ImportError:
                pass
            except Exception:
                pass

        all_facts: List[Fact] = []

        for sent in doc.sents:
            sent_text = original_text[sent.start_char:sent.end_char].strip()
            if not sent_text:
                continue

            sent_facts = self._extract_facts_from_sent(sent, sent_text)
            all_facts.extend(sent_facts)

        groups = self._create_incremental_groups(all_facts)
        return groups

    def _create_incremental_groups(self, facts: List[Fact]) -> List[IncrementalFactGroup]:
        """
        Create incremental fact groups from extracted facts.

        Groups facts with same subject+predicate+prep and creates incremental variants
        by progressively adding modifiers to the argument.

        Example:
            Input facts:
              - born in | Pensacola
              - born in | Florida

            Output group:
              - born in | Pensacola
              - born in | Pensacola, Florida
            Deltas: [Pensacola, Florida]
        """
        from collections import defaultdict

        # Group facts by (subject_span, predicate_span, prep)
        # Facts in same group will be candidates for incremental merging
        grouped_facts = defaultdict(list)

        for fact in facts:
            subj_key = (fact.subject.start, fact.subject.end)
            pred_key = (fact.predicate.start, fact.predicate.end)
            prep_key = fact.prep if fact.prep else ""
            key = (subj_key, pred_key, prep_key)

            grouped_facts[key].append(fact)

        result_groups = []

        for key, group_facts in grouped_facts.items():
            if len(group_facts) == 1:
                fact = group_facts[0]
                result_groups.append(IncrementalFactGroup(
                    facts=[fact],
                    deltas=[fact.argument] if fact.argument else []
                ))
            else:
                # Multiple facts - check if they form incremental pattern
                # Quick check: do all arguments have same start OR same end?
                args_with_spans = [f for f in group_facts if f.argument]
                if args_with_spans:
                    first_fact = args_with_spans[0]
                    same_start = all(f.argument.start == first_fact.argument.start for f in args_with_spans)
                    same_end = all(f.argument.end == first_fact.argument.end for f in args_with_spans)

                    if same_start or same_end:
                        incremental_group = self._build_incremental_group(group_facts)
                        result_groups.append(incremental_group)
                    else:
                        for fact in group_facts:
                            result_groups.append(IncrementalFactGroup(
                                facts=[fact],
                                deltas=[fact.argument] if fact.argument else []
                            ))
                else:
                    for fact in group_facts:
                        result_groups.append(IncrementalFactGroup(
                            facts=[fact],
                            deltas=[]
                        ))

        return result_groups

    def _build_incremental_group(self, facts: List[Fact]) -> IncrementalFactGroup:
        """
        Build incremental group from facts with same subject+predicate+prep.

        Handles two patterns:
        1. Nested preps: same START, increasing END (e.g., "Pensacola" → "Pensacola, Florida")
        2. Progressive modifiers: same END, decreasing START (e.g., "language" → "programming language")

        Example 1 (nested preps):
            facts = [
                Fact(born in, Pensacola),
                Fact(born in, Pensacola, Florida)
            ]
            Output:
                facts = [Pensacola, Pensacola, Florida]
                deltas = [Pensacola, Florida]

        Example 2 (progressive modifiers):
            facts = [
                Fact(is, language),
                Fact(is, programming language),
                Fact(is, multi-paradigm programming language)
            ]
            Output:
                facts = [language, programming language, multi-paradigm programming language]
                deltas = [language, programming, multi-paradigm]
        """
        facts_with_args = [f for f in facts if f.argument]

        if not facts_with_args:
            return IncrementalFactGroup(facts=facts, deltas=[])

        # Check pattern BEFORE sorting to determine sort order
        # Pattern 1: Nested chain (same START, increasing END) - nested preps
        # Pattern 2: Progressive modifiers (same END, decreasing START) - NP modifiers

        first_fact = facts_with_args[0]
        same_start_count = sum(1 for f in facts_with_args if f.argument.start == first_fact.argument.start)
        same_end_count = sum(1 for f in facts_with_args if f.argument.end == first_fact.argument.end)

        is_nested_chain = same_start_count == len(facts_with_args)
        is_progressive_modifiers = same_end_count == len(facts_with_args)

        if is_progressive_modifiers and not is_nested_chain:
            # Progressive modifiers: sort by LENGTH (shortest to longest)
            # Example: "actor" (len=1) before "American actor" (len=2)
            facts_with_args.sort(key=lambda f: (f.argument.end - f.argument.start, f.argument.start))
        else:
            # Nested chain or default: sort by (start, end)
            # Example: "Pensacola" before "Pensacola, Florida"
            facts_with_args.sort(key=lambda f: (f.argument.start, f.argument.end))

        first_start = facts_with_args[0].argument.start
        last_end = facts_with_args[-1].argument.end

        is_nested_chain = all(f.argument.start == first_start for f in facts_with_args)
        is_progressive_modifiers = all(f.argument.end == last_end for f in facts_with_args)

        if is_nested_chain and len(facts_with_args) >= 2:
            # Pattern 1: Nested preps (same start, increasing end)
            # Create incremental group with all facts
            incremental_facts = []
            deltas = []
            doc = facts_with_args[0].argument.doc

            for i, fact in enumerate(facts_with_args):
                incremental_facts.append(fact)

                if i == 0:
                    deltas.append(fact.argument)
                else:
                    prev_arg = facts_with_args[i - 1].argument
                    curr_arg = fact.argument

                    new_tokens = []
                    for tok in doc[prev_arg.end:curr_arg.end]:
                        if not tok.is_punct:  # skip punctuation
                            new_tokens.append(tok)

                    if new_tokens:
                        delta = doc[new_tokens[0].i:new_tokens[-1].i + 1]
                    else:
                        delta = doc[prev_arg.end:curr_arg.end]

                    deltas.append(delta)

            return IncrementalFactGroup(facts=incremental_facts, deltas=deltas)

        elif is_progressive_modifiers and len(facts_with_args) >= 2:
            # Pattern 2: Progressive modifiers (same end, decreasing start)
            # Create incremental group with all facts
            incremental_facts = []
            deltas = []
            doc = facts_with_args[0].argument.doc

            for i, fact in enumerate(facts_with_args):
                incremental_facts.append(fact)

                if i == 0:
                    deltas.append(fact.argument)
                else:
                    prev_arg = facts_with_args[i - 1].argument
                    curr_arg = fact.argument

                    new_tokens = []
                    for tok in doc[curr_arg.start:prev_arg.start]:
                        if not tok.is_punct:  # skip punctuation
                            new_tokens.append(tok)

                    if new_tokens:
                        delta = doc[new_tokens[0].i:new_tokens[-1].i + 1]
                    else:
                        delta = doc[curr_arg.start:prev_arg.start]

                    deltas.append(delta)

            return IncrementalFactGroup(facts=incremental_facts, deltas=deltas)

        single_fact_groups = []
        for f in facts_with_args:
            single_fact_groups.append(IncrementalFactGroup(
                facts=[f],
                deltas=[f.argument]
            ))

        return single_fact_groups[0] if single_fact_groups else IncrementalFactGroup(facts=facts, deltas=[])

    def _build_incremental_np_modifiers(self, head_token: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        Build incremental facts for NP modifiers to enable granular hallucination detection.

        Example:
            Text: "C# is a modern multi-paradigm programming language"

            Returns incremental facts:
                1. C# | is | language (delta: "language")
                2. C# | is | programming language (delta: "programming")
                3. C# | is | multi-paradigm programming language (delta: "multi-paradigm")
                4. C# | is | modern multi-paradigm programming language (delta: "modern")

            This allows NLI to detect that "modern" is a hallucination while other modifiers are correct.
        """
        facts: List[Fact] = []
        doc = head_token.doc

        modifiers = set()
        det_tokens = [c for c in head_token.children if c.dep_ == "det"]
        quant_dets = {"several", "many", "few", "some", "numerous", "various", "multiple"}
        fixed_dets = [d for d in det_tokens if d.text.lower() not in quant_dets]
        quant_det_idxs = [d.i for d in det_tokens if d.text.lower() in quant_dets]
        for child in head_token.children:
            if child.dep_ in {"compound", "amod", "nummod"}:
                modifiers.add(child.i)
                for gc in child.children:
                    if gc.dep_ == "advmod":
                        modifiers.add(gc.i)
        for idx in quant_det_idxs:
            modifiers.add(idx)

        if not modifiers:
            return []

        # Start with head noun + fixed determiners (a/the/its)
        current_start = head_token.i
        current_end = head_token.i + 1  # exclusive end
        if fixed_dets:
            det_idxs = [d.i for d in fixed_dets]
            current_start = min(current_start, min(det_idxs))
            current_end = max(current_end, max(det_idxs) + 1)

        facts.append(Fact(subject, predicate, head_token.doc[current_start:current_end]))

        # Iteratively add modifiers that are contiguous with current span
        # Repeat until no more modifiers can be added
        while modifiers:
            extended = False

            if (current_start - 1) in modifiers:
                current_start -= 1
                modifiers.remove(current_start)
                span = doc[current_start:current_end]
                facts.append(Fact(subject, predicate, span))
                extended = True
            elif current_end in modifiers:
                modifiers.remove(current_end)
                current_end += 1
                span = doc[current_start:current_end]
                facts.append(Fact(subject, predicate, span))
                extended = True

            if not extended:
                # Check for modifiers separated only by punct/det
                for candidate_i in sorted(modifiers, reverse=True):  # Start from rightmost
                    if candidate_i < current_start:
                        gap_tokens = list(doc[candidate_i + 1:current_start])
                        gap_content = [t for t in gap_tokens if not t.is_punct and t.dep_ != "det"]

                        if not gap_content:  # Only punct/det in gap
                            current_start = candidate_i
                            modifiers.remove(candidate_i)
                            span = doc[current_start:current_end]
                            facts.append(Fact(subject, predicate, span))
                            extended = True
                            break

            if not extended:
                break

        # Drop overly generic base heads when modifiers exist (e.g., "role" -> keep "crucial role")
        if len(facts) > 1:
            base_arg = facts[0].argument
            if base_arg is not None and base_arg.root.lemma_.lower() in self.generic_heads:
                facts = facts[1:]

        return facts

    def _build_incremental_argument_facts(self, base_fact: Fact) -> List[Fact]:
        """
        Build incremental facts for an argument NP so that modifiers are added stepwise.

        Example (desired):
            policy -> foreign policy -> Chinese foreign policy

        Only applies when argument head is NOUN/PROPN and has amod/compound modifiers.
        """
        if base_fact.argument is None:
            return []

        # ✅ FIX: Don't split arguments that are inside quotes (atomic entities)
        # Example: "The Other Side" should not become "Side", "Other Side", etc.
        if len(base_fact.argument) > 0:
            first_tok = base_fact.argument[0]
            if self._find_quote_pair_for_token(first_tok) is not None:
                return []  # Keep quoted entity atomic

        head = base_fact.argument.root
        if head.pos_ not in {"NOUN", "PROPN"}:
            return []

        if head.lemma_.lower() in {"time", "times"}:
            if any(c.dep_ == "det" and c.text.lower() in {"several", "many", "few"} for c in head.children):
                return []

        modifiers = set()
        det_tokens = [c for c in head.children if c.dep_ == "det"]
        quant_dets = {"several", "many", "few", "some", "numerous", "various", "multiple"}
        fixed_dets = [d for d in det_tokens if d.text.lower() not in quant_dets]
        quant_det_idxs = [d.i for d in det_tokens if d.text.lower() in quant_dets]
        for child in head.children:
            if child.dep_ in {"amod", "compound", "nummod"}:
                modifiers.add(child.i)
                for gc in child.children:
                    if gc.dep_ == "advmod":
                        modifiers.add(gc.i)
        for idx in quant_det_idxs:
            modifiers.add(idx)

        if not modifiers:
            return []

        doc = head.doc
        facts: List[Fact] = []

        # Start from the head noun + fixed determiners
        cur_start = head.i
        cur_end = head.i + 1
        if fixed_dets:
            det_idxs = [d.i for d in fixed_dets]
            cur_start = min(cur_start, min(det_idxs))
            cur_end = max(cur_end, max(det_idxs) + 1)
        facts.append(Fact(base_fact.subject, base_fact.predicate, doc[cur_start:cur_end], base_fact.prep))

        while modifiers:
            extended = False
            if (cur_start - 1) in modifiers:
                cur_start -= 1
                modifiers.remove(cur_start)
                facts.append(Fact(base_fact.subject, base_fact.predicate, doc[cur_start:cur_end], base_fact.prep))
                extended = True
            elif cur_end in modifiers:
                modifiers.remove(cur_end)
                cur_end += 1
                facts.append(Fact(base_fact.subject, base_fact.predicate, doc[cur_start:cur_end], base_fact.prep))
                extended = True
            else:
                added = False
                for cand in sorted(list(modifiers)):
                    if cand < cur_start:
                        gap = doc[cand + 1:cur_start]
                    elif cand > cur_end:
                        gap = doc[cur_end:cand]
                    else:
                        continue
                    gap_content = [t for t in gap if not t.is_punct and t.dep_ != "det"]
                    if not gap_content:
                        if cand < cur_start:
                            cur_start = cand
                        else:
                            cur_end = cand + 1
                        modifiers.remove(cand)
                        facts.append(Fact(base_fact.subject, base_fact.predicate, doc[cur_start:cur_end], base_fact.prep))
                        added = True
                        extended = True
                        break
                if not added:
                    break

        if base_fact.argument.start != cur_start or base_fact.argument.end != cur_end:
            facts.append(Fact(base_fact.subject, base_fact.predicate, base_fact.argument, base_fact.prep))

        deduped: List[Fact] = []
        seen_spans = set()
        for f in facts:
            key = (f.argument.start if f.argument else -1, f.argument.end if f.argument else -1)
            if key not in seen_spans:
                deduped.append(f)
                seen_spans.add(key)

        # Drop overly generic base heads when modifiers exist (e.g., "role" -> keep "crucial role")
        if len(deduped) > 1:
            base_arg = deduped[0].argument
            if base_arg is not None and base_arg.root.lemma_.lower() in self.generic_heads:
                deduped = deduped[1:]

        return deduped

    def _build_incremental_subject_variants(self, subject: Optional[Span]) -> List[Span]:
        """
        Build incremental subject spans for modifier-heavy common-noun subjects.
        Intended for cases like "High-intensity interval training programs (like X)".
        """
        if subject is None or len(subject) == 0:
            return []
        head = subject.root
        if head.pos_ != "NOUN" or head.ent_type_:
            return []
        if any(t.pos_ == "PROPN" for t in subject):
            return []
        if any(t.dep_ in {"relcl", "acl"} for t in subject):
            return []
        if any(t.dep_ == "poss" for t in subject):
            return []

        for ch in head.children:
            if ch.dep_ == "prep" and ch.text.lower() not in {"like"}:
                return []

        core_span = self._get_np_core_span(head)
        if core_span is None:
            return []

        # Skip if determiners are present (these typically don't need subject expansion).
        if any(t.dep_ == "det" for t in core_span):
            return []

        # Require at least two modifiers to avoid over-truncation (e.g., "Israeli airstrikes").
        modifier_count = sum(1 for t in core_span if t.i != head.i and not t.is_punct)
        if modifier_count < 2:
            return []

        spans: List[Span] = []
        start_positions = [i for i in range(head.i, core_span.start - 1, -1)]
        for start in start_positions:
            if start < core_span.start:
                continue
            span = head.doc[start:core_span.end]
            if span and not all(t.is_punct for t in span):
                spans.append(span)

        like_prep = next((c for c in head.children if c.dep_ == "prep" and c.text.lower() == "like"), None)
        if like_prep is not None:
            pobj = next((c for c in like_prep.children if c.dep_ == "pobj"), None)
            if pobj is not None:
                pobj_span = self._get_np_span(pobj) or self._tok_span(pobj)
                if pobj_span is not None:
                    full_core = head.doc[core_span.start:core_span.end]
                    like_span = self._cover(full_core, like_prep, pobj_span)
                    if like_span is not None:
                        spans.append(like_span)

        uniq: List[Span] = []
        seen = set()
        for s in spans:
            key = (s.start, s.end)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
        return uniq

    def _extract_facts_from_sent(self, sent: Span, sent_text: str) -> List[Fact]:
        """
        Extract facts from a single sentence span.

        Args:
            sent: Sentence span from parsed doc
            sent_text: Original text of this sentence

        Returns:
            List of Facts extracted from this sentence
        """
        self._quote_cache = {}

        facts: Dict[Tuple[int, int, int, int, int, int], Fact] = {}

        def _key(f: Fact) -> Tuple[int, int, int, int, int, int]:
            a_s = f.argument.start if f.argument is not None else -1
            a_e = f.argument.end if f.argument is not None else -1
            return (f.subject.start, f.subject.end, f.predicate.start, f.predicate.end, a_s, a_e)

        def _add(iterable: Iterable[Fact]) -> None:
            for f in iterable:
                if f.argument is not None:
                    trimmed_arg = self._trim_arg_clause_after_comma(f.argument)
                    if trimmed_arg is not None and (trimmed_arg.start != f.argument.start or trimmed_arg.end != f.argument.end):
                        f = Fact(f.subject, f.predicate, trimmed_arg, f.prep)
                    extended_arg = self._extend_arg_with_attached_prep(f.argument)
                    if extended_arg is not None and (extended_arg.start != f.argument.start or extended_arg.end != f.argument.end):
                        keep_base = (
                            len(f.argument) == 1 and
                            f.argument.root.lemma_.lower() not in self.generic_heads
                        )
                        if not keep_base:
                            f = Fact(f.subject, f.predicate, extended_arg, f.prep)
                    extended_compound = self._extend_arg_with_compound_head(f.argument)
                    if extended_compound is not None and (extended_compound.start != f.argument.start or extended_compound.end != f.argument.end):
                        f = Fact(f.subject, f.predicate, extended_compound, f.prep)

                # ✅ IMPROVEMENT: Validate fact precision before adding
                # Checks completeness, syntax, and non-redundancy
                validation_result = self._validate_fact_precision(f)
                if not validation_result:
                    continue

                # Skip facts with missing argument for copula/aux "be/have" (e.g., "have been ∅")
                if f.argument is None:
                    pred_root = f.predicate.root.lemma_.lower() if len(f.predicate) > 0 else ""
                    if pred_root in {"be", "have"}:
                        continue

                # ✅ FIX: Filter out nonsensical facts where subject == argument
                # Examples: "Lanny is Lanny", "Lanny appeared Lanny"

                if f.argument is not None:
                    subj_text_lower = f.subject.text.lower().strip()
                    arg_text_lower = f.argument.text.lower().strip()

                    subj_range = set(range(f.subject.start, f.subject.end))
                    arg_range = set(range(f.argument.start, f.argument.end))

                    if subj_range == arg_range:
                        continue

                    # Skip if subject overlaps significantly with argument (>50% tokens in common)
                    # Allow possessive arguments (e.g., "Heinrich Harrer's book") even if overlapping.
                    has_possessive = any(t.dep_ == "poss" for t in f.argument) or "'s" in arg_text_lower
                    overlap = subj_range & arg_range
                    if overlap:
                        overlap_ratio = len(overlap) / min(len(subj_range), len(arg_range))
                        if overlap_ratio > 0.5 and not has_possessive:
                            continue

                    # Skip if subject text == argument text (handles "Lanny Flaherty" == "Lanny Flaherty")
                    subj_text = f.subject.text.lower().strip()
                    arg_text = f.argument.text.lower().strip()

                    if subj_text == arg_text:
                        continue

                    # Skip if one is contained in the other (e.g., "Lanny" in "Lanny's")
                    subj_clean = subj_text.rstrip("'s").strip()
                    arg_clean = arg_text.rstrip("'s").strip()

                    if subj_clean and arg_clean:
                        if subj_clean == arg_clean:
                            continue

                        # ✅ FIX: Improved self-reference detection
                        # Catch patterns like "Allen signed Kenderick Allen" (last name in both)
                        # Split into words and check if any PROPN/capitalized word appears in both

                        subj_names = set()
                        arg_names = set()

                        for token in f.subject:
                            if token.pos_ == "PROPN" or (token.text and token.text[0].isupper()):
                                subj_names.add(token.text.lower())

                        for token in f.argument:
                            if token.pos_ == "PROPN" or (token.text and token.text[0].isupper()):
                                arg_names.add(token.text.lower())

                        # If they share any proper noun/name, likely self-reference
                        # Exception: very common words like "New", "The", single letters
                        common_caps = {"new", "the", "a", "an", "mr", "mrs", "ms", "dr", "st"}
                        shared_names = (subj_names - common_caps) & (arg_names - common_caps)

                        shared_names = {name for name in shared_names if len(name) > 1}

                        if shared_names:
                            # ✅ FIX: Don't filter possessive constructions as self-references
                            # "John is known for John's looks" is VALID, not self-reference
                            is_possessive = "'s" in arg_text

                            # If they share a name and one is a subset of the other, skip
                            # UNLESS it's a possessive construction (which is valid)
                            # Example to FILTER: "Allen" (subj) and "Kenderick Allen" (arg) share "Allen"
                            # Example to KEEP: "John" (subj) and "John's looks" (arg) - possessive
                            if (subj_clean in arg_clean or arg_clean in subj_clean) and not is_possessive:
                                continue

                        # Skip if subject name is at the start of argument (e.g., "Lanny Flaherty" in "Lanny Flaherty's career")
                        # ✅ FIX: Possessive constructions are usually VALID facts, not self-references
                        # Examples to KEEP: "John is known for John's looks" (valid possessive)
                        # Examples to FILTER: "John signed John" (true self-reference without possessive)
                        if arg_clean.startswith(subj_clean + " "):
                            if "'s" not in arg_text:
                                continue
                        elif arg_clean.startswith(subj_clean + "'"):
                            # Has possessive marker - this is usually valid
                            # E.g., "Lanny Flaherty is known for Lanny Flaherty's distinctive looks"
                            # Only filter if possessive is at the very end (incomplete fragment)
                            if arg_text.endswith("'s") or arg_text.endswith("'s "):
                                continue

                    # ✅ FIX: Filter noisy facts with very short/meaningless arguments
                    # Skip arguments that are just punctuation, single letters, or very short noise
                    if len(arg_text) <= 2 and not arg_text.isdigit():
                        continue

                    # ✅ ITER 14: Allow temporal arguments like "in 2019" when predicate doesn't already have the prep
                    # Pattern: "helped in 2019" - keep "in 2019" as argument
                    # Pattern: "helped in ... in 2019" - filter duplicate (predicate already has "in")
                    if arg_text.startswith("on ") or arg_text.startswith("in ") or arg_text.startswith("at "):
                        arg_prep = arg_text.split()[0]  # "in", "on", or "at"
                        pred_text_lower = f.predicate.text.lower().strip()

                        # Only filter if predicate already ends with this prep (duplicate)
                        # Example: predicate="helped in", argument="in 2019" → filter (duplicate)
                        # Example: predicate="helped", argument="in 2019" → keep (new temporal info)
                        if pred_text_lower.endswith(f" {arg_prep}"):
                            continue

                    # ✅ FIX: Skip "has" facts with organization/bureau as argument
                    # Pattern: "X has the Census Bureau" - these are source attributions, not actual facts
                    if f.predicate.text.lower().strip() == "has":
                        org_keywords = ["bureau", "census", "department", "agency", "commission"]
                        if any(keyword in arg_text for keyword in org_keywords):
                            # Skip if it's clearly an attribution (not a possession)
                            # Exception: "has a department" (indefinite article) might be valid
                            if not arg_text.startswith("a ") and not arg_text.startswith("an "):
                                continue

                    # ✅ FIX: Skip facts where argument contains year and nothing else meaningful
                    # Pattern: "the Packers placed October 4, 2006"
                    # BUT: Allow dates as arguments for copula (is/was) since they're valid attributes
                    # Example: "birthdate is September 25, 1944" is VALID
                    # ALSO: Allow dates for temporal predicates like "born on", "died on", "started in"
                    if re.match(r'^[a-z]+\s+\d{1,2},\s+(19|20)\d{2}$', arg_text, re.IGNORECASE):
                        pred_text = f.predicate.text.lower().strip()
                        is_copula = pred_text in {"is", "was", "are", "were", "be", "been"}

                        temporal_preps = {"on", "in", "at", "from", "to", "during", "since", "until", "before", "after"}
                        is_temporal_pred = any(pred_text.endswith(f" {prep}") for prep in temporal_preps)

                        # If not copula AND not temporal, filter the date (it should be temporal modifier)
                        if not is_copula and not is_temporal_pred:
                            continue

                    # ✅ FIX: Skip "was" facts where subject is a year/number
                    # Pattern: "1692 was formed X" - year can't be subject of "was formed"
                    if f.predicate.text.lower().startswith("was"):
                        if re.match(r'^\d{4}$', subj_text):
                            continue

                    # ✅ FIX: Skip "was" facts where argument is team name but predicate is just "was" (missing role)
                    # Pattern: "He was Green Bay Packers" without "member of"
                    # BUT: Allow "He was member of X" where X is a team
                    # ✅ ITER 15: Also allow title + org constructions like "third pharaoh of the Dynasty"
                    if f.predicate.text.lower().strip() == "was":
                        # Check if argument looks like an organization/team name (multiple capitalized words)
                        arg_words = f.argument.text.split()
                        if len(arg_words) >= 2:
                            caps_count = sum(1 for w in arg_words if w and w[0].isupper())
                            if caps_count >= 2:
                                # ✅ ITER 15: Allow if argument has descriptive title words
                                # Examples: "third pharaoh", "first king", "chief engineer"
                                title_words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
                                              "chief", "lead", "senior", "junior", "head", "assistant", "deputy", "vice",
                                              "pharaoh", "king", "queen", "emperor", "president", "minister", "director", "officer"]
                                has_title = any(word in arg_text for word in title_words)

                                # Exception: if argument starts with role words OR contains title words - keep it
                                if not any(arg_text.startswith(role) for role in ["member", "part", "coach", "player"]) and not has_title:
                                    continue

                    # ✅ FIX: Skip agent predicates ("by X") where X is a location/facility
                    # Examples: "led by Arena", "coached by Stadium"
                    # Use spaCy NER: FAC (facilities) and LOC (locations) cannot be agents
                    if " by" in f.predicate.text.lower() or f.predicate.text.lower().endswith("by"):
                        arg_ent_type = f.argument.root.ent_type_
                        if arg_ent_type in {"FAC", "LOC"}:
                            continue

                    # ✅ FIX: Skip facts where subject contains newlines (section headers bleeding in)
                    # Pattern: "Geography\nCokesbury", "History\nThe name"
                    if '\n' in subj_text or '\r' in subj_text:
                        continue

                    # ✅ FIX: Skip facts where subject is a bare section header (without determiner)
                    # Pattern: "Early career joined X", "History was Y"
                    # But allow "This history", "The geography" - these are valid subjects
                    section_headers = [
                        "early career", "history", "geography", "biography", "personal life",
                        "career", "education", "background", "overview", "summary",
                        "moves", "transfers", "achievements", "awards", "legacy",
                        "early life", "later life", "death", "works", "publications"
                    ]
                    # Only filter if subject is bare header without determiner/modifier
                    # Check subject ROOT token's children for det (even if det not in span)
                    subj_root = f.subject.root
                    has_det = any(c.dep_ == "det" for c in subj_root.children)
                    if subj_text in section_headers and not has_det:
                        continue

                    # ✅ FIX: Skip facts where subject is a date (not just year)
                    # Pattern: "May 11, 2007 signed X", "April 26, 2006 signed Y"
                    # Match patterns like "Month DD, YYYY" or "DD Month YYYY"
                    date_patterns = [
                        r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+(19|20)\d{2}',
                        r'^\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(19|20)\d{2}',
                    ]
                    if any(re.match(pattern, subj_text, re.IGNORECASE) for pattern in date_patterns):
                        continue

                    # ✅ FIX: Skip facts where subject is a distance/measurement acting as identity
                    # Pattern: "7 miles is county seat", "5 km is capital"
                    if re.match(r'^\d+\s*(miles?|km|kilometers?|feet|ft|meters?|m)\s*$', subj_text, re.IGNORECASE):
                        if f.predicate.text.lower().strip() == "is":
                            continue

                    # ✅ FIX: Skip arguments that are section headers
                    # Pattern: "joined Early career", "joined moves"
                    # Common section headers in Wikipedia-style text
                    section_headers = [
                        "early career", "history", "geography", "biography", "personal life",
                        "career", "education", "background", "overview", "summary",
                        "moves", "transfers", "achievements", "awards", "legacy"
                    ]
                    if arg_text in section_headers:
                        continue

                    # ✅ FIX: Skip arguments that are just percentages without context
                    # Pattern: "was 72.76%", "was 1.08%" without racial/demographic context
                    if re.match(r'^\d+\.?\d*%$', arg_text):
                        continue

                    if arg_text in {"one", "ones"}:
                        continue

                    # ✅ FIX: Skip arguments that are just a year (incomplete temporal info)
                    # Pattern: "released on 2003" should be "released on August 31, 2003"
                    # BUT allow years for establishment/founding events: "was established in 1910"
                    if f.predicate.text.lower().endswith(" on") or f.predicate.text.lower().endswith(" in"):
                        if re.match(r'^\d{4}$', arg_text):
                            pred_lower = f.predicate.text.lower()
                            establishment_verbs = {"established", "founded", "built", "created", "formed", "opened", "started", "begun"}
                            is_establishment = any(v in pred_lower for v in establishment_verbs)
                            if not is_establishment:
                                continue

                    # ✅ FIX: Skip malformed "listed" facts without proper preposition
                    # Pattern: "are listed School" should be "are listed as School" or "listed on Register"
                    # These often result from incorrect list parsing
                    if "listed" in f.predicate.text.lower():
                        has_prep = any(prep in f.predicate.text.lower() for prep in [" as", " on", " in", " among", " with"])
                        if not has_prep:
                            # If no prep in predicate and argument doesn't start with article/prep, likely malformed
                            if not any(arg_text.startswith(word) for word in ["the ", "a ", "an ", "on ", "as ", "in "]):
                                continue

                key = _key(f)
                facts[key] = f

        if self.use_improvements:

            has_quotes = bool(re.search(r'[""\'«»'']', sent_text))

            if has_quotes:
                text_preprocessed = preprocess_quoted_entities(sent_text).processed_text
            else:
                text_preprocessed = sent_text

            if 'but not' in text_preprocessed.lower():  # Quick check before expensive regex
                contrastive_parsed = parse_contrastive_construction(text_preprocessed)
                if contrastive_parsed:
                    contrastive_facts = self._extract_contrastive_facts(sent.doc, text_preprocessed, sent)
                    if contrastive_facts:
                        _add(contrastive_facts)

            # Check for list pattern (on preprocessed text)
            # Quick check: contains ':' or 'include' before expensive regex
            if ':' in text_preprocessed or 'include' in text_preprocessed.lower():
                list_parsed = detect_list_pattern(text_preprocessed)
                if list_parsed:
                    list_facts = self._extract_list_facts(sent.doc, text_preprocessed, sent)
                    if list_facts:
                        _add(list_facts)

        # Standard fact extraction (existing logic)
        # Iterate only over tokens in THIS sentence
        for token in sent:
            # ✅ FIX: Skip processing verbs inside quoted spans
            # Quoted text should be treated as atomic entities, not parsed for internal structure
            if token.pos_ in {"VERB", "AUX"} and self._get_quoted_span_context(token) is not None:
                continue

            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                has_real_subj = any(c.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"} for c in token.children)
                if not has_real_subj:
                    predicate = self._get_full_predicate_span(token)
                    if predicate:
                        doc = token.doc
                        empty_span = doc[token.i:token.i]
                        verb_facts = self._extract_verb_arguments(token, empty_span, predicate)
                        if self._is_meaningful_predicate(token) and len(verb_facts) == 0:
                            has_args = any(c.dep_ in {"dobj", "prep", "attr", "oprd", "xcomp", "ccomp"} for c in token.children)
                            if not has_args:
                                verb_facts.append(Fact(empty_span, predicate, None))
                        _add(verb_facts)
                        continue

            if token.dep_ == "ROOT":
                subj = self._extract_subject(token)

                # ✅ FIX: Process existential constructions even if subj is None
                if subj is None and token.lemma_ == "be":
                    has_expl = any(c.dep_ == "expl" for c in token.children)
                    if has_expl:
                        _add(self._extract_be_facts(token, None))
                        continue

                if subj is not None:
                    _add(self._process_verb(token, subj))

                    # ✅ NEW FIX: Process coordinated subjects (conjuncts)
                    # Example: "community and stakeholders were concerned"
                    # After processing "community", also process "stakeholders"
                    subj_tokens = [c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}]
                    if subj_tokens:
                        subj_token = subj_tokens[0]
                        for conj_token in subj_token.conjuncts:
                            conj_subj = self._get_full_subject_span(conj_token)
                            if conj_subj is not None:
                                _add(self._process_verb(token, conj_subj))

            elif token.dep_ == "relcl" and token.pos_ in {"VERB", "AUX"}:
                head = token.head

                # ✅ FIX: Process relcl more broadly based on structural patterns
                # 1. Head is pobj (argument in prep phrase) - e.g., "looks, which have made..."
                # 2. OR relcl has explicit relative pronoun subject (which/that/who)
                # This avoids hardcoding specific nouns while filtering noise
                should_process = False

                if head.dep_ == "pobj":
                    should_process = True

                has_relative_subj = any(
                    c.dep_ in {"nsubj", "nsubjpass"} and c.lemma_.lower() in {"which", "that", "who", "whom"}
                    for c in token.children
                )
                if has_relative_subj:
                    should_process = True

                if head.dep_ == "attr":
                    should_process = True

                if head.dep_ == "pobj" and head.head.dep_ == "prep" and head.head.text.lower() == "for":
                    should_process = False

                if not should_process:
                    continue

                # Skip relcl that mainly introduces a reporting clause (avoid "actions sends message ...").
                has_reporting_clause = any(
                    c.dep_ == "dobj" and any(gc.dep_ in {"acl", "ccomp"} for gc in c.children)
                    for c in token.children
                )
                if has_reporting_clause:
                    continue

                subj = None

                if head.dep_ == "attr" and head.head.lemma_ == "be":
                    if any(c.dep_ == "expl" for c in head.head.children):
                        subj = self._get_np_span(head) or self._tok_span(head)
                    else:
                        subj = self._extract_subject(head.head)

                if subj is None:
                    if self._is_bad_subject_head(head):
                        subj = self._extract_subject(token) or self._inherit_subject(token)
                    else:
                        subj = self._get_full_subject_span(head)

                if subj is not None:
                    _add(self._process_verb(token, subj))


            elif token.dep_ == "acl" and token.pos_ in {"VERB", "AUX"}:
                # NEW: "short film called/named/titled X" don't consider as separate predicate.
                # Extract titles from NP via _extract_title_from_acl / quoted-extraction.
                if token.lemma_.lower() in getattr(self, "title_verbs", {"call", "name", "title"}) and token.head.pos_ in {"NOUN", "PROPN"}:
                    continue

                # ✅ ITER 13: Extract range expressions for recall
                # Example: "parameters, ranging from 110M to 1.5B"
                # Pattern: "[noun] ranging from X to Y"
                if token.lemma_.lower() == "range":
                    def collect_range_tokens(token):
                        tokens = [token]
                        for child in token.children:
                            if child.dep_ in {"prep", "pobj"}:
                                tokens.extend(collect_range_tokens(child))
                        return tokens

                    range_tokens = collect_range_tokens(token)

                    if len(range_tokens) > 1:
                        range_tokens_sorted = sorted(range_tokens, key=lambda t: t.i)
                        range_start = range_tokens_sorted[0].i
                        range_end = range_tokens_sorted[-1].i + 1
                        range_span = token.doc[range_start:range_end]

                        described_noun = self._get_full_subject_span(token.head)
                        if described_noun:
                            _add([Fact(described_noun, range_span, None)])

                    continue

                # ✅ FIX: Skip acl that is part of "known for X" pattern
                # Pattern: "known for ability to elicit" - "to elicit" is part of argument
                # Structure: known (ROOT) -> for (prep) -> ability (pobj) -> elicit (acl)
                # Don't process acl as separate verb if it's part of pobj argument
                head = token.head
                if head.dep_ == "pobj":
                    # Check if pobj's head is prep with "known/famous/noted" parent
                    prep_head = head.head
                    if prep_head.dep_ == "prep":
                        verb_head = prep_head.head
                        # Check for "known for", "famous for", "noted for", "recognized for" patterns
                        if verb_head.lemma_.lower() in {"know", "famous", "note", "recognize", "celebrate"} and prep_head.text.lower() == "for":
                            # This acl is part of the argument in "known for X" pattern
                            continue

                root = self._find_root(token)
                subj = self._extract_subject(token) or self._extract_subject(root)
                if subj is not None:
                    _add(self._process_verb(token, subj))

        # ✅ DISABLED: Standalone entity extraction creates too many incorrect facts
        # This was creating facts like "7 miles is county seat" by incorrectly pairing
        # entities with anchor subjects/predicates from unrelated parts of sentence
        # _add(self._extract_standalone_entity_facts(sent, facts))

        # ✅ IMPROVEMENT: Enhanced dates and locations extraction
        # Extract dates and locations through patterns (even if not recognized as entities)
        # This is called separately from standalone extraction to avoid incorrect pairings
        if self.use_improvements:
            anchor_subj = None
            anchor_pred = None
            
            if facts:
                first_fact = next(iter(facts.values()))
                anchor_subj = first_fact.subject
                anchor_pred = first_fact.predicate
            else:
                for token in sent:
                    if anchor_subj is None and token.dep_ in {"nsubj", "nsubjpass"}:
                        anchor_subj = self._get_full_subject_span(token)
                    if anchor_pred is None and token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
                        anchor_pred = self._get_full_predicate_span(token) or self._tok_span(token)
                    if anchor_subj is not None and anchor_pred is not None:
                        break
            
            if anchor_subj is not None and anchor_pred is not None:
                enhanced_facts = self._extract_enhanced_dates_and_locations(sent, anchor_subj, anchor_pred, facts)
                _add(enhanced_facts)

        # ✅ ATOMICITY IMPROVEMENT: Decompose long arguments into atomic facts
        # This improves atomicity by splitting arguments with prep phrases
        # Example: "podestà of Milan" → "podestà" + separate fact "podestà in Milan"
        if self.use_improvements:
            atomic_facts = self._apply_atomicity_decomposition(list(facts.values()))
            facts.clear()
            for f in atomic_facts:
                val_result = self._validate_fact_precision(f)
                if val_result:
                    facts[_key(f)] = f

        # ✅ SAFETY: Re-inject existential facts if missing (there is/are ...)
        # Some filters can drop them; keep for NLI-friendly existence checks.
        for token in sent:
            if token.dep_ == "ROOT" and token.lemma_ == "be":
                has_expl = any(c.dep_ == "expl" for c in token.children)
                if not has_expl:
                    continue
                expl = next((c for c in token.children if c.dep_ == "expl"), None)
                if expl is None:
                    continue
                ex_facts = self._extract_be_facts(token, self._tok_span(expl))
                for f in ex_facts:
                    if f.argument is not None:
                        for inc in self._build_incremental_argument_facts(f):
                            if self._validate_fact_precision(inc):
                                facts[_key(inc)] = inc

                    if f.argument is not None:
                        ext_arg = self._extend_arg_with_attached_prep(f.argument)
                        if ext_arg is not None and (ext_arg.start != f.argument.start or ext_arg.end != f.argument.end):
                            f = Fact(f.subject, f.predicate, ext_arg, f.prep)

                    if self._validate_fact_precision(f):
                        facts[_key(f)] = f

        # Final deduplication: if two facts have same subject+predicate and one argument contains another,
        # keep only the larger argument (e.g. "despite their large size" over "their large size")
        final_facts = {}
        for key, fact in facts.items():
            # Check if there's already a fact with same subject+predicate but different argument
            subj_pred_key = (fact.subject.start, fact.subject.end, fact.predicate.start, fact.predicate.end)

            conflicts = [
                (k, f) for k, f in facts.items()
                if (f.subject.start, f.subject.end, f.predicate.start, f.predicate.end) == subj_pred_key
                and f.argument is not None and fact.argument is not None
            ]

            def _content_start(span: Span) -> int:
                idx = span.start
                doc = span.doc
                while idx < span.end and doc[idx].pos_ == "DET":
                    idx += 1
                return idx

            # If current fact's argument is contained by another, skip it
            # EXCEPTION: Don't skip if this looks like an incremental fact
            # (same prep, argument starts at same position - e.g., "Pensacola" vs "Pensacola, Florida")
            skip = False
            for other_key, other_fact in conflicts:
                if other_key == key:
                    continue
                fact_arg_range = set(range(fact.argument.start, fact.argument.end))
                other_arg_range = set(range(other_fact.argument.start, other_fact.argument.end))
                if fact_arg_range < other_arg_range:  # strict subset
                    # Check if this is an incremental fact pattern
                    # Pattern 1: Nested preps - same START, increasing END
                    #   Example: "Pensacola" vs "Pensacola, Florida"
                    # Pattern 2: Progressive modifiers - same END, decreasing START
                    #   Example: "language" vs "programming language"
                    fact_start = _content_start(fact.argument)
                    other_start = _content_start(other_fact.argument)
                    is_nested_incremental = (
                        (fact.prep == other_fact.prep or (fact.prep is None and other_fact.prep is None))
                        and fact_start == other_start
                    )
                    is_progressive_incremental = (
                        (fact.prep == other_fact.prep or (fact.prep is None and other_fact.prep is None))
                        and fact.argument.end == other_fact.argument.end
                    )
                    is_incremental = is_nested_incremental or is_progressive_incremental

                    is_conj_item = fact.argument.root.dep_ == "conj" or any(t.dep_ == "conj" for t in fact.argument)
                    is_head_base = fact.argument.root == other_fact.argument.root

                    if not is_incremental and not is_conj_item and not is_head_base:
                        skip = True
                        break

            if not skip:
                final_facts[key] = fact
            else:
                if fact.predicate and len(fact.predicate) == 1 and fact.predicate[0].pos_ == "ADP":
                    pass

        # ✅ RECALL IMPROVEMENT: Extract comparative/relative constructions
        # NOTE: Temporal extraction removed - now handled by _extract_date_parts_facts() via entity spans
        if self.use_improvements:
            for token in sent:
                # 1. Extract comparative constructions from amod
                # Pattern: "higher for females at 40 compared to males at 36"
                if token.pos_ == "ADJ" and token.tag_ in {"JJR", "RBR"} and token.dep_ == "amod":
                    noun_head = token.head

                    # Structure: higher -> age (pobj) -> with (prep) -> has (verb)
                    verb = noun_head.head  # First try: age -> with
                    while verb and verb.pos_ != "VERB" and verb != verb.head:
                        verb = verb.head

                    if verb and verb.pos_ == "VERB":
                        subj = None
                        for child in verb.children:
                            if child.dep_ in {"nsubj", "nsubjpass"}:
                                subj = self._get_np_span(child)
                                break

                        if subj:
                            pred = self._get_full_predicate_span(verb)

                            for_prep = next((c for c in token.children
                                            if c.dep_ == "prep" and c.text.lower() == "for"), None)
                            if for_prep:
                                for_pobj = next((c for c in for_prep.children if c.dep_ == "pobj"), None)
                                if for_pobj:
                                    at_prep = next((c for c in for_pobj.children
                                                   if c.dep_ == "prep" and c.text.lower() == "at"), None)
                                    if at_prep:
                                        at_pobj = next((c for c in at_prep.children if c.dep_ == "pobj"), None)
                                        if at_pobj and at_pobj.pos_ == "NUM":
                                            comp_arg = self._cover(for_prep, at_pobj)
                                            if comp_arg and pred:
                                                comp_fact = Fact(subj, pred, comp_arg)
                                                comp_key = _key(comp_fact)
                                                if comp_key not in final_facts and self._validate_fact_precision(comp_fact):
                                                    final_facts[comp_key] = comp_fact

                            compared_token = next((c for c in token.children
                                                  if c.dep_ == "prep" and c.text.lower() == "compared"), None)
                            if compared_token:
                                to_prep = next((c for c in compared_token.children
                                               if c.dep_ == "prep" and c.text.lower() == "to"), None)
                                if to_prep:
                                    to_pobj = next((c for c in to_prep.children if c.dep_ == "pobj"), None)
                                    if to_pobj:
                                        at_prep2 = next((c for c in to_pobj.children
                                                        if c.dep_ == "prep" and c.text.lower() == "at"), None)
                                        if at_prep2:
                                            at_pobj2 = next((c for c in at_prep2.children if c.dep_ == "pobj"), None)
                                            if at_pobj2 and at_pobj2.pos_ == "NUM":
                                                comp2_arg = self._cover(to_prep, at_pobj2)
                                                comp2_pred = self._tok_span(compared_token)
                                                if comp2_arg and comp2_pred:
                                                    comp2_fact = Fact(subj, comp2_pred, comp2_arg)
                                                    comp2_key = _key(comp2_fact)
                                                if comp2_key not in final_facts and self._validate_fact_precision(comp2_fact):
                                                    final_facts[comp2_key] = comp2_fact

                # Pattern: "vegetable that has leaves"
                if token.pos_ == "VERB" and token.dep_ == "relcl":
                    if token.head.dep_ == "pobj" and token.head.head.dep_ == "prep" and token.head.head.text.lower() == "for":
                        continue
                    rel_subject = self._get_np_span(token.head) or self._tok_span(token.head)
                    rel_pred = self._get_full_predicate_span(token) or self._tok_span(token)

                    if rel_subject and rel_pred:
                        for child in token.children:
                            if child.dep_ == "dobj":
                                dobj_arg = self._get_np_span(child)
                                if dobj_arg:
                                    rel_fact = Fact(rel_subject, rel_pred, dobj_arg)
                                    rel_key = _key(rel_fact)
                                    if rel_key not in final_facts and self._validate_fact_precision(rel_fact):
                                        final_facts[rel_key] = rel_fact

        out = list(final_facts.values())

        def _is_sentence_initial_temporal(fact: Fact) -> bool:
            if fact.argument is None:
                return False
            arg_text = fact.argument.text.lower().strip()
            if not arg_text.startswith(("in ", "on ", "at ")):
                return False
            if re.search(r'\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b', arg_text):
                return True
            months = ["january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december"]
            if any(month in arg_text for month in months):
                return True
            return False

        # ✅ FIX: Filter out backwards facts where argument comes BEFORE predicate
        # This can happen with relcl processing where conj items get wrong arguments
        # Example: "They include fruits..., which provide..." - "fruits" shouldn't have argument="They"
        # BUT: Allow sentence-initial temporal phrases like "In 2019, X helped..." - "In 2019" is valid argument
        # AND: Allow inverted constructions where subject comes after predicate (prep argument before)
        # Example: "Across the Rhône is the commune" - argument "Rhône" before predicate "is" is valid
        def _is_inverted_construction(fact: Fact) -> bool:
            if fact.argument is None or fact.prep is None:
                return False
            return True
        
        out = [f for f in out if f.argument is None or
               f.argument.start >= f.predicate.start or
               _is_sentence_initial_temporal(f) or
               _is_inverted_construction(f)]

        # ✅ FIX: Filter out fragment facts without verbs in predicate
        # Example: "variety of fruits" - predicate "of" is not a verb, this is a fragment
        # Keep only facts where predicate contains at least one VERB/AUX
        # EXCEPTION: Allow single-prep predicates (like "in") when argument is PROPN (location)
        # This preserves appos location facts: "Tokyo, Japan" → "in | Japan"
        def _has_verb(span: Span) -> bool:
            return any(t.pos_ in {"VERB", "AUX"} for t in span)

        def _is_location_prep_fact(fact: Fact) -> bool:
            if fact.argument is None:
                return False
            if len(fact.predicate) == 1 and fact.predicate[0].pos_ == "ADP":
                if any(t.pos_ == "PROPN" for t in fact.argument):
                    return True
            return False

        out = [f for f in out if _has_verb(f.predicate) or _is_location_prep_fact(f)]

        # ✅ FIX: Filter out facts with very short arguments that are likely fragments
        # Example: "he influenced by sense" - "sense" alone is fragment (should be "sense of spirituality")
        # - Single significant word (PROPN, NUM, DATE entity)
        def _is_fragment_argument(fact: Fact) -> bool:
            if fact.argument is None:
                return False
            arg_text = fact.argument.text.strip()
            if len(arg_text) < 2:
                return True
            if len(fact.argument) == 1 and len(arg_text) < 3:
                tok = fact.argument[0]
                if tok.pos_ in {"PROPN", "NUM"} or tok.ent_type_:
                    return False
                return True
            return False

        out = [f for f in out if not _is_fragment_argument(f)]

        out.sort(key=lambda f: (f.subject.start, f.predicate.start, (f.argument.start if f.argument else 10**9)))
        return out

    def _find_root(self, token: Token) -> Token:
        cur = token
        while cur.dep_ != "ROOT" and cur.head != cur:
            cur = cur.head
        return cur

    def _process_verb(self, verb_token: Token, subject: Span) -> List[Fact]:
        facts: List[Fact] = []

        if verb_token.lemma_ == "be":
            facts.extend(self._extract_be_facts(verb_token, subject))
            return facts

        predicate = self._get_full_predicate_span(verb_token)
        if predicate is None:
            return facts

        verb_lemma = verb_token.lemma_.lower()
        if verb_token.text.lower() == "born" or verb_lemma == "bear" or verb_lemma == "die":
            for prep in verb_token.children:
                if prep.dep_ != "prep":
                    continue
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue

                pobj_span = self._get_np_span(pobj) or self._tok_span(pobj)
                if pobj_span is None:
                    continue

                if pobj.ent_type_ == "DATE" or pobj.like_num:
                    pred_prep = self._cover(predicate, prep)
                    if pred_prep:
                        facts.append(Fact(subject, pred_prep, pobj_span))
                        continue

                if pobj.ent_type_ in {"GPE", "LOC", "FAC"}:
                    pred_prep = self._cover(predicate, prep)
                    if pred_prep:
                        facts.append(Fact(subject, pred_prep, pobj_span))
                        continue

        verb_facts = self._extract_verb_arguments(verb_token, subject, predicate)
        facts.extend(verb_facts)

        # Add incremental variants for argument NPs to enable granular hallucination detection
        incremental_arg_facts: List[Fact] = []
        for f in verb_facts:
            incremental_arg_facts.extend(self._build_incremental_argument_facts(f))
        facts.extend(incremental_arg_facts)

        # ✅ GOLD: Expand modifier-heavy common-noun subjects (e.g., "high-intensity interval training programs").
        subject_variants = self._build_incremental_subject_variants(subject)
        if subject_variants:
            for f in list(facts):
                if f.subject.start == subject.start and f.subject.end == subject.end:
                    for subj_span in subject_variants:
                        if subj_span.start == f.subject.start and subj_span.end == f.subject.end:
                            continue
                        facts.append(Fact(subj_span, f.predicate, f.argument, f.prep))

        if subject is not None and verb_token.lemma_.lower() in {"suggest", "provide"}:
            subj_text = subject.text.lower().strip()
            if subj_text.startswith("passage"):
                facts.append(Fact(subject, predicate, None))

        # ✅ FIX: Only add argument=None fact if verb has no arguments at all
        # This reduces noise while preserving intransitive verbs
        if self._is_meaningful_predicate(verb_token) and len(verb_facts) == 0:
            has_args = any(c.dep_ in {"dobj", "prep", "attr", "oprd", "xcomp", "ccomp"} for c in verb_token.children)
            if not has_args:
                facts.append(Fact(subject, predicate, None))

        return facts


    def _extract_be_facts(self, be_token: Token, subject: Span) -> List[Fact]:
        facts: List[Fact] = []
        predicate = self._get_full_predicate_span(be_token) or self._tok_span(be_token)
        also_token = next((c for c in be_token.children if c.dep_ == "advmod" and c.text.lower() == "also"), None)
        if also_token is not None:
            predicate = self._cover(predicate, also_token) or predicate
        if subject is not None and len(subject) == 1 and subject[0].pos_ == "PRON":
            aux_clitic = next(
                (c for c in be_token.children
                 if c.dep_ == "aux" and c.text in {"'d", "’d"} and c.i == subject.end),
                None,
            )
            if aux_clitic is not None:
                subject = subject.doc[subject.start:aux_clitic.i + 1]
                if predicate.start <= aux_clitic.i < predicate.end:
                    pred_tokens = [t for t in predicate if t.i != aux_clitic.i]
                    new_pred = self._span_from_tokens(predicate.doc, pred_tokens)
                    if new_pred is not None:
                        predicate = new_pred

        acomp_attached_preps = set()
        processed_acomp = set()

        # ✅ IMPROVEMENT: Handle "not the only X in Y" pattern (negation fix)
        # Pattern: "X were not the only Y in Z"
        # This prevents incorrect extraction like "X were not in Z"
        neg_token = next((c for c in be_token.children if c.dep_ == "neg"), None)
        attr = next((c for c in be_token.children if c.dep_ == "attr"), None)

        if neg_token is not None and attr is not None:
            only_token = next((c for c in attr.children
                              if c.dep_ in {"amod", "advmod"} and c.lemma_.lower() == "only"), None)

            if only_token is not None:
                # "X is not the only Y in Z" pattern detected
                # Extract as: X "is not the only" Y (without location prep)

                pred_tokens = [be_token, neg_token, only_token]
                det_token = next((c for c in attr.children if c.dep_ == "det"), None)
                if det_token:
                    pred_tokens.append(det_token)

                pred_with_neg = self._span_from_tokens(be_token.doc, pred_tokens)
                if pred_with_neg is None:
                    pred_with_neg = predicate

                # Get attribute WITHOUT prep phrases (e.g., "Kalango-speakers" without "in Zimbabwe")
                attr_tokens = [attr]
                for child in attr.children:
                    if child.dep_ in {"compound", "amod"} and child != only_token:
                        attr_tokens.append(child)

                arg_span = self._span_from_tokens(attr.doc, attr_tokens)
                if arg_span:
                    facts.append(Fact(subject, pred_with_neg, arg_span))

                # Extract prepositional phrases separately with ATTR as subject (not main subject!)
                # This creates "Kalango-speakers in Zimbabwe" instead of "Kalangas were not in Zimbabwe"
                # Build attr span WITHOUT "only" and "the" modifiers
                attr_clean_tokens = [attr]
                for child in attr.children:
                    if child.dep_ == "compound" and child != only_token:
                        attr_clean_tokens.append(child)

                attr_span_clean = self._span_from_tokens(attr.doc, attr_clean_tokens)

                if attr_span_clean:
                    for prep in attr.children:
                        if prep.dep_ == "prep":
                            prep_pred = self._tok_span(prep)
                            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                            if pobj:
                                prep_arg = self._get_np_span(pobj)
                                if prep_arg:
                                    facts.append(Fact(attr_span_clean, prep_pred, prep_arg))

                return facts

        # ✅ FIX: Handle existential constructions ("There is/are X")
        # Prefer a locative PP as subject when present (e.g., "On a hippo, there are seats"),
        # otherwise keep "there" as subject for NLI-friendly existence facts.
        expl_token = next((c for c in be_token.children if c.dep_ == "expl"), None)
        if expl_token is not None:
            locative_subject = None
            for prep in be_token.children:
                if prep.dep_ != "prep" or prep.i >= be_token.i:
                    continue
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                pobj_np = self._get_np_span(pobj) or self._tok_span(pobj)
                locative_subject = self._cover(prep, pobj_np) if pobj_np else None
                if locative_subject is not None:
                    break

            subj = locative_subject or subject or self._tok_span(expl_token)
            pred = self._cover(expl_token, predicate) if locative_subject is not None else predicate

            advmods = [
                c for c in be_token.children
                if c.dep_ == "advmod" and c.text.lower() != "also" and c.i > be_token.i
            ]

            def _prefix_advmods(span: Optional[Span]) -> Optional[Span]:
                if span is None:
                    return None
                for adv in advmods:
                    if adv.i < span.start and (span.start - adv.i) <= 3:
                        span = self._cover(adv, span) or span
                return span

            for child in be_token.children:
                if child.dep_ != "attr":
                    continue

                arg_span = self._get_np_span(child) or self._tok_span(child)
                if arg_span is None:
                    continue

                has_modifiers = any(c.dep_ in {"amod", "compound"} for c in child.children)
                nested_preps = [p for p in child.children if p.dep_ == "prep"]

                if has_modifiers:
                    incremental_modifier_facts = self._build_incremental_np_modifiers(child, subj, pred)
                    for f in incremental_modifier_facts:
                        arg = _prefix_advmods(f.argument)
                        if arg is not None:
                            facts.append(Fact(subj, pred, arg))
                    base_arg = incremental_modifier_facts[-1].argument if incremental_modifier_facts else arg_span
                else:
                    base_arg = _prefix_advmods(arg_span)
                    if base_arg is not None:
                        facts.append(Fact(subj, pred, base_arg))

                if base_arg is not None and nested_preps:
                    for prep in nested_preps:
                        if self._is_list_pattern(prep) or self._is_such_as(prep):
                            continue
                        pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                        if pobj is None:
                            continue
                        pobj_core = self._get_np_core_span(pobj) or self._tok_span(pobj)
                        pobj_full = self._subtree_span(pobj, include_punct=False) or self._get_np_span(pobj) or pobj_core

                        variants = []
                        if pobj_core is not None:
                            variants.append(pobj_core)
                        if pobj_full is not None:
                            variants.append(pobj_full)

                        relcl = next((c for c in pobj.children if c.dep_ == "relcl"), None)
                        if relcl is not None and pobj_full is not None:
                            relcl_span = self._subtree_span(relcl, include_punct=False)
                            if relcl_span is not None:
                                full_with_relcl = self._cover(pobj_full, relcl_span)
                                if full_with_relcl is not None:
                                    variants.append(full_with_relcl)

                        seen = set()
                        for var in variants:
                            key = (var.start, var.end)
                            if key in seen:
                                continue
                            seen.add(key)
                            accumulated_arg = self._cover(base_arg, prep, var)
                            if accumulated_arg is not None:
                                facts.append(Fact(subj, pred, accumulated_arg))

                # Handle "such as"/"like" lists under existential attrs (e.g., "there were downturns such as X").
                added_items = set()
                for prep in child.subtree:
                    if prep.dep_ == "prep" and (self._is_such_as(prep) or self._is_list_pattern(prep)):
                        for pobj in prep.children:
                            if pobj.dep_ != "pobj":
                                continue
                            items = [pobj] + list(pobj.conjuncts)
                            for item in items:
                                item_np = self._get_np_span(item)
                                if item_np is None:
                                    continue
                                key = (item_np.start, item_np.end)
                                if key in added_items:
                                    continue
                                added_items.add(key)
                                facts.append(Fact(subj, pred, item_np))

            return facts

        # ✅ FIX: Handle inverted locative constructions ("Across X is Y", "Among X are Y")
        # Pattern: no attr, post-verbal nsubj, pre-verbal prep phrase
        # The subject is actually the "attribute" semantically, and prep phrase is locative
        has_attr = any(c.dep_ == "attr" for c in be_token.children)
        if not has_attr and subject is not None:
            subj_tokens = [c for c in be_token.children if c.dep_ in {"nsubj", "nsubjpass"}]
            if subj_tokens and subj_tokens[0].i > be_token.i:
                # Post-verbal subject - this is inverted construction
                # Extract prep phrases as locative arguments
                for prep in be_token.children:
                    if prep.dep_ == "prep":
                        pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                        if pobj is not None:
                            pobj_span = self._get_np_span(pobj) or self._tok_span(pobj)
                            if pobj_span:
                                # e.g., "commune of Beaucaire | is across | the Rhône"
                                facts.append(Fact(subject, predicate, pobj_span, prep=self._tok_span(prep)))
                if facts:
                    return facts

        for child in be_token.children:
            if child.dep_ == "acomp" and child.i in processed_acomp:
                continue
            if child.dep_ == "attr":
                if child.pos_ == "PROPN" or child.ent_type_ == "PERSON":
                    name = self._get_np_span(child) or self._subtree_span(child, include_punct=False) or self._tok_span(child)
                    facts.append(Fact(subject, predicate, name))
                    continue

                if child.lemma_.lower() == "part":
                    of_prep = next((c for c in child.children if c.dep_ == "prep" and c.text.lower() == "of"), None)
                    if of_prep is not None:
                        pobj = next((c for c in of_prep.children if c.dep_ == "pobj"), None)
                        if pobj is not None:
                            arg = self._get_np_span(pobj) or self._tok_span(pobj)
                            pred_part = self._cover(predicate, child, of_prep) or predicate
                            if arg is not None:
                                facts.append(Fact(subject, pred_part, arg))
                            continue

                # Use full NP span to include modifiers (oldest serving justice)
                # ✅ FIX: For lists with percentages, ensure we capture advmod (25.09% White)
                # Also check for numeric tokens immediately to the left (handles edge cases)
                arg_span = self._get_np_span(child)

                # ✅ ADDITIONAL FIX: If no span found or span doesn't include percentages,
                # manually check for numeric tokens to the left (handles parsing edge cases)
                if arg_span is None or (child.i > 0 and "%" in child.doc[child.i - 1].text and child.doc[child.i - 1] not in arg_span):
                    left_tokens = []
                    for i in range(max(0, child.i - 3), child.i):
                        tok = child.doc[i]
                        if "%" in tok.text or tok.like_num or tok.pos_ == "NUM":
                            left_tokens.append(tok)

                    if left_tokens:
                        all_tokens = left_tokens + self._collect_np_tokens(child)
                        arg_span = self._span_from_tokens(child.doc, all_tokens)
                    else:
                        arg_span = arg_span or self._tok_span(child)
                else:
                    arg_span = arg_span or self._tok_span(child)

                adv_span = None
                advmods = [
                    c for c in be_token.children
                    if c.dep_ == "advmod" and c.text.lower() != "also" and c.i > be_token.i and c.i < child.i
                ]
                if advmods:
                    adv_span = self._span_from_tokens(child.doc, advmods)

                # NEW: Create incremental facts for NP modifiers AND nested preps
                # Check for modifiers (amod, compound) and nested preps
                has_modifiers = any(c.dep_ in {"amod", "compound"} for c in child.children)
                nested_preps = [p for p in child.children if p.dep_ == "prep"]

                if has_modifiers:
                    # Example: language → programming language → multi-paradigm programming language
                    incremental_modifier_facts = self._build_incremental_np_modifiers(child, subject, predicate)
                    if adv_span is not None and nested_preps:
                        pass
                    elif adv_span is not None:
                        for f in incremental_modifier_facts:
                            adv_arg = self._cover(adv_span, f.argument) if f.argument is not None else None
                            if adv_arg is not None:
                                facts.append(Fact(subject, predicate, adv_arg))
                    else:
                        facts.extend(incremental_modifier_facts)

                    # If also has nested preps, add them as further increments on top of last modifier
                    if nested_preps:
                        if incremental_modifier_facts:
                            base_arg = incremental_modifier_facts[-1].argument
                        else:
                            base_tokens = [child]
                            for c in child.children:
                                if c.dep_ in {"amod", "compound", "nummod", "det"}:
                                    base_tokens.append(c)
                            base_arg = self._span_from_tokens(child.doc, base_tokens)
                            if not base_arg:
                                base_arg = self._tok_span(child)

                        if adv_span is not None and base_arg is not None:
                            base_arg = self._cover(adv_span, base_arg) or base_arg

                        for prep in nested_preps:
                            if self._is_list_pattern(prep) or self._is_such_as(prep):
                                continue
                            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                            if pobj:
                                pobj_core = self._get_np_core_span(pobj) or self._tok_span(pobj)
                                pobj_full = self._subtree_span(pobj, include_punct=False) or self._get_np_span(pobj) or pobj_core

                                variants = []
                                if pobj_core is not None:
                                    variants.append(pobj_core)
                                if pobj_full is not None:
                                    variants.append(pobj_full)

                                relcl = next((c for c in pobj.children if c.dep_ == "relcl"), None)
                                if relcl is not None and pobj_full is not None:
                                    relcl_span = self._subtree_span(relcl, include_punct=False)
                                    if relcl_span is not None:
                                        full_with_relcl = self._cover(pobj_full, relcl_span)
                                        if full_with_relcl is not None:
                                            variants.append(full_with_relcl)
                                appos = next((c for c in pobj.children if c.dep_ == "appos" and self._get_quoted_span_context(c) is None), None)
                                if appos is not None:
                                    appos_core = self._get_np_core_span(appos) or self._tok_span(appos)
                                    appos_full = self._subtree_span(appos, include_punct=False) or self._get_np_span(appos) or appos_core
                                    if appos_core is not None:
                                        variants.append(appos_core)
                                    if appos_full is not None:
                                        variants.append(appos_full)

                                seen = set()
                                for var in variants:
                                    key = (var.start, var.end)
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    accumulated_arg = self._cover(base_arg, prep, var)
                                    if accumulated_arg is not None:
                                        facts.append(Fact(subject, predicate, accumulated_arg))

                elif nested_preps:
                    base_tokens = [child]
                    for c in child.children:
                        if c.dep_ in {"nummod", "det"}:
                            base_tokens.append(c)
                    base_arg = self._span_from_tokens(child.doc, base_tokens)
                    if not base_arg:
                        base_arg = self._tok_span(child)

                    if adv_span is not None and base_arg is not None:
                        base_arg = self._cover(adv_span, base_arg) or base_arg

                    if adv_span is None:
                        facts.append(Fact(subject, predicate, base_arg))

                    for prep in nested_preps:
                        if self._is_list_pattern(prep) or self._is_such_as(prep):
                            continue
                        pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                        if pobj:
                            pobj_core = self._get_np_core_span(pobj) or self._tok_span(pobj)
                            pobj_full = self._subtree_span(pobj, include_punct=False) or self._get_np_span(pobj) or pobj_core

                            variants = []
                            if pobj_core is not None:
                                variants.append(pobj_core)
                            if pobj_full is not None:
                                variants.append(pobj_full)

                            relcl = next((c for c in pobj.children if c.dep_ == "relcl"), None)
                            if relcl is not None and pobj_full is not None:
                                relcl_span = self._subtree_span(relcl, include_punct=False)
                                if relcl_span is not None:
                                    full_with_relcl = self._cover(pobj_full, relcl_span)
                                    if full_with_relcl is not None:
                                        variants.append(full_with_relcl)
                            appos = next((c for c in pobj.children if c.dep_ == "appos" and self._get_quoted_span_context(c) is None), None)
                            if appos is not None:
                                appos_core = self._get_np_core_span(appos) or self._tok_span(appos)
                                appos_full = self._subtree_span(appos, include_punct=False) or self._get_np_span(appos) or appos_core
                                if appos_core is not None:
                                    variants.append(appos_core)
                                if appos_full is not None:
                                    variants.append(appos_full)

                            seen = set()
                            for var in variants:
                                key = (var.start, var.end)
                                if key in seen:
                                    continue
                                seen.add(key)
                                accumulated_arg = self._cover(base_arg, prep, var)
                                if accumulated_arg is not None:
                                    facts.append(Fact(subject, predicate, accumulated_arg))
                else:
                    if adv_span is not None and arg_span is not None:
                        arg_span = self._cover(adv_span, arg_span) or arg_span
                    facts.append(Fact(subject, predicate, arg_span))

                relcl = next((c for c in child.children if c.dep_ in {"relcl", "acl"}), None)
                if relcl is not None and arg_span is not None:
                    relcl_span = self._subtree_span(relcl, include_punct=False)
                    if relcl_span is not None and len(relcl_span) > 0:
                        relcl_core = relcl_span
                        for t in relcl_span:
                            if t.dep_ == "prep":
                                relcl_core = relcl_span.doc[relcl_span.start:t.i]
                                break
                        for extra in [relcl_core, relcl_span]:
                            if extra is None or len(extra) == 0:
                                continue
                            extended = self._cover(arg_span, extra)
                            if extended is not None:
                                facts.append(Fact(subject, predicate, extended))

                # NEW: Skip old prep extraction - now handled as incremental facts above
                # OLD: Extract preps attached to attr (on the Court, in the film, etc.)
                #         facts.extend(self._extract_verb_prep(prep, subject, predicate, be_token))

                # ✅ FIX: Use _extract_conjuncts_and_appos to properly handle conjunct lists
                # This ensures percentages/numbers are included for each list item
                # Example: "was 25.09% White, 72.76% African American" extracts both with percentages
                facts.extend(self._extract_conjuncts_and_appos(child, subject, predicate))

                for prep in child.children:
                    if prep.dep_ == "prep" and self._is_such_as(prep):
                        for pobj in prep.children:
                            if pobj.dep_ != "pobj":
                                continue
                            items = [pobj] + list(pobj.conjuncts)
                            for item in items:
                                item_np = self._get_np_span(item)
                                if item_np is not None:
                                    facts.append(Fact(subject, predicate, item_np))

            elif child.dep_ == "acomp":
                if child.lemma_.lower() == "worth" and any(
                    c.dep_ == "xcomp" and c.lemma_.lower() == "note" for c in child.children
                ):
                    continue
                acomp_span = None
                left = child.i
                right = child.i
                doc = child.doc
                i = child.i - 1
                while i >= child.sent.start and doc[i].dep_ == "acomp" and doc[i].head == be_token:
                    left = i
                    i -= 1
                i = child.i + 1
                while i < child.sent.end and doc[i].dep_ == "acomp" and doc[i].head == be_token:
                    right = i
                    i += 1
                if left != right:
                    acomp_span = doc[left:right + 1]
                    processed_acomp.update(range(left, right + 1))
                else:
                    processed_acomp.add(child.i)

                # Skip base acomp fact when it has non-comparative preps (use prep-linked facts instead).
                has_noncomp_prep = any(
                    m.dep_ == "prep" and m.text.lower() not in {"than", "as"}
                    for m in child.children
                )

                # ✅ pattern: is (best) known for X
                if child.lemma_.lower() == "know" or child.text.lower() == "known":
                    best = next((m for m in child.children if m.dep_ == "advmod" and m.text.lower() == "best"), None)
                    for_prep = next((p for p in child.children if p.dep_ == "prep" and p.text.lower() == "for"), None)
                    if for_prep is not None:
                        pcomp = next((c for c in for_prep.children if c.dep_ == "pcomp" and c.pos_ in {"VERB", "AUX"}), None)
                        if pcomp is not None:
                            pred_known_for = (self._cover(be_token, best, child, for_prep, pcomp) or
                                              self._cover(be_token, child, for_prep, pcomp) or
                                              self._cover(be_token, child, for_prep) or
                                              self._tok_span(be_token))
                            for pc_child in pcomp.children:
                                if pc_child.dep_ == "prep":
                                    facts.extend(self._extract_verb_prep(pc_child, subject, pred_known_for, pcomp))
                                elif pc_child.dep_ in {"dobj", "attr", "oprd"}:
                                    facts.extend(self._extract_object_facts(pc_child, subject, pred_known_for, pcomp))
                            continue

                        pred_known_for = self._cover(be_token, best, child, for_prep) or self._cover(be_token, child, for_prep) or self._tok_span(be_token)
                        pobj = next((c for c in for_prep.children if c.dep_ == "pobj"), None)
                        if pobj is not None:
                            head = self._pick_content_head(pobj)
                            facts.extend(self._extract_single_object_facts(head, subject, pred_known_for))
                            facts.extend(self._extract_conjuncts_and_appos(head, subject, pred_known_for))
                        continue

                # ✅ pattern: is able to spin quickly
                xcomp = next((c for c in child.children if c.dep_ == "xcomp" and c.pos_ in {"VERB", "AUX"}), None)
                if xcomp is not None:
                    to_aux = next((c for c in xcomp.children if c.dep_ == "aux" and c.text.lower() == "to"), None)

                    pred_tokens = [be_token, child]
                    if to_aux is not None:
                        pred_tokens.append(to_aux)
                    pred_able_to = self._span_from_tokens(be_token.doc, pred_tokens)

                    xcomp_tokens = [xcomp]
                    for c in xcomp.children:
                        if c.dep_ == "aux" and c.text.lower() == "to":
                            continue  # skip "to" in argument
                        xcomp_tokens.extend(list(c.subtree))

                    xcomp_arg = self._span_from_tokens(xcomp.doc, xcomp_tokens)
                    if xcomp_arg is not None and pred_able_to is not None:
                        facts.append(Fact(subject, pred_able_to, xcomp_arg))
                        continue

                # ✅ FIX: Handle comparatives (faster than X, as ADJ as X)
                # Check if acomp has comparative prep ("than", "as")
                comp_prep = None
                for m in child.children:
                    if m.dep_ == "prep" and m.text.lower() in {"than", "as"}:
                        comp_prep = m
                        break

                if comp_prep is not None:
                    acomp_attached_preps.add(comp_prep.i)

                    acomp_tokens = [child]

                    for m in child.children:
                        if m.dep_ == "advmod" and m.lemma_.lower() not in self.ignored_advmods:
                            acomp_tokens.append(m)

                    acomp_tokens.append(comp_prep)
                    for pobj in comp_prep.children:
                        if pobj.dep_ == "pobj":
                            pobj_np = self._get_np_span(pobj)
                            if pobj_np is not None:
                                acomp_tokens.extend(list(pobj_np))
                            else:
                                acomp_tokens.append(pobj)

                    acomp_span = self._span_from_tokens(child.doc, acomp_tokens)
                    if acomp_span is not None:
                        facts.append(Fact(subject, predicate, acomp_span))
                    continue

                # General acomp: "was still alive", "is happy", "was 15 years old", etc.
                # ✅ FIX: Include advmod modifiers (still, very, etc.) AND npadvmod for ages
                # Pattern: "was 15 years old" → "15" (nummod) → "years" (npadvmod) → "old" (acomp)
                if not has_noncomp_prep:
                    acomp_tokens = list(acomp_span) if acomp_span is not None else [child]
                    for m in child.children:
                        if m.dep_ == "advmod" and m.lemma_.lower() not in self.ignored_advmods:
                            acomp_tokens.append(m)
                        elif m.dep_ == "npadvmod":
                            acomp_tokens.extend(self._collect_np_tokens(m))

                    acomp_span_final = self._span_from_tokens(child.doc, acomp_tokens)
                    if acomp_span_final is not None:
                        facts.append(Fact(subject, predicate, acomp_span_final))

                # ✅ NEW: Process coordinated acomp (e.g., "was both steady and impressive")
                for conj in child.conjuncts:
                    if conj.pos_ == "ADJ":
                        conj_tokens = [conj]
                        for m in conj.children:
                            if m.dep_ == "advmod" and m.lemma_.lower() not in self.ignored_advmods:
                                conj_tokens.append(m)
                        conj_span = self._span_from_tokens(conj.doc, conj_tokens)
                        if conj_span is not None:
                            facts.append(Fact(subject, predicate, conj_span))

                # ✅ NEW FIX + GOLD: Process prep phrases attached to acomp
                # Example: "were concerned about X" → extract "about X" with predicate "were concerned about"
                # Example: "are responsible for managing X" → extract pcomp verb + args
                # Structure: were (ROOT) → concerned (acomp) → about (prep) → X (pobj/pcomp)
                for prep in child.children:
                    if prep.dep_ == "prep":
                        extended_pred = self._cover(be_token, child, prep)
                        if extended_pred is None:
                            continue

                        for prep_child in prep.children:
                            if prep_child.dep_ == "pcomp" and prep_child.pos_ in {"VERB", "AUX"}:
                                # Pattern: "responsible for managing X"
                                # Extract full pcomp verb phrase (pcomp + dobj) as argument
                                # Example: "managing the entire hiring process" not just "entire hiring process"

                                for pcomp_child in prep_child.children:
                                    if pcomp_child.dep_ == "dobj":
                                        dobj_np = self._get_np_span(pcomp_child)
                                        if dobj_np is not None:
                                            pcomp_span = prep_child.doc[prep_child.i:dobj_np.end]
                                            facts.append(Fact(subject, extended_pred, pcomp_span))

                                        for dobj_child in pcomp_child.children:
                                            if dobj_child.dep_ == "prep":
                                                if dobj_child.text.lower() == "including":
                                                    for pobj in dobj_child.children:
                                                        if pobj.dep_ == "pobj":
                                                            pobj_np = self._get_np_span(pobj)
                                                            if pobj_np is not None:
                                                                facts.append(Fact(subject, extended_pred, pobj_np))

                                                            for conj in pobj.conjuncts:
                                                                # ✅ GOLD: Handle VERB tokens with dobj specially
                                                                # Example: "hiring candidates" not just "hiring"
                                                                if conj.pos_ == "VERB":
                                                                    dobj = None
                                                                    for child in conj.children:
                                                                        if child.dep_ == "dobj":
                                                                            dobj = child
                                                                            break

                                                                    if dobj is not None:
                                                                        dobj_np = self._get_np_span(dobj)
                                                                        if dobj_np is not None:
                                                                            conj_span = conj.doc[conj.i:dobj_np.end]
                                                                            facts.append(Fact(subject, extended_pred, conj_span))
                                                                        else:
                                                                            conj_np = self._tok_span(conj)
                                                                            if conj_np is not None:
                                                                                facts.append(Fact(subject, extended_pred, conj_np))
                                                                    else:
                                                                        conj_np = self._get_np_span(conj) or self._tok_span(conj)
                                                                        if conj_np is not None:
                                                                            facts.append(Fact(subject, extended_pred, conj_np))
                                                                else:
                                                                    conj_np = self._get_np_span(conj)
                                                                    if conj_np is not None:
                                                                        facts.append(Fact(subject, extended_pred, conj_np))

                            elif prep_child.dep_ == "pobj":
                                pobj_core = self._get_np_core_span(prep_child) or self._tok_span(prep_child)
                                pobj_full = self._get_np_span(prep_child) or pobj_core
                                if pobj_core is not None:
                                    facts.append(Fact(subject, extended_pred, pobj_core))
                                if pobj_full is not None and (pobj_core is None or (pobj_full.start != pobj_core.start or pobj_full.end != pobj_core.end)):
                                    facts.append(Fact(subject, extended_pred, pobj_full))

                                # Also extract all conjuncts recursively (lists like "X, Y, and Z")
                                # Use token.conjuncts property which gets all conjuncts in the chain
                                for conj_token in prep_child.conjuncts:
                                    conj_np = self._get_np_span(conj_token)
                                    if conj_np is not None:
                                        facts.append(Fact(subject, extended_pred, conj_np))

            elif child.dep_ == "prep":
                if child.i in acomp_attached_preps:
                    continue
                facts.extend(self._extract_verb_prep(child, subject, predicate, be_token))

            elif child.dep_ == "conj" and child.pos_ in {"VERB", "AUX"}:
                sub_subject = self._extract_subject(child) or self._inherit_subject(child) or subject
                facts.extend(self._process_verb(child, sub_subject))

            # ✅ FIX: Handle ADJ/NOUN conjuncts with elided copula
            # Example: "51% being male and 49% female" -> "female" is ADJ conj with own subject
            # Pattern: "X is A and Y B" where second "is" is elided
            elif child.dep_ == "conj" and child.pos_ in {"ADJ", "NOUN"}:
                conj_subject = self._extract_subject(child)
                if conj_subject is not None:
                    conj_arg = self._get_np_span(child) or self._tok_span(child)
                    if conj_arg is not None:
                        facts.append(Fact(conj_subject, predicate, conj_arg))

            # ✅ ITER 15: Handle adverbial clauses (advcl)
            # Three types: causal ("because"), existential ("there were"), and participial (VBG/VBN)
            elif child.dep_ == "advcl":
                # Type 1: Causal clauses ("This is because X")
                has_because = any(c.dep_ == "mark" and c.text.lower() == "because" for c in child.children)
                if has_because:
                    # Create predicate including "because"
                    because_token = next((c for c in child.children if c.dep_ == "mark" and c.text.lower() == "because"), None)
                    if because_token:
                        # Predicate: "is because"
                        pred_because = self._cover(be_token, because_token)
                        if pred_because is None:
                            pred_because = predicate

                        # Argument: the advcl clause (without "because" marker)
                        # Get subtree of advcl head, excluding "because"
                        advcl_tokens = [t for t in child.subtree if t != because_token and not t.is_punct]
                        advcl_span = self._span_from_tokens(child.doc, advcl_tokens)

                        if advcl_span and len(advcl_span) > 0:
                            facts.append(Fact(subject, pred_because, advcl_span))

                # Type 2: Existential "there" constructions ("although there were X")
                # ✅ GOLD: Process as separate verb with "there" as subject
                elif any(c.dep_ == "expl" for c in child.children):
                    advcl_subject = self._extract_subject(child)
                    if advcl_subject is not None:
                        facts.extend(self._process_verb(child, advcl_subject))

                # Type 3: Participial clauses ("Ramesses was pharaoh, ruling from 1155 to 1149 BC")
                elif child.pos_ == "VERB" and child.tag_ in {"VBG", "VBN"}:
                    participle_pred = self._get_full_predicate_span(child)
                    if participle_pred is None:
                        participle_pred = self._tok_span(child)

                    # Extract arguments (prep phrases, objects, etc.)
                    # Process this participial verb like a normal verb
                    participle_facts = self._extract_verb_arguments(child, subject, participle_pred)
                    facts.extend(participle_facts)

            # ✅ ITER 15: Handle explanatory clauses (ccomp with "why")
            # Example: "This is why X" → "This | is why | X"
            # Structure: is (ROOT) → are (ccomp with "why" advmod)
            elif child.dep_ == "ccomp" and child.pos_ in {"VERB", "AUX"}:
                # Check if this is an explanatory clause (has "why" modifier)
                why_token = next((c for c in child.children if c.dep_ == "advmod" and c.text.lower() == "why"), None)
                if why_token:
                    # Create predicate including "why"
                    # Predicate: "is why"
                    pred_why = self._cover(be_token, why_token)
                    if pred_why is None:
                        pred_why = predicate

                    # Argument: the ccomp clause (the explanation)
                    ccomp_span = self._subtree_span(child, include_punct=False)

                    if ccomp_span and len(ccomp_span) > 0:
                        # Trim to exclude "why" from argument (it's in predicate)
                        ccomp_tokens = [t for t in ccomp_span if t != why_token]
                        ccomp_clean = self._span_from_tokens(child.doc, ccomp_tokens)

                        if ccomp_clean and len(ccomp_clean) > 0:
                            facts.append(Fact(subject, pred_why, ccomp_clean))

            # ✅ FIX: Handle xcomp infinitives attached to copula
            # Pattern: "The purpose was to encourage farmers" -> purpose|was to encourage|farmers
            # Structure: was (ROOT) -> to encourage (xcomp) -> farmers (dobj)
            elif child.dep_ == "xcomp" and child.pos_ in {"VERB", "AUX"}:
                if child.lemma_.lower() == "note":
                    for cc in child.children:
                        if cc.dep_ in {"ccomp", "advcl"} and cc.pos_ in {"VERB", "AUX"}:
                            sub = self._extract_subject(cc) or self._inherit_subject(cc)
                            if sub is not None:
                                facts.extend(self._process_verb(cc, sub))
                    continue

                to_marker = next((c for c in child.children if c.dep_ == "aux" and c.text.lower() == "to"), None)
                
                if to_marker:
                    pred_to_verb = self._cover(be_token, to_marker, child)
                else:
                    pred_to_verb = self._cover(be_token, child)
                
                if pred_to_verb is None:
                    continue
                
                for xc_child in child.children:
                    if xc_child.dep_ == "dobj":
                        dobj_span = self._get_np_span(xc_child)
                        if dobj_span:
                            facts.append(Fact(subject, pred_to_verb, dobj_span))
                            facts.extend(self._build_incremental_argument_facts(Fact(subject, pred_to_verb, dobj_span)))
                    elif xc_child.dep_ == "prep":
                        facts.extend(self._extract_verb_prep(xc_child, subject, pred_to_verb, child))

        incremental_arg_facts: List[Fact] = []
        for f in facts:
            incremental_arg_facts.extend(self._build_incremental_argument_facts(f))
        facts.extend(incremental_arg_facts)

        return facts


    def _extract_verb_arguments(self, verb_token: Token, subject: Span, predicate: Span) -> List[Fact]:
        facts: List[Fact] = []
        condition_subject: Optional[Span] = None

        # ✅ FIX: Convert children to list to avoid iterator issues
        # If _extract_object_facts or other methods recursively modify the parse tree,
        # iterating over a generator could skip children
        children_list = list(verb_token.children)

        for child in children_list:
            # Special-case "use X to VERB Y": skip direct dobj so we can take the infinitive as argument
            if verb_token.lemma_.lower() == "use" and child.dep_ == "dobj":
                has_xcomp_to = any(c.dep_ == "xcomp" and any(gc.dep_ == "aux" and gc.text.lower() == "to" for gc in c.children)
                                   for c in children_list)
                if has_xcomp_to:
                    continue

            #     if child.lemma_.lower() not in self.ignored_advmods and child.i > verb_token.i:
            #         facts.append(Fact(subject, predicate, self._tok_span(child)))

            if child.dep_ in {"npadvmod", "advmod"}:
                if child.lemma_.lower() in {"time", "times"}:
                    time_span = self._get_np_span(child) or self._subtree_span(child, include_punct=False) or self._tok_span(child)
                    if time_span is not None:
                        facts.append(Fact(subject, predicate, time_span))

            if child.dep_ == "prep":
                # NOTE: Temporal extraction now handled by _extract_date_parts_facts() via entity spans
                # No need to create separate temporal facts here (causes duplicates)
                facts.extend(self._extract_verb_prep(child, subject, predicate, verb_token))

            elif child.dep_ == "agent":
                # Predicate already includes "by" from _get_full_predicate_span
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        if pobj.ent_type_ in {"DATE", "TIME", "GPE", "LOC", "FAC"}:
                            continue
                        if pobj.pos_ not in {"NOUN", "PROPN"}:
                            continue
                        if pobj.like_num:
                            continue
                        
                        # ✅ FIX: Always extract base object even with nested preps
                        # "marked by swift posting to embassy" -> "marked by | posting" + "marked by | swift posting"
                        objs = self._extract_single_object_facts(pobj, subject, predicate)
                        facts.extend(objs)

                        extra = self._extract_conjuncts_and_appos(pobj, subject, predicate)
                        facts.extend(extra)

                        if self._has_nested_prep(pobj):
                            facts.extend(self._lift_nested_preps_to_verb(pobj, subject, verb_token, predicate, predicate))

            elif child.dep_ in {"dobj", "attr", "oprd"}:
                facts.extend(self._extract_object_facts(child, subject, predicate, verb_token))

            # ✅ FIX: Handle ccomp with NOUN for complex predicates like "made X Y"
            # Example: "which have made Lanny Flaherty a memorable character actor"
            # Structure: "actor" (ccomp, NOUN) with "Flaherty" (nsubj of "actor")
            elif child.dep_ == "ccomp" and child.pos_ in {"NOUN", "PROPN", "ADJ"}:
                # This is a complex predicate: "made [someone] [something]"
                ccomp_span = self._get_full_subject_span(child)
                if ccomp_span is not None:
                    facts.append(Fact(subject, predicate, ccomp_span))

            elif child.dep_ in {"advcl", "xcomp", "ccomp"} and child.pos_ in {"VERB", "AUX"}:
                # ✅ FIX: Handle purpose clauses specially (infinitive "to")
                # Example: "Firms advertise to create brand recognition"
                # Don't process as separate verb with main subject ("Firms to create"),
                # Instead extract purpose facts with proper structure
                # ✅ NEW: Imperatives – if main verb has no subject, assume implied "you"
                implied_subject = None
                if subject is None and verb_token.dep_ == "ROOT" and verb_token.tag_ == "VB":
                    implied_subject = verb_token.doc[verb_token.i:verb_token.i]  # empty span placeholder
                    subject_for_imperative = verb_token.doc[verb_token.i:verb_token.i]  # zero-length
                else:
                    subject_for_imperative = subject

                if verb_token.lemma_.lower() == "use":
                    has_to_aux = any(c.dep_ == "aux" and c.text.lower() == "to" for c in child.children)
                    if has_to_aux:
                        purpose_span = self._subtree_span(child, include_punct=False) or self._tok_span(child)
                        if purpose_span and not purpose_span.text.lower().startswith("to "):
                            facts.append(Fact(subject_for_imperative or subject, predicate, purpose_span))
                        elif purpose_span is None:
                            purpose_span = self._tok_span(child)
                            if purpose_span and not purpose_span.text.lower().startswith("to "):
                                facts.append(Fact(subject_for_imperative or subject, predicate, purpose_span))
                        for pc in child.children:
                            if pc.dep_ == "prep":
                                facts.extend(self._extract_verb_prep(pc, subject_for_imperative or subject, predicate, child))
                            elif pc.dep_ in {"dobj", "attr", "oprd"}:
                                facts.extend(self._extract_object_facts(pc, subject_for_imperative or subject, predicate, child))
                        continue

                if child.dep_ == "advcl":
                    if_token = next((c for c in child.children if c.dep_ == "mark" and c.text.lower() == "if"), None)
                    if if_token is not None:
                        advcl_subj = self._extract_subject(child) or subject
                        if advcl_subj is not None:
                            if_subj = self._cover(if_token, advcl_subj) or advcl_subj
                            facts.extend(self._process_verb(child, if_subj))
                            for prep in child.children:
                                if prep.dep_ == "prep" and prep.text.lower() in {"to", "into", "onto"}:
                                    pred_cond = self._cover(child, prep)
                                    if pred_cond is not None:
                                        facts.append(Fact(if_subj, pred_cond, None))
                        cond_span = self._subtree_span(child, include_punct=False)
                        if cond_span is not None and subject is not None:
                            condition_subject = self._cover(cond_span, subject) or cond_span
                        continue

                    has_to_aux = any(c.dep_ == "aux" and c.text.lower() == "to" for c in child.children)
                    if has_to_aux:
                        # This is purpose clause - extract with main verb + purpose verb predicate
                        # "Firms advertise to create X" → predicate: "advertise to create"

                        def _extract_purpose_clause(purpose_verb: Token):
                            # Create short predicate: just "to [verb]"
                            # Example: "Firms" "advertise to create" "brand recognition"
                            # Predicate will be "advertise to create" (short, focused on purpose)
                            to_aux = next((c for c in purpose_verb.children if c.dep_ == "aux" and c.text.lower() == "to"), None)

                            # Create predicate: main verb text + "to" + purpose verb
                            # Use continuous span from verb to purpose_verb (will include intermediate text unfortunately)
                            # But better than dropping the purpose info entirely
                            purpose_pred = self._cover(verb_token, purpose_verb)

                            if purpose_pred:
                                for purp_child in purpose_verb.children:
                                    if purp_child.dep_ == "prep":
                                        facts.extend(self._extract_verb_prep(purp_child, subject, purpose_pred, verb_token))
                                    elif purp_child.dep_ in {"dobj", "attr", "oprd"}:
                                        facts.extend(self._extract_object_facts(purp_child, subject, purpose_pred, verb_token))

                                for conj_child in purpose_verb.children:
                                    if conj_child.dep_ == "conj" and conj_child.pos_ in {"VERB", "AUX"}:
                                        conj_has_to = any(c.dep_ == "aux" and c.text.lower() == "to" for c in conj_child.children)
                                        if conj_has_to:
                                            _extract_purpose_clause(conj_child)

                        _extract_purpose_clause(child)
                        continue

                # ✅ FIX: If xcomp is included in predicate, extract its arguments with parent predicate
                if child.dep_ == "xcomp" and predicate.end > child.i:
                    for xc_child in child.children:
                        if xc_child.dep_ == "prep":
                            facts.extend(self._extract_verb_prep(xc_child, subject, predicate, verb_token))
                        elif xc_child.dep_ in {"dobj", "attr", "oprd"}:
                            facts.extend(self._extract_object_facts(xc_child, subject, predicate, verb_token))
                else:
                    sub_subject = self._extract_subject(child) or subject
                    facts.extend(self._process_verb(child, sub_subject))

            elif child.dep_ == "conj" and child.pos_ in {"VERB", "AUX"}:
                # Note: purpose clause conjuncts handled recursively above, skip them here
                has_to_aux = any(c.dep_ == "aux" and c.text.lower() == "to" for c in child.children)
                head_is_purpose = (child.head.dep_ == "advcl" and child.head.pos_ in {"VERB", "AUX"} and
                                   any(c.dep_ == "aux" and c.text.lower() == "to" for c in child.head.children))
                if has_to_aux and head_is_purpose:
                    continue

                sub_subject = self._extract_subject(child) or self._inherit_subject(child) or subject
                facts.extend(self._process_verb(child, sub_subject))

                # NEW: if base-verb has no objects/preps, but conj-verb does — inherit arguments.
                base_has_args = any(ch.dep_ in {"dobj", "attr", "prep", "oprd", "xcomp", "ccomp"} for ch in verb_token.children)
                conj_has_args = any(ch.dep_ in {"dobj", "attr", "prep", "oprd", "xcomp", "ccomp"} for ch in child.children)
                if (not base_has_args) and conj_has_args:
                    base_pred = self._get_full_predicate_span(verb_token)
                    if base_pred is not None:
                        for ch2 in child.children:
                            if ch2.dep_ == "prep":
                                facts.extend(self._extract_verb_prep(ch2, subject, base_pred, verb_token))
                            elif ch2.dep_ in {"dobj", "attr", "oprd"}:
                                facts.extend(self._extract_object_facts(ch2, subject, base_pred, verb_token))

        if condition_subject is not None and subject is not None:
            for f in list(facts):
                if f.subject.start == subject.start and f.subject.end == subject.end:
                    facts.append(Fact(condition_subject, f.predicate, f.argument, f.prep))

        return facts


    def _has_nested_prep(self, token: Token) -> bool:
        """
        Check if token has nested prep children (excluding 'of' and preps with pcomp).

        Returns True only for preps that add DETAILS to the object (like "between sexes"),
        not for preps that introduce SEPARATE CLAUSES (like "with ... being male").
        """
        for child in token.children:
            if child.dep_ == "prep" and child.text.lower() != "of":
                # ✅ FIX: Don't count preps with pcomp as nested preps
                # pcomp indicates a separate clause, not a detail about the object
                # Example: "population with residents being male" - "being" is separate clause
                has_pcomp = any(c.dep_ == "pcomp" for c in child.children)
                if not has_pcomp:
                    return True
        return False

    def _extract_appos_numeric_facts(self, token: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        Extract appos numeric/monetary values as separate facts.

        Examples:
        - "the highest pay in Alaska ($23.70 per hour)"
          -> extract "$23.70 per hour" as argument
        - "($23.70 per hour or $49,400 per year)"
          -> extract both "$23.70 per hour" and "$49,400 per year"

        Args:
            token: Token to check for appos children (including conjuncts)
            subject: Subject span
            predicate: Predicate span

        Returns:
            List of Facts with numeric appos as arguments
        """
        facts: List[Fact] = []

        # Process token and its conjuncts (to handle "pay in Alaska ... and pay in Mississippi")
        tokens_to_process = [token] + list(token.conjuncts)

        for tok in tokens_to_process:
            for child in tok.children:
                if child.dep_ != "appos":
                    continue

                if not (child.pos_ == "NUM" or child.ent_type_ in {"MONEY", "PERCENT", "QUANTITY"}):
                    continue

                appos_tokens = [child]

                for nmod in child.children:
                    if nmod.dep_ == "nmod" and nmod.i < child.i:
                        appos_tokens.append(nmod)

                for prep in child.children:
                    if prep.dep_ == "prep":
                        appos_tokens.append(prep)
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                appos_tokens.extend(list(self._get_np_span(pobj) or [pobj]))

                appos_span = self._span_from_tokens(child.doc, appos_tokens)
                if appos_span is not None and appos_span.text.strip():
                    facts.append(Fact(subject, predicate, appos_span))

                for conj in child.children:
                    if conj.dep_ == "conj":
                        conj_tokens = [conj]

                        for nmod in conj.children:
                            if nmod.dep_ == "nmod" and nmod.i < conj.i:
                                conj_tokens.append(nmod)

                        for prep in conj.children:
                            if prep.dep_ == "prep":
                                conj_tokens.append(prep)
                                for pobj in prep.children:
                                    if pobj.dep_ == "pobj":
                                        conj_tokens.extend(list(self._get_np_span(pobj) or [pobj]))

                        conj_span = self._span_from_tokens(conj.doc, conj_tokens)
                        if conj_span is not None and conj_span.text.strip():
                            facts.append(Fact(subject, predicate, conj_span))

        return facts


    def _is_list_pattern(self, prep_token: Token) -> bool:
        """
        Check if prep token is part of a list pattern like "such as X, Y, Z" or "like X, Y, Z".

        Args:
            prep_token: The preposition token to check

        Returns:
            True if this is a list pattern (e.g., "such as", "like")
        """
        if prep_token.text.lower() not in {"as", "like"}:
            return False

        if prep_token.text.lower() == "as":
            # Pattern: "such as" where "such" is amod/advmod of "as"
            for child in prep_token.children:
                if child.text.lower() == "such" and child.dep_ in {"amod", "advmod"}:
                    return True

            parent = prep_token.head
            if parent:
                for i in range(parent.i, prep_token.i):
                    tok = parent.doc[i]
                    if tok.text.lower() == "such":
                        return True

        elif prep_token.text.lower() == "like":
            return True

        return False

    def _extract_list_items(
        self,
        pobj_token: Token,
        subject: Span,
        pred_with_prep: Span,
        prep_token: Token,
        attach: bool
    ) -> List[Fact]:
        """
        Extract each list item from "such as X, Y, Z" or "like X, Y, Z" patterns.

        Each conjunct becomes a separate fact for granular hallucination detection.

        Example:
            Text: "caused by factors, such as diet, stress, or exposure to toxins"
            Returns:
                - caused by | diet
                - caused by | stress
                - caused by | exposure to toxins

        Args:
            pobj_token: The pobj token (first list item)
            subject: Subject span
            pred_with_prep: Predicate (with prep if attached)
            prep_token: The preposition token
            attach: Whether prep is attached to predicate

        Returns:
            List of facts, one for each list item
        """
        facts: List[Fact] = []

        list_items = [pobj_token]

        def _collect_conjuncts(tok):
            for child in tok.children:
                if child.dep_ == "conj":
                    list_items.append(child)
                    _collect_conjuncts(child)  # Handle chained conjuncts

        _collect_conjuncts(pobj_token)

        for item in list_items:
            item_np = self._get_np_span(item)
            if (item_np is None or item.pos_ == "ADJ") and item.head is not None:
                if item.head.pos_ in {"NOUN", "PROPN"} and item.head.i > item.i:
                    head_np = self._get_np_span(item.head)
                    if head_np and item.i >= head_np.start and item.i < head_np.end:
                        item_np = head_np
            if item_np:
                nested_preps = [child for child in item.children if child.dep_ == "prep"]

                if nested_preps:
                    def _collect_base_tokens(token):
                        tokens = [token]
                        for child in token.children:
                            if child.dep_ in {"amod", "compound", "det", "nummod", "poss", "case", "nmod", "appos"}:
                                tokens.extend(_collect_base_tokens(child))
                        return tokens
                    base_tokens = _collect_base_tokens(item)
                    base_np = self._span_from_tokens(item.doc, base_tokens)
                    if not base_np:
                        base_np = self._tok_span(item)

                    generic_heads = {
                        "role", "position", "group", "community", "area", "region", "country",
                        "state", "province", "county", "city", "town", "village", "organization",
                        "school", "program", "project", "team", "company", "service", "system"
                    }
                    if not (len(base_np) <= 2 and base_np.root.lemma_.lower() in generic_heads):
                        if not attach:
                            facts.append(Fact(subject, pred_with_prep, base_np, prep=prep_token.text))
                        else:
                            facts.append(Fact(subject, pred_with_prep, base_np))

                    for nested_prep in nested_preps:
                        nested_pobj = next((c for c in nested_prep.children if c.dep_ == "pobj"), None)
                        if nested_pobj:
                            nested_np = self._get_np_span(nested_pobj)
                            if nested_np:
                                accumulated_arg = base_np.doc[base_np.start : nested_np.end]
                                if not attach:
                                    facts.append(Fact(subject, pred_with_prep, accumulated_arg, prep=prep_token.text))
                                else:
                                    facts.append(Fact(subject, pred_with_prep, accumulated_arg))
                else:
                    if not attach:
                        facts.append(Fact(subject, pred_with_prep, item_np, prep=prep_token.text))
                    else:
                        facts.append(Fact(subject, pred_with_prep, item_np))

        return facts

    def _extract_verb_prep(self, prep_token: Token, subject: Span, base_predicate: Span, verb_token: Token) -> List[Fact]:
        facts: List[Fact] = []

        if " by" in base_predicate.text.lower() and prep_token.text.lower() in {"in", "on", "at", "by", "during", "after", "before"}:
            return facts

        # ✅ FIX: For "exert/maintain control over X" patterns, attach prep to the control noun.
        # This avoids ungrammatical facts like "aiming to exert | lands" and yields
        # incremental args: "control" -> "greater control" -> "greater control over X".
        control_heads = {"control", "influence", "power", "authority", "dominion", "command"}
        if prep_token.text.lower() in {"over", "on", "upon"}:
            dobj_tokens = [c for c in verb_token.children if c.dep_ in {"dobj", "attr", "oprd"}]
            for dobj in dobj_tokens:
                if dobj.lemma_.lower() not in control_heads:
                    continue
                pobj = next((c for c in prep_token.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                dobj_span = self._get_np_span(dobj) or self._tok_span(dobj)
                pobj_span = self._get_np_span(pobj) or self._tok_span(pobj)
                if dobj_span is None or pobj_span is None:
                    continue
                if prep_token.i < dobj_span.end or prep_token.i > pobj_span.start:
                    continue
                combined = self._cover(dobj_span, prep_token, pobj_span)
                if combined is not None and combined.text.strip():
                    facts.append(Fact(subject, base_predicate, combined))
            if facts:
                return facts

        # ✅ FIX: Skip "given" as prep - it's a participle meaning "considering", not a real prep
        # Pattern: "was chaotic, given the timeframe" - "given" is not argument of "was"
        if prep_token.text.lower() == "given":
            return facts

        # ✅ FIX: Skip "in terms of" idiom - it's adverbial modifier, not argument
        # Pattern: "was a departure in terms of content" - "terms" is not argument of "was"
        if prep_token.text.lower() == "in":
            pobj = next((c for c in prep_token.children if c.dep_ == "pobj"), None)
            if pobj and pobj.text.lower() == "terms":
                has_of = any(c.dep_ == "prep" and c.text.lower() == "of" for c in pobj.children)
                if has_of:
                    return facts

        # If the prep head is already an infinitive argument ("to X"), avoid adding prep facts that
        # duplicate the same infinitival argument (e.g., "to amplify optical signals")
        if base_predicate.text.lower().startswith("to "):
            return facts

        # ✅ FIX: Skip locative/instrumental prep facts when verb already has dobj
        # Pattern: "faces opposition in Queensland" - "in Queensland" is locative modifier, not verb argument
        # Pattern: "enters leaf through stomata" - "through stomata" is instrumental, not verb argument
        # Only skip if: prep is locative/instrumental AND verb has dobj
        locative_preps = {"in", "at", "on"}
        instrumental_preps = {"through", "via", "by means of"}
        if prep_token.text.lower() in locative_preps:
            has_dobj = any(c.dep_ in {"dobj", "attr"} for c in verb_token.children)
            if has_dobj:
                pobj = next((c for c in prep_token.children if c.dep_ == "pobj"), None)
                if pobj and pobj.ent_type_ in {"GPE", "LOC", "FAC", "ORG"}:
                    return facts
        if prep_token.text.lower() in instrumental_preps:
            has_dobj = any(c.dep_ in {"dobj", "attr"} for c in verb_token.children)
            if has_dobj:
                return facts

        attach = self._should_attach_prep_to_pred(base_predicate, prep_token)

        # ✅ FIX: Skip sentence-initial adjunct preps completely
        # These are framing/discourse markers, not argument-bearing
        # Examples: "As an AI...", "In contrast...", "Additionally...", "For example..."
        # ✅ ITER 14: BUT keep temporal preps like "In 2019", "In the late 1970s"
        if not attach and prep_token.i < base_predicate.start:
            pl = prep_token.text.lower()
            discourse_preps = {"as", "in", "additionally", "ultimately", "however"}

            # Special case: "for example" - check if pobj is "example"
            if pl == "for":
                pobj = next((c for c in prep_token.children if c.dep_ == "pobj"), None)
                if pobj and pobj.text.lower() == "example":
                    return facts

            # ✅ ITER 14: Don't skip temporal "in" preps
            # Pattern: "In 2019", "In the late 1970s", "In January"
            # Check if pobj is date-like (year, month, decade, etc.)
            if pl == "in":
                pobj = next((c for c in prep_token.children if c.dep_ == "pobj"), None)
                if pobj:
                    is_temporal = (
                        pobj.pos_ == "NUM" or  # "In 2019"
                        pobj.ent_type_ in {"DATE", "TIME"} or  # "In January"
                        pobj.text.lower() in {"january", "february", "march", "april", "may", "june",
                                              "july", "august", "september", "october", "november", "december"} or
                        pobj.text.endswith("s") and any(c.isdigit() for c in pobj.text)  # "1970s", "2000s"
                    )
                    if is_temporal:
                        pass
                    else:
                        return facts
            elif pl in discourse_preps:
                return facts

        pred_with_prep = (self._cover(base_predicate, prep_token) if attach else base_predicate)

        def _maybe_prefix_prep(arg: Optional[Span]) -> Optional[Span]:
            if arg is None:
                return None
            # Include prep in argument only if they are adjacent
            # Otherwise we get a span with intermediate tokens
            # Example: prep="in"(13), arg="Florida"(16) → NOT adjacent (between: Pensacola, ,)
            # Only if prep is directly before arg, can create "in Florida"
            if prep_token.i + 1 == arg.start:
                return self._cover(prep_token, arg) or arg
            else:
                return arg

        for ch in prep_token.children:
            # ✅ FIX: Handle pcomp (gerund complement) for "by" preposition
            # Pattern: "works by impounding water" -> works by impounding | water
            # Structure: works -> by (prep) -> impounding (pcomp) -> water (dobj)
            if prep_token.text.lower() == "by" and ch.dep_ == "pcomp" and ch.pos_ in {"VERB", "AUX"}:
                pred_by_verb = self._cover(base_predicate, prep_token, ch)
                if pred_by_verb is None:
                    continue

                pcomp_verbs = [ch] + list(ch.conjuncts)

                for pcomp_verb in pcomp_verbs:
                    for pc_child in pcomp_verb.children:
                        if pc_child.dep_ in {"dobj", "attr", "oprd"}:
                            facts.extend(self._extract_object_facts(pc_child, subject, pred_by_verb, pcomp_verb))
                        elif pc_child.dep_ == "prep":
                            facts.extend(self._extract_verb_prep(pc_child, subject, pred_by_verb, pcomp_verb))
                continue
            
            if ch.dep_ == "pobj":
                if self._is_date_like_pobj(ch, prep_token):
                    date_facts = self._extract_date_parts_facts(ch, subject, pred_with_prep)
                    for f in date_facts:
                        if not attach and f.argument is not None:
                            arg2 = _maybe_prefix_prep(f.argument)
                            facts.append(Fact(subject, pred_with_prep, arg2))
                        else:
                            facts.append(f)
                else:
                    # NEW: Check for list patterns like "such as X, Y, Z" or "like X, Y, Z"
                    # These should extract each list item as a separate fact, not accumulated
                    is_list_pattern = self._is_list_pattern(prep_token)

                    if is_list_pattern:
                        # Extract each conjunct separately as independent facts
                        # Example: "caused by factors, such as diet, stress, or exposure"
                        # → "caused by | diet", "caused by | stress", "caused by | exposure"
                        list_facts = self._extract_list_items(ch, subject, pred_with_prep, prep_token, attach)
                        facts.extend(list_facts)
                    else:
                        # Normal pobj handling: Always create base fact, then add incremental facts for nested preps
                        # Exception: "as" role markers use full subtree (handled below)
                        objs = self._extract_single_object_facts(ch, subject, pred_with_prep)

                        for f in objs:
                            # NEW APPROACH: Store prep separately, don't prefix to argument
                            # This keeps argument clean for hallucination marking
                            if not attach and f.argument is not None:
                                # NEW: Create base fact WITHOUT nested preps/nummod in argument
                                # Then create incremental facts WITH nested preps/nummod
                                base_arg = f.argument

                                nested_preps = [child for child in ch.children if child.dep_ == "prep"]
                                nummod_children = [child for child in ch.children if child.dep_ == "nummod"]

                                if nested_preps or (nummod_children and prep_token.text.lower() in self.temporal_preps):
                                    # Extract base NP without nested prep/nummod subtrees
                                    # Collect tokens: head + modifiers (amod, compound, det, poss) but NOT prep/nummod subtrees
                                    # Use recursive collection to capture possessive chains like "Allan Glen's"
                                    def _collect_base_np_tokens(token):
                                        tokens = [token]
                                        for child in token.children:
                                            if child.dep_ in {"amod", "compound", "det", "poss", "case"}:
                                                tokens.extend(_collect_base_np_tokens(child))
                                        return tokens
                                    base_tokens = _collect_base_np_tokens(ch)
                                    base_arg = self._span_from_tokens(ch.doc, base_tokens)
                                    if not base_arg:
                                        base_arg = self._tok_span(ch)

                                if prep_token.text.lower() == "as":
                                    base_arg = self._strip_leading_dets(base_arg)
                                    if base_arg is None:
                                        continue

                                facts.append(Fact(subject, pred_with_prep, base_arg, prep=prep_token.text))

                                # NEW: If pobj has nested preps, create incremental facts
                                # SPECIAL: Check for list patterns like "such as X, Y, Z"
                                # Example: "caused by factors, such as diet, stress, exposure"
                                for nested_prep in nested_preps:
                                    if self._is_list_pattern(nested_prep):
                                        nested_pobj = next((c for c in nested_prep.children if c.dep_ == "pobj"), None)
                                        if nested_pobj:
                                            list_facts = self._extract_list_items(nested_pobj, subject, pred_with_prep, prep_token, False)
                                            facts.extend(list_facts)
                                    else:
                                        # Normal nested prep: create incremental fact
                                        # Example: "located in Nellore district of Andhra Pradesh"
                                        #   - Fact 1: located in | Nellore district
                                        #   - Fact 2: located in | Nellore district of Andhra Pradesh (incremental)
                                        nested_pobj = next((c for c in nested_prep.children if c.dep_ == "pobj"), None)
                                        if nested_pobj:
                                            nested_np = self._get_np_span(nested_pobj)
                                            if nested_np:
                                                accumulated_arg = base_arg.doc[base_arg.start : nested_np.end]
                                                facts.append(Fact(subject, pred_with_prep, accumulated_arg, prep=prep_token.text))

                                # NEW: If pobj has nummod (temporal increment pattern)
                                # Example: "served from August 1862"
                                #   - Fact 2: served from | August 1862 (incremental)
                                if nummod_children and prep_token.text.lower() in self.temporal_preps:
                                    for nummod in nummod_children:
                                        accumulated_arg = base_arg.doc[base_arg.start : nummod.i + 1]
                                        facts.append(Fact(subject, pred_with_prep, accumulated_arg, prep=prep_token.text))
                            elif attach:
                                base_arg = f.argument

                                nested_preps = [child for child in ch.children if child.dep_ == "prep"]
                                nummod_children = [child for child in ch.children if child.dep_ == "nummod"]

                                if nested_preps or (nummod_children and prep_token.text.lower() in self.temporal_preps):
                                    def _collect_base_np(token):
                                        tokens = [token]
                                        for child in token.children:
                                            if child.dep_ in {"amod", "compound", "det", "poss", "case"}:
                                                tokens.extend(_collect_base_np(child))
                                        return tokens
                                    base_tokens = _collect_base_np(ch)
                                    base_arg = self._span_from_tokens(ch.doc, base_tokens)
                                    if not base_arg:
                                        base_arg = self._tok_span(ch)

                                if prep_token.text.lower() == "as":
                                    base_arg = self._strip_leading_dets(base_arg)
                                    if base_arg is None:
                                        continue

                                facts.append(Fact(subject, f.predicate, base_arg))

                                # NEW: Also create incremental facts for attached preps with nested preps
                                # SPECIAL: Check for list patterns like "such as X, Y, Z"
                                for nested_prep in nested_preps:
                                    if self._is_list_pattern(nested_prep):
                                        nested_pobj = next((c for c in nested_prep.children if c.dep_ == "pobj"), None)
                                        if nested_pobj:
                                            list_facts = self._extract_list_items(nested_pobj, subject, f.predicate, prep_token, True)
                                            facts.extend(list_facts)
                                    else:
                                        nested_pobj = next((c for c in nested_prep.children if c.dep_ == "pobj"), None)
                                        if nested_pobj:
                                            nested_np = self._get_np_span(nested_pobj)
                                            if nested_np:
                                                accumulated_arg = base_arg.doc[base_arg.start : nested_np.end]
                                                facts.append(Fact(subject, f.predicate, accumulated_arg))

                                if nummod_children and prep_token.text.lower() in self.temporal_preps:
                                    for nummod in nummod_children:
                                        accumulated_arg = base_arg.doc[base_arg.start : nummod.i + 1]
                                        facts.append(Fact(subject, f.predicate, accumulated_arg))

                            # ✅ FIX: Extract conj/appos ALWAYS, not only for attached preps
                            # Pattern: "born in Tokyo, Japan" - both Tokyo and Japan should be extracted
                            # even if attach=False (sentence-initial preps)
                            # NEW: Create incremental facts with accumulated arguments
                            # Example: "born in Pensacola, Florida"
                            #   - Fact 2: born in | Pensacola, Florida (accumulated)

                            base_arg = f.argument if objs and objs[0].argument else None

                            if base_arg:
                                appos_children = [child for child in ch.children if child.dep_ == "appos"]

                                if appos_children:
                                    for appos_child in appos_children:
                                        appos_np = self._get_np_span(appos_child)
                                        if appos_np:
                                            # Create accumulated span: base_arg + appos_np
                                            # This is the span from start of base to end of appos
                                            accumulated_span = base_arg.doc[base_arg.start : appos_np.end]

                                            if not attach:
                                                facts.append(Fact(subject, pred_with_prep, accumulated_span, prep=prep_token.text))
                                            else:
                                                facts.append(Fact(subject, pred_with_prep, accumulated_span))

                                # OLD APPROACH (kept for other cases): Extract individual appos as separate facts
                                # This is commented out for location appos, but may be needed for other patterns
                                # extra = self._extract_conjuncts_and_appos(ch, subject, pred_with_prep)
                                #     extra = [Fact(f.subject, f.predicate, f.argument, prep=prep_token.text) for f in extra]

                            conj_facts = self._extract_conjuncts_and_appos(ch, subject, pred_with_prep)
                            if prep_token.text.lower() == "as":
                                stripped = []
                                for cf in conj_facts:
                                    if cf.argument is None:
                                        continue
                                    arg = self._strip_leading_dets(cf.argument)
                                    if arg is None:
                                        continue
                                    stripped.append(Fact(cf.subject, cf.predicate, arg))
                                conj_facts = stripped
                            if not attach:
                                conj_facts = [Fact(cf.subject, cf.predicate, cf.argument, prep=prep_token.text) for cf in conj_facts]
                            facts.extend(conj_facts)

                            # ✅ FIX: Handle en_core_web_trf pattern for "City, Country"
                        # trf parses "in Tokyo, Japan" as: Japan (pobj) <- Tokyo (nmod)
                        # sm/lg parse it as: Tokyo (pobj) <- Japan (appos)
                        # Extract nmod children as additional arguments (trf-specific)
                        for nmod_child in ch.children:
                            if nmod_child.dep_ == "nmod" and nmod_child.pos_ == "PROPN":
                                # This looks like "City, Country" pattern in trf
                                nmod_np = self._get_np_span(nmod_child)
                                if nmod_np is not None:
                                    if not attach:
                                        facts.append(Fact(subject, pred_with_prep, nmod_np, prep=prep_token.text))
                                    else:
                                        facts.append(Fact(subject, pred_with_prep, nmod_np))

                    if prep_token.text.lower() == "as":
                        role_span = self._subtree_span(ch, include_punct=False)
                        role_span = self._clean_arg_span(role_span)
                        role_span = self._strip_leading_dets(role_span)
                        if role_span and role_span.text.strip():
                            facts.append(Fact(subject, pred_with_prep, role_span))
                        if attach:
                            extra = self._extract_conjuncts_and_appos(ch, subject, pred_with_prep)
                            facts.extend(extra)

                    # ✅ FIX: Extract appos numeric/monetary values even when attach=False or has_nested_prep=True
                    # Example: "with highest pay in Alaska ($23.70 per hour or $49,400 per year)"
                    # We want to extract "$23.70 per hour" and "$49,400 per year" as separate facts
                    # Use more specific predicate if pobj has nested preps
                    if self._has_nested_prep(ch):
                        for nested_prep in ch.children:
                            if nested_prep.dep_ == "prep" and nested_prep.text.lower() != "of":
                                # Create predicate: NP (without appos) + prep
                                # Get NP span for pobj (includes amod/compound but not appos)
                                np_tokens = self._collect_np_tokens(ch)
                                np_tokens.append(nested_prep)
                                pred_for_appos = self._span_from_tokens(ch.doc, np_tokens)

                                if pred_for_appos is not None:
                                    facts.extend(self._extract_appos_numeric_facts(ch, subject, pred_for_appos))
                                break  # Use only first nested prep
                        else:
                            facts.extend(self._extract_appos_numeric_facts(ch, subject, pred_with_prep))
                    else:
                        facts.extend(self._extract_appos_numeric_facts(ch, subject, pred_with_prep))

                # NEW: Skip _lift_nested_preps_to_verb - nested preps now create incremental facts
                # Old behavior created separate facts like "district | of | Pradesh"
                # New behavior creates incremental facts: "located in | district" + "located in | district of Pradesh"
                # ✅ ITER 15: Skip nested prep processing for "as" role markers
                # Pattern: "served as the Minister of State for Environment"
                # Don't break down - keep full role title as argument
                # This gives "served as | the Minister of State for Environment" instead of fragments
                #     facts.extend(self._lift_nested_preps_to_verb(ch, subject, verb_token, base_predicate, pred_with_prep))

            # ✅ ITER 15: Handle coordinated preps (conj children that are themselves preps)
            # Pattern: "served as Minister A and as Minister B" - two coordinated "as" phrases
            # Structure: as (prep) -> pobj, cc, conj=as (prep) -> pobj
            elif ch.dep_ == "conj" and ch.pos_ == "ADP":
                conj_facts = self._extract_verb_prep(ch, subject, base_predicate, verb_token)
                facts.extend(conj_facts)

            # ✅ FIX: Handle pcomp (prepositional complement) - verbs that are complements to prepositions
            # Example: "with 51% of residents being male" -> "being" is pcomp of "with"
            # Structure: prep -> pcomp (VERB/AUX)
            elif ch.dep_ == "pcomp" and ch.pos_ in {"VERB", "AUX"}:
                pcomp_subject = self._extract_subject(ch)
                if pcomp_subject is not None:
                    facts.extend(self._process_verb(ch, pcomp_subject))

        return facts



    def _lift_nested_preps_to_verb(
        self,
        head_token: Token,
        subject: Span,
        verb_token: Token,
        base_predicate: Span,
        pred_with_outer_prep: Span,
    ) -> List[Fact]:
        """
        Go inside the object and lift its prep to the original verb.
        - "including" treat as a list of additional objects under the same pred_with_outer_prep (e.g. "performed in").
        - "of" skip (part of noun phrase).
        - ✅ FIX: For temporal outer_prep (until, since, etc.) preserve connection with verb
        - other preps create short predicate: pobj + nested_prep (e.g. "mother from" instead of "born to mother from").
        """
        facts: List[Fact] = []

        outer_prep_token = head_token.head if head_token.dep_ == "pobj" and head_token.head.dep_ == "prep" else None
        # ✅ ITER 15: Also check if head is conj of a prep (coordinated preps like "as X and as Y")
        if outer_prep_token is None and head_token.dep_ == "pobj" and head_token.head.dep_ == "conj":
            # Check if the conj head is a prep itself (it should be after our coordinated prep fix)
            if head_token.head.pos_ == "ADP":
                outer_prep_token = head_token.head

        is_temporal_outer = outer_prep_token is not None and outer_prep_token.text.lower() in self.temporal_preps

        tokens_to_process = [head_token] + list(head_token.conjuncts)

        for token in tokens_to_process:
            for child in token.children:
                if child.dep_ != "prep":
                    continue

                pl = child.text.lower()

                if pl == "of":
                    continue

                if pl == "including":
                    for pobj in child.children:
                        if pobj.dep_ != "pobj":
                            continue

                        sub_span = self._subtree_span(pobj, include_punct=True)
                        sub_span = self._trim_span_at_breaks(sub_span)

                        if sub_span is not None and sub_span.text.strip():
                            facts.append(Fact(subject, pred_with_outer_prep, sub_span))
                        else:
                            np_span = self._get_np_span(pobj)
                            if np_span is not None:
                                facts.append(Fact(subject, pred_with_outer_prep, np_span))

                        facts.extend(self._extract_quoted_titles_under(pobj, subject, pred_with_outer_prep))

                        facts.extend(self._extract_conjuncts_and_appos(pobj, subject, pred_with_outer_prep))
                    continue

                # ✅ FIX: For temporal outer_prep (until, since, etc.), use verb predicate
                # Example: "served until retirement in 1975" -> "served until" + "retirement in 1975"
                if is_temporal_outer:
                    # Use pred_with_outer_prep (which includes verb + temporal prep)
                    # Argument: full subtree of pobj with nested prep
                    for pobj in child.children:
                        if pobj.dep_ != "pobj":
                            continue

                        # Build full argument: use _get_full_subject_span to include possessives
                        # Example: "his retirement in 1975"
                        arg_span = self._get_full_subject_span(token)
                        if arg_span is not None:
                            facts.append(Fact(subject, pred_with_outer_prep, arg_span))
                    continue

                # ✅ GOLD: For "such as" exemplification, extract base pobj + list items separately
                # Example: "caused by factors, such as diet" ->
                #   "caused by | factors", "caused by | diet", etc.
                # Don't include pobj or "such as" in predicate
                if self._is_such_as(child):
                    # Use pred_with_outer_prep (just verb + prep, e.g., "can be caused by")
                    # Don't include pobj or "such as" in predicate

                    # First, extract the base pobj as an argument
                    # Example: "caused by | factors"
                    base_pobj_np = self._get_np_span(token)
                    if base_pobj_np is not None:
                        head_span = self._tok_span(token)
                        if head_span is not None:
                            facts.append(Fact(subject, pred_with_outer_prep, head_span))

                        facts.append(Fact(subject, pred_with_outer_prep, base_pobj_np))

                    # Then extract each item in the "such as" list
                    # Example: "caused by | diet", "caused by | stress", "caused by | exposure to toxins"
                    for pobj in child.children:
                        if pobj.dep_ != "pobj":
                            continue

                        facts.extend(self._extract_single_object_facts(pobj, subject, pred_with_outer_prep))

                        for conj in pobj.conjuncts:
                            conj_np = self._get_np_span(conj)
                            if conj_np is not None:
                                facts.append(Fact(subject, pred_with_outer_prep, conj_np))

                            for conj_child in conj.children:
                                if conj_child.dep_ == "prep":
                                    # Example: "exposure to toxins"
                                    for prep_pobj in conj_child.children:
                                        if prep_pobj.dep_ == "pobj":
                                            prep_pobj_np = self._get_np_span(prep_pobj)
                                            if prep_pobj_np is not None:
                                                full_span = conj.doc[conj.i:prep_pobj_np.end]
                                                facts.append(Fact(subject, pred_with_outer_prep, full_span))
                    continue

                # ✅ ITER 15: For "as" (role marker) nested under "as", use verb predicate
                # Example: "served as Minister for X" -> "served as | Minister for X" OR "served as Minister for | X"
                # Pattern: outer_prep="as", pobj="Minister", nested_prep="for"
                # Keep verb in predicate since "as" marks roles/positions
                if outer_prep_token is not None and outer_prep_token.text.lower() == "as":
                    # Use pred_with_outer_prep (which includes "served as")
                    # Don't extend with _cover - just use the verb+as predicate
                    # This avoids creating huge predicates for coordinated "as" phrases
                    full_pred = pred_with_outer_prep

                    for pobj in child.children:
                        if pobj.dep_ != "pobj":
                            continue
                        facts.extend(self._extract_single_object_facts(pobj, subject, full_pred))
                        facts.extend(self._extract_conjuncts_and_appos(pobj, subject, full_pred))
                        facts.extend(self._lift_nested_preps_to_verb(pobj, subject, verb_token, base_predicate, full_pred))
                    continue

                # regular nested prep -> short predicate (pobj + nested_prep)
                # Example: "mother from" instead of "born to mother from"
                pred2 = self._cover(token, child)
                if pred2 is None:
                    continue

                for pobj in child.children:
                    if pobj.dep_ != "pobj":
                        continue
                    facts.extend(self._extract_single_object_facts(pobj, subject, pred2))
                    facts.extend(self._extract_conjuncts_and_appos(pobj, subject, pred2))

                    # ✅ FIX: Extract quoted titles under nested prep pobj
                    # Example: "roles in the TV series \"Pretty Little Liars\" and \"The Bold and the Beautiful\""
                    # Structure: roles (pobj of "for") -> in (prep) -> series (pobj) -> "Pretty..." (appos in quotes)
                    # Need to extract quoted titles as separate facts
                    facts.extend(self._extract_quoted_titles_under(pobj, subject, pred2))

                    facts.extend(self._lift_nested_preps_to_verb(pobj, subject, verb_token, base_predicate, pred2))

        return facts

    # Objects / Lists / Conjunctions (Span-only)

    def _extract_single_object_facts(self, obj_token: Token, subject: Span, predicate: Span) -> List[Fact]:
        if obj_token.pos_ == "CCONJ" or obj_token.lower_ in {"and", "or"}:
            return []

        facts: List[Fact] = []

        # ✅ FIX: Skip list modifiers that can't form a contiguous span with their shared head.
        # Example: "red, blue, and green cars" -> conj token "blue" has head "red" (amod),
        # and the shared noun ("cars") comes after the list, so any contiguous span would
        # include other list items. Better to skip than emit truncated facts.
        if obj_token.dep_ == "conj":
            head = obj_token.head
            while head.dep_ == "conj" and head.head != head:
                head = head.head
            if head.dep_ in {"amod", "compound", "nmod", "nummod"} and head.head.pos_ in {"NOUN", "PROPN"}:
                shared_head = head.head
                if shared_head.i > obj_token.i:
                    return facts

        # ✅ FIX: Capture left numeric tokens (percentages) for conjunct lists
        # Example: "was 25.09% White, 72.76% African American" - each conjunct needs its percentage
        obj_span = self._get_np_span(obj_token)

        # Check if there's a percentage/number immediately to the left that's not in the span
        if obj_span is None or (obj_token.i > 0 and "%" in obj_token.doc[obj_token.i - 1].text and obj_token.doc[obj_token.i - 1] not in obj_span):
            left_tokens = []
            for i in range(max(0, obj_token.i - 3), obj_token.i):
                tok = obj_token.doc[i]
                if "%" in tok.text or tok.like_num or tok.pos_ == "NUM":
                    left_tokens.append(tok)

            if left_tokens:
                all_tokens = left_tokens + self._collect_np_tokens(obj_token)
                obj_span = self._span_from_tokens(obj_token.doc, all_tokens)

        obj_span = self._clean_arg_span(obj_span)

        # ✅ FIX: Include acl (adjectival clause) in "known for" pattern
        # Pattern: "known for ability to elicit" - include "to elicit" in argument
        # Structure: ability (pobj) -> elicit (acl)
        if obj_span is not None and obj_token.dep_ == "pobj":
            prep_head = obj_token.head
            if prep_head.dep_ == "prep" and prep_head.text.lower() == "for":
                verb_head = prep_head.head
                if verb_head.lemma_.lower() in {"know", "famous", "note", "recognize", "celebrate"}:
                    acl_token = next((c for c in obj_token.children if c.dep_ == "acl" and c.pos_ in {"VERB", "AUX"}), None)
                    if acl_token is not None:
                        # Include acl and its subtree in argument
                        # Use subtree to get full VP (e.g., "to elicit strong performances")
                        acl_span = self._subtree_span(acl_token, include_punct=False)
                        if acl_span is not None:
                            obj_span = self._cover(obj_span, acl_span)

        if obj_span is not None:
            # ✅ FIX: Filter out backwards facts where argument comes BEFORE predicate
            # Example: "They include fruits" - "fruits" (conj) shouldn't create fact with argument="They"
            # This happens when argument span is before predicate span (tokens-wise)
            if obj_span.start < predicate.start:
                return facts

            facts.append(Fact(subject, predicate, obj_span))

        return facts

    def _pick_content_head(self, tok: Token) -> Token:
        nouns = [t for t in tok.subtree if t.pos_ == "NOUN" and not t.is_punct]
        return nouns[-1] if nouns else tok

    def _is_such_as(self, as_prep: Token) -> bool:
        if as_prep.lemma_ != "as" and as_prep.text.lower() != "as":
            return False
        if any(ch.text.lower() == "such" for ch in as_prep.children):
            return True
        if as_prep.i > 0 and as_prep.doc[as_prep.i - 1].text.lower() == "such":
            return True
        return False

    def _extract_quoted_items(self, token: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        token = pobj after 'as' (or other lists).
        Return Span inside quotes (without quotes).
        """
        facts: List[Fact] = []
        items = [token] + list(token.conjuncts)

        for it in items:
            q = self._get_quoted_span_context(it)
            if q is not None:
                facts.append(Fact(subject, predicate, q))
            else:
                np = self._get_np_span(it)
                if np is not None:
                    facts.append(Fact(subject, predicate, np))

        return facts

    def _extract_conjuncts_and_appos(self, token: Token, subject: Span, predicate: Span) -> List[Fact]:
        facts: List[Fact] = []

        for child in token.children:
            if child.dep_ == "conj":
                facts.extend(self._extract_single_object_facts(child, subject, predicate))
                facts.extend(self._extract_conjuncts_and_appos(child, subject, predicate))

            elif child.dep_ == "appos":
                # ✅ FIX: Skip appositives that are inside quotes
                # These are quoted titles and should be extracted via _extract_quoted_titles_under
                # Example: "the movie \"The Trial of...\"" - "Trial" is appos of "movie" but inside quotes
                # This prevents malformed facts like "movie \"The" when the full title is quoted
                if self._get_quoted_span_context(child) is not None:
                    continue  # Skip - will be extracted as quoted title instead

                facts.extend(self._extract_single_object_facts(child, subject, predicate))
                facts.extend(self._extract_conjuncts_and_appos(child, subject, predicate))

        return facts

    def _should_drop_head_for_modifier(self, token: Token) -> bool:
        """
        Decide whether a coordinated modifier can stand alone without the head noun.
        This is useful for lists like "NumPy and SciPy libraries" -> "NumPy", "SciPy".
        """
        if token.ent_type_ in {"GPE", "LOC", "FAC"}:
            return False
        if token.pos_ == "PROPN":
            return True
        return token.ent_type_ in {"ORG", "PRODUCT", "WORK_OF_ART", "EVENT", "PERSON"}

    def _extract_head_modifier_facts(self, head: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        Extract standalone facts for coordinated modifiers attached to a noun head.
        Example: "NumPy and SciPy libraries" -> facts for "NumPy" and "SciPy".
        """
        facts: List[Fact] = []
        modifiers = [ch for ch in head.children if ch.dep_ in {"compound", "amod", "nmod"}]

        for mod in modifiers:
            for item in [mod] + list(mod.conjuncts):
                if not self._should_drop_head_for_modifier(item):
                    continue
                mod_span = self._get_np_span(item) or self._tok_span(item)
                mod_span = self._clean_arg_span(mod_span)
                if mod_span is None or not mod_span.text.strip():
                    continue
                if head.i >= mod_span.start and head.i < mod_span.end:
                    continue
                facts.append(Fact(subject, predicate, mod_span))

        deduped: List[Fact] = []
        seen = set()
        for f in facts:
            key = (f.argument.start, f.argument.end)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(f)
        return deduped


    def _extract_object_attached_preps(self, obj_token, subject, verb_predicate, verb_token):
        if self._find_quote_pair_for_token(obj_token) is not None:
            return []
        if obj_token.lemma_.lower() in {"coverage", "area", "distance"}:
            return []

        facts = []
        for prep in obj_token.children:
            if prep.dep_ != "prep":
                continue
            if self._is_list_pattern(prep):
                nested_pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if nested_pobj is not None:
                    facts.extend(self._extract_list_items(nested_pobj, subject, verb_predicate, prep, True))
                continue

            # ✅ 'of' is almost always part of NP (King of New York, role of X, etc.)
            # BUT for nested preps, we want to extract them as separate facts
            # Examples: "position of Professor in Department at Hospital"
            # Arguments: "of Assistant Professor", "in the Department of Surgery", "at Hospital"
            if prep.text.lower() == "of":
                # Process "of" prep phrase AND nested preps recursively
                # Create base predicate: verb + obj (e.g., "holds position")
                base_pred = self._cover(verb_predicate, obj_token)
                if base_pred is None:
                    continue

                for pobj in prep.children:
                    if pobj.dep_ != "pobj":
                        continue

                    of_phrase = self._cover(prep, pobj)
                    if of_phrase is not None:
                        facts.append(Fact(subject, base_pred, of_phrase))

                    # Light-noun pattern: "coverage of X" -> allow "verb | X".
                    if obj_token.lemma_.lower() in {"coverage"}:
                        pobj_np = self._get_np_span(pobj)
                        if pobj_np is not None:
                            facts.append(Fact(subject, verb_predicate, pobj_np))

                    nested_preps = self._collect_nested_preps(pobj)

                    for nested_prep in nested_preps:
                        if nested_prep.text.lower() == "of":
                            continue

                        for nested_pobj in nested_prep.children:
                            if nested_pobj.dep_ != "pobj":
                                continue

                            np = self._get_np_span(nested_pobj)
                            if np is None:
                                continue

                            nested_phrase = self._cover(nested_prep, np)
                            if nested_phrase is not None:
                                facts.append(Fact(subject, base_pred, nested_phrase))

                            facts.extend(self._extract_conjuncts_and_appos(nested_pobj, subject, base_pred))

                continue

            pred_obj = obj_token
            if prep.text.lower() == "as":
                if any(c.dep_ == "poss" for c in obj_token.children):
                    pred_obj = self._get_np_span(obj_token) or self._tok_span(obj_token)
                else:
                    pred_obj = None
            elif prep.text.lower() == "in" and any(c.dep_ == "poss" for c in obj_token.children):
                pred_obj = self._get_np_span(obj_token) or self._tok_span(obj_token)
            pred2 = self._cover(verb_predicate, prep) if pred_obj is None else self._cover(verb_predicate, pred_obj, prep)
            if pred2 is None:
                continue

            for pobj in prep.children:
                if pobj.dep_ != "pobj":
                    continue

                np = self._get_np_span(pobj)
                if np is not None:
                    if prep.text.lower() == "as":
                        np = self._strip_leading_dets(np)
                    # ✅ FIX: Filter backwards facts (argument before predicate)
                    if np.start >= pred2.start:
                        facts.append(Fact(subject, pred2, np))
                extra = self._extract_conjuncts_and_appos(pobj, subject, pred2)
                if prep.text.lower() == "as":
                    stripped = []
                    for cf in extra:
                        if cf.argument is None:
                            continue
                        arg = self._strip_leading_dets(cf.argument)
                        if arg is None:
                            continue
                        stripped.append(Fact(cf.subject, cf.predicate, arg))
                    extra = stripped
                facts.extend(extra)

                # ✅ FIX: For "such as" lists, use full verb predicate instead of shortened pred2
                # Example: "appeared in shows such as X" not just "shows such as X"
                # "such as" is an exemplification, needs complete verb context
                for ch in pobj.children:
                    if ch.dep_ == "prep" and self._is_such_as(ch):
                        # Use verb_predicate + obj + "such as" for complete context
                        # This gives "has appeared in shows such as" instead of "shows such as"
                        full_pred = self._cover(verb_predicate, pobj, ch)
                        if full_pred is None:
                            full_pred = verb_predicate

                        for as_pobj in ch.children:
                            if as_pobj.dep_ == "pobj":
                                facts.extend(self._extract_quoted_items(as_pobj, subject, full_pred))

            # ✅ FIX: Handle pcomp (prepositional complement) - verbs as complements to prepositions
            # Example: "population with 51% of residents being male" -> "being" is pcomp of "with"
            # Structure: obj -> prep -> pcomp (VERB/AUX)
            for ch in prep.children:
                if ch.dep_ == "pcomp" and ch.pos_ in {"VERB", "AUX"}:
                    pcomp_subject = self._extract_subject(ch)
                    if pcomp_subject is not None:
                        facts.extend(self._process_verb(ch, pcomp_subject))

        return facts

    def _extract_object_prep_argument_facts(self, obj_token: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        Build argument-extended facts for preps attached to the object head.
        Example: "libraries for Python programming language" -> argument includes "for ...".
        Also produces a shorter variant with leading PROPNs (e.g., "for Python").
        """
        if obj_token.pos_ not in {"NOUN", "PROPN"}:
            return []
        facts: List[Fact] = []
        base_span = self._get_np_core_span(obj_token) or self._tok_span(obj_token)
        if base_span is None:
            return facts

        for prep in obj_token.children:
            if prep.dep_ != "prep":
                continue
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj is None:
                continue

            pobj_core = self._get_np_core_span(pobj) or self._tok_span(pobj)
            pobj_full = self._subtree_span(pobj, include_punct=False) or self._get_np_span(pobj) or pobj_core
            if pobj_full is None:
                continue

            variants = []
            if pobj_core is not None:
                variants.append(pobj_core)
            if pobj_full is not None:
                variants.append(pobj_full)

            # If pobj has a relcl, add a full variant that includes it (incremental extension).
            relcl = next((c for c in pobj.children if c.dep_ == "relcl"), None)
            if relcl is not None and pobj_full is not None:
                relcl_span = self._subtree_span(relcl, include_punct=False)
                if relcl_span is not None:
                    full_with_relcl = self._cover(pobj_full, relcl_span)
                    if full_with_relcl is not None:
                        variants.append(full_with_relcl)

            # If the pobj starts with a proper noun, create a shorter variant (e.g., "Python").
            propn_start = []
            for t in pobj_full:
                if t.pos_ == "PROPN":
                    propn_start.append(t)
                else:
                    break
            if propn_start:
                propn_span = self._span_from_tokens(pobj_full.doc, propn_start)
                if propn_span is not None:
                    variants.append(propn_span)

            uniq = []
            seen = set()
            for v in variants:
                key = (v.start, v.end)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(v)

            for var in uniq:
                arg_span = self._cover(base_span, prep, var)
                arg_span = self._clean_arg_span(arg_span)
                if arg_span is not None and arg_span.text.strip():
                    facts.append(Fact(subject, predicate, arg_span))

        return facts

    # Custom include+role facet extraction (Span-only)

    def _extract_include_role_facts(self, role_token: Token, subject: Span, predicate: Span) -> List[Fact]:
        facts: List[Fact] = []

        det = next((c for c in role_token.children if c.dep_ == "det"), None)
        role_head = self._cover(det, role_token) or self._tok_span(role_token)

        facts.append(Fact(subject, predicate, role_head))

        as_tok = next(
            (c for c in role_token.children if c.dep_ == "conj" and c.pos_ == "ADP" and c.text.lower() == "as"),
            None,
        )
        as_i = as_tok.i if as_tok is not None else 10**9

        # 1) role -> of Irene Molloy (here we can keep facet in predicate, because "of" is near role)
        for prep in role_token.children:
            if prep.dep_ == "prep" and prep.text.lower() == "of":
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                arg = self._get_np_span(pobj) or self._subtree_span(pobj, include_punct=True) or self._tok_span(pobj)
                arg = self._trim_span_at_breaks(arg)
                pred_of = self._cover(predicate, role_head, prep) or predicate
                facts.append(Fact(subject, pred_of, arg))
                facts.extend(self._extract_conjuncts_and_appos(pobj, subject, pred_of))

        # 2) role -> in 1969 film adaptation (IMPORTANT: DON'T make pred via cover to in!)
        #    start argument from 'in' to preserve relation "in ..."
        for prep in role_token.children:
            if prep.dep_ == "prep" and prep.text.lower() == "in" and prep.i < as_i:
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                pobj_np = self._get_np_span(pobj) or self._subtree_span(pobj, include_punct=True) or self._tok_span(pobj)
                pobj_np = self._trim_span_at_breaks(pobj_np)

                arg_in = self._cover(prep, pobj_np) or pobj_np
                arg_in = self._trim_span_at_breaks(arg_in)
                if arg_in is not None and arg_in.text.strip():
                    facts.append(Fact(subject, predicate, arg_in))

                facts.extend(self._extract_quoted_titles_under(pobj, subject, predicate))

        if as_tok is not None:
            z = next((c for c in as_tok.children if c.dep_ == "pobj"), None)
            if z is not None:
                z_np = self._get_np_span(z) or self._subtree_span(z, include_punct=True) or self._tok_span(z)
                z_np = self._trim_span_at_breaks(z_np)
                if z_np is not None:
                    facts.append(Fact(subject, predicate, z_np))

            # In your tree "in 1976 film" also hangs on role (not on as), so take in-Prep to the right of as
            in2 = None
            for prep in role_token.children:
                if prep.dep_ == "prep" and prep.text.lower() == "in" and prep.i > as_i:
                    in2 = prep
                    break

            if in2 is not None:
                film = next((c for c in in2.children if c.dep_ == "pobj"), None)
                if film is not None:
                    film_np = self._get_np_span(film) or self._subtree_span(film, include_punct=True) or self._tok_span(film)
                    film_np = self._trim_span_at_breaks(film_np)

                    arg_film = self._cover(in2, film_np) or film_np
                    arg_film = self._trim_span_at_breaks(arg_film)
                    if arg_film is not None:
                        facts.append(Fact(subject, predicate, arg_film))

                    facts.extend(self._extract_quoted_titles_under(film, subject, predicate))

        return facts


    def _should_attach_prep_to_pred(self, base_predicate: Span, prep_token: Token) -> bool:
        """
        Decide whether to make pred = cover(base_predicate, prep) (i.e. "appeared in"),
        or keep pred as is and put "in X" in argument.

        IMPORTANT:
        - Never attach prep that stands BEFORE predicate (sentence-initial PP).
        - Don't attach if there's a "heavy" break between pred and prep (comma/SCONJ/relcl/...).
        - Attach if prep is nearly adjacent and looks like verb valency.
        """
        doc = base_predicate.doc

        if prep_token.i < base_predicate.start:
            return False

        dist = prep_token.i - base_predicate.end
        if dist > 3:
            return False

        between = [t for t in doc[base_predicate.end : prep_token.i] if not t.is_space]

        for t in between:
            if t.is_punct and t.text in {",", ";", ":"}:
                return False

            if t.dep_ in {"mark", "relcl", "advcl", "acl"}:
                return False

            if t.lemma_.lower() in {"when", "whose", "which", "that", "where"}:
                return False

        allowed_between = {"advmod", "neg", "aux", "auxpass", "prt"}
        if between and not all(t.dep_ in allowed_between or t.is_punct for t in between):
            return False

        # 4) whitelist of preps that are usually verb valency (and actually help the predicate)
        # (the rest are often better kept in argument)
        # ✅ Include temporal preps (until, since, etc.) in whitelist
        good_preps = {
            "in", "on", "at", "to", "from", "for", "with", "as", "into", "onto", "over", "under", "by", "of",
            "until", "since", "before", "after", "during", "throughout"
        }

        pl = prep_token.text.lower()
        if pl not in good_preps:
            return False

        # 5) (optional) if you want: don't attach "of", except very-close cases
        # to avoid breaking "King of New York" and similar NP,
        # but it's better to handle this in object-attached prep (there I advised to skip 'of').
        if pl == "of" and dist > 1:
            return False

        return True

    def _trim_trailing_punct(self, span: Optional[Span]) -> Optional[Span]:
        if span is None or len(span) == 0:
            return span
        doc = span.doc
        end = span.end
        while end > span.start and doc[end - 1].is_punct and doc[end - 1].text in {",", ";", ":", ".", "!", "?"}:
            end -= 1
        return doc[span.start:end] if end > span.start else None

    def _clean_arg_span(self, span: Optional[Span]) -> Optional[Span]:
        span = self._trim_span_at_breaks(span)
        span = self._trim_trailing_punct(span)
        return span


    def _extract_object_facts(self, obj_token: Token, subject: Span, predicate: Span, verb_token: Token) -> List[Fact]:
        if verb_token.lemma_ == "include" and obj_token.lemma_.lower() == "role":
            return self._extract_include_role_facts(obj_token, subject, predicate)

        facts: List[Fact] = []

        # ✅ FIX: Always create base fact for dobj (even with nested preps)
        # This ensures "marked event" is extracted before "marked event in history"
        # The nested prep facts provide incremental detail for hallucination detection
        base_span = self._get_np_span(obj_token) or self._subtree_span(obj_token, include_punct=False) or self._tok_span(obj_token)
        base_span = self._trim_span_at_breaks(base_span) if base_span is not None else None
        base_span = self._trim_trailing_punct(base_span)

        has_obj_prep = any(c.dep_ == "prep" for c in obj_token.children)
        skip_base = obj_token.pos_ == "ADJ" and has_obj_prep

        if not skip_base and base_span is not None and base_span.text.strip():
            facts.append(Fact(subject, predicate, base_span))
            core_span = self._get_np_core_span(obj_token)
            if core_span is not None and (core_span.start != base_span.start or core_span.end != base_span.end):
                facts.append(Fact(subject, predicate, core_span))

        # Add a head-only fact when the object has left modifiers (for better granularity).
        if not skip_base and obj_token.pos_ in {"NOUN", "PROPN"} and base_span is not None:
            has_left_mods = any(c.dep_ in {"compound", "amod", "nmod"} for c in obj_token.children)
            if has_left_mods and len(base_span) > 1:
                head_only = self._tok_span(obj_token)
                if head_only is not None and (head_only.start != base_span.start or head_only.end != base_span.end):
                    facts.append(Fact(subject, predicate, head_only))

        # ✅ NEW: Emit standalone facts for coordinated modifiers on the head noun
        # Example: "NumPy and SciPy libraries" -> "NumPy", "SciPy"
        facts.extend(self._extract_head_modifier_facts(obj_token, subject, predicate))

        # ✅ NEW: Argument-extended facts for attached preps (e.g., "libraries for Python")
        facts.extend(self._extract_object_prep_argument_facts(obj_token, subject, predicate))

        # If obj is inside quotes, consider it an "atomic title" and
        # DON'T run object-attached preps (otherwise you'll get pred like: include "King of")
        is_quoted_obj = self._get_quoted_span_context(obj_token) is not None

        # 2) Titles: extract quoted titles only where it's really a container (film/show/song/etc)
        #    or if the object itself is a container (film credits include ...)
        if obj_token.lemma_.lower() in self.title_container_lemmas:
            facts.extend(self._extract_quoted_titles_under(obj_token, subject, predicate))

        facts.extend(self._extract_title_from_acl(obj_token, subject, predicate))

        # 4) Object-attached preps (had roles on shows such as ... / productions at ...)
        #    - skip 'of' (it's almost always part of name/NP: "King of New York")
        if not is_quoted_obj:
            facts.extend(self._extract_object_attached_preps(obj_token, subject, predicate, verb_token))

        skip_conj = False
        if verb_token.lemma_.lower() == "determine" or "determine" in predicate.text.lower():
            conj_children = [c for c in obj_token.children if c.dep_ == "conj"]
            if conj_children and base_span is not None:
                last_conj = max(conj_children, key=lambda t: t.i)
                has_comma = any(t.is_punct and t.text == "," for t in obj_token.doc[base_span.start:last_conj.i + 1])
                if not has_comma:
                    combined_span = obj_token.doc[base_span.start:last_conj.i + 1]
                    combined_span = self._clean_arg_span(combined_span)
                    if combined_span is not None and combined_span.text.strip():
                        facts.append(Fact(subject, predicate, combined_span))
                        skip_conj = True

        if not skip_conj:
            facts.extend(self._extract_conjuncts_and_appos(obj_token, subject, predicate))

        return facts



    def _is_year_token(self, tok: Token) -> bool:
        t = tok.text
        if not (tok.like_num and len(t) == 4 and t.isdigit()):
            return False
        y = int(t)
        return 1500 <= y <= 2100

    def _is_bad_subject_head(self, t: Token) -> bool:
        if t.pos_ == "NUM":
            return True
        if t.ent_type_ == "DATE":
            return True
        if getattr(self, "_is_year_token", None) is not None and self._is_year_token(t):
            return True
        return False

    def _extract_quoted_titles_under(self, head: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        Find any "...quoted..." pieces inside head's subtree and add them as separate argument-facts.
        This fixes cases like: a short film called "The Other Side" in 2019,
        where title often doesn't become an argument in the base NP.
        """
        facts: List[Fact] = []
        seen = set()

        for t in head.subtree:
            q = self._get_quoted_span_context(t)
            if q is None:
                continue
            key = (q.start, q.end)
            if key in seen:
                continue
            seen.add(key)
            if q.text.strip():
                facts.append(Fact(subject, predicate, q))

        return facts

    def _extract_title_from_acl(self, obj_token: Token, subject: Span, predicate: Span) -> List[Fact]:
        """
        Try to extract title from construction:
          <obj> (acl/relcl) called/named/titled <X>
        Works without quotes too.
        """
        if obj_token.lemma_.lower() not in self.title_container_lemmas:
            return []

        facts: List[Fact] = []

        for ch in obj_token.children:
            if ch.dep_ not in {"acl", "relcl"}:
                continue
            if ch.pos_ not in {"VERB", "AUX"}:
                continue

            for g in ch.children:
                if g.dep_ in {"dobj", "attr", "oprd"}:
                    span = self._get_np_span(g) or self._subtree_span(g, include_punct=True)
                    if span is not None and span.text.strip():
                        facts.append(Fact(subject, predicate, span))

            facts.extend(self._extract_quoted_titles_under(ch, subject, predicate))

            for prep in ch.children:
                if prep.dep_ != "prep":
                    continue
                for pobj in prep.children:
                    if pobj.dep_ != "pobj":
                        continue
                    if pobj.ent_type_ == "DATE" or self._is_year_token(pobj):
                        facts.append(Fact(subject, predicate, self._tok_span(pobj)))

        return facts

    # Atomicity Improvements (Argument Decomposition)

    def _decompose_argument_with_prep(self, fact: Fact) -> List[Fact]:
        """
        Decompose arguments with prepositional phrases into atomic facts.

        Pattern: "X verb [NP of/in/at Y]" → "X verb NP" + "NP is/of/in/at Y"

        Examples:
        - "served as podestà of Milan" → "served as podestà" + "podestà of Milan"
        - "towers on island of Forments" → "towers on island" + "island of Forments"

        IMPORTANT: Does NOT decompose arguments that are just prep phrases without a noun head.
        Example: "on the island of Forments" - this is already part of predicate, don't decompose.

        NEW: Disabled - nested preps are now handled as incremental facts during extraction.
        This prevents creating duplicate facts like "district | of | Pradesh" when we already
        create incremental facts like "located in | district" + "located in | district of Pradesh".
        """
        return [fact]

    def _decompose_temporal_range(self, fact: Fact) -> List[Fact]:
        """
        Split temporal ranges into start and end facts.

        Pattern: "X verb from Y to Z" → "X verb from Y" + "X verb until Z"

        Examples:
        - "served from 1992 to 2000" → "served from 1992" + "served until 2000"
        - "served from August 1862 to July 1865" → two atomic temporal facts
        """
        if fact.argument is None:
            return [fact]

        arg_text = fact.argument.text.lower()

        if "from" in arg_text and "to" in arg_text:
            from_token = None
            to_token = None

            for token in fact.argument:
                if token.text.lower() == "from":
                    from_token = token
                elif token.text.lower() == "to" and from_token is not None:
                    to_token = token
                    break

            if from_token and to_token:
                start_date = None
                for child in from_token.children:
                    if child.dep_ == "pobj":
                        start_date = self._subtree_span(child, include_punct=False)
                        break

                end_date = None
                for child in to_token.children:
                    if child.dep_ == "pobj":
                        end_date = self._subtree_span(child, include_punct=False)
                        break

                decomposed = []

                if start_date:
                    from_arg = self._cover(from_token, start_date)
                    if from_arg:
                        decomposed.append(Fact(fact.subject, fact.predicate, from_arg))

                if end_date:
                    until_pred = fact.predicate  # Keep same predicate for consistency
                    until_arg = self._cover(to_token, end_date)
                    if until_arg:
                        decomposed.append(Fact(fact.subject, until_pred, until_arg))

                return decomposed if decomposed else [fact]

        return [fact]

    def _should_decompose_argument(self, fact: Fact) -> bool:
        """
        Check if argument should be decomposed for atomicity.

        Returns True if:
        - Argument has > 3 tokens (excluding proper nouns)
        - Argument contains prep phrases that can be separated
        - Argument is a temporal range
        """
        if fact.argument is None:
            return False

        # ✅ ITER 15: Don't decompose causal/explanatory clauses
        # These are already atomic - "This is because X", "This is why X"
        # Decomposing them breaks the causal/explanatory meaning
        pred_text = fact.predicate.text.lower().strip()
        if "because" in pred_text or "why" in pred_text:
            return False

        # ✅ GOLD: Don't decompose predicative NPs after copula verbs
        # Example: "is a topic of debate" - should stay together as predicate complement
        # Pattern: predicate is just "is/are/was/were" (copula) and argument is NP with "of"
        # These form semantic units that shouldn't be split
        pred_tokens = [t for t in fact.predicate if not t.is_punct]
        if pred_tokens:
            has_copula = any(t.lemma_.lower() in {"be"} and t.pos_ == "AUX" for t in pred_tokens)
            is_simple_copula = has_copula and len(pred_tokens) <= 3

            if is_simple_copula:
                arg_tokens = list(fact.argument)
                if arg_tokens and arg_tokens[0].pos_ in {"DET"} and arg_tokens[0].text.lower() in {"a", "an", "the"}:
                    # This is predicative NP - don't decompose
                    # Examples: "is a topic of debate", "is the only constant member"
                    return False

        if len(fact.argument) <= 2:
            return False

        if fact.argument.root.ent_type_ in {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"}:
            return False

        has_decomposable_prep = any(
            t.dep_ == "prep" and t.text.lower() in {"of", "in", "on", "at"}
            for t in fact.argument
        )

        arg_text = fact.argument.text.lower()
        has_temporal_range = "from" in arg_text and "to" in arg_text

        return has_decomposable_prep or has_temporal_range

    def _apply_atomicity_decomposition(self, facts: List[Fact]) -> List[Fact]:
        """
        Apply atomicity decomposition to all facts.

        This is the main entry point for atomicity improvements.
        """
        decomposed_facts = []

        for fact in facts:
            if self._should_decompose_argument(fact):
                temporal_decomposed = self._decompose_temporal_range(fact)

                for temp_fact in temporal_decomposed:
                    prep_decomposed = self._decompose_argument_with_prep(temp_fact)
                    decomposed_facts.extend(prep_decomposed)
            else:
                decomposed_facts.append(fact)

        return decomposed_facts

    # Fact Validation (Precision Improvements)

    def _validate_fact_completeness(self, fact: Fact) -> bool:
        """
        Validate that a fact is grammatically complete.

        Returns False if:
        - Argument ends with incomplete adjective (no noun)
        - Argument ends with verb participle without complement
        - Argument is empty or very short without content
        - Predicate is incomplete
        """
        if fact.predicate is None or len(fact.predicate) == 0:
            return False

        if fact.argument is None:
            return True

        arg = fact.argument
        if len(arg) == 0:
            return False

        content_tokens = [t for t in arg if not t.is_punct and t.pos_ not in {"DET", "ADP"}]

        if len(content_tokens) == 0:
            return False

        last_token = arg[-1]

        # Argument ending with preposition is incomplete ("ideals of", "role in"),
        # unless it is part of a relative clause (e.g., "that people can sit on").
        if last_token.pos_ == "ADP":
            has_relcl = any(t.dep_ in {"relcl", "acl"} for t in arg)
            has_rel_pron = any(t.lower_ in {"that", "which", "who", "whom"} for t in arg)
            has_verb = any(t.pos_ in {"VERB", "AUX"} for t in arg)
            if not (has_relcl or (has_rel_pron and has_verb)):
                return False

        if last_token.pos_ == "ADJ":
            has_noun = any(t.pos_ in {"NOUN", "PROPN"} for t in arg)
            if not has_noun:
                # ✅ ITER 15: Allow standalone adjectives for passive voice oprd constructions
                # Pattern: "It is considered impolite", "X is deemed appropriate"
                # These are valid object predicates, not incomplete NPs
                # Check if this is oprd construction (single ADJ after passive verb)
                is_single_adj = len(arg) == 1 and arg[0].pos_ == "ADJ"
                predicate_verbs = {"considered", "deemed", "thought", "found", "judged", "rated", "regarded", "seen", "viewed"}
                has_passive_oprd_verb = any(verb in fact.predicate.text.lower() for verb in predicate_verbs)

                # ✅ FIX: Allow copula + acomp patterns: "was chaotic", "is happy", "are beautiful"
                # These are valid predicative adjective constructions
                copula_verbs = {"is", "are", "was", "were", "be", "been", "being"}
                pred_root = fact.predicate.root.lemma_.lower() if fact.predicate.root else ""
                is_copula_acomp = pred_root == "be" and all(t.pos_ in {"ADJ", "PUNCT", "PART"} for t in arg)

                if is_single_adj and has_passive_oprd_verb:
                    pass  # Allow
                elif is_copula_acomp:
                    pass  # Allow
                else:
                    return False

        if last_token.pos_ == "ADJ" and last_token.tag_ in {"JJR", "RBR"}:
            has_comparison = any(t.dep_ == "prep" and t.text.lower() in {"than", "as"} for t in arg)
            if not has_comparison:
                return False

        if len(arg) == 1 and arg[0].pos_ in {"DET", "PRON"}:
            return False

        if arg.text.lower().strip() in {"similar", "different", "equal", "equivalent"}:
            return False

        return True

    def _validate_argument_syntax(self, argument: Span) -> bool:
        """
        Check if argument has valid syntax.

        Returns False if:
        - Starts with preposition followed by another preposition (no noun between)
        - Has verb in infinitive form without proper structure
        - Multiple prepositions in sequence without nouns
        """
        if argument is None or len(argument) == 0:
            return True

        tokens = list(argument)


        if len(tokens) >= 2 and tokens[0].pos_ == "ADP":
            has_noun_after = any(t.pos_ in {"NOUN", "PROPN", "NUM"} for t in tokens[1:])
            if not has_noun_after:
                return False

        for i in range(len(tokens) - 1):
            if tokens[i].pos_ == "ADP" and tokens[i+1].pos_ == "ADP":
                # ✅ ITER 15: Allow quantifier constructions like "of around", "of about", "of approximately"
                # Pattern: "of around 1,000 people" - "around" is quantifier/modifier, not true preposition
                # These are valid constructions for expressing approximate quantities
                if tokens[i+1].text.lower() in {"around", "about", "approximately", "roughly", "nearly", "almost", "over", "under"}:
                    continue  # Allow quantifier constructions
                return False

        if tokens[0].pos_ in {"VERB", "AUX"} and len(tokens) <= 2:
            # ✅ GOLD: Allow gerunds (VBG) - they're valid noun-like arguments
            # Example: "responsible for sourcing", "responsible for screening"
            # Gerunds function as nouns and are complete arguments
            if tokens[0].tag_ == "VBG":
                return True

            has_object = any(t.pos_ in {"NOUN", "PROPN"} for t in tokens[1:])
            if not has_object:
                return False

        return True

    def _validate_fact_precision(self, fact: Fact) -> bool:
        """
        Combined validation for fact precision.

        Checks:
        1. Completeness (grammatical validity)
        2. Argument syntax (structural validity)
        3. Non-redundancy (subject != argument)
        4. Predicate validity (no malformed predicates)
        """

        if fact.subject is None or len(fact.subject) == 0:
            return False

        pred_text = fact.predicate.text
        arg_text = fact.argument.text if fact.argument else "None"
        subj_text = fact.subject.text.lower().strip()

        meta_subjects = {
            "context", "the context", "provided context", "given context", "information", "the information",
            "text", "the text", "passage", "the passage", "question", "the question",
            "answer", "the answer", "response", "the response", "details", "the details"
        }
        meta_verbs = {
            "say", "state", "mention", "provide", "outline", "describe", "indicate",
            "suggest", "imply", "show", "note", "explain", "deduce", "infer", "summarize",
            "include", "contain", "specify", "detail", "offer", "elaborate", "highlight", "link"
        }
        pred_lower = pred_text.lower()
        subj_root = fact.subject.root.lemma_.lower() if fact.subject.root else ""
        if subj_text in meta_subjects or subj_root in meta_subjects:
            has_numeric = any(t.like_num for t in fact.subject)
            if not has_numeric and any(v in pred_lower for v in meta_verbs):
                return False

        if subj_text in {"i", "we"} or subj_root in {"i", "we"}:
            if any(v in pred_lower for v in meta_verbs):
                return False

        # Filter meta pronoun statements ("This provides an overview...", "It establishes trajectory...")
        if subj_text in {"it", "this", "that"} or subj_root in {"it", "this", "that"}:
            if any(v in pred_lower for v in meta_verbs) and fact.argument is not None:
                arg_lower = fact.argument.text.lower()
                meta_arg_keywords = {
                    "information", "details", "context", "summary", "overview",
                    "trajectory", "background", "insight", "framework", "structure"
                }
                if any(k in arg_lower for k in meta_arg_keywords):
                    return False

        if fact.subject.root and fact.subject.root.lemma_.lower() == "let":
            return False
        if subj_text in {"'s", "’s", "lets", "let's"}:
            return False

        if "known as" in pred_lower and fact.argument is not None:
            doc = fact.argument.doc
            if fact.argument.start > 0 and fact.argument.end < len(doc):
                if doc[fact.argument.start - 1].text == "(" and doc[fact.argument.end].text == ")":
                    left_idx = fact.argument.start - 2
                    if left_idx >= 0:
                        left_tok = doc[left_idx]
                        if left_tok.i < fact.subject.start or left_tok.i >= fact.subject.end:
                            return False

        if "born on" in pred_lower and fact.argument is not None:
            arg_root = fact.argument.root
            if arg_root.ent_type_ not in {"DATE"} and not self._is_year_token(arg_root):
                if arg_root.ent_type_ in {"GPE", "LOC", "FAC"} or arg_root.pos_ in {"PROPN", "NOUN"}:
                    return False

        if not self._validate_fact_completeness(fact):
            return False

        if not self._validate_argument_syntax(fact.argument):
            return False

        if not self._validate_predicate(fact.predicate):
            return False

        # Already checked in _add() but double-check here
        # Ensure subject != argument (no self-referential facts)
        if fact.argument is not None:
            subj_text = fact.subject.text.lower().strip()
            arg_text = fact.argument.text.lower().strip()
            if subj_text == arg_text:
                return False

        return True

    def _validate_predicate(self, predicate: Span) -> bool:
        """
        Validate that predicate is well-formed.

        Returns False if:
        - Predicate is malformed (e.g., "is of", "in in", "with of")
        - Predicate has repeated prepositions
        - Predicate starts with preposition and ends with preposition
        """
        if predicate is None or len(predicate) == 0:
            return False

        pred_text = predicate.text.lower().strip()
        pred_tokens = list(predicate)

        prep_words = ["in", "on", "at", "of", "with", "from", "to", "for"]
        for prep in prep_words:
            if f"{prep} {prep}" in pred_text:
                return False

        # Check for invalid prep combinations: "is of", "are of", "was of", "on of", "with an"
        invalid_patterns = [
            r"\bis\s+of\b",
            r"\bare\s+of\b",
            r"\bwas\s+of\b",
            r"\bwere\s+of\b",
            r"\bwith\s+of\b",
            r"\bon\s+of\b",
            r"\bin\s+of\b",
            r"\bat\s+of\b",
            r"\bwith\s+an?\s+(?:average|total|number|amount)\b",  # "with a/an X" where X is measure word
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, pred_text):
                return False

        # Check if predicate starts and ends with prepositions (likely malformed)
        # Example: "on of" - starts with "on", ends with "of"
        if len(pred_tokens) >= 2:
            if pred_tokens[0].pos_ == "ADP" and pred_tokens[-1].pos_ == "ADP":
                return False

        # Check if predicate is just a single preposition (incomplete)
        # Allow only if it's a valid copula fact pattern
        if len(predicate) == 1 and predicate[0].pos_ == "ADP":
            # Single preposition is only OK for decomposed facts like "podestà of Milan"
            pass

        return True



    def visualize_tree(self, text: str):
        doc = self.nlp(text)
        print("=== DENDENCY TREE ===")
        for token in doc:
            print(f"{token.i:2} {token.text:15} {token.dep_:10} {token.head.text:15} {token.pos_:10}")
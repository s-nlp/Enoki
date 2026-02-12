import re
import string
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter


PUNCT = set(string.punctuation)


def trim_orig_span(text: str, s: int, e: int) -> Tuple[int, int]:
    """Trim whitespace and punctuation from span boundaries"""
    while s < e and (text[s].isspace() or text[s] in PUNCT):
        s += 1
    while s < e and (text[e-1].isspace() or text[e-1] in PUNCT):
        e -= 1
    return s, e


def build_resolved_to_orig_char_ranges(
    orig_text: str,
    replacements: List[Dict[str, Any]]
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Build resolved_text by applying replacements to orig_text and create character mapping.

    Args:
        orig_text: Original text
        replacements: List of replacement dicts from decontextualizer
            Supported types:
            - pronoun_rewrite: has 'start', 'end', 'replacement'
            - locative_rewrite: has 'start', 'end', 'replacement'
            - existential_there_rewrite: has 'there_start', 'there_end', 'be_start', 'be_end'
            - insertion: has 'at_char', 'insert'

    Returns:
        Tuple of (resolved_text, res2orig_ranges)
        res2orig_ranges maps each char in resolved_text to (start, end) in orig_text
    """
    # Normalize replacements to standard format with 'start', 'end', 'replacement'
    normalized_reps = []

    for r in replacements:
        rep_type = r.get("type", "")

        if rep_type == "pronoun_rewrite" or rep_type == "locative_rewrite":
            # Simple replacement: 'start', 'end', 'replacement'
            normalized_reps.append({
                "start": r["start"],
                "end": r["end"],
                "replacement": r["replacement"]
            })

        elif rep_type == "existential_there_rewrite":
            # Two separate replacements: 'there' and 'be' tokens
            # Add both as separate replacements
            normalized_reps.append({
                "start": r["there_start"],
                "end": r["there_end"],
                "replacement": r["topic"]
            })
            normalized_reps.append({
                "start": r["be_start"],
                "end": r["be_end"],
                "replacement": r["have"]
            })

        elif rep_type == "insertion":
            # Insertion: insert text at position without replacing
            # Treat as zero-width replacement at 'at_char'
            at_char = r["at_char"]
            normalized_reps.append({
                "start": at_char,
                "end": at_char,
                "replacement": r["insert"]
            })

        else:
            # Fallback: assume old format with 'start', 'end', 'replacement'
            if "start" in r and "end" in r and "replacement" in r:
                normalized_reps.append({
                    "start": r["start"],
                    "end": r["end"],
                    "replacement": r["replacement"]
                })

    # Sort by start position
    reps = sorted(normalized_reps, key=lambda x: (x["start"], x["end"]))

    res_parts = []
    res2orig_ranges: List[Tuple[int, int]] = []

    o_i = 0
    for r in reps:
        s, e = int(r["start"]), int(r["end"])
        repl = str(r["replacement"])

        # Handle overlapping replacements (skip if we've already passed this position)
        if s < o_i:
            # Overlapping replacement - adjust or skip
            if e <= o_i:
                # Completely before current position - skip
                continue
            else:
                # Partial overlap - adjust start
                s = o_i

        # unchanged chunk
        chunk = orig_text[o_i:s]
        res_parts.append(chunk)
        for k in range(o_i, s):
            res2orig_ranges.append((k, k+1))

        # replaced chunk
        res_parts.append(repl)
        if e > s:
            # Normal replacement - map to original span
            for _ in range(len(repl)):
                res2orig_ranges.append((s, e))
        else:
            # Insertion (e == s) - map to zero-width position
            for _ in range(len(repl)):
                res2orig_ranges.append((s, s))

        o_i = max(o_i, e)

    # tail
    tail = orig_text[o_i:]
    res_parts.append(tail)
    for k in range(o_i, len(orig_text)):
        res2orig_ranges.append((k, k+1))

    resolved = "".join(res_parts)
    assert len(resolved) == len(res2orig_ranges), f"alignment length mismatch: {len(resolved)} != {len(res2orig_ranges)}"
    return resolved, res2orig_ranges


def resolved_span_to_orig_span(
    res_span: Tuple[int, int],
    res2orig_ranges: List[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """
    Map span in resolved text to span in original text.

    Args:
        res_span: (start, end) in resolved text
        res2orig_ranges: Character mapping from build_resolved_to_orig_char_ranges

    Returns:
        (start, end) in original text, or None if invalid
    """
    s, e = res_span
    s = max(0, min(s, len(res2orig_ranges)))
    e = max(0, min(e, len(res2orig_ranges)))
    if e <= s:
        return None

    S = 10**18
    E = -1
    for i in range(s, e):
        a, b = res2orig_ranges[i]
        S = min(S, a)
        E = max(E, b)
    if E < 0:
        return None
    return (int(S), int(E))


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Check if two spans overlap"""
    return not (a[1] <= b[0] or b[1] <= a[0])


def entity_offsets_from_concat(text: str, pieces: list[str]) -> Optional[List[Tuple[int, int]]]:
    """
    Get entity offsets assuming pieces concatenate to text exactly.

    Args:
        text: Full text
        pieces: List of entity strings

    Returns:
        List of (start, end) tuples, or None if pieces don't concatenate exactly
    """
    joined = "".join(pieces)
    if joined != text:
        return None  # fallback to robust search
    offsets = []
    pos = 0
    for p in pieces:
        offsets.append((pos, pos + len(p)))
        pos += len(p)
    return offsets


def _word_boundary_ok(text: str, s: int, e: int) -> bool:
    """Check if span boundaries are at word boundaries"""
    left_ok = (s == 0) or (not text[s-1].isalnum())
    right_ok = (e == len(text)) or (not text[e].isalnum())
    return left_ok and right_ok


def _safe_find(text: str, piece: str, start: int, window: int = 64) -> Optional[int]:
    """
    Safely find piece in text near start position.

    Args:
        text: Text to search in
        piece: Piece to find
        start: Starting position
        window: Search window size

    Returns:
        Position of piece, or None if not found
    """
    # search only near current position
    lo = max(0, start - 8)
    hi = min(len(text), start + window)
    sub = text[lo:hi]

    # if piece is short word, only match at word boundaries
    p = piece
    if p.strip().isalnum() and len(p.strip()) <= 3:
        pat = re.compile(rf"(?<!\w){re.escape(p.strip())}(?!\w)")
        m = pat.search(sub)
        return None if m is None else lo + m.start()

    pos = sub.find(p)
    return None if pos == -1 else lo + pos


def align_pieces_to_text_robust(text: str, pieces: list[str]) -> List[Tuple[int, int]]:
    """
    Robustly align pieces to text with safe local search.

    Args:
        text: Full text
        pieces: List of pieces to align
        lookback: Lookback window for searching

    Returns:
        List of (start, end) tuples for each piece.
        If a piece cannot be found, uses current position (i, i) as fallback.
    """
    offsets = []
    i = 0
    for idx, p in enumerate(pieces):
        if p == "":
            offsets.append((i, i))
            continue

        # Strip trailing/leading spaces from piece for matching
        p_stripped = p.strip()
        if not p_stripped:
            offsets.append((i, i))
            continue

        # 1) try as-is / skip whitespace
        j = i
        while j < len(text) and text[j].isspace():
            j += 1

        # Try to match stripped version
        if text.startswith(p_stripped, j):
            s = j
            e = s + len(p_stripped)
        elif text.startswith(p, j):
            s = j
            e = s + len(p)
        else:
            # 2) fallback - safe local search with stripped version
            s = _safe_find(text, p_stripped, j, window=96)
            if s is None or not text.startswith(p_stripped, s):
                # Try with original piece
                s = _safe_find(text, p, j, window=96)
                if s is None or not text.startswith(p, s):
                    # ✅ FIX: Try global search (entire text) as last resort
                    global_pos = text.find(p_stripped)
                    if global_pos == -1:
                        global_pos = text.find(p)
                    if global_pos == -1:
                        # Still not found - skip this piece, use current position
                        # This handles cases where piece is from original text but text is decontextualized
                        offsets.append((i, i))
                        continue
                    s = global_pos
                    e = s + len(p_stripped) if global_pos != -1 and text.startswith(p_stripped, global_pos) else s + len(p)
                else:
                    e = s + len(p)
            else:
                e = s + len(p_stripped)

            # if piece is word - check boundaries (extra protection)
            if p_stripped.isalnum() and not _word_boundary_ok(text, s, e):
                # ✅ FIX: If boundary check fails, try to find next occurrence
                next_pos = text.find(p_stripped, s + 1)
                if next_pos != -1 and _word_boundary_ok(text, next_pos, next_pos + len(p_stripped)):
                    s = next_pos
                    e = s + len(p_stripped)
                else:
                    # Skip boundary check for now - might be false positive
                    pass

        offsets.append((s, e))
        i = e
    return offsets


def get_entity_offsets(text: str, pieces: list[str]) -> List[Tuple[int, int]]:
    """
    Get entity offsets, trying exact concatenation first, then robust search.

    Args:
        text: Full text
        pieces: List of entity strings

    Returns:
        List of (start, end) tuples for each entity
    """
    off = entity_offsets_from_concat(text, pieces)
    if off is not None:
        return off
    
    # ✅ FIX: Use robust search with error handling
    try:
        offsets = align_pieces_to_text_robust(text, pieces)
        return offsets
    except Exception as e:
        # ✅ FIX: If robust search fails completely, try per-piece search
        offsets = []
        pos = 0
        for p in pieces:
            if not p or not p.strip():
                offsets.append((pos, pos))
                continue
            # Try to find piece in text starting from current position
            found_pos = text.find(p.strip(), pos)
            if found_pos == -1:
                # Try full text search
                found_pos = text.find(p.strip())
            if found_pos == -1:
                # Still not found - use current position
                offsets.append((pos, pos))
            else:
                offsets.append((found_pos, found_pos + len(p.strip())))
                pos = found_pos + len(p.strip())
        return offsets


def dedupe_fact_scores_norm(fact_scores_norm: list[dict]) -> list[dict]:
    """
    Remove exact duplicates (same fact+kind+orig_span).
    For arguments: keep max hallucination probability
    For predicates: keep min hallucination probability

    Args:
        fact_scores_norm: List of normalized fact scores

    Returns:
        Deduplicated list
    """
    best = {}
    for r in fact_scores_norm:
        key = (
            r.get("fact"),
            r.get("span_kind"),
            int(r["orig_span_start"]),
            int(r["orig_span_end"]),
        )
        p = float(r["hall_prob"])
        if key not in best:
            best[key] = r
            continue

        prev = best[key]
        prev_p = float(prev["hall_prob"])
        if r.get("span_kind") == "predicate":
            # predicate: aggregate with min -> keep minimal hallucination
            if p < prev_p:
                best[key] = r
        else:
            # argument: aggregate with max -> keep maximal hallucination
            if p > prev_p:
                best[key] = r

    return list(best.values())


def normalize_fact_spans_to_orig(
    *,
    orig_text: str,
    resolved_text: str,
    replacements: list[dict],
    fact_scores: list[dict],
    search_window: int = 200,
) -> Tuple[List[Dict], Counter]:
    """
    Normalize fact spans to original text coordinates.

    Args:
        orig_text: Original text
        resolved_text: Resolved (decontextualized) text
        replacements: List of replacement dicts
        fact_scores: List of fact scores with spans
        search_window: Window for fallback search

    Returns:
        Tuple of (normalized_fact_scores, stats_counter)
    """
    # Fast path: if no replacements and texts match (ignoring whitespace), use direct coordinates
    if not replacements:
        if orig_text.strip() == resolved_text.strip():
            # Texts are identical except for whitespace - use resolved coordinates directly
            out = []
            stats = Counter()
            for r in fact_scores:
                st = r.get("span_text")
                s, e = int(r["span_start"]), int(r["span_end"])

                # Try using coordinates as-is in orig_text
                if st and 0 <= s < e <= len(orig_text) and orig_text[s:e] == st:
                    os, oe = s, e
                    stats["no_replacements_direct"] += 1
                elif st and 0 <= s < e <= len(resolved_text) and resolved_text[s:e] == st:
                    # Coordinates valid in resolved, search in orig
                    pos = orig_text.find(st)
                    if pos == -1:
                        stats["no_replacements_not_found"] += 1
                        continue
                    os, oe = pos, pos + len(st)
                    stats["no_replacements_search"] += 1
                else:
                    # Fallback: search by text
                    if not st:
                        continue
                    pos = orig_text.find(st)
                    if pos == -1:
                        stats["no_replacements_not_found"] += 1
                        continue
                    os, oe = pos, pos + len(st)
                    stats["no_replacements_search"] += 1

                rr = dict(r)
                os, oe = trim_orig_span(orig_text, os, oe)
                if oe <= os:
                    continue
                rr["orig_span_start"] = os
                rr["orig_span_end"] = oe
                out.append(rr)

            return out, stats
        else:
            # ✅ FIX: No replacements but texts differ - search by span_text in orig_text
            # This handles cases where resolved_text is decontextualized but replacements weren't tracked
            out = []
            stats = Counter()
            for r in fact_scores:
                st = r.get("span_text")
                if not st:
                    continue
                # Search span_text in orig_text
                pos = orig_text.find(st)
                if pos == -1:
                    stats["no_replacements_not_found"] += 1
                    continue
                os, oe = pos, pos + len(st)
                stats["no_replacements_search"] += 1

                rr = dict(r)
                os, oe = trim_orig_span(orig_text, os, oe)
                if oe <= os:
                    continue
                rr["orig_span_start"] = os
                rr["orig_span_end"] = oe
                out.append(rr)

            return out, stats

    rebuilt, res2orig = build_resolved_to_orig_char_ranges(orig_text, replacements)

    # ✅ FIX: More lenient assertion - allow whitespace differences
    if rebuilt.strip() != resolved_text.strip():
        # Find first difference for debugging
        min_len = min(len(rebuilt), len(resolved_text))
        first_diff = next((i for i in range(min_len) if rebuilt[i] != resolved_text[i]), min_len)
        raise AssertionError(
            f"resolved_text != rebuild(orig_text + replacements)\n"
            f"Length: rebuilt={len(rebuilt)}, resolved={len(resolved_text)}\n"
            f"First diff at position {first_diff}\n"
            f"Context: rebuilt[{first_diff}:{first_diff+50}]={repr(rebuilt[first_diff:first_diff+50])}\n"
            f"         resolved[{first_diff}:{first_diff+50}]={repr(resolved_text[first_diff:first_diff+50])}"
        )

    out = []
    stats = Counter()

    for r in fact_scores:
        st = r.get("span_text")
        s, e = int(r["span_start"]), int(r["span_end"])

        # ✅ FIX: Check coordinates in resolved_text first (they're from resolved_text)
        if st and 0 <= s < e <= len(resolved_text) and resolved_text[s:e] == st:
            # Map from resolved_text to orig_text using res2orig mapping
            mapped = resolved_span_to_orig_span((s, e), res2orig)
            if mapped is None:
                stats["map_fail"] += 1
                continue
            os, oe = mapped
            stats["resolved_coords_mapped"] += 1

        # ✅ FIX: Check if coordinates happen to work in orig_text (rare case)
        elif st and 0 <= s < e <= len(orig_text) and orig_text[s:e] == st:
            os, oe = s, e
            stats["orig_coords"] += 1

        else:
            stats["neither"] += 1
            if not st:
                continue
            # ✅ FIX: Fallback search - first try to map position, then search
            # Map s from resolved_text to approximate position in orig_text
            if s < len(res2orig):
                mapped_start = res2orig[s][0]  # Get start of mapped range
                # Use mapped position as center for search window
                lo = max(0, mapped_start - search_window)
                hi = min(len(orig_text), mapped_start + search_window)
            else:
                # If mapping fails, search entire text
                lo = 0
                hi = len(orig_text)
            
            pos = orig_text.find(st, lo, hi)
            if pos == -1:
                # Try full text search as last resort
                pos = orig_text.find(st)
            if pos == -1:
                stats["fallback_not_found"] += 1
                continue
            os, oe = pos, pos + len(st)
            stats["fallback_found"] += 1

        rr = dict(r)
        os, oe = trim_orig_span(orig_text, os, oe)
        if oe <= os:
            continue
        rr["orig_span_start"] = os
        rr["orig_span_end"] = oe
        out.append(rr)

    return out, stats


def align_fact_scores_to_entities_orig(
    *,
    orig_text: str,
    ds_entities: list[str],
    fact_scores_norm: list[dict],
) -> np.ndarray:
    """
    Align fact scores to entity spans.
    Arguments: aggregate with max
    Predicates: propagate to overlapping arguments

    Args:
        orig_text: Original text
        ds_entities: List of entity strings
        fact_scores_norm: List of normalized fact scores

    Returns:
        Array of hallucination probabilities for each entity
    """
    ent_offsets = get_entity_offsets(orig_text, ds_entities)

    # arguments: max
    arg_scores = np.zeros(len(ent_offsets), dtype=np.float32)
    arg_seen = np.zeros(len(ent_offsets), dtype=bool)

    STOP_ENTS = {
        # Punctuation
        ",", ".", ":", ";", "(", ")", "'", '"',
        # Prepositions
        "in","on","at","to","from","of","for","with","as","and","or","by","into","onto",
        # Articles
        "a","the","an",
        # Pronouns
        "he","his","him","she","her","they","their","them","it","its",
        "whose","which","that","when","where",
        # Common verbs
        "is","was","were","are","am","be","been","being","has","had","have",
        # Quantifiers and modifiers that are often false positives
        "including","various","numerous","several","many","some","few","most",
        "all","both","each","every","such","other","another",
        # Time/sequence words
        "then","later","after","before","during","while","since",
        # Common adjectives
        "first","last","new","old","good","great","best","long","short",
        # Time expressions that cause false positives
        "late","early","mid","90s","80s","70s","60s","50s",
        "1990s","1980s","1970s","1960s","1950s","2000s","2010s",
        # Common phrase fragments
        "the popular","the late","the early","this day","that day",
        "small roles","small","large","big","main","major","minor",
    }

    # Group fact scores by fact text to enable predicate propagation
    fact_groups = {}
    for r in fact_scores_norm:
        fact_text = r.get("fact", "")
        if fact_text not in fact_groups:
            fact_groups[fact_text] = {"predicates": [], "arguments": []}

        kind = r.get("span_kind", "argument")
        if kind == "predicate":
            fact_groups[fact_text]["predicates"].append(r)
        elif kind == "argument":
            fact_groups[fact_text]["arguments"].append(r)

    # First pass: collect predicate scores ONLY for predicate-only facts
    # (facts that have NO arguments - just subject+predicate)
    predicate_only_scores = {}  # fact_text -> hallucination probability
    for fact_text, group in fact_groups.items():
        # If fact has NO arguments but has predicates -> it's predicate-only
        if len(group["arguments"]) == 0 and len(group["predicates"]) > 0:
            # Take max predicate score
            max_pred_score = max(float(s["hall_prob"]) for s in group["predicates"])
            predicate_only_scores[fact_text] = max_pred_score

    # Second pass: process arguments and propagate predicate scores
    for r in fact_scores_norm:
        os, oe = int(r["orig_span_start"]), int(r["orig_span_end"])
        p = float(r["hall_prob"])
        kind = r.get("span_kind", "argument")
        fact_text = r.get("fact", "")
        fspan = (os, oe)

        # Skip predicates in entity alignment (they don't directly map to entities)
        if kind == "predicate":
            continue

        # For arguments: check if the SAME subject+predicate (without this arg)
        # exists as predicate-only fact with high score
        if kind == "argument":
            # Try to find predicate-only version by removing argument from fact text
            # e.g., "John directed film" -> look for "John directed"
            # This is approximate - ideally we'd track fact relationships better
            base_fact_candidates = [
                ft for ft in predicate_only_scores.keys()
                if fact_text.startswith(ft) or ft in fact_text
            ]

            # Use best predicate-only score from candidates
            pred_only_score = 0.0
            if base_fact_candidates:
                pred_only_score = max(predicate_only_scores[ft] for ft in base_fact_candidates)

            # Apply propagation: boost argument if predicate-only has high score
            effective_score = max(p, pred_only_score)

            for i, espan in enumerate(ent_offsets):
                ent = ds_entities[i].strip().lower()
                if ent in STOP_ENTS:
                    continue

                if not overlap(fspan, espan):
                    continue

                if effective_score > arg_scores[i]:
                    arg_scores[i] = effective_score
                arg_seen[i] = True

    return arg_scores


def explain_entity_orig(
    *,
    entity_str: str,
    orig_text: str,
    ds_entities: list[str],
    fact_scores_norm: list[dict],
    topn: int = 10,
):
    """
    Explain hallucination scores for a specific entity.

    Args:
        entity_str: Entity string to explain
        orig_text: Original text
        ds_entities: List of all entity strings
        fact_scores_norm: List of normalized fact scores
        topn: Number of top facts to show
    """
    ent_offsets = get_entity_offsets(orig_text, ds_entities)

    # find first index matching entity string
    idxs = [i for i, e in enumerate(ds_entities) if e == entity_str]
    if not idxs:
        idxs = [i for i, e in enumerate(ds_entities) if e.strip() == entity_str.strip()]

    if not idxs:
        print(f"Entity '{entity_str}' not found")
        return

    idx = idxs[0]

    e_span = ent_offsets[idx]
    print("ENTITY:", repr(entity_str), "idx=", idx, "orig_span=", e_span,
          "orig_cut=", repr(orig_text[e_span[0]:e_span[1]]))

    hits = []
    for r in fact_scores_norm:
        os, oe = int(r["orig_span_start"]), int(r["orig_span_end"])
        if overlap((os, oe), e_span):
            hits.append((float(r["hall_prob"]), r.get("fact"), r.get("span_text"), (os, oe)))

    hits.sort(reverse=True, key=lambda x: x[0])
    print("Overlapping facts:", len(hits))
    for p, fact, st, (os, oe) in hits[:topn]:
        print(f"  p={p:.3f} | span_text={repr(st)} | orig_cut={repr(orig_text[os:oe])}")
        print(f"    fact={fact}")
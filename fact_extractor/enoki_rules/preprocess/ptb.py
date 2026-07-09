"""PTB de-tokenization.

The LSOIE / OpenIE4 corpora ship Penn-Treebank-tokenized text:
bracket escapes (``-LRB-``), directional quotes (`` `` `` / ``''``),
and punctuation / clitics split off as their own tokens. Fed verbatim
to spaCy, ``-LRB-`` parses as a noun ``compound`` and detached
punctuation distorts dependency attachment (verified: it pollutes
subject spans and breaks appositive detection).

Two helpers, both **index-preserving** (1 token in -> 1 token out) so
gold span offsets stay valid:

- :func:`normalize_ptb_token` — map a single PTB token to its surface
  form (``-LRB-`` -> ``(``).
- :func:`ptb_detokenize` — join normalized tokens with natural
  spacing (no space before ``,`` / ``.`` / clitics, none after an
  open bracket) so spaCy parses clean prose.
"""

from __future__ import annotations

from typing import List, Sequence

_TOKEN_MAP = {
    "-LRB-": "(", "-RRB-": ")",
    "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}",
    "``": '"', "''": '"', "`": "'",
}

# Clitic / suffix tokens that attach to the preceding word with no space.
_NO_LEADING_SPACE = {
    "n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'", "’",
    ",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "''", '"',
}
# Tokens after which the next token gets no space.
_NO_TRAILING_SPACE = {"(", "[", "{", "``", "`", "$", "#"}


def normalize_ptb_token(tok: str) -> str:
    """Map one PTB token to its surface form (identity if not special)."""
    return _TOKEN_MAP.get(tok, tok)


def ptb_detokenize(tokens: Sequence[str]) -> str:
    """Join (already-normalized) tokens into natural, spaced prose."""
    out: List[str] = []
    prev = ""
    for raw in tokens:
        t = normalize_ptb_token(raw)
        if not out:
            out.append(t)
        elif t in _NO_LEADING_SPACE or prev in _NO_TRAILING_SPACE:
            out.append(t)
        else:
            out.append(" " + t)
        prev = t
    return "".join(out)


def normalize_ptb_tokens(tokens: Sequence[str]) -> List[str]:
    """1:1 normalize a token list (bracket/quote escapes only)."""
    return [normalize_ptb_token(t) for t in tokens]

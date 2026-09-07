"""Penn Treebank de-tokenization for the LSOIE and OpenIE4 corpora.

Maps bracket escapes such as ``-LRB-`` and PTB quote marks back to their surface
forms and joins tokens with natural spacing. Both helpers are index-preserving
(one token in, one token out) so gold span offsets stay valid.
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

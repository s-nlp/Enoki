"""NLI checker module for hallucination detection."""

import os

# Disable progress bars and verbose logging
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set transformers logging to error only
try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass

from nli.base import BaseNLIChecker
from nli.modernbert_nli import ModernBERTEncoderNLI
from nli.utils import (
    hallucination_prob_from_nli,
    is_context_valid,
    get_nli_checker,
    check_nli_batch_fast,
    argument_core_span,
    predicate_core_span,
    score_facts_with_nli,
    score_preextracted_with_nli,
    group_incremental_triplets,
)

__all__ = [
    "BaseNLIChecker",
    "ModernBERTEncoderNLI",
    "hallucination_prob_from_nli",
    "is_context_valid",
    "get_nli_checker",
    "check_nli_batch_fast",
    "argument_core_span",
    "predicate_core_span",
    "score_facts_with_nli",
    "score_preextracted_with_nli",
    "group_incremental_triplets",
]

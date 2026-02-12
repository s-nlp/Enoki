"""
Fact Extractor Module

A comprehensive fact extraction system using spaCy dependency parsing.
Extracts subject-predicate-argument triples from text with support for:
- Incremental fact building for granular hallucination detection
- Quoted entity handling
- Contrastive constructions
- List/enumeration parsing
"""

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
from .extractor import FactExtractor

__all__ = [
    # Main class
    "FactExtractor",

    # Models
    "TokOrSpan",
    "Fact",
    "IncrementalFactGroup",
    "QuotedEntityMapping",
    "ContrastiveParse",
    "ListParse",

    # Utilities
    "preprocess_quoted_entities",
    "restore_quoted_entities",
    "parse_contrastive_construction",
    "split_list_items",
    "detect_list_pattern",
    "split_enumeration",
    "extract_name_from_context",
]

__version__ = "1.0.0"

"""Fact extraction backends: Stanford OpenIE, MinIE, EnokiEncoder, EnokiLLM, and EnokiRules."""

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
from .stanford_extractor import StanfordFactExtractor
from .minie_extractor import (
    MinIEFactExtractor,
    MinIEFactExtractorSafe,
    MinIEFactExtractorComplete,
    MinIEFactExtractorAggressive,
    MinIEFactExtractorDictionary,
)
from .enoki_encoder_extractor import ModernOpenIEExtractor
from .enoki_llm_extractor import PreExtractedFactExtractor
from .enoki_rules_extractor import EnokiRulesFactExtractor

__all__ = [
    "StanfordFactExtractor",
    "MinIEFactExtractor",
    "MinIEFactExtractorSafe",
    "MinIEFactExtractorComplete",
    "MinIEFactExtractorAggressive",
    "MinIEFactExtractorDictionary",
    "ModernOpenIEExtractor",
    "PreExtractedFactExtractor",
    "EnokiRulesFactExtractor",

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

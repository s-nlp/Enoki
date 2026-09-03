"""Fact extraction backends.

Backend classes are imported lazily so installing one Enoki inference extra
does not pull the optional dependencies of every other extractor.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "StanfordFactExtractor": (".stanford_extractor", "StanfordFactExtractor"),
    "MinIEFactExtractor": (".minie_extractor", "MinIEFactExtractor"),
    "MinIEFactExtractorSafe": (".minie_extractor", "MinIEFactExtractorSafe"),
    "MinIEFactExtractorComplete": (".minie_extractor", "MinIEFactExtractorComplete"),
    "MinIEFactExtractorAggressive": (
        ".minie_extractor",
        "MinIEFactExtractorAggressive",
    ),
    "MinIEFactExtractorDictionary": (
        ".minie_extractor",
        "MinIEFactExtractorDictionary",
    ),
    "ModernOpenIEExtractor": (".enoki_encoder_extractor", "ModernOpenIEExtractor"),
    "PreExtractedFactExtractor": (
        ".enoki_llm_extractor",
        "PreExtractedFactExtractor",
    ),
    "EnokiRulesFactExtractor": (
        ".enoki_rules_extractor",
        "EnokiRulesFactExtractor",
    ),
    "TokOrSpan": (".models", "TokOrSpan"),
    "QuotedEntityMapping": (".models", "QuotedEntityMapping"),
    "ContrastiveParse": (".models", "ContrastiveParse"),
    "ListParse": (".models", "ListParse"),
    "Fact": (".models", "Fact"),
    "IncrementalFactGroup": (".models", "IncrementalFactGroup"),
    "preprocess_quoted_entities": (".utils", "preprocess_quoted_entities"),
    "restore_quoted_entities": (".utils", "restore_quoted_entities"),
    "parse_contrastive_construction": (".utils", "parse_contrastive_construction"),
    "split_list_items": (".utils", "split_list_items"),
    "detect_list_pattern": (".utils", "detect_list_pattern"),
    "split_enumeration": (".utils", "split_enumeration"),
    "extract_name_from_context": (".utils", "extract_name_from_context"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

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

__version__ = "0.1.0"

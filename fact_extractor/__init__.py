"""Fact extraction backends.

Backend classes are imported lazily so importing this package does not
initialize heavyweight extractor runtimes.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "StanfordFactExtractor": (".stanford_extractor", "StanfordFactExtractor"),
    "EnokiEncoderFactExtractor": (
        ".enoki_encoder_extractor",
        "EnokiEncoderFactExtractor",
    ),
    "EnokiLLMFactExtractor": (
        ".enoki_llm_extractor",
        "EnokiLLMFactExtractor",
    ),
    "EnokiRulesFactExtractor": (
        ".enoki_rules_extractor",
        "EnokiRulesFactExtractor",
    ),
    "Fact": (".models", "Fact"),
    "IncrementalFactGroup": (".models", "IncrementalFactGroup"),
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
    "EnokiEncoderFactExtractor",
    "EnokiLLMFactExtractor",
    "EnokiRulesFactExtractor",

    # Models
    "Fact",
    "IncrementalFactGroup",
]

__version__ = "0.1.0"

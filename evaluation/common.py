"""
Common utilities for evaluation.

Provides setup functions for logging, model loading, and decontextualization.
"""

import os
import sys
import warnings
import logging
from pathlib import Path

import spacy
from typing import Optional

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_logging():
    """Suppress verbose logging from libraries."""
    warnings.filterwarnings('ignore')

    # Environment variables
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_DATASETS_VERBOSITY'] = 'error'

    # Logging configuration
    logging.basicConfig(level=logging.ERROR)
    for logger_name in ['transformers', 'datasets', 'gliner', 'gliner_spacy',
                         'fastcoref', 'spacy', 'urllib3', 'filelock']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
        logging.getLogger(logger_name).propagate = False


def load_fact_extractor(
    extractor_method: str,
    use_gliner: bool = True,
    incremental: bool = True,
    use_preprocessing: bool = True,
    max_workers: int = 1,
    dataset: str = 'mushroom',
    checkpoint: Optional[str] = None,
    pre_extracted_facts_file: Optional[str] = None,
):
    """Load spaCy model and fact extractor."""
    print("Loading spaCy model and fact extractor...")
    if extractor_method == 'cycleoie':
        nlp = None
    else:
        nlp = spacy.load('en_core_web_trf')

    if extractor_method == 'stanford':
        from fact_extractor import StanfordFactExtractor
        ext = StanfordFactExtractor(nlp=nlp)
    elif extractor_method == 'minie':
        from fact_extractor import MinIEFactExtractor
        ext = MinIEFactExtractor(nlp=nlp, max_workers=max_workers)
    elif extractor_method == 'minie_safe':
        from fact_extractor import MinIEFactExtractorSafe
        ext = MinIEFactExtractorSafe(nlp=nlp, max_workers=max_workers)
    elif extractor_method == 'minie_complete':
        from fact_extractor import MinIEFactExtractorComplete
        ext = MinIEFactExtractorComplete(nlp=nlp, max_workers=max_workers)
    elif extractor_method == 'minie_aggressive':
        from fact_extractor import MinIEFactExtractorAggressive
        ext = MinIEFactExtractorAggressive(nlp=nlp, max_workers=max_workers)
    elif extractor_method == 'minie_dictionary':
        from fact_extractor import MinIEFactExtractorDictionary
        ext = MinIEFactExtractorDictionary(nlp=nlp, max_workers=max_workers)
    elif extractor_method == 'enoki_encoder':
        if checkpoint is None:
            raise ValueError("--checkpoint is required for enoki_encoder extractor")
        from fact_extractor import ModernOpenIEExtractor
        return ModernOpenIEExtractor(checkpoint=checkpoint, nlp=nlp, incremental=incremental)
    elif extractor_method == 'cycleoie':
        from fact_extractor.enoki_llm_extractor import PreExtractedFactExtractor
        if pre_extracted_facts_file is not None:
            facts_file = Path(pre_extracted_facts_file)
        else:
            facts_file = ROOT / "pre_extracted_facts" / f"{dataset}-refchecker-v1-triples.jsonl"
        return PreExtractedFactExtractor(facts_file)
    elif extractor_method == 'enoki_rules':
        from fact_extractor import EnokiRulesFactExtractor
        return EnokiRulesFactExtractor(nlp=nlp)
    else:
        raise NotImplementedError(f"Unknown extractor method: {extractor_method!r}")

    if use_preprocessing:
        from fact_extractor.preprocessor import PreprocessingWrapper
        ext = PreprocessingWrapper(ext, use_gliner=use_gliner)

    return ext


def load_decontextualizer(enabled: bool = False):
    """
    Load decontextualizer if enabled.

    Args:
        enabled: Whether to enable coreference resolution

    Returns:
        FastCorefDecontextualizer instance or None
    """
    if not enabled:
        print("Coreference resolution: DISABLED")
        return None

    from evaluation.decontextualizer import FastCorefDecontextualizer

    print("Loading FastCoref decontextualizer...")
    decontextualizer = FastCorefDecontextualizer()
    print("Coreference resolution: ENABLED")
    return decontextualizer


def print_header(title: str, width: int = 70):
    """Print formatted header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)

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
    encoder_model: Optional[str] = None,
    llm_model: Optional[str] = None,
):
    """Load spaCy model and fact extractor."""
    print("Loading spaCy model and fact extractor...")
    nlp = spacy.load('en_core_web_trf')

    if extractor_method == 'stanford':
        from fact_extractor import StanfordFactExtractor
        ext = StanfordFactExtractor(nlp=nlp)
    elif extractor_method == 'enoki_encoder':
        from fact_extractor import EnokiEncoderFactExtractor
        return EnokiEncoderFactExtractor(nlp=nlp, model=encoder_model)
    elif extractor_method == 'enoki_llm':
        from fact_extractor import EnokiLLMFactExtractor
        return EnokiLLMFactExtractor(nlp=nlp, model=llm_model)
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

"""
Decontextualizer for resolving coreferences and pronouns in text using fastcoref.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict


class FastCorefDecontextualizer:
    """Decontextualizer using fastcoref (biu-nlp/f-coref)."""

    def __init__(self):
        # Suppress warnings and verbose logging
        warnings.filterwarnings('ignore')

        # Set environment variables
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['HF_DATASETS_VERBOSITY'] = 'error'
        os.environ['TQDM_DISABLE'] = '1'

        # Suppress verbose logging from all relevant libraries
        for logger_name in ['transformers', 'gliner', 'gliner_spacy', 'datasets',
                           'fastcoref', 'spacy', 'urllib3', 'filelock']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            logging.getLogger(logger_name).propagate = False

        # Suppress progress bars from datasets
        try:
            import datasets
            datasets.logging.set_verbosity_error()
            datasets.disable_progress_bar()
        except:
            pass

        # Disable tqdm before importing fastcoref
        try:
            import functools
            import tqdm as tqdm_module
            original_tqdm = tqdm_module.tqdm
            tqdm_module.tqdm = functools.partial(original_tqdm, disable=True)
            tqdm_module.trange = functools.partial(original_tqdm, disable=True)
        except:
            pass

        from fastcoref import FCoref
        self.model = FCoref(device='cuda')

    def decontextualize(self, text: str) -> Dict[str, Any]:
        """Resolve coreferences using fastcoref.

        Only replaces pronouns with their antecedents to avoid aggressive expansion.
        """
        try:
            preds = self.model.predict(texts=[text])
            if preds and len(preds) > 0:
                # Track replacements for span alignment
                replacements = []
                clusters = preds[0].get_clusters(as_strings=False)

                # Pronouns to replace (only these will be expanded)
                pronouns = {'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their',
                           'him', 'them', 'himself', 'herself', 'itself', 'themselves',
                           'who', 'whom', 'whose', 'which', 'that'}

                for cluster in clusters:
                    if len(cluster) > 1:
                        # First mention is the antecedent
                        antecedent_span = cluster[0]
                        antecedent_text = text[antecedent_span[0]:antecedent_span[1]]
                        for mention_span in cluster[1:]:
                            mention_text = text[mention_span[0]:mention_span[1]]
                            # Only replace if it's a pronoun
                            if mention_text.lower() in pronouns:
                                replacements.append({
                                    'start': mention_span[0],
                                    'end': mention_span[1],
                                    'replacement': antecedent_text
                                })

                # Build resolved text by applying replacements in reverse order
                resolved = text
                for rep in sorted(replacements, key=lambda x: x['start'], reverse=True):
                    resolved = resolved[:rep['start']] + rep['replacement'] + resolved[rep['end']:]

                return {
                    "resolved_text": resolved,
                    "replacements": replacements
                }
        except Exception as e:
            print(f"Warning: fastcoref failed: {e}")
        return {"resolved_text": text, "replacements": []}

"""Base NLI checker abstract class."""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseNLIChecker(ABC):
    """Abstract base class for NLI checkers."""

    @abstractmethod
    def check_batch(
        self,
        premise: str,
        hypotheses: List[str],
        **kwargs
    ) -> List[Dict[str, float]]:
        """
        Check NLI for a batch of hypotheses given a shared premise.

        Args:
            premise: The context/premise text
            hypotheses: List of hypothesis strings to check
            **kwargs: Additional parameters specific to implementation

        Returns:
            List of dicts with 'entailment', 'neutral', 'contradiction' probabilities
        """
        pass

    def get_premise_chunks(
        self,
        premise: str,
        hypotheses: List[str],
        *,
        max_length: int = 2048,
        overlap_sents: int = 1,
    ) -> List[str]:
        """Return premise chunks that each fit within max_length paired with any hypothesis.

        Default implementation returns a single chunk (no splitting). Subclasses that
        support sentence-level chunking should override this.
        """
        return [premise]

    def check_batch_flat(
        self,
        premises: List[str],
        hypotheses: List[str],
        *,
        max_length: int = 2048,
    ) -> List[Dict[str, float]]:
        """Score parallel (premise, hypothesis) pairs in one forward pass.

        Each entry in *premises* and *hypotheses* is an independent pair (premises
        may differ across rows). This enables cross-sample GPU batching.

        Default implementation falls back to looping over unique premises and calling
        check_batch. Subclasses should override for an efficient batched forward pass.
        """
        results: List[Dict[str, float]] = [{}] * len(premises)
        # Group by premise to reuse check_batch where possible
        from collections import defaultdict
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(premises):
            groups[p].append(i)
        for prem, idxs in groups.items():
            hyps = [hypotheses[i] for i in idxs]
            scores = self.check_batch(prem, hyps, max_length=max_length)
            for i, score in zip(idxs, scores):
                results[i] = score
        return results

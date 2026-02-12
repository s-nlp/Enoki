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
        Check NLI for a batch of hypotheses given a premise.

        Args:
            premise: The context/premise text
            hypotheses: List of hypothesis strings to check
            **kwargs: Additional parameters specific to implementation

        Returns:
            List of dicts with 'entailment', 'neutral', 'contradiction' probabilities
        """
        pass

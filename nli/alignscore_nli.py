"""AlignScore-based NLI checker."""

import torch
from typing import List, Dict

from nli.base import BaseNLIChecker

class AlignScoreNLI(BaseNLIChecker):
    """
    NLI checker using AlignScore.
    Based on AlignScore-base model with roberta-base backbone.
    """

    def __init__(
        self,
        model: str = "roberta-base",
        batch_size: int = 32,
        device: str = None,
        ckpt_path: str = "models/alignscore/AlignScore-base.ckpt",
        evaluation_mode: str = "nli_sp",
    ):
        """
        Initialize AlignScore NLI checker.

        Args:
            model: Base model architecture (default: roberta-base)
            batch_size: Batch size for inference
            device: Device to use (default: auto-detect cuda:0/cpu)
            ckpt_path: Path to AlignScore checkpoint
            evaluation_mode: Evaluation mode (default: nli_sp for sentence-pair NLI)
        """
        self.model_name = model
        self.batch_size = batch_size
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.evaluation_mode = evaluation_mode
        self._scorer = None

    def _load_model(self):
        """Lazy load AlignScore model."""
        if self._scorer is not None:
            return

        # Fix tqdm monkey-patching issue that breaks pytorch_lightning
        # evaluate.py patches tqdm.tqdm with functools.partial, but pytorch_lightning
        # needs it to be a class for inheritance
        try:
            import tqdm as tqdm_module
            if hasattr(tqdm_module.tqdm, 'func'):  # It's a functools.partial
                # Restore original tqdm class
                tqdm_module.tqdm = tqdm_module.tqdm.func
                if hasattr(tqdm_module.trange, 'func'):
                    tqdm_module.trange = tqdm_module.trange.func
        except:
            pass

        try:
            from alignscore import AlignScore
        except ImportError:
            raise ImportError(
                "AlignScore not installed. Install from: "
                "pip install git+https://github.com/yuh-zha/AlignScore.git"
            )

        print(f"Loading AlignScore model on device: {self.device}")
        print(f"  Model: {self.model_name}")
        print(f"  Checkpoint: {self.ckpt_path}")
        print(f"  Evaluation mode: {self.evaluation_mode}")

        self._scorer = AlignScore(
            model=self.model_name,
            batch_size=self.batch_size,
            device=self.device,
            ckpt_path=self.ckpt_path,
            evaluation_mode=self.evaluation_mode,
        )

        print("AlignScore model loaded successfully")

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
            **kwargs: Additional parameters (ignored for AlignScore)

        Returns:
            List of dicts with 'entailment', 'neutral', 'contradiction' probabilities
        """
        if self._scorer is None:
            self._load_model()

        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        hypotheses = [str(h) for h in hypotheses]
        if not hypotheses:
            return []

        try:
            # AlignScore expects contexts and claims as lists
            contexts = [premise] * len(hypotheses)

            # Call inference() directly to get tri_label (3-class NLI)
            # Returns: (reg, bin, tri) where tri has shape [N, 3]
            _, _, tri_probs = self._scorer.model.inference(contexts, hypotheses)

            # tri_probs has shape [N, 3] where:
            # - Class 0 = entailment
            # - Class 1 = neutral
            # - Class 2 = contradiction
            results = []
            for probs in tri_probs:
                results.append({
                    "entailment": float(probs[0]),
                    "neutral": float(probs[1]),
                    "contradiction": float(probs[2]),
                })

            return results

        except Exception as e:
            print(f"[AlignScore] Error during inference: {e}")
            import traceback
            traceback.print_exc()
            # Return uniform distribution as fallback
            return [{"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33} for _ in hypotheses]



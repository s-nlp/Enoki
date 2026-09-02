"""LLM-based NLI checkers using vLLM."""

import os
import math
import torch
from typing import List, Dict, Optional

from nli.base import BaseNLIChecker

try:
    from vllm import LLM, SamplingParams, TokensPrompt
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None
    SamplingParams = None
    TokensPrompt = None

class LLM_NLI(BaseNLIChecker):
    """
    NLI checker using vLLM directly (local inference).

    Base class for all LLM-based NLI methods.
    Creates vLLM instance directly instead of connecting to server.
    Based on QwenNLIOpenAI_ragtruth_v5 which showed best performance.
    """

    LETTER_TO_CLASS = {"C": "contradiction", "N": "neutral", "E": "entailment"}
    LETTERS = ["C", "N", "E"]

    SYSTEM_PROMPT = (
        "You are a Natural Language Inference (NLI) classifier.\n"
        "Decide whether the Hypothesis is entailed by the Premise.\n\n"
        "The Premise is a passage provided as context. References in the Hypothesis to "
        "\"the passage\", \"the text\", \"the provided context\", or similar "
        "expressions refer to the Premise itself.\n\n"
        "Definitions:\n"
        "- E (entailment): Hypothesis follows from Premise. Paraphrases, summaries, logical inferences = E.\n"
        "- C (contradiction): Hypothesis CONFLICTS with Premise - directly or via logical implication.\n"
        "- N (neutral): Premise provides no relevant information about Hypothesis.\n\n"
        "Key Rules:\n"
        "- Same meaning, different wording = E (entailment).\n"
        "- If Premise IMPLIES something that makes Hypothesis false = C.\n"
        "- Meta-references to \"the passage\" or \"the text\" refer to the Premise = E if content matches.\n"
        "- Only use N for completely unrelated topics.\n"
        "- Generic statements that don't contradict the passage should be marked as ENTAILED (E).\n"
        "- Only mark as C if there is a CLEAR factual conflict.\n"
        "- Output exactly one letter: C, N, or E.\n\n"
        "Examples:\n"
        "Premise: The company reported revenue of $5 million.\n"
        "Hypothesis: The company's revenue was $5 million.\nAnswer: E\n\n"
        "Premise: John owns a dog named Max.\n"
        "Hypothesis: John has a pet.\nAnswer: E\n\n"
        "Premise: John is a vegetarian.\n"
        "Hypothesis: John eats meat regularly.\nAnswer: C\n\n"
        "Premise: The guide lists only three steps: 1. Open app. 2. Click settings. 3. Save.\n"
        "Hypothesis: Click the Advanced tab.\nAnswer: C\n\n"
        "Premise: The event was held in Paris in 2019.\n"
        "Hypothesis: The event took place in 2020.\nAnswer: C\n\n"
        "Premise: Mary bought a red car.\n"
        "Hypothesis: Mary purchased a vehicle.\nAnswer: E\n\n"
        "Premise: The building has 50 floors.\n"
        "Hypothesis: The building is very old.\nAnswer: N\n\n"
        "Premise: Automotive technicians can be paid hourly wages or commissions depending on employer.\n"
        "Hypothesis: Based on the passage, technicians can be paid hourly wages.\nAnswer: E\n\n"
        "Premise: Succinate is derived from succinic acid and used in drug manufacturing.\n"
        "Hypothesis: According to the provided text, succinate comes from succinic acid.\nAnswer: E\n"
    )

    def __init__(
        self,
        model: str = None,
        max_tokens: int = 10,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = None,
    ):
        """
        Initialize LLM-based NLI checker with direct vLLM instance.

        Args:
            model: Model name from HuggingFace (default: Qwen/Qwen3-8B)
            max_tokens: Max tokens for generation
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model context length (default: model's max)
        """
        if not HAS_VLLM:
            raise ImportError("vllm package required. Install with: pip install vllm")

        self.model_name = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        self._llm: Optional[LLM] = None
        self._sampling_params: Optional[SamplingParams] = None
        self._tokenizer = None  # Cache tokenizer for chunking

    def _get_tokenizer(self):
        """Get tokenizer (lazy load and cache)."""
        if self._tokenizer is None:
            if self._llm is None:
                self._load_model()
            self._tokenizer = self._llm.get_tokenizer()
        return self._tokenizer

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for chunking long premises.

        Prefer spaCy sentencizer when available; fall back to a simple regex splitter.
        """
        text = (text or "").strip()
        if not text:
            return []

        try:
            import spacy

            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            return sents if sents else [text]
        except Exception:
            import re

            parts = re.split(r"(?<=[.!?])\s+", text)
            parts = [p.strip() for p in parts if p.strip()]
            return parts if parts else [text]

    def _count_prompt_tokens(self, premise: str, hypothesis: str) -> int:
        """Count tokens in the full prompt (including chat template)."""
        tokenizer = self._get_tokenizer()
        messages = [
            {"role": "user", "content": f"{self.SYSTEM_PROMPT}\n\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:"},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        return len(tokens)

    def _chunk_premise_by_sentences(
        self,
        premise: str,
        *,
        hypotheses: List[str],
        max_length: int,
        overlap_sents: int = 1,
    ) -> List[str]:
        """
        Chunk premise into N parts (sentence-based) so that each (premise_chunk, hypothesis)
        fits into max_length without truncation.
        """
        if not premise.strip():
            return [premise]

        # Use the longest hypothesis (by token length) as worst-case
        tokenizer = self._get_tokenizer()
        if hypotheses:
            hyp_lens = [
                (h, len(tokenizer.encode(h, add_special_tokens=False)))
                for h in hypotheses
            ]
            longest_hyp = max(hyp_lens, key=lambda x: x[1])[0]
        else:
            longest_hyp = ""

        # Fast-path: already fits
        if self._count_prompt_tokens(premise, longest_hyp) <= max_length:
            return [premise]

        sents = self._split_into_sentences(premise)
        if not sents:
            return [premise]

        chunks: List[str] = []
        cur: List[str] = []

        def flush():
            nonlocal cur
            if cur:
                chunks.append(" ".join(cur).strip())
                if overlap_sents > 0:
                    cur = cur[-overlap_sents:]
                else:
                    cur = []

        for sent in sents:
            candidate = (" ".join(cur + [sent])).strip() if cur else sent.strip()
            if not candidate:
                continue

            if self._count_prompt_tokens(candidate, longest_hyp) <= max_length:
                cur.append(sent.strip())
                continue

            # Candidate too long: flush current and try sentence alone
            flush()

            if self._count_prompt_tokens(sent, longest_hyp) <= max_length:
                cur.append(sent.strip())
                continue

            # Single sentence doesn't fit: last-resort split by tokens
            tok_ids = tokenizer.encode(sent, add_special_tokens=False)

            # Calculate budget based on max_length minus overhead
            empty_tokens = self._count_prompt_tokens("", longest_hyp)
            budget = max(1, max_length - empty_tokens)

            for i in range(0, len(tok_ids), budget):
                sub_ids = tok_ids[i : i + budget]
                sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True).strip()
                if sub_text:
                    chunks.append(sub_text)

        flush()
        return [c for c in chunks if c]

    def _load_model(self):
        """Lazy load vLLM model."""
        if self._llm is not None:
            return

        print(f"Loading vLLM model: {self.model_name}")
        print(f"  Tensor parallel size: {self.tensor_parallel_size}")
        print(f"  GPU memory utilization: {self.gpu_memory_utilization}")

        # Use v0 engine to avoid multiprocessing issues with CUDA
        os.environ["VLLM_USE_V1"] = "0"

        vllm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }

        if self.max_model_len is not None:
            vllm_kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**vllm_kwargs)

        # Create sampling params
        self._sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.0,
            logprobs=10,  # Get top 10 logprobs to find C/N/E
        )

        print("Model loaded successfully")

    def _build_prompt(self, premise: str, hypothesis: str) -> tuple:
        """
        Build prompt for the model - with passage framing (v5 approach).

        Returns:
            Tuple of (prompt_text, prompt_tokens) where prompt_tokens is tokenized with enable_thinking=False
        """
        framed_premise = (
            f"The following passage is given as context:\n\n{premise}\n\nEnd of passage.\n\n"
            f"Note: Generic statements that don't contradict the passage should be marked as ENTAILED (E). "
            f"Only mark as C if there is a CLEAR factual conflict. "
            f"Meta-references to 'the passage' or 'the text' refer to the Context above."
        )

        # Get tokenizer and apply chat template with enable_thinking=False
        tokenizer = self._get_tokenizer()

        messages = [
            {"role": "user", "content": f"{self.SYSTEM_PROMPT}\n\nPremise: {framed_premise}\nHypothesis: {hypothesis}\nAnswer:"},
        ]

        # Apply chat template with thinking disabled - get text first
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Then tokenize manually
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        return prompt_text, prompt_tokens

    def _parse_answer(self, text: str) -> Optional[str]:
        """Parse answer letter from response text."""
        import re

        text = text.strip()

        # Look for letter at start or end
        for pattern in [
            r"^([CNE])[\s\.\,]?$",
            r"^([CNE])\b",
            r"\b([CNE])\s*$",
        ]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Last resort: find any C/N/E in last 30 chars
        last_part = text[-30:] if len(text) > 30 else text
        for char in reversed(last_part):
            if char.upper() in "CNE":
                return char.upper()

        return None

    def _softmax_from_scores(self, letter_logprobs: Dict[str, float]) -> Dict[str, float]:
        """Convert letter logprobs to softmax probabilities."""
        import math

        scores = {k: v for k, v in letter_logprobs.items() if k in self.LETTERS}
        if not scores:
            return {"C": 0.33, "N": 0.34, "E": 0.33}

        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())

        # Handle division by zero (when all logprobs are extremely low)
        if total == 0.0:
            return {"C": 0.33, "N": 0.34, "E": 0.33}

        return {k: v / total for k, v in exp_scores.items()}

    def _extract_probs_from_output(self, output) -> Dict[str, float]:
        """Extract probabilities from vLLM output."""
        try:
            # Get first token's logprobs
            if not output.outputs or not output.outputs[0].logprobs:
                # Fallback: parse text
                text = output.outputs[0].text if output.outputs else ""
                answer = self._parse_answer(text)
                if answer is None:
                    return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

                probs = {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.05}
                probs[self.LETTER_TO_CLASS[answer]] = 0.9
                return probs

            # Extract C/N/E logprobs from first few tokens (not just first)
            # Some models may generate whitespace or other tokens before the letter
            letter_logprobs = {}

            # Check first 3 tokens for C/N/E
            max_tokens_to_check = min(3, len(output.outputs[0].logprobs))
            for tok_idx in range(max_tokens_to_check):
                token_logprobs = output.outputs[0].logprobs[tok_idx]

                for token_id, logprob_obj in token_logprobs.items():
                    # logprob_obj has .logprob and .decoded_token
                    token = logprob_obj.decoded_token.strip().upper()
                    if token in self.LETTERS:
                        # Use highest logprob for each letter across all positions
                        if token not in letter_logprobs or logprob_obj.logprob > letter_logprobs[token]:
                            letter_logprobs[token] = logprob_obj.logprob

            if letter_logprobs:
                probs_by_letter = self._softmax_from_scores(letter_logprobs)
                return {
                    "contradiction": probs_by_letter.get("C", 0.0),
                    "neutral": probs_by_letter.get("N", 0.0),
                    "entailment": probs_by_letter.get("E", 0.0),
                }

            # Fallback: parse text
            text = output.outputs[0].text
            answer = self._parse_answer(text)
            if answer is None:
                return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

            probs = {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.05}
            probs[self.LETTER_TO_CLASS[answer]] = 0.9
            return probs

        except Exception as e:
            print(f"[LLM_NLI] Error extracting probs: {e}")
            return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

    def check_batch(
        self,
        premise: str,
        hypotheses: List[str],
        **kwargs
    ) -> List[Dict[str, float]]:
        """Check NLI for a batch of hypotheses using direct vLLM inference."""
        if self._llm is None:
            self._load_model()

        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        hypotheses = [str(h) for h in hypotheses]
        if not hypotheses:
            return []

        # Get max_length from kwargs or use model default
        max_length = kwargs.get('max_length', 40960)  # Qwen3 default max
        overlap_sents = kwargs.get('premise_chunk_overlap_sents', 1)

        # Chunk premise if needed
        premise_chunks = self._chunk_premise_by_sentences(
            premise,
            hypotheses=hypotheses,
            max_length=max_length,
            overlap_sents=overlap_sents,
        )

        # Log chunking info
        if len(premise_chunks) > 1:
            premise_len = len(premise.split())
            print(f"[LLM_NLI] Premise too long ({premise_len} words), split into {len(premise_chunks)} chunks with {overlap_sents}-sentence overlap")

        def _run_once(prem: str) -> List[Dict[str, float]]:
            """Run NLI for all hypotheses with given premise chunk."""
            # Build prompts for all hypotheses - get both text and tokens
            prompt_data = [self._build_prompt(prem, h) for h in hypotheses]
            prompt_token_ids = [tokens for _, tokens in prompt_data]

            # Run inference using token IDs directly
            try:
                from vllm import TokensPrompt
                token_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_token_ids]

                outputs = self._llm.generate(token_prompts, self._sampling_params, use_tqdm=False)

                # Extract probabilities from each output
                results = []
                for output in outputs:
                    probs = self._extract_probs_from_output(output)
                    results.append(probs)

                return results

            except Exception as e:
                print(f"[LLM_NLI] Batch inference error: {e}")
                # Return uniform distribution for all
                return [{"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33} for _ in hypotheses]

        # If single chunk, just run once
        if len(premise_chunks) == 1:
            return _run_once(premise_chunks[0])

        # Multiple chunks: aggregate by taking chunk with MIN hallucination probability
        # (same logic as ModernBERT)
        print(f"[LLM_NLI] Processing {len(hypotheses)} hypotheses across {len(premise_chunks)} chunks...")
        best: List[Optional[Dict[str, float]]] = [None] * len(hypotheses)
        best_key: List[Tuple[float, float]] = [(1e9, 1e9)] * len(hypotheses)

        for chunk_idx, chunk in enumerate(premise_chunks, 1):
            print(f"[LLM_NLI]   Chunk {chunk_idx}/{len(premise_chunks)}...")
            scores = _run_once(chunk)
            for i, s in enumerate(scores):
                hall_prob = s["contradiction"] + s["neutral"]
                key = (hall_prob, s["contradiction"])
                if key < best_key[i]:
                    best_key[i] = key
                    best[i] = s

        print(f"[LLM_NLI] Aggregation complete - selected best scores from {len(premise_chunks)} chunks")

        return [
            b if b is not None else {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}
            for b in best
        ]


# ============================================================================
# ALIGNSCORE NLI
# ============================================================================

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


# ============================================================================
# QWEN SPECIFIC MODELS
# ============================================================================

class QwenNLI_06B(LLM_NLI):
    """Qwen3 0.6B model for NLI."""
    def __init__(self, **kwargs):
        kwargs.setdefault('model', 'Qwen/Qwen3-0.6B')
        super().__init__(**kwargs)


class QwenNLI_4B(LLM_NLI):
    """Qwen3 4B model for NLI."""
    def __init__(self, **kwargs):
        kwargs.setdefault('model', 'Qwen/Qwen3-4B')
        super().__init__(**kwargs)


class QwenNLI_8B(LLM_NLI):
    """Qwen3 8B model for NLI."""
    def __init__(self, **kwargs):
        kwargs.setdefault('model', 'Qwen/Qwen3-8B')
        super().__init__(**kwargs)



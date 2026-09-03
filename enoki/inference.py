"""Unified runtime inference for the three Enoki OpenIE backends.

The module intentionally imports backend dependencies lazily. Installing the
encoder extra should not require spaCy, while rules-only users should not need
PyTorch or an OpenAI client.
"""

from __future__ import annotations

from typing import Any, Sequence


DEFAULT_ENCODER_MODEL = "s-nlp/enoki-openie-encoder"
DEFAULT_LLM_MODEL = "gpt-oss-120b"
SUPPORTED_METHODS = ("encoder", "llm", "rules")


class EnokiPipeline:
    """Extract subject-predicate-object triples with an Enoki backend.

    Args:
        method: ``"encoder"``, ``"llm"``, or ``"rules"``.
        model: Hugging Face model ID for the encoder or API model name for LLM.
        device: Torch device for the encoder. ``"auto"`` selects CUDA, MPS,
            then CPU. Ignored by the other methods.
        min_confidence: Minimum encoder confidence.
        top_k: Maximum encoder triples per input text.
        temperature: LLM sampling temperature.
        prompt: LLM prompt variant: ``"incremental"`` or ``"original"``.
        max_retries: LLM retries after a failed API request.
        request_timeout: LLM request timeout in seconds.

    The return value of :meth:`extract` has the same shape for every method::

        [{
            "text": "Apple acquired Beats.",
            "triples": [{
                "subject": "Apple",
                "predicate": "acquired",
                "object": "Beats",
                "confidence": 0.98,
            }],
        }]

    LLM triples do not expose a calibrated score, so their ``confidence`` is
    ``None``.
    """

    def __init__(
        self,
        method: str = "encoder",
        *,
        model: str | None = None,
        device: str = "auto",
        min_confidence: float = 0.7,
        top_k: int = 10,
        temperature: float = 0.0,
        prompt: str = "incremental",
        max_retries: int = 3,
        request_timeout: float = 120.0,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
    ) -> None:
        normalized_method = method.lower().replace("_", "-").removeprefix("enoki-")
        if normalized_method not in SUPPORTED_METHODS:
            choices = ", ".join(SUPPORTED_METHODS)
            raise ValueError(f"Unknown Enoki method {method!r}; choose one of: {choices}")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if prompt not in {"incremental", "original"}:
            raise ValueError("prompt must be 'incremental' or 'original'")

        self.method = normalized_method
        self.model = model
        self.device = device
        self.min_confidence = min_confidence
        self.top_k = top_k
        self.temperature = temperature
        self.prompt = prompt
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self._backend: Any = None

    def __call__(self, texts: str | Sequence[str]) -> list[dict[str, Any]]:
        return self.extract(texts)

    def extract(self, texts: str | Sequence[str]) -> list[dict[str, Any]]:
        """Extract triples from one string or a sequence of strings."""
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items or any(not isinstance(text, str) or not text.strip() for text in items):
            raise ValueError("texts must contain one or more non-empty strings")

        if self._backend is None:
            self._backend = self._load_backend()
        return self._backend.extract(items)

    def _load_backend(self) -> Any:
        if self.method == "encoder":
            return _EncoderBackend(
                model=self.model or DEFAULT_ENCODER_MODEL,
                device=self.device,
                min_confidence=self.min_confidence,
                top_k=self.top_k,
            )
        if self.method == "llm":
            return _LLMBackend(
                model=self.model or DEFAULT_LLM_MODEL,
                temperature=self.temperature,
                prompt=self.prompt,
                max_retries=self.max_retries,
                request_timeout=self.request_timeout,
                enable_thinking=self.enable_thinking,
                max_tokens=self.max_tokens,
            )
        return _RulesBackend()


class _EncoderBackend:
    def __init__(
        self,
        *,
        model: str,
        device: str,
        min_confidence: float,
        top_k: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "Encoder inference requires the 'encoder' dependencies. "
                "Install them with: pip install -e '.[encoder]'"
            ) from error

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.min_confidence = min_confidence
        self.top_k = top_k
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.model.to(device).eval()

    def extract(self, texts: list[str]) -> list[dict[str, Any]]:
        raw_results = self.model.extract_triples(
            texts,
            min_confidence=self.min_confidence,
            top_k=self.top_k,
        )
        results: list[dict[str, Any]] = []
        for text, raw in zip(texts, raw_results):
            triples = [
                _triple(
                    subject=item.get("subject", ""),
                    predicate=item.get("predicate", item.get("relation", "")),
                    object_text=item.get("object", ""),
                    confidence=item.get("confidence"),
                )
                for item in raw.get("triples", [])
            ]
            results.append({"text": text, "triples": triples})
        return results


class _LLMBackend:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        prompt: str,
        max_retries: int,
        request_timeout: float,
        enable_thinking: bool,
        max_tokens: int | None,
    ) -> None:
        try:
            from fact_extractor.llm_backend import (
                SYSTEM_PROMPT_INCREMENTAL,
                SYSTEM_PROMPT_ORIGINAL,
                call_model_with_retries,
                parse_model_output,
                split_sentences_with_spans,
            )
        except ImportError as error:
            raise RuntimeError(
                "LLM inference requires the 'llm' dependencies. "
                "Install them with: pip install -e '.[llm]'"
            ) from error

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.system_prompt = (
            SYSTEM_PROMPT_INCREMENTAL if prompt == "incremental" else SYSTEM_PROMPT_ORIGINAL
        )
        self._call = call_model_with_retries
        self._parse = parse_model_output
        self._split = split_sentences_with_spans

    def extract(self, texts: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for text in texts:
            triples: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            for sentence in self._split(text):
                response = self._call(
                    model=self.model,
                    sentence=sentence.text,
                    system_prompt=self.system_prompt,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                    request_timeout=self.request_timeout,
                    model_params=0.0,
                    enable_thinking=self.enable_thinking,
                    max_tokens=self.max_tokens,
                )
                parsed, _, _ = self._parse(response.raw)
                for subject, predicate, object_text in parsed:
                    key = (subject.casefold(), predicate.casefold(), object_text.casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(_triple(subject, predicate, object_text, None))
            results.append({"text": text, "triples": triples})
        return results


class _RulesBackend:
    def __init__(self) -> None:
        try:
            from fact_extractor.enoki_rules.pipeline import Pipeline
        except ImportError as error:
            raise RuntimeError(
                "Rules inference requires the 'rules' dependencies. "
                "Install them with: pip install -e '.[rules]'"
            ) from error

        try:
            self.pipeline = Pipeline()
        except OSError as error:
            raise RuntimeError(
                "The rules backend requires the en_core_web_trf spaCy model. "
                "Install it with: python -m spacy download en_core_web_trf"
            ) from error

    def extract(self, texts: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for text in texts:
            triples = []
            for item in self.pipeline.extract(text):
                predicate = item.predicate_surface
                if item.negated:
                    predicate = f"NOT {predicate}"
                triples.append(
                    _triple(
                        item.subject.text,
                        predicate,
                        item.argument.span.text if item.argument is not None else "",
                        item.confidence,
                    )
                )
            results.append({"text": text, "triples": triples})
        return results


def _triple(
    subject: str,
    predicate: str,
    object_text: str,
    confidence: float | None,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_text,
        "confidence": confidence,
    }

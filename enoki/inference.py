"""Unified runtime inference for the three Enoki OpenIE backends.

The module intentionally imports backend dependencies lazily. Installing the
encoder extra should not require spaCy, while rules-only users should not need
PyTorch or an OpenAI client.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ENCODER_MODEL = "s-nlp/enoki-openie-encoder"
DEFAULT_LLM_MODEL = "gpt-oss-120b"
SUPPORTED_METHODS = ("encoder", "llm", "rules")


class EnokiPipeline:
    """Extract subject-predicate-object triples with an Enoki backend.

    Args:
        method: ``"encoder"`` (default), ``"llm"``, or ``"rules"``.
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

    def detect(
        self,
        *,
        context: str,
        answer: str,
        nli_method: str = "modernbert",
        max_length: int = 2048,
        return_all: bool = False,
    ) -> list[dict[str, Any]]:
        """Score the probability that each answer fact lacks support in ``context``.

        Facts are extracted with this pipeline's selected backend and verified
        with the selected NLI checker. Each result has a plain-text answer
        ``span``, its ``start`` and ``end`` character offsets, a structured SPO
        ``fact``, and its hallucination ``probability``. By default, only
        facts with ``probability > 0.5`` are returned; set ``return_all=True``
        to inspect every scored fact.
        """
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a non-empty string")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        triples = self.extract(answer)[0]["triples"]
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for triple in triples:
            fact = {
                part: str(triple.get(part, "")).strip()
                for part in ("subject", "predicate", "object")
            }
            span, delta_span, base_triple = self._answer_span(answer, triple, triples)
            if span is None:
                continue
            if delta_span is not None and base_triple is not None:
                base_predicate = str(base_triple.get("predicate", "")).strip()
                if fact["predicate"].casefold() != base_predicate.casefold():
                    fact["object"] = answer[delta_span[0]:delta_span[1]]
                    span = delta_span
            hypothesis = " ".join(value for value in fact.values() if value)
            if not hypothesis:
                continue
            key = (hypothesis, *span)
            if key not in seen:
                seen.add(key)
                candidates.append({"fact": fact, "hypothesis": hypothesis, "span": span})

        if not candidates:
            return []

        # Keep this import lazy: extraction alone does not need an NLI model.
        from nli import check_nli_batch_fast, hallucination_prob_from_nli

        scores = check_nli_batch_fast(
            context,
            [candidate["hypothesis"] for candidate in candidates],
            method=nli_method,
            max_length=max_length,
        )
        results = []
        for candidate, score in zip(candidates, scores):
            start, end = candidate["span"]
            probability = hallucination_prob_from_nli(score)
            if not return_all and probability <= 0.5:
                continue
            results.append(
                {
                    "span": answer[start:end],
                    "start": start,
                    "end": end,
                    "fact": candidate["fact"],
                    "probability": probability,
                }
            )
        return results

    @staticmethod
    def _answer_span(
        answer: str,
        triple: dict[str, Any],
        triples: Sequence[dict[str, Any]],
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None, dict[str, Any] | None]:
        """Anchor an object, preserving an incremental delta when available."""
        object_text = str(triple.get("object", "")).strip()
        if object_text:
            object_span, exact = EnokiPipeline._text_span(answer, object_text)
            if object_span is None:
                # Never mark an unrelated subject when an object was predicted
                # but cannot be anchored in the answer.
                return None, None, None
            if exact:
                return object_span, None, None

            delta_span, base_triple = EnokiPipeline._incremental_delta_span(
                answer, triple, triples, object_span
            )
            return delta_span or object_span, delta_span, base_triple

        for part in ("predicate", "subject"):
            text = str(triple.get(part, "")).strip()
            if not text:
                continue
            span, _ = EnokiPipeline._text_span(answer, text)
            if span is not None:
                return span, None, None
        return None, None, None

    @staticmethod
    def _text_span(answer: str, text: str) -> tuple[tuple[int, int] | None, bool]:
        """Find ``text`` exactly, or as answer words separated by short glue."""
        normalized_answer = answer.casefold()
        normalized_text = text.casefold()
        start = normalized_answer.find(normalized_text)
        if start >= 0:
            return (start, start + len(text)), True

        words = re.findall(r"\w+", normalized_text, flags=re.UNICODE)
        if not words:
            return None, False
        first = re.search(rf"\b{re.escape(words[0])}\b", normalized_answer)
        if first is None:
            return None, False
        span_start, previous_end = first.start(), first.end()
        for word in words[1:]:
            current = re.search(rf"\b{re.escape(word)}\b", normalized_answer[previous_end:])
            if current is None:
                return None, False
            current_start = previous_end + current.start()
            separator = answer[previous_end:current_start]
            if len(separator) > 24 or re.search(r"[.;:!?]", separator):
                return None, False
            previous_end += current.end()
        return (span_start, previous_end), False

    @staticmethod
    def _incremental_delta_span(
        answer: str,
        triple: dict[str, Any],
        triples: Sequence[dict[str, Any]],
        object_span: tuple[int, int],
    ) -> tuple[tuple[int, int] | None, dict[str, Any] | None]:
        """Find the newly added object words relative to the closest base fact."""
        target = re.findall(r"\w+", str(triple.get("object", "")).casefold())
        if len(target) < 2:
            return None, None
        subject = str(triple.get("subject", "")).casefold().strip()
        relation = EnokiPipeline._relation_stem(str(triple.get("predicate", "")))
        best: tuple[list[str], dict[str, Any]] | None = None
        for other in triples:
            if other is triple or str(other.get("subject", "")).casefold().strip() != subject:
                continue
            if EnokiPipeline._relation_stem(str(other.get("predicate", ""))) != relation:
                continue
            base = re.findall(r"\w+", str(other.get("object", "")).casefold())
            if not base or len(base) >= len(target):
                continue
            if target[:len(base)] == base and (best is None or len(base) > len(best[0])):
                best = (base, other)
        if best is None:
            return None, None

        delta = " ".join(target[len(best[0]):])
        delta_span, _ = EnokiPipeline._text_span(answer[object_span[0]:object_span[1]], delta)
        if delta_span is None:
            return None, None
        start = object_span[0] + delta_span[0]
        end = object_span[0] + delta_span[1]
        return (start, end), best[1]

    @staticmethod
    def _relation_stem(predicate: str) -> str:
        words = predicate.casefold().split()
        while words and words[-1] in {"at", "by", "for", "from", "in", "of", "on", "to", "with"}:
            words.pop()
        return " ".join(words)

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
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "Enoki-Encoder could not import PyTorch. This usually means the "
                "PyTorch installation is incomplete or incompatible with this Python. "
                "Create a virtual environment, then run: "
                "python -m pip install --upgrade --force-reinstall torch && "
                "python -m pip install -e ."
            ) from error
        try:
            from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version
        except ImportError as error:
            raise RuntimeError(
                "Encoder inference requires Transformers. Install Enoki with: "
                "python -m pip install -e ."
            ) from error

        if transformers_version != "4.57.6":
            raise RuntimeError(
                "The published Enoki-Encoder requires transformers==4.57.6. "
                "Reinstall Enoki's dependencies with: python -m pip install -e ."
            )

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        from model.export import is_local_encoder_model
        if is_local_encoder_model(model):
            self._local = _LocalEncoderBackend(
                model_dir=Path(model),
                device=device,
                min_confidence=min_confidence,
                top_k=top_k,
            )
            self.model = self._local.model
            return

        self.min_confidence = min_confidence
        self.top_k = top_k
        self._local = None
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            use_fast=True,
        )
        self.model.to(device).eval()

    def extract(self, texts: list[str]) -> list[dict[str, Any]]:
        if self._local is not None:
            return self._local.extract(texts)
        raw_results = self.model.extract_triples(
            texts,
            tokenizer=self.tokenizer,
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


class _LocalEncoderBackend:
    """Inference backend for directories exported by ``enoki train encoder``."""

    def __init__(self, *, model_dir: Path, device: str, min_confidence: float, top_k: int) -> None:
        import torch
        from transformers import AutoTokenizer

        from model.export import MANIFEST_NAME
        from model.model import IGLModel

        manifest = json.loads((model_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != "enoki-lightning-v1":
            raise ValueError(f"Unsupported local Enoki-Encoder format in {model_dir}")

        self.device = torch.device(device)
        self.min_confidence = min_confidence
        self.top_k = top_k
        self.model = IGLModel.load_from_checkpoint(
            model_dir / manifest["checkpoint"],
            map_location=self.device,
            init_from_config_only=True,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def extract(self, texts: list[str]) -> list[dict[str, Any]]:
        from model.predict import extract

        raw_results = extract(
            texts,
            self.model,
            self.tokenizer,
            top_k=self.top_k,
            batch_size=1,
            device=self.device,
            min_conf=self.min_confidence,
        )
        return [
            {
                "text": text,
                "triples": [
                    _triple(subject, predicate, object_text, confidence)
                    for confidence, subject, predicate, object_text in triples
                ],
            }
            for text, (_, triples) in zip(texts, raw_results)
        ]


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
                "LLM inference requires Enoki's dependencies. "
                "Install them with: pip install -e ."
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
                "Rules inference requires Enoki's dependencies. "
                "Install them with: pip install -e ."
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
                if item.argument is not None and item.argument.prep:
                    prep = item.argument.prep
                    if not predicate.casefold().endswith(f" {prep.casefold()}"):
                        predicate = f"{predicate} {prep}"
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

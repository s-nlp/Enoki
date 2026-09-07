"""Unified runtime inference for the three Enoki OpenIE backends.

The module intentionally imports backend dependencies lazily. The default installation supports
encoder extraction and NLI. Rules, LLM extraction, evaluation and training
have optional dependencies.
"""

from __future__ import annotations

import json
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
        from fact_extractor.llm_backend import split_sentences_with_spans
        from enoki.projection import PARTS, text_spans

        results = []
        for text in items:
            sentences = split_sentences_with_spans(text)
            # Use original slices: the splitter's .text normalizes whitespace.
            inputs = [text[s.start:s.end] for s in sentences]
            extracted = self._backend.extract(inputs)
            if len(extracted) != len(sentences):
                raise RuntimeError("Extractor returned an unexpected number of sentences")
            triples = []
            for sentence, result in zip(sentences, extracted):
                original = text[sentence.start:sentence.end]
                for raw in result["triples"]:
                    triple = dict(raw)
                    spans = raw.get("spans")
                    if spans is None:
                        spans = {part: text_spans(original, raw.get(part, "")) for part in PARTS}
                    triple["spans"] = {
                        part: [[start + sentence.start, end + sentence.start] for start, end in spans.get(part, [])]
                        for part in PARTS
                    }
                    triple["sentence_start"] = sentence.start
                    triple["sentence_end"] = sentence.end
                    triples.append(triple)
            results.append({"text": text, "triples": triples})
        return results

    def detect(
        self,
        *,
        context: str,
        answer: str,
        nli_method: str = "modernbert",
        max_length: int = 2048,
        return_all: bool = False,
        return_stats: bool = False,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Verify full facts and project failures to original answer tokens.

        By default return localized unsupported facts. ``return_all`` includes
        every scored fact (including ones without a unique source span).
        ``return_stats`` returns ``{"results": [...], "stats": {...}}``.
        Scores mean lack of support in the provided context, not factual falsity.
        """
        if not isinstance(context, str) or not context.strip():
            raise ValueError("context must be a non-empty string")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        from fact_extractor.llm_backend import split_sentences_with_spans
        from enoki.projection import PARTS, project_facts

        triples = self.extract(answer)[0]["triples"]
        candidates = [t for t in triples if t["subject"].strip() and t["predicate"].strip()]
        probabilities = []
        if candidates:
            from nli import check_nli_batch_fast, hallucination_prob_from_nli
            scores = check_nli_batch_fast(
                context,
                [" ".join(t[p] for p in PARTS if t[p]) for t in candidates],
                method=nli_method,
                max_length=max_length,
            )
            if len(scores) != len(candidates):
                raise RuntimeError("NLI returned an unexpected number of fact scores")
            probabilities = [hallucination_prob_from_nli(score) for score in scores]
        projections = project_facts(answer, candidates, probabilities, threshold)
        results = []
        for triple, probability, (spans, suppressed) in zip(candidates, probabilities, projections):
            if not return_all and (probability <= threshold or suppressed or not spans):
                continue
            fact = {part: triple[part] for part in PARTS}
            # Preserve discontiguous projections as separate exact source spans.
            for bounds in spans or [None]:
                start, end = bounds if bounds is not None else (None, None)
                results.append({
                    "span": answer[start:end] if bounds is not None else None,
                    "start": start, "end": end,
                    "fact": fact, "probability": probability,
                })
        sentences = split_sentences_with_spans(answer)
        checked_sentences = {t["sentence_start"] for t in candidates}
        localized = sum(p > threshold and bool(spans) and not suppressed
                        for p, (spans, suppressed) in zip(probabilities, projections))
        stats = {
            "sentences_total": len(sentences),
            "sentences_with_checked_facts": len(checked_sentences),
            "sentences_without_checked_facts": len(sentences) - len(checked_sentences),
            "facts_extracted": len(triples),
            "facts_checked": len(candidates),
            "facts_skipped": len(triples) - len(candidates),
            "facts_supported": sum(p <= threshold for p in probabilities),
            "facts_unsupported": sum(p > threshold for p in probabilities),
            "facts_unlocalized": sum(not spans for spans, _ in projections),
            "facts_localized_unsupported": localized,
            "facts_suppressed_by_base": sum(p > threshold and suppressed
                                            for p, (_, suppressed) in zip(probabilities, projections)),
            "threshold": threshold,
            "status": ("no_facts" if not candidates else "partial" if
                       len(checked_sentences) < len(sentences) or any(not spans for spans, _ in projections)
                       or len(candidates) < len(triples) else "checked"),
        }
        return {"results": results, "stats": stats} if return_stats else results

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
                "Create a fresh Poetry environment and run: poetry install"
            ) from error
        try:
            from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version
        except ImportError as error:
            raise RuntimeError(
                "Encoder inference requires Transformers. Install Enoki with: "
                "poetry install"
            ) from error

        if transformers_version != "4.57.6":
            raise RuntimeError(
                "The published Enoki-Encoder requires transformers==4.57.6. "
                "Reinstall Enoki's dependencies with: poetry install"
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
        from enoki.encoder_decode import extract_anchored
        return extract_anchored(
            texts, self.model, self.tokenizer,
            min_confidence=self.min_confidence, top_k=self.top_k,
        )


class _LocalEncoderBackend:
    """Inference backend for directories exported by ``enoki train encoder``."""

    def __init__(self, *, model_dir: Path, device: str, min_confidence: float, top_k: int) -> None:
        import torch
        from transformers import AutoTokenizer

        from model.export import MANIFEST_NAME
        try:
            from model.model import IGLModel
        except ImportError as error:
            raise RuntimeError(
                "Loading a Lightning checkpoint requires poetry install -E train. "
                "The published Hugging Face model only needs inference dependencies."
            ) from error

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
        from enoki.encoder_decode import extract_anchored
        return extract_anchored(
            texts, self.model, self.tokenizer,
            min_confidence=self.min_confidence, top_k=self.top_k, local=True,
        )


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
                "LLM inference requires the LLM dependencies. "
                "Install them with: poetry install -E llm"
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
                    sentence=text[sentence.start:sentence.end],
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
                "Rules inference requires spaCy. "
                "Install it with: poetry install -E rules"
            ) from error

        try:
            self.pipeline = Pipeline()
        except OSError as error:
            raise RuntimeError(
                "The rules backend requires the en_core_web_trf spaCy model. "
                "Install it with: poetry run python -m spacy download en_core_web_trf"
            ) from error

    def extract(self, texts: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for text in texts:
            triples = []
            for item in self.pipeline.extract(text):
                triples.append(_rule_triple(item))
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


def _has_negation_token(predicate) -> bool:
    try:
        return any(getattr(token, "dep_", "") == "neg" for token in predicate)
    except TypeError:
        return False


def _rule_triple(item):
    predicate = item.predicate_surface
    # The predicate span is contiguous, so a ``neg`` token between an
    # auxiliary and the verb ("could not pay") is already part of the
    # surface; only prefix NOT when the negation is not visible in it.
    if item.negated and not _has_negation_token(item.predicate):
        predicate = f"NOT {predicate}"
    if item.argument is not None and item.argument.prep:
        prep = item.argument.prep
        if not predicate.casefold().endswith(f" {prep.casefold()}"):
            predicate = f"{predicate} {prep}"
    triple = _triple(
        item.subject.text, predicate,
        item.argument.span.text if item.argument is not None else "",
        item.confidence,
    )
    triple["spans"] = {
        part: [[token.idx, token.idx + len(token.text)] for token in span]
        if span is not None else []
        for part, span in {
            "subject": item.subject,
            "predicate": item.predicate if item.predicate_text is None else None,
            "object": item.argument.span if item.argument is not None else None,
        }.items()
    }
    return triple

"""Transformers-native Enoki OpenIE model with built-in triplet extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.utils import ModelOutput

try:
    from .configuration_enoki import EnokiOpenIEConfig
except ImportError:  # Support direct imports while building the local release.
    from configuration_enoki import EnokiOpenIEConfig


LABEL2ID = {
    "NONE": 0,
    "ARG1": 1,
    "REL": 2,
    "ARG2": 3,
    "LOC_TMP": 4,
    "TYPE": 5,
}


@dataclass
class EnokiOpenIEOutput(ModelOutput):
    """Token-label predictions and confidence for every IGL depth."""

    predictions: torch.LongTensor | None = None
    confidences: torch.FloatTensor | None = None


def _detect_arch(encoder: nn.Module) -> str:
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return "bert"
    if hasattr(encoder, "layers"):
        return "modernbert"
    raise ValueError(
        f"Unsupported encoder architecture: {type(encoder).__name__}. "
        "Expected BERT or ModernBERT."
    )


def _word_starts(word_ids: list[int | None], max_words: int) -> list[int]:
    starts: list[int] = []
    seen: set[int] = set()
    for position, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in seen:
            starts.append(position)
            seen.add(word_id)
    return starts + [0] * (max_words - len(starts))


class EnokiOpenIEModel(PreTrainedModel):
    """ModernBERT IGL model that extracts OpenIE subject-relation-object triples."""

    config_class = EnokiOpenIEConfig
    base_model_prefix = "_encoder"
    main_input_name = "input_ids"

    def __init__(self, config: EnokiOpenIEConfig) -> None:
        super().__init__(config)
        if not config.encoder_config:
            raise ValueError("config.encoder_config is required")

        encoder_values = dict(config.encoder_config)
        encoder_model_type = encoder_values.pop("model_type")
        encoder_config = AutoConfig.for_model(encoder_model_type, **encoder_values)
        self._encoder = AutoModel.from_config(
            encoder_config,
            attn_implementation="sdpa",
        )

        if config.vocab_size > 0:
            self._encoder.resize_token_embeddings(config.vocab_size)

        hidden_size = self._encoder.config.hidden_size
        self._arch = _detect_arch(self._encoder)

        all_layers = (
            list(self._encoder.encoder.layer)
            if self._arch == "bert"
            else list(self._encoder.layers)
        )
        iterative_layers = config.iterative_layers
        self._iterative = nn.ModuleList(all_layers[-iterative_layers:])
        if self._arch == "bert":
            self._encoder.encoder.layer = nn.ModuleList(
                all_layers[:-iterative_layers]
            )
        else:
            self._encoder.layers = nn.ModuleList(all_layers[:-iterative_layers])

        self._dropout = nn.Dropout(config.dropout)
        self._label_emb = nn.Embedding(config.num_labels + 1, hidden_size)
        self._merge = nn.Linear(hidden_size, config.labelling_dim)
        self._out = nn.Linear(config.labelling_dim, config.num_labels)
        self._tokenizer = None

    def _base_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        output = self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = output.last_hidden_state

        if self._arch == "bert":
            extended_mask = self._encoder.get_extended_attention_mask(
                attention_mask,
                input_ids.shape,
            )
            return hidden, (extended_mask, None, None)

        attn_mask, sliding_window_mask = self._encoder._update_attention_mask(
            attention_mask.bool(),
            output_attentions=False,
        )
        position_ids = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        return hidden, (attn_mask, sliding_window_mask, position_ids)

    def _iter_step(self, hidden: torch.Tensor, mask_context: tuple) -> torch.Tensor:
        attention_mask, sliding_window_mask, position_ids = mask_context
        for layer in self._iterative:
            if self._arch == "bert":
                hidden = layer(hidden, attention_mask=attention_mask)[0]
            else:
                hidden = layer(
                    hidden,
                    attention_mask=attention_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                )[0]
        return hidden

    @staticmethod
    def _gather_words(
        hidden: torch.Tensor,
        word_starts: torch.Tensor,
    ) -> torch.Tensor:
        hidden_size = hidden.shape[-1]
        indices = word_starts.unsqueeze(-1).expand(-1, -1, hidden_size)
        return torch.gather(hidden, 1, indices)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        num_words: torch.Tensor | None = None,
        n_real_words: torch.Tensor | None = None,
        **_: Any,
    ) -> EnokiOpenIEOutput:
        """Predict one token-label sequence per IGL extraction depth."""
        max_depth = self.config.max_depth
        batch_size = input_ids.shape[0]
        n_unused = len(self.config.unused_tokens)

        hidden, mask_context = self._base_encode(input_ids, attention_mask)
        all_scores: list[torch.Tensor] = []
        previous_predictions: torch.Tensor | None = None
        histories: list[list[tuple[int, ...]]] = [[] for _ in range(batch_size)]
        valid_depths = [[False] * max_depth for _ in range(batch_size)]

        for depth in range(max_depth):
            hidden = self._dropout(self._iter_step(hidden, mask_context))
            word_hidden = self._gather_words(hidden, word_starts)
            labels = (
                previous_predictions
                if previous_predictions is not None
                else word_hidden.new_full(
                    word_hidden.shape[:2],
                    self.config.num_labels,
                    dtype=torch.long,
                )
            )
            scores = self._out(self._merge(word_hidden + self._label_emb(labels)))
            all_scores.append(scores)
            previous_predictions = scores.argmax(dim=-1)

            for batch_index in range(batch_size):
                if n_real_words is not None:
                    real_words = int(n_real_words[batch_index])
                elif num_words is not None:
                    real_words = int(num_words[batch_index]) - n_unused
                else:
                    real_words = word_starts.shape[1]

                prediction = previous_predictions[batch_index, :real_words]
                signature = tuple(prediction.tolist())
                valid = (
                    prediction.eq(LABEL2ID["ARG1"]).any()
                    and prediction.eq(LABEL2ID["REL"]).any()
                )
                if valid and signature not in histories[batch_index]:
                    histories[batch_index].append(signature)
                    valid_depths[batch_index][depth] = True

        width = all_scores[0].shape[1]
        if n_real_words is not None:
            real_word_counts = n_real_words
        elif num_words is not None:
            real_word_counts = (num_words - n_unused).clamp(min=0)
        else:
            real_word_counts = None

        word_mask = None
        if real_word_counts is not None:
            word_mask = (
                torch.arange(width, device=input_ids.device).unsqueeze(0)
                < real_word_counts.unsqueeze(1)
            )

        predictions: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        for depth, scores in enumerate(all_scores):
            log_probabilities = torch.log_softmax(scores, dim=-1)
            max_log_probabilities, prediction = log_probabilities.max(dim=-1)
            for batch_index in range(batch_size):
                if not valid_depths[batch_index][depth]:
                    prediction[batch_index] = 0

            if word_mask is not None:
                max_log_probabilities = max_log_probabilities * word_mask.float()

            non_none = prediction != LABEL2ID["NONE"]
            if word_mask is not None:
                non_none = non_none & word_mask
            non_none_float = non_none.float()
            non_none_count = non_none_float.sum(-1)
            confidence = torch.exp(
                (max_log_probabilities * non_none_float).sum(-1)
                / non_none_count.clamp(min=1.0)
            )
            confidence = torch.where(
                non_none_count > 0,
                confidence,
                torch.zeros_like(confidence),
            )
            predictions.append(prediction.unsqueeze(1))
            confidences.append(confidence.unsqueeze(1))

        return EnokiOpenIEOutput(
            predictions=torch.cat(predictions, dim=1),
            confidences=torch.cat(confidences, dim=1),
        )

    def _get_tokenizer(self):
        if self._tokenizer is None:
            source = self.config.name_or_path or self.config.base_model_name
            self._tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
        return self._tokenizer

    @staticmethod
    def _tokenize_words(sentence: str) -> list[str]:
        try:
            import nltk
        except ImportError as exc:
            raise ImportError(
                "Enoki tokenization requires NLTK. Install it with `pip install nltk`."
            ) from exc
        return nltk.word_tokenize(sentence, preserve_line=True)

    @staticmethod
    def _detokenize(words: list[str]) -> str:
        if not words:
            return ""
        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer
        except ImportError:
            return " ".join(words)
        return TreebankWordDetokenizer().detokenize(words)

    @staticmethod
    def _fit_words(
        words: list[str],
        tokenizer,
        unused_tokens: list[str],
        max_length: int,
    ) -> list[str]:
        """Keep a maximal word prefix while reserving space for sentinels."""
        def encoded_length(prefix_size: int) -> int:
            encoded = tokenizer(
                words[:prefix_size] + unused_tokens,
                is_split_into_words=True,
                truncation=False,
            )
            return len(encoded["input_ids"])

        if encoded_length(len(words)) <= max_length:
            return words

        low, high = 0, len(words)
        while low < high:
            middle = (low + high + 1) // 2
            if encoded_length(middle) <= max_length:
                low = middle
            else:
                high = middle - 1
        return words[:low]

    @torch.inference_mode()
    def extract_triples(
        self,
        texts: str | list[str],
        *,
        tokenizer=None,
        min_confidence: float = 0.7,
        top_k: int = 10,
        batch_size: int = 1,
        max_length: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract OpenIE triples from one string or a list of English sentences."""
        sentences = [texts] if isinstance(texts, str) else list(texts)
        if not sentences or any(not text.strip() for text in sentences):
            raise ValueError("texts must contain one or more non-empty sentences")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if top_k < 1 or batch_size < 1:
            raise ValueError("top_k and batch_size must be at least 1")

        tokenizer = tokenizer or self._get_tokenizer()
        unused_tokens = list(self.config.unused_tokens)
        max_length = max_length or self.config.max_length
        device = next(self.parameters()).device
        self.eval()

        results: list[dict[str, Any]] = []
        for offset in range(0, len(sentences), batch_size):
            batch_sentences = sentences[offset : offset + batch_size]
            words_batch = [self._tokenize_words(text) for text in batch_sentences]
            words_batch = [
                self._fit_words(words, tokenizer, unused_tokens, max_length)
                for words in words_batch
            ]
            model_words = [words + unused_tokens for words in words_batch]
            encoding = tokenizer(
                model_words,
                is_split_into_words=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            word_counts = [len(words) + len(unused_tokens) for words in words_batch]
            max_words = max(word_counts)
            starts = [
                _word_starts(encoding.word_ids(i), max_words)
                for i in range(len(batch_sentences))
            ]
            output = self(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                word_starts=torch.tensor(starts, dtype=torch.long, device=device),
                num_words=torch.tensor(word_counts, dtype=torch.long, device=device),
            )

            for batch_index, sentence in enumerate(batch_sentences):
                words = words_batch[batch_index]
                n_real = len(words)
                n_total = word_counts[batch_index]
                candidates: list[dict[str, Any]] = []

                for depth in range(output.predictions.shape[1]):
                    confidence = float(output.confidences[batch_index, depth])
                    if confidence < min_confidence:
                        continue
                    row = output.predictions[
                        batch_index, depth, :n_total
                    ].tolist()
                    subject_words: list[str] = []
                    relation_words: list[str] = []
                    object_words: list[str] = []

                    for word, label_id in zip(words, row[:n_real]):
                        if label_id == LABEL2ID["ARG1"]:
                            subject_words.append(word)
                        elif label_id == LABEL2ID["REL"]:
                            relation_words.append(word)
                        elif label_id in (
                            LABEL2ID["ARG2"],
                            LABEL2ID["LOC_TMP"],
                            LABEL2ID["TYPE"],
                        ):
                            object_words.append(word)

                    relation_case = 0
                    for sentinel_index, label_id in enumerate(
                        row[n_real : n_real + len(unused_tokens)]
                    ):
                        if label_id == LABEL2ID["REL"]:
                            relation_case = sentinel_index + 1
                            break

                    subject = self._detokenize(subject_words)
                    relation = self._detokenize(relation_words)
                    object_text = self._detokenize(object_words).lstrip(" ,;:")
                    if relation_case == 1:
                        relation = f"is {relation}".strip()
                    elif relation_case == 2:
                        relation = f"is {relation} of".strip()
                    elif relation_case == 3:
                        relation = f"is {relation} from".strip()

                    if subject and relation:
                        candidates.append(
                            {
                                "subject": subject,
                                "relation": relation,
                                "object": object_text,
                                "confidence": confidence,
                            }
                        )

                unique: list[dict[str, Any]] = []
                seen: set[tuple[str, str, str]] = set()
                for triple in sorted(
                    candidates,
                    key=lambda item: item["confidence"],
                    reverse=True,
                ):
                    key = (
                        triple["subject"].lower(),
                        triple["relation"].lower(),
                        triple["object"].lower(),
                    )
                    if key not in seen:
                        seen.add(key)
                        unique.append(triple)

                results.append(
                    {
                        "sentence": sentence,
                        "triples": unique[:top_k],
                    }
                )

        return results

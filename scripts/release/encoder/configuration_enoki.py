"""Hugging Face configuration for the Enoki OpenIE encoder."""

from __future__ import annotations

from transformers import PretrainedConfig


LABEL2ID = {
    "NONE": 0,
    "ARG1": 1,
    "REL": 2,
    "ARG2": 3,
    "LOC_TMP": 4,
    "TYPE": 5,
}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


class EnokiOpenIEConfig(PretrainedConfig):
    """Configuration for a ModernBERT-based IGL OpenIE model."""

    model_type = "enoki-openie"

    def __init__(
        self,
        base_model_name: str = "answerdotai/ModernBERT-large",
        encoder_config: dict | None = None,
        vocab_size: int = 50368,
        num_labels: int = 6,
        max_depth: int = 14,
        iterative_layers: int = 2,
        labelling_dim: int = 300,
        dropout: float = 0.1,
        max_length: int = 128,
        unused_tokens: list[str] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("id2label", ID2LABEL)
        kwargs.setdefault("label2id", LABEL2ID)
        kwargs.setdefault("architectures", ["EnokiOpenIEModel"])
        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.encoder_config = encoder_config or {}
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.max_depth = max_depth
        self.iterative_layers = iterative_layers
        self.labelling_dim = labelling_dim
        self.dropout = dropout
        self.max_length = max_length
        self.unused_tokens = unused_tokens or [
            "[unused1]",
            "[unused2]",
            "[unused3]",
        ]


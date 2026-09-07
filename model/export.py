"""Portable local model artifacts for Enoki-Encoder."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


MANIFEST_NAME = "enoki_encoder.json"


def is_local_encoder_model(source: str | Path) -> bool:
    """Whether *source* is a directory exported by ``enoki train encoder``."""
    return Path(source).is_dir() and (Path(source) / MANIFEST_NAME).is_file()


def export_encoder_model(checkpoint: str | Path, tokenizer, output_dir: str | Path) -> Path:
    """Write a trained IGL checkpoint as a portable Enoki-Encoder directory."""
    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_name = "model.ckpt"
    shutil.copy2(checkpoint_path, output_path / checkpoint_name)
    tokenizer.save_pretrained(output_path)
    (output_path / MANIFEST_NAME).write_text(
        json.dumps({"format": "enoki-lightning-v1", "checkpoint": checkpoint_name}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output_path

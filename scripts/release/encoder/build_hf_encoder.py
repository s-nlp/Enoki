#!/usr/bin/env python3
"""Convert an Enoki Lightning checkpoint into a self-contained HF model repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from configuration_enoki import EnokiOpenIEConfig
from modeling_enoki import EnokiOpenIEModel


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("hf_enoki_openie_encoder"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading checkpoint metadata from {checkpoint_path}", flush=True)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    hyperparameters = checkpoint["hyper_parameters"]
    base_model_name = hyperparameters["model_name"]

    print(f"Loading config and tokenizer for {base_model_name}", flush=True)
    base_config = AutoConfig.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    unused_tokens = ["[unused1]", "[unused2]", "[unused3]"]
    tokenizer.add_special_tokens({"additional_special_tokens": unused_tokens})

    if len(tokenizer) != hyperparameters["vocab_size"]:
        raise RuntimeError(
            "Tokenizer/checkpoint vocabulary mismatch: "
            f"{len(tokenizer)} != {hyperparameters['vocab_size']}"
        )

    config = EnokiOpenIEConfig(
        base_model_name=base_model_name,
        encoder_config=base_config.to_dict(),
        vocab_size=hyperparameters["vocab_size"],
        num_labels=hyperparameters["num_labels"],
        max_depth=hyperparameters["max_depth"],
        iterative_layers=hyperparameters["iterative_layers"],
        labelling_dim=hyperparameters["labelling_dim"],
        dropout=hyperparameters["dropout"],
        max_length=128,
        unused_tokens=unused_tokens,
    )

    print("Building the Transformers model", flush=True)
    model = EnokiOpenIEModel(config)
    load_result = model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
        assign=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"State mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()

    EnokiOpenIEConfig.register_for_auto_class()
    EnokiOpenIEModel.register_for_auto_class("AutoModel")

    print(f"Writing safetensors model to {output_path}", flush=True)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(output_path)

    release_dir = Path(__file__).resolve().parent
    shutil.copy2(release_dir / "inference_hf.py", output_path / "inference.py")
    shutil.copy2(
        release_dir / "requirements-hf-encoder.txt",
        output_path / "requirements.txt",
    )
    model_card = (release_dir / "README.template").read_text(encoding="utf-8")
    (output_path / "README.md").write_text(
        model_card.replace(
            "{{BANNER_PATH}}",
            "./enoki-openie-banner.png",
        ),
        encoding="utf-8",
    )
    shutil.copy2(
        REPO_ROOT / "assets" / "enoki-openie-banner.png",
        output_path / "enoki-openie-banner.png",
    )

    metadata = {
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
    (output_path / "conversion_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Done. Test with:", flush=True)
    print(
        "  python scripts/release/encoder/inference_hf.py "
        f"--model {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

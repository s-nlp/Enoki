#!/usr/bin/env python3
"""Train the IGL Open IE encoder on OIE4-format label files (stage 3 objective).

Label file format (one sentence block per blank-separated group):
    The company was founded in 1990 . [unused1] [unused2] [unused3]
    NONE ARG1 ARG1 REL ARG2 ARG2 NONE NONE NONE NONE

Download OIE4 labels (Zenodo 4094228):
    pip install zenodo-get && zenodo_get 4094228
    tar xzf data.tar.gz   # produces openie4_labels/train_labels, dev_labels

Usage:
    enoki train encoder --train data/openie4_labels
    enoki train encoder --train data/stage3_labels --dev data/stage3_dev
    enoki train encoder --train data/openie4_labels --hungarian
    enoki train encoder --train data/openie4_labels --checkpoint checkpoints/best.ckpt
"""
import argparse
import os

import torch
import lightning as L

torch.set_float32_matmul_precision("high")

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model.oie4_data import OIE4DataModule
from model.model import IGLModel


def run_training(
    train_fp: str = "data/openie4_labels",
    dev_fp: str | None = None,
    dev_split: float = 0.002,
    allow_negatives: bool = True,
    model_name: str = "answerdotai/ModernBERT-large",
    iter_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 5e-5,
    max_depth: int = 14,
    max_length: int = 128,
    patience: int = 3,
    warmup_steps: int = 500,
    hungarian: bool = True,
    empty_weight: float = 0.1,
    workers: int = 4,
    seed: int = 42,
    gpus: int = 1,
    out: str = "checkpoints/",
    checkpoint: str | None = None,
    save_weights_only: bool = True,
) -> None:

    L.seed_everything(seed)

    dm = OIE4DataModule(
        tokenizer_name=model_name,
        train_fp=train_fp,
        dev_fp=dev_fp,
        dev_split=dev_split,
        allow_negatives=allow_negatives,
        batch_size=batch_size,
        max_depth=max_depth,
        max_length=max_length,
        num_workers=workers,
    )
    dm.setup()

    steps_per_epoch = len(dm.train_dataloader())
    total_steps = steps_per_epoch * epochs

    model = IGLModel(
        model_name=model_name,
        vocab_size=len(dm.tokenizer),
        max_depth=max_depth,
        iterative_layers=iter_layers,
        dropout=dropout,
        lr=lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        hungarian=hungarian,
        empty_weight=empty_weight,
    )

    ckpt_path = None
    if checkpoint:
        _raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt_in_out_dir = os.path.abspath(checkpoint).startswith(os.path.abspath(out))
        is_resume = "optimizer_states" in _raw and ckpt_in_out_dir
        if is_resume:
            ckpt_path = checkpoint
        else:
            missing, unexpected = model.load_state_dict(_raw["state_dict"], strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint mismatch — missing: {missing}  unexpected: {unexpected}"
                )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else "auto",
        precision="bf16-mixed",
        callbacks=[
            ModelCheckpoint(
                dirpath=out,
                filename=f"igl-{model_name.split('/')[-1]}" + "-{epoch:02d}-{val_f1:.3f}",
                monitor="val_f1",
                mode="max",
                save_top_k=1,
                save_weights_only=save_weights_only,
            ),
            EarlyStopping(monitor="val_f1", patience=patience, mode="max"),
        ],
        logger=TensorBoardLogger("logs/", name="igl_openie"),
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )
    trainer.fit(model, dm, ckpt_path=ckpt_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train IGL OpenIE encoder")
    p.add_argument("--train",          default="data/openie4_labels",
                   help="OIE4 label file (train split)")
    p.add_argument("--dev",            default=None,
                   help="Separate dev label file (auto-split from train if omitted)")
    p.add_argument("--dev-split",      type=float, default=0.002,
                   help="Fraction of training data for auto dev split (default 0.002 ≈ 5K sentences)")
    p.add_argument("--no-negatives",   action="store_true",
                   help="Filter out sentences with no extractions")
    p.add_argument("--model",          default="answerdotai/ModernBERT-large")
    p.add_argument("--iter-layers",    type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--epochs",         type=int,   default=15)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--max-depth",      type=int,   default=14)
    p.add_argument("--max-length",     type=int,   default=128)
    p.add_argument("--patience",       type=int,   default=3)
    p.add_argument("--warmup-steps",   type=int,   default=500)
    p.add_argument("--hungarian",      action="store_true",
                   help="Use Hungarian matching for optimal depth-to-triple assignment")
    p.add_argument("--empty-weight",   type=float, default=0.1)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--gpus",           type=int,   default=1)
    p.add_argument("--out",            default="checkpoints/")
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--save-weights-only", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        train_fp=args.train,
        dev_fp=args.dev,
        dev_split=args.dev_split,
        allow_negatives=not args.no_negatives,
        model_name=args.model,
        iter_layers=args.iter_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_depth=args.max_depth,
        max_length=args.max_length,
        patience=args.patience,
        warmup_steps=args.warmup_steps,
        hungarian=args.hungarian,
        empty_weight=args.empty_weight,
        workers=args.workers,
        seed=args.seed,
        gpus=args.gpus,
        out=args.out,
        checkpoint=args.checkpoint,
        save_weights_only=args.save_weights_only,
    )

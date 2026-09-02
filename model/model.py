"""Iterative Grid Labeling (IGL) model for Open IE.

Architecture
------------
  1. Encoder (BERT-style or ModernBERT) with last `iterative_layers` blocks
     detached.
  2. At each depth step d (up to max_depth):
       a. Re-run the detached blocks on the current hidden states.
       b. Gather first-subword hidden state for each word.
       c. If d > 0: add label embeddings from the previous step's greedy
          predictions so the model "sees" what was already extracted.
       d. Project: H → labelling_dim → num_labels.
  3. Loss: cross-entropy at every depth step, -100 positions ignored.

Supported architectures
-----------------------
  Classic BERT  – layers at model.encoder.layer
  ModernBERT    – layers at model.layers  (requires position_ids + sliding_window_mask)
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

_N_UNUSED = 3  # number of [unused] sentinel tokens appended to every sentence

LABEL2ID = {"NONE": 0, "ARG1": 1, "REL": 2, "ARG2": 3, "LOC_TMP": 4, "TYPE": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def _detect_arch(encoder) -> str:
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return "bert"
    if hasattr(encoder, "layers"):
        return "modernbert"
    raise ValueError(
        f"Unsupported encoder architecture: {type(encoder).__name__}. "
        "Expected a BERT-style model (encoder.layer) or ModernBERT (layers)."
    )


class IGLModel(L.LightningModule):

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-large",
        vocab_size: int = 0,
        num_labels: int = NUM_LABELS,
        max_depth: int = 5,
        iterative_layers: int = 2,
        labelling_dim: int = 300,
        dropout: float = 0.0,
        lr: float = 2e-5,
        warmup_steps: int = 0,
        total_steps: int = 0,
        hungarian: bool = False,     # use Hungarian matching for depth-to-triple assignment
        empty_weight: float = 0.1,   # weight for EMPTY-matched depths in Hungarian loss
    ):
        super().__init__()
        self.save_hyperparameters()

        self._encoder = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        if vocab_size > 0:
            self._encoder.resize_token_embeddings(vocab_size)
        H = self._encoder.config.hidden_size
        self._arch = _detect_arch(self._encoder)

        # Detach last n transformer blocks; they are re-applied at every depth step
        n = iterative_layers
        if self._arch == "bert":
            all_layers = list(self._encoder.encoder.layer)
            self._iterative = nn.ModuleList(all_layers[-n:])
            self._encoder.encoder.layer = nn.ModuleList(all_layers[:-n])
        else:  # modernbert
            all_layers = list(self._encoder.layers)
            self._iterative = nn.ModuleList(all_layers[-n:])
            self._encoder.layers = nn.ModuleList(all_layers[:-n])

        self._dropout = nn.Dropout(p=dropout)
        # Index `num_labels` acts as "no-label-yet" embedding at depth 0
        self._label_emb = nn.Embedding(num_labels + 1, H)
        self._merge = nn.Linear(H, labelling_dim)
        self._out = nn.Linear(labelling_dim, num_labels)

        self._loss = nn.CrossEntropyLoss(ignore_index=-100)

    # ------------------------------------------------------------------
    # Architecture-aware helpers
    # ------------------------------------------------------------------

    def _base_encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Run base encoder (minus iterative layers) and return hidden + mask context."""
        out = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                                    # (B, T, H)

        if self._arch == "bert":
            ext_mask = self._encoder.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )                                                             # (B, 1, 1, T)
            return hidden, (ext_mask, None, None)

        else:  # modernbert
            attn_mask, sw_mask = self._encoder._update_attention_mask(
                attention_mask.bool(), output_attentions=False
            )
            pos_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)                # (B, T)
            return hidden, (attn_mask, sw_mask, pos_ids)

    def _iter_step(self, hidden: torch.Tensor, mask_ctx: tuple) -> torch.Tensor:
        """Apply the iterative transformer blocks once."""
        attn_mask, sw_mask, pos_ids = mask_ctx
        for layer in self._iterative:
            if self._arch == "bert":
                hidden = layer(hidden, attention_mask=attn_mask)[0]
            else:
                hidden = layer(
                    hidden,
                    attention_mask=attn_mask,
                    sliding_window_mask=sw_mask,
                    position_ids=pos_ids,
                )[0]
        return hidden

    def _gather_words(self, hidden: torch.Tensor, word_starts: torch.Tensor) -> torch.Tensor:
        # word_starts: (B, W) — token position of first subword per word
        H = hidden.shape[-1]
        idx = word_starts.unsqueeze(-1).expand(-1, -1, H)                # (B, W, H)
        return torch.gather(hidden, 1, idx)                               # (B, W, H)

    # ------------------------------------------------------------------
    # CE loss helpers
    # ------------------------------------------------------------------

    def _ce_loss(
        self,
        all_scores: list,                           # list of (B, W, L), length = actual depth steps run
        label_matrix: torch.Tensor,                 # (B, max_depth, W)
        num_words: torch.Tensor | None = None,      # (B,) — total words incl. [unused] sentinels
        n_real_words: torch.Tensor | None = None,   # (B,) — real words only, excl. [unused]
    ) -> torch.Tensor:
        if self.hparams.hungarian:
            return self._ce_loss_hungarian(
                all_scores, label_matrix, num_words=num_words, n_real_words=n_real_words,
            )
        loss = torch.zeros(1, device=label_matrix.device, dtype=torch.float).squeeze()
        n_active = 0
        for d, sc in enumerate(all_scores):
            flat = label_matrix[:, d, :].reshape(-1)
            if (flat == -100).all():
                continue
            loss = loss + self._loss(sc.reshape(-1, sc.shape[-1]), flat)
            n_active += 1
        return loss / max(n_active, 1)

    def _ce_loss_hungarian(
        self,
        all_scores: list,
        label_matrix: torch.Tensor,
        num_words: torch.Tensor | None = None,
        n_real_words: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from scipy.optimize import linear_sum_assignment

        none_id      = LABEL2ID["NONE"]
        empty_weight = self.hparams.empty_weight

        B = label_matrix.shape[0]
        D = len(all_scores)
        W = all_scores[0].shape[1]
        L = all_scores[0].shape[2]
        device = label_matrix.device

        scores_stacked = torch.stack(all_scores, dim=0)  # (D, B, W, L)

        total_loss = scores_stacked.new_tensor(0.0)
        n_sentences = 0

        for b in range(B):
            gold_idx = (label_matrix[b] != -100).any(dim=-1).nonzero(as_tuple=True)[0]  # (G,)
            G = gold_idx.shape[0]

            sc_b = scores_stacked[:, b, :, :]  # (D, W, L)

            def _make_empty_row(valid_pos: torch.Tensor) -> torch.Tensor:
                """NONE (none_id) at valid word positions, -100 at padding."""
                return torch.where(
                    valid_pos,
                    label_matrix.new_full((W,), none_id),
                    label_matrix.new_full((W,), -100),
                )

            if G > 0:
                gold_b = label_matrix[b, gold_idx]  # (G, W)

                if G < D:
                    valid_pos = (gold_b != -100).any(dim=0)          # (W,)
                    empty_row = _make_empty_row(valid_pos)            # (W,)
                    n_empty   = D - G
                    targets   = torch.cat(
                        [gold_b, empty_row.unsqueeze(0).expand(n_empty, W)], dim=0
                    )                                                  # (D, W)
                elif G > D:
                    if not hasattr(IGLModel, "_gd_overflow_seen"):
                        IGLModel._gd_overflow_seen = 0
                    IGLModel._gd_overflow_seen += 1
                    if IGLModel._gd_overflow_seen == 1:
                        warnings.warn(
                            f"G={G} > D={D}: gold triples beyond the depth budget will be "
                            "ignored. Consider increasing --max-depth. "
                            "Further occurrences silenced; final count logged at fit end.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    targets = gold_b                                   # (G, W)
                else:
                    targets = gold_b                                   # (D, W)

            else:
                if n_real_words is not None:
                    nw = int(n_real_words[b])
                elif num_words is not None:
                    nw = int(num_words[b]) - _N_UNUSED
                else:
                    raise ValueError(
                        f"G==0 for sentence {b} but neither n_real_words nor num_words "
                        "is provided; cannot construct EMPTY target rows."
                    )
                valid_pos = torch.arange(W, device=device) < nw
                empty_row = _make_empty_row(valid_pos)
                targets   = empty_row.unsqueeze(0).expand(D, W)       # (D, W)

            T = targets.shape[0]

            sc_exp  = sc_b.unsqueeze(1).expand(D, T, W, L).reshape(D * T * W, L)
            tgt_exp = targets.unsqueeze(0).expand(D, T, W).reshape(D * T, W)

            ce_flat = F.cross_entropy(
                sc_exp,
                tgt_exp.reshape(D * T * W),
                ignore_index=-100,
                reduction='none',
            )                                                          # (D*T*W,)

            valid_t  = (tgt_exp != -100).float()                       # (D*T, W)
            cost_mat = (
                (ce_flat.reshape(D * T, W) * valid_t).sum(-1)
                / valid_t.sum(-1).clamp(min=1)
            ).reshape(D, T)                                            # (D, T)

            row_ind, col_ind = linear_sum_assignment(cost_mat.detach().cpu().numpy())

            row_t = torch.as_tensor(row_ind, device=device, dtype=torch.long)
            col_t = torch.as_tensor(col_ind, device=device, dtype=torch.long)

            matched = cost_mat[row_t, col_t]                           # (n_matched,)

            if G == 0:
                sent_loss = empty_weight * matched.mean()
            elif G < D and empty_weight < 1.0:
                is_empty = col_t >= G
                weights  = torch.where(
                    is_empty,
                    matched.new_full(matched.shape, empty_weight),
                    matched.new_ones(matched.shape),
                )
                sent_loss = (matched * weights).sum() / weights.sum().clamp(min=1e-6)
            else:
                sent_loss = matched.mean()

            total_loss  = total_loss + sent_loss
            n_sentences += 1

        return total_loss / max(n_sentences, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        label_matrix: torch.Tensor | None = None,
        num_words: torch.Tensor | None = None,
        n_real_words: torch.Tensor | None = None,
        return_scores: bool = False,
    ):
        """
        Train  (label_matrix not None):
            return_scores=False (default): Returns scalar loss.
            return_scores=True:            Returns (loss, all_depth_scores).

        Eval   (label_matrix=None):
            Returns (predictions, confidences).
        """
        is_train = label_matrix is not None
        max_depth = self.hparams.max_depth
        B = input_ids.shape[0]

        hidden, mask_ctx = self._base_encode(input_ids, attention_mask)

        all_scores: list[torch.Tensor] = []
        prev_preds: torch.Tensor | None = None
        sent_pred_history: list[list[tuple]] = [[] for _ in range(B)]
        valid_depth_mask = [[False] * max_depth for _ in range(B)]
        active = list(range(B))  # used only when not Hungarian

        for d in range(max_depth):
            hidden = self._iter_step(hidden, mask_ctx)
            hidden = self._dropout(hidden)

            word_h_raw = self._gather_words(hidden, word_starts)           # (B, W, H)

            lbl    = prev_preds if prev_preds is not None else \
                word_h_raw.new_full(word_h_raw.shape[:2], self.hparams.num_labels, dtype=torch.long)
            word_h = word_h_raw + self._label_emb(lbl)

            scores = self._out(self._merge(word_h))                       # (B, W, L)
            all_scores.append(scores)
            prev_preds = scores.argmax(dim=-1)                            # (B, W)

            if not is_train:
                def _get_nw(b: int) -> int:
                    if n_real_words is not None:
                        return int(n_real_words[b])
                    if num_words is not None:
                        return int(num_words[b]) - _N_UNUSED
                    return word_starts.shape[1]

                check_set = range(B) if self.hparams.hungarian else active
                next_active = []
                for b in check_set:
                    nw = _get_nw(b)
                    pred_nw = prev_preds[b, :nw]
                    pt    = tuple(pred_nw.tolist())
                    valid = pred_nw.eq(LABEL2ID["ARG1"]).any() and \
                            pred_nw.eq(LABEL2ID["REL"]).any()
                    if valid and pt not in sent_pred_history[b]:
                        sent_pred_history[b].append(pt)
                        valid_depth_mask[b][d] = True
                        if not self.hparams.hungarian:
                            next_active.append(b)
                if not self.hparams.hungarian:
                    active = next_active
                    if not active:
                        break

        if is_train:
            loss = self._ce_loss(
                all_scores, label_matrix, num_words=num_words, n_real_words=n_real_words,
            )
            if self.training and self._trainer is not None:
                self.log("ce_loss", loss.detach(), on_step=True, on_epoch=False)
                self.log("total_loss", loss.detach(), on_step=True, on_epoch=False)
            return (loss, all_scores) if return_scores else loss

        # Eval: predictions + per-depth confidence.
        preds_list: list[torch.Tensor] = []
        confs_list: list[torch.Tensor] = []
        W = all_scores[0].shape[1]
        if n_real_words is not None:
            real_word_counts = n_real_words
        elif num_words is not None:
            real_word_counts = (num_words - _N_UNUSED).clamp(min=0)
        else:
            real_word_counts = None
        if real_word_counts is not None:
            word_mask = (torch.arange(W, device=input_ids.device).unsqueeze(0)
                         < real_word_counts.unsqueeze(1))                 # (B, W) bool
        else:
            word_mask = None

        for d, sc in enumerate(all_scores):
            lp = torch.log_softmax(sc, dim=-1)
            mp, p = lp.max(dim=-1)                                        # (B, W)
            for b in range(B):
                if not valid_depth_mask[b][d]:
                    p[b] = 0    # all-NONE → filtered downstream
            if word_mask is not None:
                p  = p  * word_mask.long()
                mp = mp * word_mask.float()
            preds_list.append(p.unsqueeze(1))
            non_none       = (p != 0).float()
            non_none_count = non_none.sum(-1)
            conf = torch.exp(
                (mp * non_none).sum(-1) / non_none_count.clamp(min=1.0)
            )
            conf = torch.where(non_none_count > 0, conf, torch.zeros_like(conf))
            confs_list.append(conf.unsqueeze(1))

        preds_out = torch.cat(preds_list, dim=1)
        confs_out = torch.cat(confs_list, dim=1)
        return preds_out, confs_out

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        self._encoder.train()
        self._iterative.train()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_starts"],
            label_matrix=batch["label_matrix"],
            num_words=batch.get("num_words"),
            n_real_words=batch.get("n_real_words"),
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        opt = self.optimizers()
        self.log("lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        total_norm = torch.norm(
            torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None])
        )
        self.log("grad_norm", total_norm, on_step=True, on_epoch=False)

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        loss, all_scores = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_starts"],
            label_matrix=batch["label_matrix"],
            num_words=batch.get("num_words"),
            n_real_words=batch.get("n_real_words"),
            return_scores=True,
        )
        self.log("val_loss", loss, prog_bar=False, on_epoch=True,
                 batch_size=batch["input_ids"].shape[0])

        correct = total = 0
        tp_extr = fp_extr = fn_extr = 0
        for d, sc in enumerate(all_scores):
            gold_d = batch["label_matrix"][:, d, :]
            mask = gold_d != -100
            if not mask.any():
                continue
            pred_d = sc.detach().argmax(-1)
            correct += (pred_d[mask] == gold_d[mask]).sum().item()
            total   += mask.sum().item()

            pred_flat = pred_d[mask]
            gold_flat = gold_d[mask]
            pred_extr = pred_flat != 0
            gold_extr = gold_flat != 0
            tp_extr += (pred_extr & gold_extr & (pred_flat == gold_flat)).sum().item()
            fp_extr += (pred_extr & ~gold_extr).sum().item()
            fn_extr += (~pred_extr & gold_extr).sum().item()

        prec = tp_extr / max(tp_extr + fp_extr, 1)
        rec  = tp_extr / max(tp_extr + fn_extr, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)

        bs = batch["input_ids"].shape[0]
        self.log("val_acc",  correct / max(total, 1), prog_bar=False, on_epoch=True, batch_size=bs)
        self.log("val_prec", prec,                    prog_bar=False, on_epoch=True, batch_size=bs)
        self.log("val_rec",  rec,                     prog_bar=False, on_epoch=True, batch_size=bs)
        self.log("val_f1",   f1,                      prog_bar=True,  on_epoch=True, batch_size=bs)

        return {"val_loss": loss}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "norm"]
        params = [
            {
                "params": [p for n, p in self.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(params, lr=self.hparams.lr)
        if self.hparams.total_steps > 0:
            sched = get_linear_schedule_with_warmup(
                opt,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_steps,
            )
            return [opt], [{"scheduler": sched, "interval": "step"}]
        return opt

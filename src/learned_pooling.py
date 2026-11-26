# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Learned (attention) pooling over a history window, trained on the fraud label.

``mean`` and ``last_token`` pool with fixed weights. Attention inside the
decoder mixes the window too, but its weights were trained to predict the next
token. This head learns *which transactions of the window matter for fraud*:

1. The frozen decoder runs over the window; each transaction becomes one vector
   (the mean of its field tokens' hidden states) -> ``(window, dim)``.
2. A small scorer gives every transaction a weight (plus a learned bias by
   distance from the scored transaction, so recency can matter); softmax over
   the real transactions; weighted sum -> the embedding ``(dim,)``.
3. A linear layer on that embedding gives the fraud logit, trained with
   class-weighted cross-entropy. Only the head trains; the decoder is frozen.

The embedding then goes to XGBoost like any other pooling (``pooling="learned"``
once :func:`use_head` registered it), and the logit is saved as a score.

Leakage: rows the head was fitted on would get optimistic embeddings, so the
head is fitted on a random share of the *training customers* only
(``fraction``); notebook 05 trains XGBoost -- every arm -- on the other
customers' rows. Early stopping uses a customer-disjoint slice of the head's
own share, so val/test stay untouched.
"""

import zlib
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .decoder_inference import register_pooling, run_halving


def transaction_states(hidden: torch.Tensor, attention_mask: torch.Tensor,
                       tokens_per_txn: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(b, L, d)`` window hidden states -> ``(b, W, d)`` per transaction + ``(b, W)`` mask.

    Layout (``HistoryWindowEncoder``): ``<bos>`` then W blocks of
    ``f1 .. fT <sep|eos|pad>``, transactions left-aligned, padding on the right.
    """
    b, L, d = hidden.shape
    period = tokens_per_txn + 1
    W = (L - 1) // period
    blocks = hidden[:, 1:1 + W * period].reshape(b, W, period, d)
    valid = attention_mask[:, 1:1 + W * period].reshape(b, W, period)[:, :, 0].bool()
    return blocks[:, :, :tokens_per_txn].float().mean(dim=2), valid


class AttentionPoolingHead(nn.Module):
    def __init__(self, dim: int, window: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.config = {"dim": dim, "window": window, "hidden": hidden, "dropout": dropout}
        self.norm = nn.LayerNorm(dim)
        self.scorer = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.recency = nn.Parameter(torch.zeros(window))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)
        # Start from uniform weights -- exactly mean pooling, which already
        # carries the signal -- and let training sharpen them. From a random
        # scorer the attention can lock onto the wrong transactions before the
        # readout has learned anything (seen on a synthetic task).
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, states: torch.Tensor, valid: torch.Tensor):
        """-> (embedding (b, d), logit (b,), weights (b, W))."""
        x = self.norm(states)
        W = x.size(1)
        k = valid.sum(dim=1, keepdim=True)                       # real transactions
        dist = (k - 1 - torch.arange(W, device=x.device)).clamp(0, W - 1)
        score = self.scorer(x).squeeze(-1) + self.recency[dist]
        weights = score.masked_fill(~valid, float("-inf")).softmax(dim=1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return pooled, self.out(self.drop(pooled)).squeeze(-1), weights


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """``evaluation.safe_auc``: NaN (never an exception) on degenerate input."""
    from .evaluation import safe_auc

    return safe_auc(labels, scores)


@torch.no_grad()
def collect_states(inference, encoder, batch_size: int = 256,
                   show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Per-transaction states for every window of *encoder*: fp16 ``(n, W, d)``, bool ``(n, W)``."""
    from tqdm import tqdm

    n, W, T = len(encoder), encoder.window, encoder.tokens_per_txn
    states = np.empty((n, W, inference._embedding_dim_base), dtype=np.float16)
    valid = np.empty((n, W), dtype=bool)
    batches = encoder.iter_batches(batch_size)
    if show_progress:
        batches = tqdm(batches, total=(n + batch_size - 1) // batch_size,
                       desc="Transaction states for the head")
    def states_of(batch_ids):
        hidden, mask = inference._get_hidden_states(batch_ids)
        return transaction_states(hidden, mask, T)

    for lo, hi, ids in batches:
        s, v = run_halving(states_of, torch.from_numpy(ids).to(inference.device))
        states[lo:hi] = s.half().cpu().numpy()
        valid[lo:hi] = v.cpu().numpy()
    return states, valid


def customer_share(groups: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Deterministic ``fraction`` of distinct group keys -> boolean row mask."""
    keys = np.asarray(groups).astype(str)
    uniq = np.unique(keys)
    draw = np.array([zlib.crc32(f"{seed}|{k}".encode()) / 2**32 for k in uniq])
    chosen = set(uniq[draw < fraction])
    return np.fromiter((k in chosen for k in keys), dtype=bool, count=len(keys))


def train_head(states: np.ndarray, valid: np.ndarray, labels: np.ndarray,
               stop_mask: np.ndarray, device=None, epochs: int = 30, batch_size: int = 1024,
               lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 4,
               hidden: int = 128, dropout: float = 0.1, seed: int = 0,
               warmup_epochs: int = 3, log=print) -> Tuple[AttentionPoolingHead, Dict]:
    """Fit on rows where ``~stop_mask``, early-stop on AUC over ``stop_mask`` rows.

    For the first ``warmup_epochs`` the attention is frozen at uniform (mean
    pooling) and only the readout trains. Otherwise the attention sharpens
    while the readout is still random, follows noise and locks onto the wrong
    transactions -- on a synthetic task, 2 of 5 seeds ended *below* plain mean
    pooling that way.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = AttentionPoolingHead(states.shape[2], states.shape[1], hidden, dropout).to(device)

    fit_idx, stop_idx = np.where(~stop_mask)[0], np.where(stop_mask)[0]
    if labels[stop_idx].sum() == 0 or labels[fit_idx].sum() == 0:
        raise ValueError(
            f"Both the fit and the early-stop rows need positives (fit: "
            f"{int(labels[fit_idx].sum())}, early-stop: {int(labels[stop_idx].sum())}). "
            "Raise HEAD_FRACTION or change HEAD_SEED."
        )
    y = torch.as_tensor(labels, dtype=torch.float32)
    n_pos = float(labels[fit_idx].sum())
    pos_weight = torch.tensor((len(fit_idx) - n_pos) / max(n_pos, 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    def batch(idx):
        return (torch.as_tensor(states[idx], device=device).float(),
                torch.as_tensor(valid[idx], device=device))

    @torch.no_grad()
    def stop_auc():
        head.eval()
        out = []
        for i in range(0, len(stop_idx), batch_size):
            s, v = batch(stop_idx[i:i + batch_size])
            out.append(head(s, v)[1].float().cpu().numpy())
        head.train()
        return rank_auc(np.concatenate(out), labels[stop_idx])

    attention = list(head.scorer.parameters()) + [head.recency]

    history = {"epoch": [], "train_loss": [], "stop_auc": []}
    best, best_auc, waited = None, -1.0, 0
    for epoch in range(epochs):
        for param in attention:
            param.requires_grad_(epoch >= warmup_epochs)
        order = rng.permutation(fit_idx)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = np.sort(order[i:i + batch_size])
            s, v = batch(idx)
            loss = loss_fn(head(s, v)[1], y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        auc = stop_auc()
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(total / len(order))
        history["stop_auc"].append(auc)
        phase = "readout only" if epoch < warmup_epochs else "attention + readout"
        log(f"  head epoch {epoch + 1:>2} ({phase}): loss {total / len(order):.4f}  "
            f"early-stop AUC {auc:.4f}")
        if epoch < warmup_epochs:
            continue                  # early stopping starts once attention trains
        if auc > best_auc:
            best_auc, waited = auc, 0
            best = {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best is None:
        # No epoch beat -1: the early-stop slice had no positives (AUC is NaN)
        # or training never left the warm-up. Keep the last state, say so.
        log("  !! early stopping never scored an epoch (no positives in the early-stop "
            "slice, or epochs <= warmup_epochs): keeping the last epoch's head")
        best_auc = float("nan")
    else:
        head.load_state_dict(best)
    head.eval()
    history["best_stop_auc"] = best_auc
    return head, history


def use_head(head: AttentionPoolingHead, tokens_per_txn: int, name: str = "learned") -> None:
    """Register *head* as pooling strategy *name* (embedding = attention-pooled vector)."""
    param = next(head.parameters())

    @register_pooling(name)
    def _pool(hidden, attention_mask, ctx=None):
        s, v = transaction_states(hidden, attention_mask, tokens_per_txn)
        return head(s.to(param.dtype), v)[0]


@torch.no_grad()
def head_scores(head: AttentionPoolingHead, embeddings: np.ndarray) -> np.ndarray:
    """Fraud logits from saved ``learned`` embeddings (the head's output layer)."""
    w = head.out.weight.detach().float().cpu().numpy().ravel()
    b = float(head.out.bias.detach().float().cpu())
    return embeddings.astype(np.float32) @ w + b


def save_head(head: AttentionPoolingHead, path, extra: Optional[Dict] = None) -> None:
    torch.save({"config": head.config, "state_dict": head.state_dict(),
                "extra": extra or {}}, path)


def load_head(path, device=None) -> AttentionPoolingHead:
    blob = torch.load(path, map_location=device or "cpu")
    head = AttentionPoolingHead(**blob["config"])
    head.load_state_dict(blob["state_dict"])
    return head.to(device or "cpu").eval()

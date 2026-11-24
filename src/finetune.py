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
Fine-tuning the pretrained model on the fraud label, end to end with the head.

Learned pooling (``learned_pooling.py``) trains an attention head on top of a
*frozen* model. Fine-tuning goes one step further: part of the model itself
adapts to the label. Selected by ``finetune`` (notebook 04 / experiment
``embeddings.finetune``):

``none``      the head only -- learned pooling as before
``top:N``     the last N transformer layers + final norm
``lora:R``    rank-R low-rank adapters on the attention projections; the base
              weights stay frozen (few trainable weights: a regulariser when
              positives are few). Hand-written, no extra dependency.
``full``      every weight of the model

Procedure (``finetune_with_head``):

1. The head is first trained on the frozen model (``train_head``), the stable
   warm-up that learned pooling already uses.
2. The selected weights then train jointly with the head, model at a much
   lower learning rate than the head, early-stopped on the customer-disjoint
   slice.
3. Fine-tuning is kept only if it beats step 1 on that slice; otherwise the
   model's original weights are restored. It cannot end worse than the head
   alone on the early-stop slice.

The model is changed in place, so the embedding extraction that follows uses
the fine-tuned weights. Same leakage rule as the head: only the head's share
of training customers is used; XGBoost trains on the others.

Memory: one window batch forward + backward. A batch that runs out of memory
is retried as smaller micro-batches with gradient accumulation, which leaves
the optimisation unchanged.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .finetune_spec import parse_finetune
from .learned_pooling import AttentionPoolingHead, rank_auc, transaction_states

LORA_TARGETS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """``base(x) + (alpha / r) * B(A(x))``, B starting at zero (identity at init)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: Optional[float] = None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = (alpha or 2 * rank) / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(base.weight.device, base.weight.dtype)
        self.lora_b.to(base.weight.device, base.weight.dtype)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_b(self.lora_a(x))


def decoder_layers(backbone: nn.Module) -> nn.ModuleList:
    """The stack of transformer blocks (``layers`` in Llama/Qwen2/Mistral, ``h`` in GPT-2)."""
    n = getattr(getattr(backbone, "config", None), "num_hidden_layers", None)
    for module in backbone.children():
        if isinstance(module, nn.ModuleList) and (n is None or len(module) == n):
            return module
    raise ValueError(f"no transformer block list found in {type(backbone).__name__}")


def prepare_trainable(model: nn.Module, backbone: nn.Module, spec: str) -> List[nn.Parameter]:
    """Freeze everything, then unfreeze / add what *spec* trains. Returns those parameters."""
    mode, value = parse_finetune(spec)
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "none":
        return []
    if mode == "full":
        params = list(backbone.parameters())
    elif mode == "top":
        layers = decoder_layers(backbone)
        if value > len(layers):
            raise ValueError(f"top:{value} but the model has {len(layers)} layers")
        params = [p for layer in list(layers)[-value:] for p in layer.parameters()]
        final_norm = getattr(backbone, "norm", None) or getattr(backbone, "ln_f", None)
        if final_norm is not None:
            params += list(final_norm.parameters())
    else:  # lora
        params = []
        for layer in decoder_layers(backbone):
            for parent in layer.modules():
                for name, child in list(parent.named_children()):
                    if name in LORA_TARGETS and isinstance(child, nn.Linear):
                        wrapped = LoRALinear(child, value)
                        setattr(parent, name, wrapped)
                        params += [wrapped.lora_a.weight, wrapped.lora_b.weight]
        if not params:
            raise ValueError(f"lora: no {LORA_TARGETS} linear layers in this architecture "
                             "(GPT-2 uses a fused c_attn); use top:N or full")
    for p in params:
        p.requires_grad_(True)
    return params


def finetune_with_head(inference, encoder, labels: np.ndarray, stop_mask: np.ndarray,
                       head: AttentionPoolingHead, spec: str, head_auc: float,
                       lr: float = 1e-5, head_lr: float = 1e-4, epochs: int = 3,
                       batch_size: int = 32, patience: int = 2, seed: int = 0,
                       log=print) -> Tuple[AttentionPoolingHead, Dict]:
    """Step 2-3 above. *encoder* yields the windows of the rows *labels* refers to."""
    mode, _ = parse_finetune(spec)
    history = {"spec": spec, "head_only_auc": head_auc, "epoch": [], "train_loss": [],
               "stop_auc": [], "kept": False}
    if mode == "none":
        return head, history

    model, device = inference.model, inference.device
    backbone = inference._backbone
    original = copy.deepcopy(backbone.state_dict())       # restored if fine-tuning loses
    head_original = copy.deepcopy(head.state_dict())
    params = prepare_trainable(model, backbone, spec)
    n_train = sum(p.numel() for p in params)
    log(f"  fine-tune {spec}: {n_train / 1e6:.2f}M trainable weights "
        f"(of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)")

    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": params, "lr": lr},
                             {"params": head.parameters(), "lr": head_lr}], weight_decay=1e-4)
    T = encoder.tokens_per_txn
    fit_idx, stop_idx = np.where(~stop_mask)[0], np.where(stop_mask)[0]
    n_pos = float(labels[fit_idx].sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(fit_idx) - n_pos) / max(n_pos, 1.0), device=device))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    rng = np.random.default_rng(seed)
    micro = batch_size

    def windows(rows):
        return torch.from_numpy(encoder.select(encoder.row_index[rows]).to_array()).to(device)

    def logits_of(ids):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            hidden, mask = inference._get_hidden_states(ids)
            states, valid = transaction_states(hidden, mask, T)
            return head(states.float(), valid)[1].float()

    @torch.no_grad()
    def stop_auc():
        model.eval()
        head.eval()
        out = []
        for i in range(0, len(stop_idx), batch_size):
            out.append(logits_of(windows(stop_idx[i:i + batch_size])).cpu().numpy())
        head.train()
        return rank_auc(np.concatenate(out), labels[stop_idx])

    best_auc = head_auc if np.isfinite(head_auc) else -np.inf      # NaN would block all
    best, waited = None, 0
    y = torch.as_tensor(labels, dtype=torch.float32)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(fit_idx)
        total = 0.0
        for i in range(0, len(order), batch_size):
            rows = np.sort(order[i:i + batch_size])
            opt.zero_grad()
            while True:          # micro-batches; halved on out-of-memory
                try:
                    batch_loss = 0.0
                    for j in range(0, len(rows), micro):
                        part = rows[j:j + micro]
                        loss = loss_fn(logits_of(windows(part)), y[part].to(device))
                        (loss * len(part) / len(rows)).backward()
                        batch_loss += loss.item() * len(part)
                    total += batch_loss       # only once the whole batch went through
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro //= 2
                    log(f"  out of memory: micro-batch -> {micro} (same effective batch)")
            torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
            opt.step()
        auc = stop_auc()
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(total / len(order))
        history["stop_auc"].append(auc)
        log(f"  fine-tune epoch {epoch + 1}: loss {total / len(order):.4f}  "
            f"early-stop AUC {auc:.4f} (head only {head_auc:.4f})")
        if auc > best_auc:
            best_auc, waited = auc, 0
            best = (copy.deepcopy(backbone.state_dict()), copy.deepcopy(head.state_dict()))
        else:
            waited += 1
            if waited >= patience:
                break

    model.eval()
    head.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if best is None:
        log("  fine-tuning did not beat the head alone on the early-stop slice: "
            "original model weights restored")
        _restore(backbone, original)
        head.load_state_dict(head_original)
        history["kept"] = False
        return head, history
    backbone.load_state_dict(best[0])
    head.load_state_dict(best[1])
    history.update(kept=True, best_stop_auc=best_auc, trainable_weights=n_train)
    return head, history


def _restore(backbone: nn.Module, original: Dict[str, torch.Tensor]) -> None:
    """Original weights back; LoRA adapters (absent from *original*) zeroed."""
    state = backbone.state_dict()
    for key in state:
        if key in original:
            state[key].copy_(original[key])
        elif "lora_b" in key:
            state[key].zero_()
        else:
            # A LoRA-wrapped layer's base weight lives at '<name>.base.weight'.
            plain = key.replace(".base.", ".")
            if plain in original:
                state[key].copy_(original[plain])

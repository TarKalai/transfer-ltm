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
Model architectures, selected by name (``model.architecture`` in the YAML).

The model is trained from scratch on the transaction vocabulary, so what is
swappable is the *architecture* -- attention, normalisation, positional
encoding, feed-forward -- not pretrained text weights (their vocabulary is
words, and they are 15-20x larger). One size description is written in
Llama's terms (``model.config``: hidden_size, num_hidden_layers, ...); each
entry here translates it into its own config class, so switching architecture
keeps the model the same size.

=============  =============================================================
``llama``      RMSNorm, RoPE, SwiGLU, grouped-query attention (the default)
``qwen2``      Llama-like; bias on the Q/K/V projections
``mistral``    Llama-like; sliding-window attention (off here: windows fit)
``gpt2``       LayerNorm, learned absolute positions, GELU MLP, full multi-head
``modernbert`` encoder (sees both sides) for ``model.objective: mlm``: LayerNorm,
               RoPE, GeGLU, full multi-head attention on every layer
=============  =============================================================

The first four are decoders (``model.objective: clm``, next-token prediction);
``modernbert`` is the only encoder (``ENCODERS``).

New one: add an entry to ``ARCHITECTURES`` (config class + translation).
"""

from typing import Any, Callable, Dict, Tuple

#: Keys of the shared size description that only rotary/RMSNorm models use.
_LLAMA_ONLY = ("num_key_value_heads", "rope_theta", "rms_norm_eps", "hidden_act")


def _same(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg)


def _mistral(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["sliding_window"] = None          # every window here fits: plain attention
    return out


def _qwen2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["use_sliding_window"] = False
    return out


def _gpt2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in cfg.items() if k not in _LLAMA_ONLY}
    rename = {"hidden_size": "n_embd", "num_hidden_layers": "n_layer",
              "num_attention_heads": "n_head", "intermediate_size": "n_inner",
              "max_position_embeddings": "n_positions", "attention_dropout": "attn_pdrop"}
    for old, new in rename.items():
        if old in out:
            out[new] = out.pop(old)
    out.setdefault("resid_pdrop", 0.0)
    out.setdefault("embd_pdrop", 0.0)
    out["activation_function"] = "gelu_new"
    return out


def _modernbert(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in cfg.items() if k not in _LLAMA_ONLY}
    if "rms_norm_eps" in cfg:
        out["norm_eps"] = cfg["rms_norm_eps"]
    # Every layer attends to the whole sequence, as in the decoders. transformers
    # 5.8 refuses a ModernBERT whose layers are all "full_attention" (its RoPE
    # setup needs both layer types), so the "sliding" layers keep their type but
    # get a window twice the longest sequence -- the same as full attention --
    # and the same RoPE as the full layers.
    theta = float(cfg.get("rope_theta", 160_000.0))
    out["rope_parameters"] = {t: {"rope_type": "default", "rope_theta": theta}
                              for t in ("full_attention", "sliding_attention")}
    out["local_attention"] = 2 * int(cfg.get("max_position_embeddings", 8192))
    # Its default special ids (50281...) are past our vocabulary; the model does
    # not use them, but keep them valid.
    out.setdefault("cls_token_id", cfg.get("bos_token_id", 1))
    out.setdefault("sep_token_id", cfg.get("eos_token_id", 2))
    return out


#: name -> (config class, translation of the shared Llama-style description)
ARCHITECTURES: Dict[str, Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = {
    "llama": ("transformers.LlamaConfig", _same),
    "qwen2": ("transformers.Qwen2Config", _qwen2),
    "mistral": ("transformers.MistralConfig", _mistral),
    "gpt2": ("transformers.GPT2Config", _gpt2),
    "modernbert": ("transformers.ModernBertConfig", _modernbert),
}

#: Architectures trained with masked-token prediction (model.objective: mlm).
ENCODERS = {"modernbert"}


def architecture_config(name: str, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``("transformers.Qwen2Config", {...kwargs})`` for architecture *name*."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}")
    target, translate = ARCHITECTURES[name]
    cfg = {k: v for k, v in cfg.items() if k != "_target_"}
    return target, translate(cfg)

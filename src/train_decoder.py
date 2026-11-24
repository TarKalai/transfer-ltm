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

"""Plain-PyTorch decoder pretraining loop — replaces NeMo AutoModel.

Why this exists
---------------
NVIDIA's blueprint drove training with NeMo AutoModel's
``TrainFinetuneRecipeForNextTokenPrediction``. On a single L40S with a ~29M
parameter model, everything that recipe contributes is either trivially
replaceable or inapplicable:

===========================  =================================================
NeMo component               Replacement here
===========================  =================================================
``NeMoAutoModelForCausalLM``  ``AutoModelForCausalLM.from_config`` (already HF)
``MaskedCrossEntropy``        HF causal LMs compute it from ``labels`` with the
                              ``-100`` ignore convention, which
                              ``FinancialCLMDataset`` already emits
``StatefulDataLoader``        ``torch.utils.data.DataLoader``
``FSDP2Manager``              irrelevant at 29M params; DDP covers multi-GPU
``save_consolidated: true``   ``model.save_pretrained()`` — the same
                              ``config.json`` + safetensors layout notebook 04
                              loads with ``from_pretrained``
===========================  =================================================

Unblocking the NeMo path instead would mean moving ``torch`` (the import error
is torchao's float8 ``ScalingType`` against an older torch), and torch is
load-bearing for the cuDF/CuPy stack that notebooks 01 and 02 already run.

Config
------
The **same** ``configs/pretrain_bank_decoder.yaml`` is the source of truth.
``_target_`` keys naming NeMo classes (``distributed``, ``loss_fn``,
``dataloader``, ``dist_env``) are ignored; the plain fields
(``model.config``, ``dataset``, ``step_scheduler``, ``optimizer``,
``lr_scheduler``, ``rng``, ``checkpoint``) drive this loop. There is
deliberately no second config file: a value that lives in several places
ends up with several values.

Memory note
-----------
Loss is computed over every position: ``local_batch_size`` x ``seq_length`` x
``vocab_size`` logits, upcast to fp32 by cross-entropy. At 16 x 4096 x 8232
that is ~1.1 GB bf16 + ~2.2 GB fp32, comfortable on 48 GB. If it is tight,
drop ``local_batch_size`` to 4 — ``global_batch_size`` is held constant by
gradient accumulation, so the optimizer math does not change.
"""

import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

DEFAULT_CLIP_GRAD_NORM = 1.0

#: Written *after* ``save_pretrained()`` returns, so its presence means the
#: checkpoint finished writing. ``save_pretrained`` emits ``config.json``
#: before the weights, so a run that dies mid-serialization (a full disk, for
#: instance) leaves a directory that looks loadable but holds truncated
#: weights. Discovery keys on this marker, not on ``config.json``.
CHECKPOINT_MARKER = "training_state.json"
#: Seed of the validation subset: fixed, so every run sees the same sequences.
VAL_SUBSET_SEED = 12345


def completed_checkpoints(ckpt_dir) -> Dict[int, Path]:
    """Map ``step -> directory`` for checkpoints that finished writing.

    Directories without :data:`CHECKPOINT_MARKER` are partial writes and are
    deliberately invisible here.
    """
    found: Dict[int, Path] = {}
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return found
    for marker in ckpt_dir.glob(f"step_*/{CHECKPOINT_MARKER}"):
        try:
            found[int(marker.parent.name.split("_", 1)[1])] = marker.parent
        except (IndexError, ValueError):  # not a step_<N> directory
            continue
    return found


# ======================================================================
# Config loading
# ======================================================================


def load_config(path) -> Dict[str, Any]:
    """Read a YAML config into a plain nested dict."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to read the training config. `pip install pyyaml`."
        ) from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _coerce(text: str) -> Any:
    """Parse a command-line override value into a Python scalar."""
    low = text.lower()
    if low in ("none", "null", "~"):
        return None
    if low in ("true", "false"):
        return low == "true"
    if text.strip().startswith(("[", "{")):
        # Lists/maps (e.g. --dataset.loss_ignore_tokens '[]'); a bare string
        # would otherwise be iterated character by character downstream.
        return json.loads(text.replace("'", '"'))
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``--a.b.c value`` overrides in place, NeMo's CLI syntax.

    Accepts both ``--a.b value`` and ``--a.b=value``. Unknown keys are created
    rather than rejected, matching the forward-compatible behaviour of the
    recipe this replaces.
    """
    items: List[Tuple[str, str]] = []
    i = 0
    overrides = list(overrides)
    while i < len(overrides):
        tok = overrides[i]
        if not tok.startswith("--"):
            raise ValueError(f"Expected an option starting with '--', got {tok!r}")
        tok = tok[2:]
        if "=" in tok:
            key, value = tok.split("=", 1)
        else:
            if i + 1 >= len(overrides):
                raise ValueError(f"Option --{tok} has no value")
            key, value = tok, overrides[i + 1]
            i += 1
        items.append((key, value))
        i += 1

    for key, value in items:
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _coerce(value)
    return cfg


def _resolve_target(target: str):
    """Resolve a ``_target_`` string to a Python object.

    Supports both forms used in the YAML:
      ``transformers.LlamaConfig``            -> dotted import
      ``src/clm_data.py:build_bank_clm_dataset`` -> file path + attribute
    """
    if ".py:" in target:
        file_part, attr = target.rsplit(":", 1)
        file_path = Path(file_part)
        if not file_path.is_absolute():
            file_path = _SRC_DIR.parent / file_path
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr)

    module_name, attr = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


# ======================================================================
# Component construction
# ======================================================================


def build_model(model_section: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate a randomly initialised HF model from ``model.config``.

    ``model.objective`` picks the kind: ``clm`` (default) a causal LM (next-token
    prediction, a decoder architecture), ``mlm`` a masked LM (masked-token
    prediction, an encoder architecture: ``modernbert``).

    ``model._target_`` (NeMo's wrapper) is ignored; the architecture comes
    from ``model.config._target_``, so swapping ``transformers.LlamaConfig``
    for another HF config class in the YAML still works.

    Two memory knobs live alongside it in the ``model`` section:

    ``attn_implementation`` (default ``"sdpa"``)
        Eager attention materialises a ``(batch, heads, seq, seq)`` score
        matrix -- at 32 x 8 x 4096 x 4096 that is 8.6 GB per layer in bf16 and
        OOMs immediately. SDPA / flash never form it. Pinned explicitly rather
        than left to whatever the installed transformers defaults to.
    ``gradient_checkpointing`` (default False)
        Recompute layer activations in the backward pass instead of storing
        them: ~30% more compute for a large activation saving. At seq 4096 the
        stored activations, not the logits, are the memory ceiling, so this is
        usually the difference between local_batch 8 and 32 -- and the bigger
        batch is often faster overall despite the recompute.
    """
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

    section = dict(model_section)
    section.pop("_target_", None)
    objective = model_objective(section)
    section.pop("objective", None)
    grad_ckpt = bool(section.pop("gradient_checkpointing", False))
    attn_impl = section.pop("attn_implementation", "sdpa")

    cfg_section = dict(section["config"])
    architecture = section.pop("architecture", None)
    if architecture:
        # Named architecture (src/architectures.py): the size description in
        # model.config is translated into that architecture's config class.
        from src.architectures import architecture_config
        target, cfg_section = architecture_config(architecture, cfg_section)
    else:
        target = cfg_section.pop("_target_", "transformers.LlamaConfig")
    from src.architectures import ENCODERS
    if objective == "mlm" and architecture not in ENCODERS:
        raise ValueError(f"model.objective: mlm needs an encoder model.architecture "
                         f"({sorted(ENCODERS)}), got {architecture!r}")
    if objective == "clm" and architecture in ENCODERS:
        raise ValueError(f"model.architecture: {architecture} is an encoder: set "
                         "model.objective: mlm")
    config_cls = _resolve_target(target)
    hf_config = config_cls(**cfg_section)
    auto_cls = AutoModelForMaskedLM if objective == "mlm" else AutoModelForCausalLM

    try:
        model = auto_cls.from_config(
            hf_config, **({"attn_implementation": attn_impl} if attn_impl else {})
        )
    except (TypeError, ValueError) as exc:
        # Older transformers, or a backend this build cannot provide.
        print(f"  NOTE: attn_implementation={attn_impl!r} unavailable ({exc}); "
              "falling back to the library default.")
        model = auto_cls.from_config(hf_config)

    if grad_ckpt:
        model.config.use_cache = False   # incompatible with checkpointing
        # use_reentrant=False is the supported form and is required under DDP.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def model_objective(model_section: Optional[Dict[str, Any]]) -> str:
    """``model.objective``: "clm" (next-token, the default) or "mlm" (masked-token)."""
    objective = str((model_section or {}).get("objective") or "clm").lower()
    if objective not in ("clm", "mlm"):
        raise ValueError(f"model.objective={objective!r}; choose 'clm' or 'mlm'")
    return objective


def build_dataset(section: Optional[Dict[str, Any]], **extra):
    """Build a dataset from a YAML section, or return None if unconfigured.

    ``extra`` is passed to the builder on top of the section (the masking
    settings for ``model.objective: mlm``).
    """
    if not section:
        return None
    kwargs = dict(section)
    kwargs.update(extra)
    target = kwargs.pop("_target_", "src/clm_data.py:build_bank_clm_dataset")
    if kwargs.get("data_path") in (None, ""):
        return None
    builder = _resolve_target(target)
    return builder(**kwargs)


def batch_attention_mask(batch: Dict[str, Any], labels):
    """Attention mask for a batch.

    Taken from the batch when the dataset provides one. ``labels != -100`` is
    only a fallback for datasets that predate it: it marks the real tokens only
    while padding is the sole ignored position, and would hide any token that
    is excluded from the loss (``loss_ignore_tokens``) from attention as well.
    """
    mask = batch.get("attention_mask")
    if mask is None:
        return (labels != -100).long()
    return mask.to(labels.device, non_blocking=True)


def make_lr_lambda(warmup_steps: int, total_steps: int, base_lr: float,
                   min_lr: float, decay_style: str = "cosine") -> Callable[[int], float]:
    """Linear warmup then decay to ``min_lr`` (not to zero).

    ``LambdaLR`` multiplies ``base_lr``, so the floor is expressed as a ratio.
    """
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        if decay_style == "constant":
            return 1.0
        if decay_style == "linear":
            shape = 1.0 - progress
        else:  # cosine
            shape = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * shape

    return lr_lambda


def make_wsd_lr_lambda(warmup_steps: int, total_steps: int, base_lr: float, min_lr: float,
                       cooldown_steps: int, state: Dict[str, Any]) -> Callable[[int], float]:
    """Warmup, then constant, then a linear decay to ``min_lr`` over ``cooldown_steps``
    (warmup-stable-decay), for early stopping.

    The decay starts at ``state["start"]``, which the training loop sets once validation
    stops improving, and at the latest ``total_steps - cooldown_steps``, so a run that
    reaches ``max_steps`` also ends decayed. Unlike cosine, nothing depends on when the
    run will stop, so a run stopped early is not left at a high learning rate.
    """
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
    latest = max(warmup_steps, total_steps - cooldown_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        start = latest if state.get("start") is None else min(state["start"], latest)
        if cooldown_steps <= 0 or step < start:
            return 1.0
        progress = min(1.0, (step - start) / cooldown_steps)
        return min_ratio + (1.0 - min_ratio) * (1.0 - progress)

    return lr_lambda


# ======================================================================
# Distributed helpers
# ======================================================================


def _dist_info() -> Tuple[int, int, int]:
    """(rank, local_rank, world_size) from the torchrun environment."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def _seed_everything(seed: int, rank: int, ranked: bool) -> None:
    effective = seed + rank if ranked else seed
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


# ======================================================================
# Training
# ======================================================================


from src.clm_data import vocab_of  # noqa: E402  (vocab.json's one rule)


def _vocab_payload(vocab):
    """A parsed vocab.json: given as a path (read here) or already parsed (a dict)."""
    if isinstance(vocab, dict):
        return vocab
    if not vocab or not Path(vocab).exists():
        return None
    return json.loads(Path(vocab).read_text())


def check_vocab_fits(vocab, vocab_size: int, vocab_path=None) -> None:
    """Raise if the tokenizer's ids do not fit the model's embedding table.

    ``vocab``: vocab.json's path, or its parsed content (``{"vocab": {token:
    id}, ...}`` or a bare ``{token: id}`` map).
    """
    payload = _vocab_payload(vocab)
    if payload is None:
        return
    vocab_path = vocab_path or (vocab if not isinstance(vocab, dict) else "vocab.json")
    needed = max(int(i) for i in vocab_of(payload).values()) + 1
    if needed > vocab_size:
        raise ValueError(
            f"model.config.vocab_size={vocab_size} but {vocab_path} uses ids up to "
            f"{needed - 1}. Set --model.config.vocab_size {needed} (notebook 03 "
            "passes the corpus report's vocab_size)."
        )


def token_loss_weights(vocab, field_weights: Dict[str, float], vocab_size: int):
    """Per-token-id loss weight from ``{field prefix: weight}`` (others 1.0).

    A token's field is its prefix (``AMT_3`` -> ``AMT``, ``TXT_17`` -> ``TXT``).
    Returns None when no weights are configured, so the default path is unchanged.
    """
    if not field_weights:
        return None
    payload = _vocab_payload(vocab)
    if payload is None:
        raise ValueError(f"loss_weights needs dataset.vocab_path (got {vocab!r}) to map "
                         "fields to token ids")
    from src.token_fields import field_of_token

    # Field list and filler tokens as recorded by the tokenizer (save_vocab).
    field_list = payload.get("field_prefixes")
    if not field_list:
        raise ValueError("loss_weights needs a vocab.json that records its fields "
                         "(field_prefixes): this one predates them -- re-run notebook 02")
    fillers = set(payload.get("filler_tokens") or [])
    vocab_map = vocab_of(payload)
    weights = torch.ones(vocab_size, dtype=torch.float32)
    prefixes = {str(k).upper(): float(v) for k, v in field_weights.items()}
    found = set()
    for token, idx in vocab_map.items():
        field = field_of_token(token, field_list)
        if field is None or field.upper() not in prefixes or int(idx) >= vocab_size:
            continue
        found.add(field.upper())
        weight = prefixes[field.upper()]
        # Padding gets the field's weight but never more than 1: it is never
        # upweighted, and never outweighs the real tokens of a downweighted field.
        weights[int(idx)] = min(weight, 1.0) if token in fillers else weight
    missing = set(prefixes) - found
    if missing:
        raise ValueError(f"loss_weights names field(s) {sorted(missing)} with no token in "
                         "vocab.json")
    return weights


def weighted_lm_loss(logits, labels, weights, return_plain: bool = False):
    """Next-token cross-entropy with a weight per target token, normalised by total weight.

    Same shift and ignore rule (-100) as the HF causal-LM loss, so with all
    weights 1 it equals ``out.loss``. ``return_plain`` also returns the
    unweighted loss of the same batch -- what gets logged, so the training
    curve stays comparable with validation and with unweighted runs.
    """
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
        ignore_index=-100, reduction="none",
    )
    valid = shift_labels.reshape(-1) != -100
    w = weights.to(ce.device)[shift_labels.reshape(-1).clamp(min=0)] * valid
    plain = ce[valid].sum() / valid.sum().clamp(min=1)
    # All targets of this batch weighted 0: train on the plain loss rather than
    # silently on nothing (a loss of 0 has no gradient).
    weighted = torch.where(w.sum() > 0, (ce * w).sum() / w.sum().clamp(min=1e-8), plain)
    if return_plain:
        return weighted, plain
    return weighted


def fit_local_batch(cfg: Dict[str, Any], device, local_batch: int, global_batch: int,
                    world_size: int = 1, headroom: float = 0.90, log=print) -> int:
    """Largest ``local_batch`` (halving) whose training step fits on this GPU.

    One real forward + backward at full ``seq_length`` with a throwaway model:
    sequences are padded to that length, so this is the step's peak. A batch
    passes only if the peak stays under ``headroom`` x device memory -- the
    validation pass (fp32 per-field loss) and fragmentation need the rest.
    The global batch is unchanged: accumulation absorbs the difference.
    """
    if device.type != "cuda":
        return local_batch
    seq_length = int((cfg.get("dataset") or {}).get("seq_length", 4096))
    total = torch.cuda.get_device_properties(device).total_memory
    use_bf16 = torch.cuda.is_bf16_supported()
    b = local_batch
    while b >= 1:
        if global_batch % (b * world_size) != 0:
            b //= 2
            continue
        model = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = build_model(cfg["model"]).to(device)
            vocab = int(model.config.vocab_size)
            ids = torch.randint(5, vocab, (b, seq_length), device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if cfg.get("loss_weights"):     # the weighted path holds more logits
                    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
                    loss = weighted_lm_loss(out.logits, ids, torch.ones(vocab))
                else:
                    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 labels=ids).loss
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
            fits = peak <= headroom * total
            log(f"  Memory probe : local_batch={b} -> peak {peak / 2**30:.1f} of "
                f"{total / 2**30:.1f} GiB {'(fits)' if fits else '(over headroom)'}")
        except torch.cuda.OutOfMemoryError:
            fits = False
            log(f"  Memory probe : local_batch={b} -> out of memory")
        finally:
            del model
            torch.cuda.empty_cache()
        if fits:
            return b
        b //= 2
    raise RuntimeError(
        f"Even local_batch=1 does not fit at seq_length={seq_length}: lower "
        "dataset.seq_length or enable model.gradient_checkpointing."
    )


def train_bank_decoder(
    cfg: Dict[str, Any],
    log_every_steps: int = 10,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Pretrain the decoder with next-token prediction.

    Parameters
    ----------
    cfg
        Config dict, as returned by :func:`load_config` (plus any overrides).
    log_every_steps
        Print a training-loss line every N optimizer steps on rank 0.
    progress_cb
        Called on rank 0 after every optimizer step with a dict holding
        ``step``, ``loss``, ``lr`` and elapsed ``seconds``. Lets a notebook
        draw a live curve without parsing stdout.

    Returns
    -------
    dict
        ``steps``/``train_loss``/``lr``, ``val_steps``/``val_loss``,
        ``checkpoints`` (surviving ones only — pruning removes entries),
        ``baseline_loss`` (= ``ln(vocab_size)``, the random-init expectation),
        ``final_checkpoint``, and ``best_step``/``best_val_loss``/
        ``best_checkpoint``. **Prefer ``best_checkpoint`` for notebook 04**:
        the last step is not the best model once validation starts to flatten.
    """
    rank, local_rank, world_size = _dist_info()
    is_main = rank == 0
    distributed = world_size > 1

    if distributed:
        torch.distributed.init_process_group(
            backend=cfg.get("dist_env", {}).get("backend", "nccl")
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng_cfg = cfg.get("rng", {}) or {}
    _seed_everything(int(rng_cfg.get("seed", 42)), rank, bool(rng_cfg.get("ranked", True)))

    sched = cfg.get("step_scheduler", {}) or {}
    global_batch = int(sched.get("global_batch_size", 16))
    local_batch = int(sched.get("local_batch_size", global_batch))
    max_steps = sched.get("max_steps")
    num_epochs = int(sched.get("num_epochs", 1))
    val_every = int(sched.get("val_every_steps", 0) or 0)
    ckpt_every = int(sched.get("ckpt_every_steps", 0) or 0)
    val_batches = sched.get("val_batches")  # optional cap on the val pass
    # Early stopping (off when patience is 0): see step_scheduler in the config.
    es_patience = int(sched.get("early_stop_patience", 0) or 0)
    es_min_delta = float(sched.get("early_stop_min_delta", 0.0) or 0.0)
    es_cooldown = int(sched.get("early_stop_cooldown_steps", 0) or 0)
    if es_patience and not val_every:
        raise ValueError("step_scheduler.early_stop_patience needs val_every_steps > 0 "
                         "(it counts validations without improvement)")

    # -- Objective ------------------------------------------------------
    # mlm: masked-token prediction. Its <mask> token is not in the tokenizer: it
    # gets the first id after the vocabulary, and the model one more row.
    objective = model_objective(cfg.get("model"))
    mlm_cfg = dict(cfg.get("mlm") or {})
    data_extra: Dict[str, Any] = {}
    val_extra: Dict[str, Any] = {}
    mask_token_id = None
    if objective == "mlm":
        if cfg.get("loss_weights"):
            raise ValueError("loss_weights is for next-token prediction only (objective clm)")
        from src.clm_data import VocabTokenizer
        vocab_path_ = (cfg.get("dataset") or {}).get("vocab_path")
        payload_ = _vocab_payload(vocab_path_)
        if payload_ is None:
            raise ValueError(f"model.objective: mlm needs dataset.vocab_path ({vocab_path_})")
        mask_token_id = VocabTokenizer(vocab_of(payload_)).vocab_size
        mcfg = cfg["model"]["config"]
        mcfg["vocab_size"] = max(int(mcfg.get("vocab_size", 0) or 0), mask_token_id + 1)
        data_extra = {"objective": "mlm", "mask_token_id": mask_token_id,
                      "mask_prob": float(mlm_cfg.get("mask_prob", 0.15)),
                      "mask_unit": str(mlm_cfg.get("mask_unit", "token"))}
        # Validation: the same masks every time (and the same for every run).
        val_extra = dict(data_extra, mask_seed=int(mlm_cfg.get("val_mask_seed", 1234)))

    # Shrink the per-step batch until a training step fits this GPU (global
    # batch and optimizer maths unchanged). step_scheduler.auto_fit_batch: false
    # turns it off.
    configured_local_batch = local_batch      # before any memory fitting (val subset size)
    if sched.get("auto_fit_batch", True):
        fitted = fit_local_batch(cfg, device, local_batch, global_batch, world_size,
                                 log=print if rank == 0 else (lambda *a: None))
        if fitted != local_batch:
            if rank == 0:
                print(f"  local_batch_size {local_batch} -> {fitted} to fit GPU memory; "
                      f"global batch stays {global_batch} (more accumulation)")
            local_batch = fitted

    per_step = local_batch * world_size
    if global_batch % per_step != 0:
        raise ValueError(
            f"global_batch_size={global_batch} is not divisible by "
            f"local_batch_size={local_batch} x world_size={world_size}."
        )
    accum_steps = global_batch // per_step

    # -- Data ----------------------------------------------------------
    train_ds = build_dataset(cfg.get("dataset"), **data_extra)
    if train_ds is None:
        raise ValueError("dataset.data_path is not set — nothing to train on.")
    val_ds = build_dataset(cfg.get("validation_dataset"), **val_extra)

    # -- Validation subsample: fixed *and* random ----------------------
    # The corpus is written by groupby(group_cols + ["_chunk_id"]), which sorts,
    # so lines are ordered by customer and then chronologically. Capping by
    # "first N batches" of an unshuffled loader would therefore evaluate only
    # the lowest-sorted customers' earliest history -- a systematic slice, not a
    # sample, and biased by whatever customer_id encodes (branch, segment,
    # onboarding cohort).
    #
    # Instead draw a seeded random subset once, up front:
    #   * representative  -- uniform over all validation sequences
    #   * reproducible    -- same subset at every validation, so the curve is
    #                        comparable across steps and best-checkpoint
    #                        selection stays meaningful
    #   * prefix-safe     -- kept in permuted order (NOT sorted), so the first
    #                        N batches, which is what the per-field diagnostic
    #                        consumes, are also a uniform sample
    # All ranks draw the same subset (the seed here is deliberately not
    # rank-offset) because the per-rank losses are averaged.
    val_total = len(val_ds) if val_ds is not None else 0
    val_sampled = val_total
    if val_ds is not None and val_batches:
        # Sized by the *configured* local batch and drawn with a fixed seed --
        # not rng.seed, not the memory-fitted batch -- so runs that differ in
        # seed or model size are validated on the very same sequences.
        cap = int(val_batches) * configured_local_batch * max(1, world_size)
        if cap < len(val_ds):
            picks = np.random.default_rng(VAL_SUBSET_SEED).permutation(len(val_ds))[:cap]
            val_ds = torch.utils.data.Subset(val_ds, picks.tolist())
            val_sampled = len(val_ds)

    loader_workers = int((cfg.get("dataloader") or {}).get("num_workers", 0))

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=local_batch,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=local_batch,
            shuffle=False,
            drop_last=False,
            num_workers=loader_workers,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None
        else None
    )

    # -- Model ---------------------------------------------------------
    model = build_model(cfg["model"]).to(device)
    vocab_size = int(model.config.vocab_size)
    if mask_token_id is not None:
        model.config.mask_token_id = mask_token_id   # saved in config.json
    # The YAML's vocab_size is only a default; the tokenizer decides the real
    # size (tail labels, counterparty head, text slots all add tokens). A token
    # id past the embedding table is a CUDA assert mid-batch -- fail here instead.
    vocab_path = (cfg.get("dataset") or {}).get("vocab_path")
    vocab_payload = _vocab_payload(vocab_path)           # read once, used twice
    check_vocab_fits(vocab_payload, vocab_size, vocab_path)
    # Optional per-field weights on the training loss (validation stays unweighted,
    # so val loss remains comparable across runs).
    token_weights = token_loss_weights(
        vocab_payload if vocab_payload is not None else vocab_path,
        cfg.get("loss_weights") or {}, vocab_size)
    if token_weights is not None:
        token_weights = token_weights.to(device)
        if rank == 0:
            print(f"  Loss weights : {cfg.get('loss_weights')} (training loss only; the "
                  "logged train loss is the unweighted one)")
    n_params = sum(p.numel() for p in model.parameters())

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    core = model.module if distributed else model

    # -- Optimizer / schedule ------------------------------------------
    opt_cfg = dict(cfg.get("optimizer", {}) or {})
    opt_cfg.pop("_target_", None)
    clip_grad_norm = float(opt_cfg.pop("clip_grad_norm", DEFAULT_CLIP_GRAD_NORM))
    base_lr = float(opt_cfg.pop("lr", 2.0e-4))
    betas = tuple(opt_cfg.pop("betas", (0.9, 0.95)))
    weight_decay = float(opt_cfg.pop("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, betas=betas, weight_decay=weight_decay, **opt_cfg
    )

    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total_steps = int(max_steps) if max_steps else steps_per_epoch * num_epochs

    lrs_cfg = cfg.get("lr_scheduler", {}) or {}
    # Early stopping: the loop sets cooldown["start"] when validation stops improving,
    # and stops cooldown_steps later (or at total_steps).
    cooldown: Dict[str, Any] = {"start": None}
    if es_patience:
        lr_lambda = make_wsd_lr_lambda(
            warmup_steps=int(lrs_cfg.get("lr_warmup_steps", 0) or 0),
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=float(lrs_cfg.get("min_lr", 0.0) or 0.0),
            cooldown_steps=es_cooldown,
            state=cooldown,
        )
    else:
        lr_lambda = make_lr_lambda(
            warmup_steps=int(lrs_cfg.get("lr_warmup_steps", 0) or 0),
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=float(lrs_cfg.get("min_lr", 0.0) or 0.0),
            decay_style=str(lrs_cfg.get("lr_decay_style", "cosine")),
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -- Precision -----------------------------------------------------
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_kwargs = dict(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)

    # -- Diagnostics ---------------------------------------------------
    # Per-field-position validation loss. The corpus is rigidly periodic --
    # <bos> then [T field tokens, <sep>] repeated -- so bucketing the per-token
    # loss by (position - offset) % period reports one number per tokenizer
    # step. That separates "learned the schema" (low-entropy fields, <sep>,
    # which fall fast and mean little) from "learned behaviour" (counterparty,
    # amount, time gap), which a single aggregate curve cannot show.
    diag_cfg = cfg.get("diagnostics", {}) or {}
    field_period = int(diag_cfg.get("field_loss_period") or 0)
    field_offset = int(diag_cfg.get("field_loss_offset", 1) or 0)
    field_batches = int(diag_cfg.get("field_loss_batches", 20) or 0)
    field_row_chunk = max(1, int(diag_cfg.get("field_loss_row_chunk", 4) or 4))
    field_names = diag_cfg.get("field_names") or None
    want_field_loss = field_period > 1

    # -- Checkpointing -------------------------------------------------
    ckpt_cfg = cfg.get("checkpoint", {}) or {}
    ckpt_enabled = bool(ckpt_cfg.get("enabled", True))
    ckpt_dir = Path(ckpt_cfg.get("checkpoint_dir", "models/bank-decoder/checkpoints"))
    keep_last_n = ckpt_cfg.get("keep_last_n")
    keep_last_n = int(keep_last_n) if keep_last_n else None
    save_best = bool(ckpt_cfg.get("save_best", True))
    if is_main and ckpt_enabled:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Checkpoints of an earlier run in the same directory would be pruned by
        # step number together with this run's, and a step_N/ overwritten here
        # would keep the old run's completion marker during the write.
        stale = sorted(ckpt_dir.glob("step_*"))
        if stale and not ckpt_cfg.get("overwrite", False):
            raise FileExistsError(
                f"{ckpt_dir} already holds {len(stale)} checkpoint(s) from an earlier run "
                f"({stale[0].name} ...). Use a new checkpoint_dir, delete them, or set "
                "checkpoint.overwrite: true to delete them automatically."
            )
        for old in stale:
            shutil.rmtree(old)

    baseline_loss = math.log(vocab_size)
    if is_main:
        print("=" * 62)
        print("Bank Transfer Decoder Pretraining  (plain PyTorch)")
        print("=" * 62)
        if objective == "mlm":
            print(f"  Objective    : masked-token prediction (mlm): "
                  f"{data_extra['mask_prob']:.0%} of {data_extra['mask_unit']}s hidden "
                  f"per sequence, <mask> = id {mask_token_id}")
        else:
            print("  Objective    : next-token prediction (clm)")
        print(f"  Architecture : {core.config.model_type} "
              f"({core.config.num_hidden_layers}L / {core.config.hidden_size}d)")
        print(f"  Parameters   : {n_params / 1e6:.1f}M")
        print(f"  Vocab size   : {vocab_size}  (random-init loss = ln V = {baseline_loss:.2f})")
        print(f"  Device       : {device}  bf16={use_bf16}  world_size={world_size}")
        print(f"  Memory       : attn={getattr(core.config, '_attn_implementation', '?')}"
              f"  grad_ckpt={bool(getattr(core, 'is_gradient_checkpointing', False))}")
        print(f"  Batch        : {global_batch} global = {local_batch} local "
              f"x {accum_steps} accum x {world_size} rank(s)")
        print(f"  Steps        : {total_steps}  "
              f"(val every {val_every or '-'}, ckpt every {ckpt_every or '-'})")
        if es_patience:
            print(f"  Early stop   : after {es_patience} validations without an improvement "
                  f"> {es_min_delta:g}, then {es_cooldown} decay steps "
                  f"({total_steps} = ceiling; lr constant after warmup, "
                  "lr_decay_style ignored)")
        if val_total:
            how = ("all" if val_sampled >= val_total
                   else f"seeded random sample of {val_total:,}")
            print(f"  Validation   : {val_sampled:,} sequences ({how})")
        print(f"  Checkpoints  : {ckpt_dir}  "
              f"(keep_last_n={keep_last_n or 'all'}, save_best={save_best})")
        print("=" * 62 + "\n")

    history: Dict[str, Any] = {
        "steps": [], "train_loss": [], "lr": [],
        "val_steps": [], "val_loss": [],
        "checkpoints": [], "baseline_loss": baseline_loss,
        "final_checkpoint": None,
        "best_step": None, "best_val_loss": None, "best_checkpoint": None,
        "field_loss": None, "field_loss_counts": None, "field_loss_step": None,
        "field_names": list(field_names) if field_names else None,
        "field_period": field_period if want_field_loss else None,
        "early_stop_step": None,   # step where validation stopped improving (decay starts)
        "objective": objective,
        "mask_unit": data_extra.get("mask_unit"),
    }

    # ``step`` / ``val_loss`` are the true best seen; ``ckpt_step`` is the step
    # whose checkpoint is actually on disk. They diverge only if a best-model
    # save failed (full disk), which is warned about rather than fatal.
    best: Dict[str, Any] = {"step": None, "val_loss": None,
                            "ckpt_step": None, "checkpoint": None}
    saved_steps: set = set()

    def accumulate_field_loss(logits, labels, sums, counts) -> None:
        """Bucket per-token cross-entropy by field position within a transaction.

        Causal shift: ``logits[:, k]`` predicts ``labels[:, k + 1]``, so the
        target at sequence position ``p = k + 1`` belongs to field
        ``(p - offset) % period`` (mlm: no shift, ``logits[:, p]`` predicts
        ``labels[:, p]``, only at the hidden positions). With ``offset=1`` (position 0 is ``<bos>``)
        and ``period = tokens_per_transaction + 1``, index ``j`` is the j-th
        tokenizer step and the last index is ``<sep>`` (and ``<eos>``, which
        lands in the same slot). Padding carries ``-100`` and is excluded.

        Rows are processed in chunks because the fp32 upcast of a
        ``(rows, seq, vocab)`` logit tensor is the memory peak here.
        """
        seq_len = labels.size(1)
        shift = 0 if objective == "mlm" else 1
        pos = torch.arange(shift, seq_len, device=labels.device)
        field = (pos - field_offset) % field_period          # (seq_len - shift,)

        for i in range(0, labels.size(0), field_row_chunk):
            lg = logits[i:i + field_row_chunk, :seq_len - shift, :].float()
            tgt = labels[i:i + field_row_chunk, shift:]
            ce = torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.size(-1)),
                tgt.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(tgt.shape)
            valid = tgt != -100
            if not bool(valid.any()):
                continue
            idx = field.unsqueeze(0).expand_as(tgt)[valid]
            sums.index_add_(0, idx, ce[valid].double())
            counts.index_add_(
                0, idx, torch.ones(idx.numel(), device=idx.device, dtype=torch.float64)
            )

    def run_validation(step: int) -> Optional[float]:
        if val_loader is None:
            return None
        model.eval()
        total, count = 0.0, 0
        f_sums = f_counts = None
        if want_field_loss:
            f_sums = torch.zeros(field_period, device=device, dtype=torch.float64)
            f_counts = torch.zeros(field_period, device=device, dtype=torch.float64)

        with torch.no_grad():
            # No break on val_batches: the loader already covers exactly the
            # sampled subset built at setup.
            for i, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(**autocast_kwargs):
                    out = model(input_ids=input_ids,
                                attention_mask=batch_attention_mask(batch, labels),
                                labels=labels)
                total += out.loss.detach().float().item()
                count += 1
                # Capped separately from val_batches: the fp32 per-token loss is
                # the expensive part, and a couple of hundred sequences already
                # give a stable per-field mean.
                if want_field_loss and (not field_batches or i < field_batches):
                    accumulate_field_loss(out.logits, labels, f_sums, f_counts)

        model.train()
        if count == 0:
            return None
        mean = total / count
        if distributed:
            t = torch.tensor([mean], device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            mean = t.item()

        if want_field_loss and f_counts is not None:
            if distributed:
                torch.distributed.all_reduce(f_sums)
                torch.distributed.all_reduce(f_counts)
            history["field_loss"] = (f_sums / f_counts.clamp(min=1)).tolist()
            history["field_loss_counts"] = f_counts.tolist()
            history["field_loss_step"] = step
        return mean

    def save_checkpoint(step: int, reason: str = "interval") -> Optional[str]:
        """Write ``step_<N>/`` and return its path, or None if it did not write.

        A failed write is **not** fatal: a full disk at step 3500 should not
        destroy the preceding hours of training. The partial directory is
        removed (which is also how the space is reclaimed) and training goes on.
        """
        if not (is_main and ckpt_enabled):
            return None
        if step in saved_steps:
            return str(ckpt_dir / f"step_{step}")

        out_dir = ckpt_dir / f"step_{step}"
        try:
            core.save_pretrained(out_dir, safe_serialization=True)
            # Marker last: see CHECKPOINT_MARKER.
            (out_dir / CHECKPOINT_MARKER).write_text(
                json.dumps(
                    {
                        "step": step,
                        "reason": reason,
                        "vocab_size": vocab_size,
                        "global_batch_size": global_batch,
                        "objective": objective,
                        "val_loss": best["val_loss"] if reason == "best" else None,
                    },
                    indent=2,
                )
            )
        except OSError as exc:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(
                f"  WARNING: could not write checkpoint at step {step}: {exc}\n"
                f"           Removed the partial directory; training continues. "
                f"Free space under {ckpt_dir}, or set "
                f"checkpoint.keep_last_n to bound how many are retained."
            )
            return None

        saved_steps.add(step)
        path = str(out_dir)
        if path not in history["checkpoints"]:
            history["checkpoints"].append(path)
        return path

    def prune_checkpoints(protect: Iterable[int] = ()) -> None:
        """Keep the ``keep_last_n`` newest checkpoints plus ``protect``; delete the rest.

        ``protect`` is the current step and the best-validation checkpoint, so a
        long run can never discard the model you actually want. Retaining the
        *newest* N rather than deleting the oldest N matters: an old protected
        checkpoint must not push a recent one out of the budget. Total on disk
        is therefore at most ``keep_last_n + len(protect)``.

        Partial directories are invisible to ``completed_checkpoints`` and are
        left alone; ``save_checkpoint`` removes its own on failure.
        """
        if not (is_main and ckpt_enabled and keep_last_n):
            return
        completed = completed_checkpoints(ckpt_dir)
        keep = set(sorted(completed)[-keep_last_n:])
        keep |= {s for s in protect if s is not None}
        for step_n in sorted(completed):
            if step_n in keep:
                continue
            shutil.rmtree(completed[step_n], ignore_errors=True)
            saved_steps.discard(step_n)
            gone = str(completed[step_n])
            if gone in history["checkpoints"]:
                history["checkpoints"].remove(gone)
            print(f"  pruned {gone}  (keep_last_n={keep_last_n})")

    def record_validation(step: int, vloss: float) -> None:
        """Append the val loss, track the best, and checkpoint on improvement."""
        history["val_steps"].append(step)
        history["val_loss"].append(vloss)
        improved = best["val_loss"] is None or vloss < best["val_loss"]
        if improved:
            best["val_loss"] = vloss
            best["step"] = step
        if is_main:
            marker = "  <- best" if improved else ""
            print(f"  step {step:>5}/{total_steps}  VAL loss {vloss:.4f}  "
                  f"(ppl {math.exp(min(vloss, 20)):.1f}){marker}")
        if improved and save_best:
            path = save_checkpoint(step, reason="best")
            if path:
                best["ckpt_step"] = step
                best["checkpoint"] = path
        # Early stopping. vloss is averaged over ranks, so every rank decides the same.
        if es_patience and cooldown["start"] is None:
            if es["val_loss"] is None or vloss < es["val_loss"] - es_min_delta:
                es["val_loss"], es["bad"] = vloss, 0
            else:
                es["bad"] += 1
                if es["bad"] >= es_patience:
                    cooldown["start"] = step
                    history["early_stop_step"] = step
                    if is_main:
                        print(f"  step {step:>5}: no improvement > {es_min_delta:g} in "
                              f"{es_patience} validations -> lr decays over {es_cooldown} "
                              "steps, then stop")

    es: Dict[str, Any] = {"val_loss": None, "bad": 0}   # early-stopping state

    # -- Loop ----------------------------------------------------------
    model.train()
    step = 0
    micro = 0
    running = 0.0
    t0 = time.time()
    stop = False

    for epoch in range(num_epochs):
        if stop:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(**autocast_kwargs):
                if token_weights is None:
                    out = model(input_ids=input_ids,
                                attention_mask=batch_attention_mask(batch, labels),
                                labels=labels)
                    step_loss = reported = out.loss
                else:
                    # Field-weighted loss (loss_weights): trains on the weighted
                    # loss, reports the plain one (comparable with validation).
                    out = model(input_ids=input_ids,
                                attention_mask=batch_attention_mask(batch, labels))
                    step_loss, reported = weighted_lm_loss(
                        out.logits, labels, token_weights, return_plain=True)
                loss = step_loss / accum_steps

            loss.backward()
            running += reported.detach().float().item()
            micro += 1

            if micro % accum_steps != 0:
                continue

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            train_loss = running / accum_steps
            running = 0.0
            lr_now = optimizer.param_groups[0]["lr"]

            history["steps"].append(step)
            history["train_loss"].append(train_loss)
            history["lr"].append(lr_now)

            if is_main:
                if progress_cb is not None:
                    progress_cb({"step": step, "total_steps": total_steps,
                                 "loss": train_loss, "lr": lr_now,
                                 "seconds": time.time() - t0})
                if step % log_every_steps == 0 or step == 1 or step == total_steps:
                    print(f"  step {step:>5}/{total_steps}  loss {train_loss:.4f}  "
                          f"lr {lr_now:.2e}  {time.time() - t0:.0f}s")

            if val_every and step % val_every == 0:
                vloss = run_validation(step)
                if vloss is not None:
                    record_validation(step, vloss)

            if ckpt_every and step % ckpt_every == 0:
                path = save_checkpoint(step)
                if path and is_main:
                    print(f"  saved {path}")

            # Only after something was actually written this step, so the
            # directory is not globbed on every optimizer step.
            if step in saved_steps:
                prune_checkpoints(protect=(step, best["ckpt_step"]))

            if step >= total_steps or (cooldown["start"] is not None
                                       and step >= cooldown["start"] + es_cooldown):
                stop = True
                break

    # -- Final validation + checkpoint ---------------------------------
    if val_loader is not None and (not history["val_steps"] or history["val_steps"][-1] != step):
        vloss = run_validation(step)
        if vloss is not None:
            record_validation(step, vloss)

    # Always keep the final step, whatever the interval schedule did.
    final_path = save_checkpoint(step, reason="final")
    if final_path and is_main:
        print(f"  saved {final_path}")
    history["final_checkpoint"] = final_path or (
        history["checkpoints"][-1] if history["checkpoints"] else None
    )

    # The final step is protected here, so keep_last_n never discards it.
    prune_checkpoints(protect=(step, best["ckpt_step"]))

    history["best_step"] = best["step"]
    history["best_val_loss"] = best["val_loss"]
    history["best_checkpoint"] = best["checkpoint"]

    if is_main:
        print(f"\nTraining complete: {step} steps in {time.time() - t0:.0f}s")
        if es_patience and history["early_stop_step"]:
            print(f"  early stop: validation stopped improving at step {history['early_stop_step']}")
        elif es_patience:
            print("  early stop: not triggered, max_steps reached (still improving: raise it)")
        if history["train_loss"]:
            print(f"  loss {history['train_loss'][0]:.3f} -> "
                  f"{history['train_loss'][-1]:.3f}  (baseline ln V = {baseline_loss:.2f})")
        if best["step"] is not None:
            print(f"  best val loss {best['val_loss']:.4f} at step {best['step']}")
            if best["ckpt_step"] != best["step"]:
                print("  NOTE: the best step has no checkpoint on disk (its save "
                      f"failed); best available is step {best['ckpt_step']}.")
            print(f"  use for NB04 -> {best['checkpoint'] or final_path}")

        if history["field_loss"]:
            print(f"\n  Validation loss by field position "
                  f"(step {history['field_loss_step']}, period {field_period}):")
            names = list(field_names) if field_names else []
            for j, value in enumerate(history["field_loss"]):
                if j < len(names):
                    label = names[j]
                elif j == field_period - 1:
                    label = "<sep>/<eos>"
                else:
                    label = f"field {j}"
                n = int(history["field_loss_counts"][j])
                if n == 0:
                    print(f"    {label:<24}    --    (excluded from the loss)")
                    continue
                print(f"    {label:<24} {value:6.3f}   ({n:,} targets)")
            print("    (a field far below ln(its own vocab) is one the model "
                  "actually learned; see notebook 03)")

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return history


def train_from_config(config_path, overrides: Sequence[str] = (), **kwargs) -> Dict[str, Any]:
    """Convenience wrapper: load YAML, apply ``--a.b c`` overrides, train."""
    cfg = apply_overrides(load_config(config_path), overrides)
    return train_bank_decoder(cfg, **kwargs)
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
Decoder-Only Model Inference Wrapper

Embedding extraction from HuggingFace causal LM models (GPT-2, Llama, Mistral, etc.).

Embedding strategies (``pooling=``), registered in ``POOLINGS``:
- ``last_token``: the hidden state at the last non-padding position (``<eos>``).
  Standard for decoder-only embedding models (NV-Embed-v2, E5-Mistral); under
  next-token training it is a *prediction* of what comes next.
- ``mean``: average over all non-padding positions -- a profile of the window.
- ``last_txn``: average over the field tokens of the last (current) transaction,
  each of which has attended to the whole history before it. Needs
  ``tokens_per_txn``.
- ``learned``: attention pooling trained on the label (src/learned_pooling.py);
  registered by ``use_head()`` once the head is trained.
- ``surprise``: not a pooled hidden state but an anomaly profile -- how
  unexpected the scored transaction is to the model given its history (its
  negative log-likelihood, overall and per field) against how unexpected the
  window's earlier transactions were. ``tokens_per_txn + 4`` values.
- Combinations with ``+``, e.g. ``"mean+surprise"``: the parts are concatenated
  (embedding_dim is the sum of the parts' sizes).

New strategies: decorate ``fn(hidden, mask, ctx) -> (batch, dim)`` with
``@register_pooling("name")`` (pass ``dim=`` if the output is not hidden-sized).
``ctx`` is a :class:`PoolingContext` (tokens per transaction, input ids, LM head).
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PoolingContext:
    """What a pooling strategy may need beyond the hidden states."""
    tokens_per_txn: Optional[int] = None
    input_ids: Optional[torch.Tensor] = None
    lm_head: Optional[Callable] = None        # hidden -> vocabulary logits


def _ctx(ctx: Any) -> PoolingContext:
    """Accept a PoolingContext or, for brevity, just tokens_per_txn."""
    return ctx if isinstance(ctx, PoolingContext) else PoolingContext(tokens_per_txn=ctx)


#: name -> fn(hidden (b, L, d), attention_mask (b, L), ctx) -> (b, dim)
POOLINGS: Dict[str, Callable] = {}
#: name -> fn(hidden_size, tokens_per_txn) -> dim; absent means hidden_size.
POOLING_DIMS: Dict[str, Callable[[int, Optional[int]], int]] = {}


def register_pooling(name: str, dim: Optional[Callable[[int, Optional[int]], int]] = None):
    def deco(fn):
        POOLINGS[name] = fn
        if dim is not None:
            POOLING_DIMS[name] = dim
        return fn
    return deco


def pooling_dims(spec: str, hidden_size: int, tokens_per_txn: Optional[int]) -> List[int]:
    """Output size of each part of *spec*, in order."""
    return [POOLING_DIMS.get(p, lambda h, t: h)(hidden_size, tokens_per_txn)
            for p in pooling_parts(spec)]


def pooling_slice(spec: str, part: str, hidden_size: int,
                  tokens_per_txn: Optional[int]) -> slice:
    """Columns of *part* inside a concatenated *spec* embedding."""
    parts = pooling_parts(spec)
    dims = pooling_dims(spec, hidden_size, tokens_per_txn)
    start = sum(dims[:parts.index(part)])
    return slice(start, start + dims[parts.index(part)])


#: Strategies registered at run time (after training), so accepted by name
#: before they exist.
DEFERRED_POOLINGS = {"learned"}


def pooling_parts(spec: str) -> List[str]:
    """``"mean+last_txn"`` -> ["mean", "last_txn"], validated."""
    parts = [p.strip() for p in str(spec).split("+") if p.strip()]
    unknown = [p for p in parts if p not in POOLINGS and p not in DEFERRED_POOLINGS]
    if not parts or unknown:
        raise ValueError(f"Unknown pooling {spec!r}; choose from "
                         f"{sorted(set(POOLINGS) | DEFERRED_POOLINGS)} or combine them with '+'")
    return parts


def resolve_pooling(spec: str) -> List[Callable]:
    """``"mean+last_txn"`` -> [mean, last_txn]."""
    parts = pooling_parts(spec)
    deferred = [p for p in parts if p not in POOLINGS]   # validated, not yet registered
    if deferred:
        raise ValueError(f"Pooling {deferred} is not registered yet: train the head and "
                         "call src.learned_pooling.use_head() first")
    return [POOLINGS[p] for p in parts]


def run_halving(fn: Callable, ids: torch.Tensor):
    """``fn(ids)``, splitting the batch in halves (recursively) on CUDA OOM.

    Outputs are concatenated along dim 0; ``fn`` may return a tensor or a
    tuple of tensors. Only a single row that does not fit raises.
    """
    try:
        return fn(ids)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(ids) <= 1:
            raise
        mid = len(ids) // 2
        a, b = run_halving(fn, ids[:mid]), run_halving(fn, ids[mid:])
        if isinstance(a, tuple):
            return tuple(torch.cat([x, y]) for x, y in zip(a, b))
        return torch.cat([a, b])


def _last_index(attention_mask):
    """Position of the last non-padding token (right padding)."""
    return (attention_mask.sum(dim=1) - 1).clamp(min=0)


@register_pooling("last_token")
def _pool_last_token(hidden, attention_mask, ctx=None):
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, _last_index(attention_mask), :]


@register_pooling("mean")
def _pool_mean(hidden, attention_mask, ctx=None):
    mask = attention_mask.unsqueeze(-1).float()  # sums in fp32 even for bf16 hidden
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@register_pooling("last_txn")
def _pool_last_txn(hidden, attention_mask, ctx=None):
    tokens_per_txn = _ctx(ctx).tokens_per_txn
    if not tokens_per_txn:
        raise ValueError("pooling 'last_txn' needs tokens_per_txn")
    # Layout: ... <sep> f1 .. fT <eos> <pad>...; the fields sit just before <eos>.
    end = _last_index(attention_mask)
    offsets = torch.arange(tokens_per_txn, device=hidden.device) - tokens_per_txn
    idx = (end[:, None] + offsets[None, :]).clamp(min=0)            # (b, T)
    picked = hidden.gather(1, idx[:, :, None].expand(-1, -1, hidden.size(-1)))
    return picked.mean(dim=1)


@register_pooling("surprise", dim=lambda hidden_size, tokens_per_txn: tokens_per_txn + 4)
def _pool_surprise(hidden, attention_mask, ctx=None, row_chunk: Optional[int] = None,
                   budget_bytes: int = 1 << 30):
    """Anomaly profile of the scored (last) transaction given its history.

    For every field token, the model's negative log-likelihood of the actual
    token given everything before it (what pretraining optimised). Returns:

    ``[0]``        mean NLL of the scored transaction's fields
    ``[1]``        mean NLL of the window's earlier transactions (= [0] if none)
    ``[2]``        [0] - [1]: unusual *for this history*, not just unusual
    ``[3]``        history depth: earlier transactions / window size
    ``[4:4+T]``    NLL of each field of the scored transaction (which part surprised)
    """
    ctx = _ctx(ctx)
    T = ctx.tokens_per_txn
    if not T or ctx.input_ids is None or ctx.lm_head is None:
        raise ValueError("pooling 'surprise' needs tokens_per_txn, input_ids and lm_head")
    b, L, _ = hidden.shape
    ids = ctx.input_ids.to(hidden.device)

    # nll[:, p] = -log P(ids[p] | ids[<p]), from the logits at p-1. Vocabulary
    # logits are large (rows x L x V, fp32, plus log_softmax's copy), so they
    # are built a few rows at a time, sized to ``budget_bytes``.
    if row_chunk is None:
        vocab = int(getattr(ctx.lm_head, "out_features", 0)
                    or ctx.lm_head(hidden[:1, :1]).shape[-1])
        row_chunk = max(1, int(budget_bytes // (L * vocab * 4 * 3)))
    nll = torch.zeros(b, L, device=hidden.device, dtype=torch.float32)
    for lo in range(0, b, row_chunk):
        hi = min(lo + row_chunk, b)
        logp = torch.log_softmax(ctx.lm_head(hidden[lo:hi, :-1]).float(), dim=-1)
        nll[lo:hi, 1:] = -logp.gather(-1, ids[lo:hi, 1:, None]).squeeze(-1)

    period = T + 1
    W = (L - 1) // period
    fields = nll[:, 1:1 + W * period].reshape(b, W, period)[:, :, :T]        # (b, W, T)
    valid = attention_mask[:, 1:1 + W * period].reshape(b, W, period)[:, :, 0].bool()
    per_txn = fields.mean(dim=2)                                               # (b, W)
    k = valid.sum(dim=1)                                                       # real txns
    rows = torch.arange(b, device=hidden.device)
    last = (k - 1).clamp(min=0)
    last_fields = fields[rows, last]                                           # (b, T)
    last_mean = per_txn[rows, last]
    earlier = (per_txn * valid).sum(dim=1) - last_mean
    n_earlier = (k - 1).clamp(min=0)
    hist_mean = torch.where(n_earlier > 0, earlier / n_earlier.clamp(min=1), last_mean)
    return torch.cat([
        last_mean[:, None], hist_mean[:, None], (last_mean - hist_mean)[:, None],
        (n_earlier.float() / max(W - 1, 1))[:, None], last_fields,
    ], dim=1)


class HuggingFaceDecoderInference:
    """
    Fast inference wrapper for any HuggingFace causal LM (GPT-2, Llama, etc.).

    Extracts embeddings from the last hidden layer using last-token
    or mean pooling.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer,
        device: Optional[str] = None,
        pooling: str = "last_token",
        use_flash_attention: bool = False,
        tokens_per_txn: Optional[int] = None,
    ):
        """
        Args:
            model_path: Path to HuggingFace model directory or checkpoint.
            tokenizer: Tokenizer instance with pad_token_id attribute.
            device: Device to load model on (defaults to cuda if available).
            pooling: Embedding pooling strategy: a name in ``POOLINGS`` or
                several joined with ``+`` (concatenated).
            use_flash_attention: Enable Flash Attention 2 if available.
            tokens_per_txn: Tokens per transaction (needed by ``last_txn``).
        """
        from transformers import AutoModelForCausalLM

        self.model_path = Path(model_path)
        self.tokenizer = tokenizer
        self.pooling = pooling
        pooling_parts(pooling)   # validate now; resolved per call (see learned)
        self.tokens_per_txn = tokens_per_txn

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing HuggingFaceDecoderInference on {self.device}")

        load_kwargs = {}
        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")

        # A masked-LM checkpoint (model.objective: mlm) is loaded as one; the
        # embeddings come from its encoder exactly as from a decoder's stack.
        model_cfg = json.loads((self.model_path / "config.json").read_text()) \
            if (self.model_path / "config.json").exists() else {}
        self.masked_lm = any(str(a).endswith("ForMaskedLM")
                             for a in (model_cfg.get("architectures") or []))
        if self.masked_lm and "surprise" in pooling_parts(pooling):
            raise ValueError("pooling 'surprise' scores next-token predictions: it needs a "
                             "model trained with model.objective: clm, not mlm")
        if self.masked_lm:
            from transformers import AutoModelForMaskedLM as _Auto
        else:
            _Auto = AutoModelForCausalLM
        self.model = _Auto.from_pretrained(
            str(self.model_path), **load_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

        self._embedding_dim_base = self.model.config.hidden_size
        logger.info(
            f"Model loaded: {self.model.config.num_hidden_layers} layers, "
            f"{self.embedding_dim} embedding dim, pooling={pooling}"
        )

    @property
    def embedding_dim(self) -> int:
        """Follows the current ``pooling`` (several parts are concatenated)."""
        return sum(pooling_dims(self.pooling, self._embedding_dim_base, self.tokens_per_txn))

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return last-layer hidden states."""
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Run the base transformer, not the causal-LM wrapper: the wrapper also
        # projects every position onto the vocabulary, a (batch, seq, vocab)
        # logit tensor nothing here uses -- 4.4 GB in fp32 at 256 x 545 x 7,966,
        # and the reason a longer HISTORY_WINDOW ran out of memory. The wrapper
        # returns the base model's hidden_states unchanged, so the embeddings
        # are identical.
        outputs = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        return hidden_states, attention_mask

    @property
    def _backbone(self):
        """The decoder stack without the LM head (``model.model`` for Llama)."""
        prefix = getattr(self.model, "base_model_prefix", "")
        return getattr(self.model, prefix, None) or self.model

    def _pool_embeddings(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the pooling strategy (or concatenate several)."""
        model = getattr(self, "model", None)
        ctx = PoolingContext(
            tokens_per_txn=self.tokens_per_txn,
            input_ids=input_ids,
            lm_head=model.get_output_embeddings() if model is not None else None,
        )
        parts = [fn(hidden_states, attention_mask, ctx)
                 for fn in resolve_pooling(self.pooling)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states, attention_mask = self._get_hidden_states(input_ids)
        return self._pool_embeddings(hidden_states, attention_mask, input_ids)

    @torch.no_grad()
    def extract_embeddings(
        self,
        input_ids: torch.Tensor,
        return_numpy: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract embeddings from input sequences.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len).
            return_numpy: If True, return numpy array; else return tensor.

        Returns:
            Embeddings of shape (batch, embedding_dim).
        """
        input_ids = input_ids.to(self.device)
        hidden_states, attention_mask = self._get_hidden_states(input_ids)
        embeddings = self._pool_embeddings(hidden_states, attention_mask, input_ids)

        if return_numpy:
            return embeddings.float().cpu().numpy()
        return embeddings

    @torch.no_grad()
    def extract_embeddings_batched(
        self,
        padded_ids: np.ndarray,
        batch_size: int = 4096,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a large dataset in batches.

        Optimized for throughput with:
        - Pinned memory for async transfers
        - GPU accumulation before CPU transfer

        Args:
            padded_ids: Pre-padded token IDs array of shape (n_samples, seq_len).
            batch_size: Batch size for inference.
            show_progress: Whether to show progress bar.

        Returns:
            Embeddings array of shape (n_samples, embedding_dim).
        """
        from tqdm import tqdm

        n_samples = len(padded_ids)
        embed_dim = self.embedding_dim

        input_tensor = torch.from_numpy(padded_ids).pin_memory()

        gpu_embeddings = torch.empty(
            (n_samples, embed_dim),
            dtype=torch.float32,
            device=self.device,
        )

        n_batches = (n_samples + batch_size - 1) // batch_size
        iterator = range(0, n_samples, batch_size)

        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Extracting embeddings ({self.pooling})",
                total=n_batches,
            )

        for i in iterator:
            batch_end = min(i + batch_size, n_samples)
            batch_ids = input_tensor[i:batch_end].to(self.device, non_blocking=True)

            batch_embeddings = run_halving(self._embed_ids, batch_ids)

            gpu_embeddings[i:batch_end] = batch_embeddings.float()

        torch.cuda.synchronize()
        embeddings = gpu_embeddings.cpu().numpy()

        del gpu_embeddings
        torch.cuda.empty_cache()

        return embeddings

    @torch.no_grad()
    def extract_embeddings_windowed(
        self,
        encoder,
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings from lazily-built history windows.

        Same output as :meth:`extract_embeddings_batched`, but the token IDs of
        each batch are assembled on the fly by a
        :class:`~src.tokenizer.history.HistoryWindowEncoder`, so a
        ``(n_rows, window * tokens_per_txn)`` array never has to exist in
        memory.

        Args:
            encoder: Object exposing ``__len__``, ``sequence_length`` and
                ``iter_batches(batch_size)`` yielding ``(lo, hi, ids)``.
            batch_size: Rows per forward pass.
            show_progress: Whether to show a progress bar.

        Returns:
            Embeddings array of shape (n_rows, embedding_dim).
        """
        from tqdm import tqdm

        n_samples = len(encoder)
        embeddings = np.empty((n_samples, self.embedding_dim), dtype=np.float32)

        iterator = encoder.iter_batches(batch_size)
        if show_progress:
            n_batches = (n_samples + batch_size - 1) // batch_size
            iterator = tqdm(
                iterator,
                desc=f"Embedding history windows (L={encoder.sequence_length})",
                total=n_batches,
            )

        for lo, hi, batch_ids in iterator:
            input_ids = torch.from_numpy(batch_ids).to(self.device, non_blocking=True)
            batch_embeddings = run_halving(self._embed_ids, input_ids)
            embeddings[lo:hi] = batch_embeddings.float().cpu().numpy()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return embeddings

"""Transfer-level pretraining objectives: NPPR and CoLES.

``model.objective: clm`` and ``mlm`` predict tokens. The two objectives here work
on whole transfers instead; the transformer is the same, only the training signal
changes. Both read the corpus layout: ``<bos>`` then, per transfer, ``period - 1``
field tokens closed by ``<sep>`` (``<eos>`` for the last one).

NPPR (Skalski, Sutton & Burrell, "Towards a Foundation Purchasing Model:
Pretrained Generative Autoregression on Transaction Sequences", 2023)
    e_t = the hidden state at the token that closes transfer t (it has seen
    transfers 0..t only: NPPR needs a decoder). Two heads read it:

    * next event prediction: an MLP predicts every field of transfer t+1 at once,
      one cross-entropy per field, summed over fields;
    * past reconstruction: an MLP gets e_t and the time gap to transfer t-k and
      rebuilds every field of transfer t-k, for k = 1..K, weighted by
      exp(-gap / decay_days).

    loss = (1 - alpha) * NP + alpha * PR. The paper's GRU is replaced by our
    transformer. Its exact time gaps are replaced by the middle of each TDIF
    time-gap bin, summed between the two transfers (the corpus holds binned gaps
    only). All fields are categorical tokens, so every field is a cross-entropy
    (the paper's numeric fields use squared error).

CoLES (Babaev et al., "CoLES: Contrastive Learning for Event Sequences with
Self-Supervision", SIGMOD 2022)
    ``slices`` random slices (contiguous runs of transfers, ``min_txns`` to
    ``max_txns`` long) are drawn from each sequence; a slice's vector is the mean
    of its token states, scaled to unit length. Slices of the same sequence are
    pulled together, slices of other sequences pushed beyond ``margin``: the
    contrastive loss with hard negatives (for each slice, the ``neg_count``
    closest slices of other sequences), as in the paper's best setting. The
    paper's GRU and last state are replaced by our transformer and the mean
    (notebook 04's mean pooling). Negatives come from the same micro-batch.

Neither objective trains the language-model head: a checkpoint records
``pretrain_objective`` in config.json and notebook 04 refuses ``surprise`` pooling
for it.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

EVENT_OBJECTIVES = ("nppr", "coles")

SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------- layout


def n_transfers(length: int, period: int) -> int:
    """Complete transfers in a sequence of ``length`` tokens (``<bos>`` first)."""
    return max(0, (length - 1) // period)


def slot_token_ids(sequences: Sequence[Sequence[int]], period: int,
                   vocab_size: int, closing_ids: Optional[Sequence[int]] = None
                   ) -> List[np.ndarray]:
    """Token ids seen at each field position of a transfer, in the training corpus.

    Each field position has its own vocabulary (AMT_*, TDIF_*, ...); a head that
    predicts a field only scores that field's tokens.
    """
    n_fields = period - 1
    seen = np.zeros((n_fields, vocab_size), dtype=bool)
    rows = np.arange(n_fields)[None, :]
    closing = np.zeros(vocab_size, dtype=bool)
    if closing_ids is not None:
        closing[list(closing_ids)] = True
    n_blocks = n_closed = 0
    for seq in sequences:
        n = n_transfers(len(seq), period)
        if n:
            block = np.asarray(seq[1:1 + n * period], dtype=np.int64).reshape(n, period)
            seen[rows, block[:, :n_fields]] = True
            n_blocks += n
            n_closed += int(closing[block[:, -1]].sum())
    if closing_ids is not None and n_blocks and n_closed < 0.99 * n_blocks:
        raise ValueError(
            f"period={period} does not match the corpus: only {n_closed / n_blocks:.0%} of "
            "the transfers end with <sep>/<eos> at that position. "
            "diagnostics.field_loss_period must be tokens per transfer + 1 (notebook 03 sets it).")
    return [np.flatnonzero(seen[j]) for j in range(n_fields)]


def gap_seconds_table(state_path, vocab: Dict[str, int], vocab_size: int,
                      field: str = "TDIF") -> np.ndarray:
    """``table[token id]`` = the middle of that time-gap bin, in seconds (0 elsewhere).

    TimeDeltaTokenizer bins log(1 + gap seconds) between fixed boundaries; with
    ``zero_based_bins`` token i covers [b_i, b_i+1), without it [b_i-1, b_i).
    The last token also takes every gap beyond (clipped): its range ends at the
    last boundary. ``TDIF_FIRST`` (no previous transfer) counts as 0.
    """
    state = json.loads(Path(state_path).read_text())
    step = (state.get("steps") or {}).get(field)
    if not step:
        raise ValueError(f"{state_path} has no {field} step: NPPR needs the time-gap field "
                         "(tokenizer include_time_delta)")
    init = step.get("init_params") or {}
    bounds = np.asarray((step.get("fitted_state") or {}).get("boundaries"), dtype=np.float64)
    zero_based = bool(init.get("zero_based_bins", False))
    table = np.zeros(vocab_size, dtype=np.float64)
    last = len(bounds) - 1
    for i in range(int(init.get("num_bins", last))):
        tid = vocab.get(f"{field}_{i}")
        if tid is None or tid >= vocab_size:
            continue
        lo = i if zero_based else i - 1
        lo = min(max(lo, 0), last)
        hi = min(lo + 1, last)
        if not zero_based and i == int(init.get("num_bins", last)) - 1:
            hi = last           # clipped: covers [b_N-2, b_N]
        table[tid] = np.expm1((bounds[lo] + bounds[hi]) / 2.0)
    return table


class EventLayout:
    """Where the transfers are in a corpus line, and what each field position holds."""

    def __init__(self, period: int, slot_ids: List[np.ndarray], vocab_size: int,
                 closing_ids: Sequence[int], gap_seconds: Optional[np.ndarray] = None,
                 gap_slot: Optional[int] = None):
        if period < 2:
            raise ValueError(f"period={period}: needs at least one field token per transfer")
        self.period = int(period)
        self.n_fields = self.period - 1
        self.sizes = [max(1, len(ids)) for ids in slot_ids]
        lookup = torch.full((self.n_fields, vocab_size), -1, dtype=torch.long)
        for j, ids in enumerate(slot_ids):
            if len(ids):
                lookup[j, torch.as_tensor(ids, dtype=torch.long)] = torch.arange(len(ids))
        self.lookup = lookup
        self.closing_ids = torch.as_tensor(sorted(set(closing_ids)), dtype=torch.long)
        self.gap_seconds = (None if gap_seconds is None
                            else torch.as_tensor(gap_seconds, dtype=torch.float64))
        self.gap_slot = gap_slot

    @classmethod
    def build(cls, sequences, period: int, vocab: Dict[str, int], vocab_size: int,
              state_path=None, need_gaps: bool = False, log=print) -> "EventLayout":
        t0 = time.time()
        closing = [vocab[t] for t in ("<sep>", "<eos>") if t in vocab]
        slot_ids = slot_token_ids(sequences, period, vocab_size, closing_ids=closing)
        gap_seconds = gap_slot = None
        if need_gaps:
            if state_path is None or not Path(state_path).exists():
                raise FileNotFoundError(f"NPPR needs the tokenizer state for the time-gap "
                                        f"bins: {state_path} not found")
            gap_seconds = gap_seconds_table(state_path, vocab, vocab_size)
            gap_ids = set(np.flatnonzero(gap_seconds).tolist())
            gap_ids |= {vocab[t] for t in ("TDIF_FIRST",) if t in vocab}
            share = [len(gap_ids.intersection(ids.tolist())) / max(1, len(ids)) for ids in slot_ids]
            gap_slot = int(np.argmax(share))
            if share[gap_slot] < 0.99:
                raise ValueError("could not find the TDIF position in the corpus "
                                 f"(best match {share[gap_slot]:.0%})")
        log(f"  Transfer layout: {period - 1} field tokens + <sep>; field vocabularies "
            f"{sum(len(i) for i in slot_ids):,} ids in total"
            + (f"; time gap at position {gap_slot}" if gap_slot is not None else "")
            + f" ({time.time() - t0:.0f}s)")
        return cls(period, slot_ids, vocab_size, closing, gap_seconds, gap_slot)


# ---------------------------------------------------------------- heads


class FieldDecoder(torch.nn.Module):
    """MLP from one vector to every field of one transfer (one softmax per field)."""

    def __init__(self, in_dim: int, hidden: int, sizes: Sequence[int]):
        super().__init__()
        self.sizes = list(sizes)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, sum(self.sizes)),
        )

    def field_ce(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy per row and field, ``(rows, fields)``; 0 where target is -1."""
        logits = self.mlp(x).float()
        out = []
        for j, part in enumerate(torch.split(logits, self.sizes, dim=-1)):
            out.append(F.cross_entropy(part, targets[:, j], ignore_index=-1, reduction="none"))
        return torch.stack(out, dim=1)


class _Pretrainer(torch.nn.Module):
    def __init__(self, lm: torch.nn.Module):
        super().__init__()
        self.lm = lm

    def hidden(self, input_ids, attention_mask):
        return self.lm.base_model(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state


class NPPRPretrainer(_Pretrainer):
    """Next event prediction + past reconstruction on the transformer's transfer states."""

    def __init__(self, lm, layout: EventLayout, alpha: float = 0.1, past_events: int = 8,
                 decay_days: float = 60.0, decoder_hidden: int = 512):
        super().__init__(lm)
        if layout.gap_seconds is None:
            raise ValueError("NPPR needs the time-gap table (EventLayout need_gaps=True)")
        h = int(lm.config.hidden_size)
        self.alpha, self.past_events, self.decay_days = float(alpha), int(past_events), float(decay_days)
        self.period, self.n_fields, self.gap_slot = layout.period, layout.n_fields, layout.gap_slot
        self.register_buffer("lookup", layout.lookup, persistent=False)
        self.register_buffer("closing", layout.closing_ids, persistent=False)
        self.register_buffer("gap_table", layout.gap_seconds.float(), persistent=False)
        self.next_head = FieldDecoder(h, decoder_hidden, layout.sizes)
        self.past_head = FieldDecoder(h + 1, decoder_hidden, layout.sizes)

    def forward(self, input_ids, attention_mask, **_) -> Dict[str, Any]:
        P, Fn = self.period, self.n_fields
        B, L = input_ids.shape
        n = n_transfers(L, P)
        hid = self.hidden(input_ids, attention_mask)
        close = slice(P, n * P + 1, P)                       # token closing transfer t
        e = hid[:, close]                                   # (B, n, H)
        valid = (attention_mask[:, close] > 0) & torch.isin(input_ids[:, close], self.closing)
        fields = input_ids[:, 1:1 + n * P].reshape(B, n, P)[..., :Fn]
        tgt = self.lookup[torch.arange(Fn, device=input_ids.device).view(1, 1, Fn), fields]
        tgt = tgt.masked_fill(~valid.unsqueeze(-1), -1)
        # Time of each transfer from the binned gaps (days since the sequence start).
        gap = self.gap_table[fields[..., self.gap_slot]] * valid
        days = torch.cumsum(gap, dim=1) / SECONDS_PER_DAY

        # Next event: e_t -> fields of t+1.
        pair = valid[:, :-1] & valid[:, 1:]
        np_ce = self.next_head.field_ce(e[:, :-1][pair], tgt[:, 1:][pair])   # (pairs, F)
        n_pairs = max(1, int(pair.sum()))
        np_loss = np_ce.sum() / n_pairs

        # Past reconstruction: e_t and the gap -> fields of t-k, weight exp(-gap/decay).
        n_events = max(1, int(valid.sum()))
        pr_total = e.new_zeros((), dtype=torch.float32)
        for k in range(1, min(self.past_events, n - 1) + 1):
            ok = valid[:, k:] & valid[:, :-k]
            if not bool(ok.any()):
                continue
            dt = (days[:, k:] - days[:, :-k])[ok].clamp(min=0)
            x = torch.cat([e[:, k:][ok], torch.log1p(dt).unsqueeze(-1).to(e.dtype)], dim=-1)
            w = torch.exp(-dt / self.decay_days)
            t = tgt[:, :-k][ok]
            if torch.is_grad_enabled():
                ce = checkpoint(self.past_head.field_ce, x, t, use_reentrant=False)
            else:
                ce = self.past_head.field_ce(x, t)
            pr_total = pr_total + (w * ce.sum(dim=1)).sum()
        pr_loss = pr_total / n_events

        loss = (1.0 - self.alpha) * np_loss + self.alpha * pr_loss
        loss = loss + 0.0 * e.sum()       # keeps the graph when a batch has no pairs
        counts = (tgt[:, 1:][pair] >= 0).sum(dim=0)
        return {"loss": loss, "np_loss": np_loss.detach(), "pr_loss": pr_loss.detach(),
                "field_ce_sum": np_ce.detach().sum(dim=0), "field_count": counts}


def contrastive_loss(emb: torch.Tensor, owner: torch.Tensor, margin: float = 0.5,
                     neg_count: int = 5) -> torch.Tensor:
    """CoLES loss on unit vectors: d^2 for same-owner pairs, relu(margin - d)^2 for
    the ``neg_count`` closest other-owner vectors of each one. Mean over pairs.

    In fp32 even under autocast: in bf16 a cosine above ~0.996 rounds to distance
    0, where the negatives give no gradient (a collapsed start could not recover)."""
    with torch.autocast(device_type=emb.device.type, enabled=False):
        return _contrastive_fp32(emb.float(), owner, margin, neg_count)


def _contrastive_fp32(emb, owner, margin, neg_count):
    sim = emb @ emb.t()
    d2 = (2.0 - 2.0 * sim).clamp(min=0.0)
    same = owner[:, None] == owner[None, :]
    upper = torch.triu(torch.ones_like(same), diagonal=1)
    pos = d2[same & upper]
    n_other = int((~same).sum(dim=1).min())
    k = min(int(neg_count), n_other)
    if k < 1:            # one sequence only (a last validation batch): no negatives
        return pos.mean() if pos.numel() else d2.sum() * 0.0
    d = torch.sqrt(d2.clamp(min=1e-6))
    hardest = d.masked_fill(same, float("inf")).topk(k, dim=1, largest=False).values
    neg = F.relu(margin - hardest).pow(2).flatten()
    return torch.cat([pos, neg]).mean()


class CoLESPretrainer(_Pretrainer):
    """Contrastive learning on slices: input (B, K, L), K slices of B sequences."""

    def __init__(self, lm, margin: float = 0.5, neg_count: int = 5):
        super().__init__(lm)
        self.margin, self.neg_count = float(margin), int(neg_count)

    def embed(self, input_ids, attention_mask):
        hid = self.hidden(input_ids, attention_mask).float()     # mean in fp32
        m = attention_mask.unsqueeze(-1).float()
        mean = (hid * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return F.normalize(mean, dim=-1)

    def forward(self, input_ids, attention_mask, **_) -> Dict[str, Any]:
        B, K, L = input_ids.shape
        emb = self.embed(input_ids.reshape(B * K, L), attention_mask.reshape(B * K, L))
        owner = torch.arange(B, device=input_ids.device).repeat_interleave(K)
        return {"loss": contrastive_loss(emb, owner, self.margin, self.neg_count)}


def build_pretrainer(objective: str, lm, cfg: Dict[str, Any],
                     layout: Optional[EventLayout] = None) -> _Pretrainer:
    if objective == "nppr":
        c = cfg.get("nppr") or {}
        return NPPRPretrainer(lm, layout, alpha=float(c.get("alpha", 0.1)),
                              past_events=int(c.get("past_events", 8)),
                              decay_days=float(c.get("decay_days", 60.0)),
                              decoder_hidden=int(c.get("decoder_hidden", 512)))
    if objective == "coles":
        c = cfg.get("coles") or {}
        return CoLESPretrainer(lm, margin=float(c.get("margin", 0.5)),
                               neg_count=int(c.get("neg_count", 5)))
    raise ValueError(f"objective={objective!r} is not a transfer-level objective")


# ---------------------------------------------------------------- data


class BankCoLESDataset(Dataset):
    """Each item: ``slices`` random slices of one sequence, ``(slices, slice_len)``.

    A slice is ``<bos>`` plus ``min_txns``..``max_txns`` consecutive transfers, at a
    random start. Sequences shorter than ``min_txns`` transfers are left out.
    ``slice_seed`` fixes the slices per item (validation).
    """

    def __init__(self, sequences, period: int, pad_token_id: int, bos_token_id: int,
                 slices: int = 5, min_txns: int = 8, max_txns: int = 64,
                 seq_length: int = 4096, slice_seed: Optional[int] = None):
        if not 1 <= min_txns <= max_txns:
            raise ValueError(f"coles: need 1 <= min_txns ({min_txns}) <= max_txns ({max_txns})")
        if slices < 2:
            raise ValueError("coles.slices must be at least 2 (positives are slice pairs)")
        max_fit = (seq_length - 1) // period
        if max_txns > max_fit:
            raise ValueError(f"coles.max_txns={max_txns} transfers do not fit in "
                             f"seq_length={seq_length} ({max_fit} do)")
        self.period, self.pad, self.bos = int(period), int(pad_token_id), int(bos_token_id)
        self.slices, self.min_txns, self.max_txns = int(slices), int(min_txns), int(max_txns)
        self.slice_len = 1 + self.max_txns * self.period
        self.slice_seed = slice_seed
        self.sequences = [s for s in sequences if n_transfers(len(s), self.period) >= self.min_txns]
        self.dropped = len(sequences) - len(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        P = self.period
        n = n_transfers(len(seq), P)
        gen = None
        if self.slice_seed is not None:
            gen = torch.Generator().manual_seed(int(self.slice_seed) * 1_000_003 + int(idx))
        hi = min(self.max_txns, n)
        lengths = torch.randint(self.min_txns, hi + 1, (self.slices,), generator=gen)
        ids = torch.full((self.slices, self.slice_len), self.pad, dtype=torch.long)
        mask = torch.zeros((self.slices, self.slice_len), dtype=torch.long)
        body = torch.as_tensor(seq, dtype=torch.long)
        for i, ln in enumerate(lengths.tolist()):
            start = int(torch.randint(0, n - ln + 1, (1,), generator=gen))
            part = body[1 + start * P: 1 + (start + ln) * P]
            ids[i, 0] = self.bos
            ids[i, 1:1 + len(part)] = part
            mask[i, :1 + len(part)] = 1
        return {"input_ids": ids, "attention_mask": mask}

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
Trailing-history windows for per-transaction embedding extraction.

``TokenizerPipeline.encode()`` turns each row into a standalone sequence
(``<bos> f1 ... fN <eos>``), so the resulting embedding sees a single
transaction -- none of the sequential structure the model was pretrained on.

This module builds, for every row *i*, the sequence of the **last K
transactions of the same account ending at row i**::

    <bos> txn[i-K+1] <sep> txn[i-K+2] <sep> ... <sep> txn[i] <eos> <pad> ...

Padding stays on the right, so last-token pooling
(``attention_mask.sum(1) - 1``) lands on ``<eos>`` -- the position that has
attended to the whole window.  Windows are truncated, never padded on the
left, so early transactions in an account simply get shorter histories.

Rows must be ordered by ``(group_cols..., time)`` -- exactly what
``BankTransferTokenizerPipeline.preprocess()`` produces.  Windows are built
lazily per batch to keep memory flat on multi-million-row splits.
"""

import warnings
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


class HistoryWindowEncoder:
    """Lazily builds right-padded history windows of token IDs.

    Parameters
    ----------
    token_ids : np.ndarray, shape (n_rows, tokens_per_txn), int
        Per-transaction token IDs, in the pipeline's column order.
    positions : np.ndarray, shape (n_rows,), int
        0-based index of each row within its account sequence (``cumcount``).
    window : int
        Maximum number of transactions per window (including the current one).
    max_length : int, optional
        Sequence length. Defaults to the exact length needed by *window*;
        a smaller value shrinks the effective window.
    row_index : np.ndarray, optional
        Rows to emit windows for (default: all).  History is always read from
        the full frame, so embedding a sampled subset still sees that subset's
        true predecessors -- pass the *full* split here and select the labelled
        rows with :meth:`select`.
    """

    def __init__(
        self,
        token_ids: np.ndarray,
        positions: np.ndarray,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        sep_id: int = 3,
        window: int = 32,
        max_length: Optional[int] = None,
        validate: bool = True,
        row_index: Optional[np.ndarray] = None,
    ):
        self.token_ids = np.ascontiguousarray(token_ids, dtype=np.int64)
        self.positions = np.ascontiguousarray(positions, dtype=np.int64)
        if len(self.token_ids) != len(self.positions):
            raise ValueError(
                f"token_ids has {len(self.token_ids)} rows but positions has "
                f"{len(self.positions)}"
            )

        self.pad_id, self.bos_id, self.eos_id, self.sep_id = (
            int(pad_id), int(bos_id), int(eos_id), int(sep_id)
        )
        self.tokens_per_txn = int(self.token_ids.shape[1])
        self.window = max(1, int(window))

        needed = self.required_length(self.window, self.tokens_per_txn)
        if max_length is None:
            self.max_length = needed
        else:
            self.max_length = int(max_length)
            fits = (self.max_length - 1) // (self.tokens_per_txn + 1)
            if fits < self.window:
                warnings.warn(
                    f"max_length={self.max_length} fits only {fits} transactions "
                    f"of {self.tokens_per_txn} tokens; reducing window from "
                    f"{self.window} to {max(1, fits)}.",
                    stacklevel=2,
                )
                self.window = max(1, fits)
        if self.max_length < self.required_length(self.window, self.tokens_per_txn):
            raise ValueError(
                f"max_length={self.max_length} is too small for even one "
                f"transaction of {self.tokens_per_txn} tokens."
            )

        self.n_rows = len(self.token_ids)
        if row_index is None:
            self.row_index = np.arange(self.n_rows, dtype=np.int64)
        else:
            self.row_index = np.ascontiguousarray(row_index, dtype=np.int64)
            if len(self.row_index) and (
                self.row_index.min() < 0 or self.row_index.max() >= self.n_rows
            ):
                raise ValueError("row_index contains out-of-range row positions")

        if validate:
            self._validate_order()

        # Per-window column layout, precomputed once.
        j = np.arange(self.window, dtype=np.int64)
        self._offsets = j
        self._base = 1 + j * (self.tokens_per_txn + 1)
        self._field_cols = (
            self._base[:, None] + np.arange(self.tokens_per_txn, dtype=np.int64)[None, :]
        ).reshape(-1)
        self._sep_cols = self._base + self.tokens_per_txn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def required_length(window: int, tokens_per_txn: int) -> int:
        """``<bos>`` + window * (fields + separator/eos)."""
        return 1 + window * (tokens_per_txn + 1)

    def _validate_order(self) -> None:
        """Check rows are grouped and time-ordered (pos increments by 1)."""
        if len(self.positions) < 2:
            return
        cur, prev = self.positions[1:], self.positions[:-1]
        continuing = cur > 0
        if not np.all(cur[continuing] == prev[continuing] + 1):
            raise ValueError(
                "Rows are not ordered by (account, time): a row with position "
                "p>0 must directly follow position p-1. Build the frame with "
                "BankTransferTokenizerPipeline.preprocess() (which sorts) and "
                "keep that row order."
            )

    def __len__(self) -> int:
        return len(self.row_index)

    @property
    def sequence_length(self) -> int:
        return self.max_length

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self), self.max_length)

    def select(self, rows) -> "HistoryWindowEncoder":
        """Return an encoder that emits windows only for *rows*.

        Shares the token matrix, so histories still come from the full frame.
        """
        clone = object.__new__(HistoryWindowEncoder)
        clone.__dict__.update(self.__dict__)
        clone.row_index = np.ascontiguousarray(rows, dtype=np.int64)
        if len(clone.row_index) and (
            clone.row_index.min() < 0 or clone.row_index.max() >= self.n_rows
        ):
            raise ValueError("select() got out-of-range row positions")
        return clone

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def batch(self, lo: int, hi: int) -> np.ndarray:
        """Windows for selected rows ``[lo, hi)`` as int64 ``(b, max_length)``."""
        lo, hi = int(lo), int(hi)
        rows = self.row_index[lo:hi]
        b = len(rows)
        out = np.full((b, self.max_length), self.pad_id, dtype=np.int64)
        if b == 0:
            return out
        out[:, 0] = self.bos_id

        # k = transactions available in this window (>=1, <= window)
        k = np.minimum(self.positions[rows] + 1, self.window)

        # Source row for slot j: i - k + 1 + j  (left-aligned, valid for j < k)
        src = (rows - k + 1)[:, None] + self._offsets[None, :]
        valid = self._offsets[None, :] < k[:, None]
        np.clip(src, 0, self.n_rows - 1, out=src)

        fields = self.token_ids[src]                       # (b, window, T)
        flat = fields.reshape(b, -1)                       # (b, window*T)
        field_mask = np.repeat(valid, self.tokens_per_txn, axis=1)
        out[:, self._field_cols] = np.where(field_mask, flat, self.pad_id)

        # <sep> after every transaction except the last valid one
        sep_mask = self._offsets[None, :] < (k[:, None] - 1)
        out[:, self._sep_cols] = np.where(sep_mask, self.sep_id, self.pad_id)

        # <eos> right after the current (last) transaction
        eos_pos = 1 + (k - 1) * (self.tokens_per_txn + 1) + self.tokens_per_txn
        out[np.arange(b), eos_pos] = self.eos_id
        return out

    def iter_batches(
        self, batch_size: int = 1024
    ) -> Iterator[Tuple[int, int, np.ndarray]]:
        for lo in range(0, len(self), batch_size):
            hi = min(lo + batch_size, len(self))
            yield lo, hi, self.batch(lo, hi)

    def to_array(self) -> np.ndarray:
        """Materialize every window (only for small splits -- O(n x L) memory)."""
        return self.batch(0, len(self))

    # ------------------------------------------------------------------
    # Construction from a tokenized frame
    # ------------------------------------------------------------------

    @classmethod
    def from_token_frame(
        cls,
        token_df,
        meta_df,
        group_cols: Sequence[str],
        vocab: Dict[str, int],
        unk_id: int = 4,
        window: int = 32,
        max_length: Optional[int] = None,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        sep_id: int = 3,
        columns: Optional[List[str]] = None,
        row_index: Optional[np.ndarray] = None,
    ) -> "HistoryWindowEncoder":
        """Build an encoder from ``pipeline.transform()`` output.

        Parameters
        ----------
        token_df : cudf.DataFrame
            Token *strings*, one column per field (output of ``transform``).
        meta_df : cudf.DataFrame
            Preprocessed frame (same row order) holding *group_cols*.
        group_cols : sequence of str
            Sequence key, e.g. ``["customer_id", "account_number"]``.
        vocab : dict
            ``token string -> global id`` map from the fitted pipeline.
        """
        cols = list(columns or token_df.columns)
        n = len(token_df)
        ids = np.full((n, len(cols)), unk_id, dtype=np.int64)
        for c, name in enumerate(cols):
            host = token_df[name]
            host = host.to_pandas() if hasattr(host, "to_pandas") else host
            ids[:, c] = host.map(vocab).fillna(unk_id).astype(np.int64).values

        pos = meta_df.groupby(list(group_cols)).cumcount()
        pos = pos.to_pandas() if hasattr(pos, "to_pandas") else pos
        positions = np.asarray(pos.values, dtype=np.int64)

        return cls(
            token_ids=ids,
            positions=positions,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            sep_id=sep_id,
            window=window,
            max_length=max_length,
            row_index=row_index,
        )

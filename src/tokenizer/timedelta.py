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
Log-compressed time-delta tokenizer (GPU-accelerated).

Bins time differences (in seconds) into logarithmically-spaced buckets.
Useful for capturing recency signals — small deltas (seconds/minutes) get
fine-grained bins while large deltas (months) share coarser bins.

Two options fix defects of the original binning. Both default to the original
behaviour, so a state file written before they existed reloads unchanged and
its checkpoint keeps meaning what it meant; ``BankTokenizerConfig`` turns them
on for new fits.

``zero_based_bins``
    The original indexed bins with ``digitize(x, edges)``, which returns 1 for
    the first interval, so ``{prefix}_0`` could never be emitted, and a gap at
    or beyond the horizon (index ``num_bins + 1``) was clipped into the same
    token as the last real bin. Subtracting one maps interval *i* to token *i*,
    and every one of the ``num_bins`` tokens is reachable.
``first_token``
    Adds ``{prefix}_FIRST`` for a row with no previous transaction, passed in
    as a missing gap (NaN). Without it the caller filled that case with 0, so
    "first transaction of the sequence" and "booked in the same second as the
    previous one" shared a token.

Data is never stored in the constructor.
"""

import cudf
import cupy as cp

from .base import BaseTokenizer

_SECONDS_PER_JULIAN_YEAR = 31556951.999999996


class TimeDeltaTokenizer(BaseTokenizer):
    """Log-scale binning for time deltas (in seconds)."""

    def __init__(
        self,
        num_bins: int = 32,
        special_token: str = "TDIF",
        max_years: float = 2.5,
        stream: cp.cuda.Stream = None,
        zero_based_bins: bool = False,
        first_token: bool = False,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.special_token = special_token
        self.max_years = max_years
        self.stream = stream
        self.zero_based_bins = zero_based_bins
        self.first_token = first_token

        self.max_horizon = int(max_years * _SECONDS_PER_JULIAN_YEAR)
        log_max = cp.log(float(self.max_horizon) + 1.0)
        self.boundaries = cp.linspace(0, log_max, self.num_bins + 1)
        self._vocab_built = False

    @property
    def first_label(self) -> str:
        return f"{self.special_token}_FIRST"

    def build_vocab(self, column_data=None) -> None:
        if self.stream:
            with self.stream:
                self._idx_to_token = self._make_vocab()
        else:
            self._idx_to_token = self._make_vocab()
        self._vocab_built = True

    def _make_vocab(self) -> dict:
        vocab = {i: f"{self.special_token}_{i}" for i in range(self.num_bins)}
        if self.first_token:
            vocab[self.num_bins] = self.first_label
        return vocab

    def tokenize(self, column_data) -> cudf.Series:
        if self.stream:
            with self.stream:
                return self._tokenize_internal(column_data)
        return self._tokenize_internal(column_data)

    def _tokenize_internal(self, column_data) -> cudf.Series:
        gaps = column_data.astype("float64")
        is_first = gaps.isna()
        # A missing gap only reaches here when the caller kept it missing, i.e.
        # with first_token on; without it fill as the original did (gap 0).
        clamped = gaps.fillna(0.0).clip(0, self.max_horizon)
        log_vals = cp.log(cp.asarray(clamped.values, dtype=cp.float64) + 1.0)
        token_ids = cp.digitize(log_vals, self.boundaries)
        if self.zero_based_bins:
            token_ids = token_ids - 1
        token_ids = cp.clip(token_ids, 0, self.num_bins - 1)
        tokens = cudf.Series(token_ids, index=column_data.index).map(self._idx_to_token)
        if self.first_token:
            tokens = tokens.where(~is_first, self.first_label)
        return tokens

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"TimeDeltaTokenizer(token={self.special_token}, "
            f"bins={self.num_bins}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "num_bins": self.num_bins,
            "special_token": self.special_token,
            "max_years": self.max_years,
            "stream": None,
            "zero_based_bins": self.zero_based_bins,
            "first_token": self.first_token,
        }

    def _get_fitted_state(self) -> dict:
        return {
            "boundaries": (
                self.boundaries.get()
                if isinstance(self.boundaries, cp.ndarray)
                else self.boundaries
            ),
            "max_horizon": self.max_horizon,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        self.boundaries = cp.array(state["boundaries"])
        self.max_horizon = state["max_horizon"]
        self._vocab_built = state.get("_vocab_built", False)

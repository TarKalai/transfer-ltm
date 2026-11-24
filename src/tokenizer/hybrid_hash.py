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
Hash buckets with a collision-free head of shared counterparties.

``CategoricalHashTokenizer`` puts every value into ``hash % vocab_limit``. For
counterparty accounts that means ~37 unrelated IBANs per bucket, so even a
counterparty that hundreds of customers pay -- a utility, the tax office, a
salary payer -- has no identity of its own: its token is shared with dozens of
one-off private accounts.

This step gives the ``top_k`` counterparties used by the most **distinct
customers** in the training split a dedicated token (``CPTY_T0``,
``CPTY_T1``, ...), and hashes everything else exactly as before
(``CPTY_<hash % vocab_limit>``, identical tokens to ``CategoricalHashTokenizer``).

Why distinct customers and not rows: ranking by rows would hand slots to the
private counterparties of a few very active customers -- memorising those
customers rather than learning shared behaviour, and putting individual
relationships into the vocabulary. ``min_customers`` sets the floor.

Input is two columns, ``[hash, customer]``: the integer hash from
``_hash.stable_hash`` (so the hash stays pinned, see that module) and the
customer key used only for counting at fit time. A single hash column also
works and then falls back to counting rows.

The fitted state stores the integer hashes of the head counterparties, never
the raw account strings.
"""

from typing import Dict, List, Optional

import cudf
import cupy as cp

from .base import BaseTokenizer


class HybridHashTokenizer(BaseTokenizer):
    """Dedicated tokens for widely shared values, hash buckets for the rest."""

    def __init__(
        self,
        vocab_limit: int = 4096,
        special_token: str = "CPTY",
        top_k: int = 2048,
        min_customers: int = 10,
        top_label: str = "T",
        stream: cp.cuda.Stream = None,
    ):
        """
        Parameters
        ----------
        vocab_limit : int
            Number of hash buckets for the tail (as ``CategoricalHashTokenizer``).
        special_token : str
            Token prefix, e.g. ``"CPTY"``.
        top_k : int
            Maximum number of values with a dedicated token.
        min_customers : int
            A value needs at least this many distinct customers (rows, if no
            customer column is given) in the training data to get one.
        top_label : str
            Marker for head tokens: ``{special_token}_{top_label}{rank}``.
        """
        super().__init__()
        self.vocab_limit = vocab_limit
        self.special_token = special_token
        self.top_k = top_k
        self.min_customers = min_customers
        self.top_label = top_label
        self.stream = stream

        self._top_hashes: List[int] = []
        self._top_counts: List[int] = []
        self._row_coverage: Optional[float] = None
        self._hash_to_token: Dict[int, str] = {}
        self._bucket_tokens: Dict[int, str] = {}
        self._vocab_built = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split(column_data):
        """``(hashes, customers or None)`` from a one- or two-column input."""
        if hasattr(column_data, "columns"):
            hashes = column_data[column_data.columns[0]]
            keys = column_data[column_data.columns[1]] if len(column_data.columns) > 1 else None
            return hashes, keys
        return column_data, None

    def _assign_tokens(self) -> None:
        self._bucket_tokens = {
            i: f"{self.special_token}_{i}" for i in range(self.vocab_limit)
        }
        head = [
            f"{self.special_token}_{self.top_label}{rank}"
            for rank in range(len(self._top_hashes))
        ]
        self._hash_to_token = dict(zip(self._top_hashes, head))
        # Bucket tokens keep local index == bucket, head tokens follow.
        self._idx_to_token = dict(self._bucket_tokens)
        for rank, token in enumerate(head):
            self._idx_to_token[self.vocab_limit + rank] = token
        self._vocab = None

    @property
    def top_values(self) -> int:
        return len(self._top_hashes)

    @property
    def row_coverage(self) -> Optional[float]:
        """Share of training rows whose value got a dedicated token."""
        return self._row_coverage

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def build_vocab(self, column_data=None) -> None:
        self._top_hashes, self._top_counts, self._row_coverage = [], [], None

        if column_data is not None and self.top_k > 0:
            hashes, keys = self._split(column_data)
            if keys is not None:
                pairs = cudf.DataFrame({"h": hashes, "k": keys}).drop_duplicates()
                counts = pairs["h"].value_counts()
            else:
                counts = hashes.value_counts()
            host = counts.to_pandas() if hasattr(counts, "to_pandas") else counts

            frame = host.rename("n").reset_index()
            frame.columns = ["h", "n"]
            frame = frame[frame["n"] >= self.min_customers]
            # Deterministic: most customers first, ties broken by hash value.
            frame = frame.sort_values(["n", "h"], ascending=[False, True]).head(self.top_k)

            self._top_hashes = [int(h) for h in frame["h"].tolist()]
            self._top_counts = [int(n) for n in frame["n"].tolist()]
            if len(hashes):
                covered = hashes.isin(self._top_hashes).sum()
                self._row_coverage = float(covered) / float(len(hashes))

        self._assign_tokens()
        self._vocab_built = True

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def tokenize(self, column_data) -> cudf.Series:
        if not self._vocab_built:
            raise ValueError(
                f"HybridHashTokenizer(prefix={self.special_token}) is not fitted."
            )
        hashes, _ = self._split(column_data)
        tokens = (hashes % self.vocab_limit).map(self._bucket_tokens)
        if self._hash_to_token:
            head = hashes.map(self._hash_to_token)
            tokens = head.fillna(tokens)
        return tokens

    def __repr__(self) -> str:
        if not self._vocab_built:
            status = "not built"
        else:
            cov = "" if self._row_coverage is None else f", {self._row_coverage:.1%} of rows"
            status = f"{self.top_values} dedicated{cov}"
        return (
            f"HybridHashTokenizer(token={self.special_token}, "
            f"buckets={self.vocab_limit}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "vocab_limit": self.vocab_limit,
            "special_token": self.special_token,
            "top_k": self.top_k,
            "min_customers": self.min_customers,
            "top_label": self.top_label,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        return {
            "top_hashes": [int(h) for h in self._top_hashes],
            "top_counts": [int(n) for n in self._top_counts],
            "row_coverage": self._row_coverage,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        self._top_hashes = [int(h) for h in state.get("top_hashes", [])]
        self._top_counts = [int(n) for n in state.get("top_counts", [])]
        self._row_coverage = state.get("row_coverage")
        self._vocab_built = state.get("_vocab_built", False)
        if self._vocab_built:
            self._assign_tokens()

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
Data-driven binning tokenizer for continuous columns (GPU-accelerated).

Replaces the cuML ``KBinsDiscretizer`` approach: the fitted state here is
just a list of bin edges, which (a) needs no dependency beyond cuDF/CuPy and
(b) is JSON-serializable, so the pipeline can be fitted once on the training
split and reloaded for val/test/inference without pickling an estimator.

Strategies:

``"quantile"`` (default)
    Edges at the ``num_bins`` quantiles, so every bin holds the same share of
    the training rows.  A skewed amount distribution (hundreds of thousands of
    distinct EUR values) is spread evenly over the available tokens instead of
    piling up in one fixed-threshold bucket.
``"uniform"``
    Equal-width edges between the observed minimum and maximum.  Rarely what
    you want for money, but useful for bounded, roughly flat quantities.

Handles negative values and zeros; ``signed_log=True`` additionally applies
``sign(x) * log1p(|x|)`` before binning, which spreads the mass of a
heavy-tailed distribution when ``num_bins`` is small.
"""

from typing import List, Optional, Sequence

import cudf
import cupy as cp

from .base import BaseTokenizer


class QuantileBinTokenizer(BaseTokenizer):
    """Token = ``{prefix}_{bin_index}`` with data-driven quantile edges."""

    STRATEGIES = ("quantile", "uniform")

    def __init__(
        self,
        prefix: str = "AMT",
        num_bins: int = 32,
        signed_log: bool = False,
        clip_quantiles: Optional[Sequence[float]] = None,
        stream: cp.cuda.Stream = None,
        strategy: str = "quantile",
        missing_label: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        prefix : str
            Token prefix, e.g. ``"AMT"``.
        num_bins : int
            Number of bins (upper bound -- duplicate edges are dropped, so a
            near-constant column yields fewer used bins).
        signed_log : bool
            Apply ``sign(x) * log1p(|x|)`` before computing/applying edges.
        clip_quantiles : (lo, hi), optional
            Winsorize at these quantiles before binning, e.g. ``(0.001, 0.999)``.
        strategy : {"quantile", "uniform"}
            Equal-mass or equal-width bins.
        missing_label : str, optional
            If set, missing values get their own ``{prefix}_{missing_label}``
            token instead of being binned as 0 (the original behaviour) -- for
            values that are undefined rather than zero, e.g. "no history yet".
        """
        super().__init__()
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.STRATEGIES}, got {strategy!r}"
            )
        self.prefix = prefix
        self.num_bins = num_bins
        self.signed_log = signed_log
        self.clip_quantiles = tuple(clip_quantiles) if clip_quantiles else None
        self.stream = stream
        self.strategy = strategy
        self.missing_label = missing_label

        self.edges: List[float] = []
        self.clip_lo: Optional[float] = None
        self.clip_hi: Optional[float] = None
        self._vocab_built = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, column_data) -> cp.ndarray:
        vals = column_data.astype("float64").fillna(0.0).values
        vals = cp.asarray(vals, dtype=cp.float64)
        if self.clip_lo is not None:
            vals = cp.clip(vals, self.clip_lo, self.clip_hi)
        if self.signed_log:
            vals = cp.sign(vals) * cp.log1p(cp.abs(vals))
        return vals

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def build_vocab(self, column_data=None) -> None:
        self._idx_to_token = {
            i: f"{self.prefix}_{i}" for i in range(self.num_bins)
        }
        if self.missing_label:
            self._idx_to_token[self.num_bins] = f"{self.prefix}_{self.missing_label}"
        self._vocab = None

        if column_data is not None:
            series = column_data.astype("float64").dropna()
            if self.clip_quantiles is not None:
                lo_q, hi_q = self.clip_quantiles
                self.clip_lo = float(series.quantile(lo_q))
                self.clip_hi = float(series.quantile(hi_q))

            values = self._prepare(series)
            if self.strategy == "quantile":
                probs = [i / self.num_bins for i in range(1, self.num_bins)]
                host = cudf.Series(values).quantile(probs).to_pandas().tolist()
            else:  # uniform: equal-width edges between the observed extremes
                lo, hi = float(values.min()), float(values.max())
                width = (hi - lo) / self.num_bins
                host = [lo + width * i for i in range(1, self.num_bins)]

            edges: List[float] = []
            for edge in host:
                edge = float(edge)
                if edge != edge:  # NaN
                    continue
                if not edges or edge > edges[-1]:
                    edges.append(edge)
            self.edges = edges

        self._vocab_built = True

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def tokenize(self, column_data) -> cudf.Series:
        if not self._vocab_built:
            raise ValueError(f"QuantileBinTokenizer(prefix={self.prefix}) is not fitted.")

        vals = self._prepare(column_data)
        if self.edges:
            edges = cp.asarray(self.edges, dtype=cp.float64)
            bins = cp.searchsorted(edges, vals, side="right")
        else:
            bins = cp.zeros(len(vals), dtype=cp.int32)
        bins = cp.clip(bins, 0, self.num_bins - 1).astype(cp.int32)
        if self.missing_label:
            missing = cp.asarray(column_data.isna().values)
            bins = cp.where(missing, self.num_bins, bins).astype(cp.int32)
        return cudf.Series(bins, index=column_data.index).map(self._idx_to_token)

    @property
    def used_bins(self) -> int:
        """Number of bins actually reachable given the fitted edges."""
        return min(len(self.edges) + 1, self.num_bins) if self._vocab_built else 0

    def __repr__(self) -> str:
        status = f"{self.used_bins}/{self.num_bins} bins" if self._vocab_built else "not built"
        return (
            f"QuantileBinTokenizer(prefix={self.prefix}, "
            f"strategy={self.strategy}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "prefix": self.prefix,
            "num_bins": self.num_bins,
            "signed_log": self.signed_log,
            "clip_quantiles": list(self.clip_quantiles) if self.clip_quantiles else None,
            "stream": None,
            "strategy": self.strategy,
            "missing_label": self.missing_label,
        }

    def _get_fitted_state(self) -> dict:
        return {
            "edges": [float(e) for e in self.edges],
            "clip_lo": self.clip_lo,
            "clip_hi": self.clip_hi,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        self.edges = [float(e) for e in state.get("edges", [])]
        self.clip_lo = state.get("clip_lo")
        self.clip_hi = state.get("clip_hi")
        self._vocab_built = state.get("_vocab_built", False)

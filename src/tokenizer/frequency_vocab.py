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
Frequency-ranked categorical tokenizer (GPU-accelerated, data-driven).

For mid-cardinality categorical columns whose value set is *not* known in
advance -- counterparty BIC (thousands), country (hundreds), employee
(dozens), movement code (hundreds).  The vocabulary is the ``max_vocab``
most frequent values observed during ``build_vocab()``; everything else
(including values unseen at fit time) is handled by a pluggable **tail
strategy** (``tail=``, see ``tail.py``): the original single
``{prefix}_OTHER`` token (``"other"``, the default so saved states reload
unchanged), rarity bands (``"rare_bins"``) or hash buckets (``"hash"``).

Why not hashing?  Hash buckets mix unrelated values at *every* frequency
level.  Frequency ranking keeps the head of the distribution collision-free
and pushes only the rare tail into one bucket, which is what a transaction
model can actually learn from.

IMPORTANT -- this tokenizer is *fitted*: it must be built once on the
training split and then re-used (via ``TokenizerPipeline.save`` /
``load``) for validation, test and inference.  Re-fitting per split would
assign different token IDs to the same value.

Values are compared as strings; the column is cast internally, so integer
code columns (e.g. movement_code) work unchanged.
"""

import re
from typing import List, Optional

import cudf
import cupy as cp
import pandas as pd

from .base import BaseTokenizer
from .tail import make_tail

_UNSAFE = re.compile(r"[^A-Z0-9]+")
_SYMBOL_NAMES = {
    "+": "PLUS", "-": "NEG", "=": "EQ", "*": "STAR", "/": "SLASH",
    "%": "PCT", "&": "AND", "#": "NUM", "@": "AT", "<": "LT", ">": "GT",
}


def safe_label(value: str, max_chars: int = 24) -> str:
    """Make *value* usable inside a whitespace-separated token string.

    Uppercases and collapses every run of non-alphanumeric characters to
    ``-``.  A leading sign is spelled out (``"-1"`` -> ``"NEG1"``) and
    values made purely of symbols keep a readable name (``"+"`` ->
    ``"PLUS"``), so signed codes stay distinguishable from unsigned ones.
    """
    raw = str(value).strip().upper()
    prefix = ""
    if raw[:1] in _SYMBOL_NAMES and _UNSAFE.sub("", raw[1:]):
        prefix = _SYMBOL_NAMES[raw[:1]]
        raw = raw[1:]

    label = _UNSAFE.sub("-", raw).strip("-")
    if not label:
        label = "".join(_SYMBOL_NAMES.get(ch, "") for ch in raw)
    label = (prefix + label)[:max_chars].strip("-")
    return label or "NA"


class FrequencyVocabTokenizer(BaseTokenizer):
    """Top-``max_vocab`` values by frequency, plus a shared ``OTHER`` token."""

    def __init__(
        self,
        prefix: str,
        max_vocab: int = 1024,
        min_count: int = 1,
        other_label: str = "OTHER",
        readable: bool = True,
        stream: cp.cuda.Stream = None,
        tail: str = "other",
        tail_buckets: int = 16,
    ):
        """
        Parameters
        ----------
        prefix : str
            Token prefix, e.g. ``"BIC"``, ``"CTRY"``, ``"MVC"``.
        max_vocab : int
            Maximum number of distinct values that get their own token.
            Vocabulary size is ``min(n_distinct, max_vocab) + 1`` (``OTHER``).
        min_count : int
            Values observed fewer than this many times are pushed to ``OTHER``
            even if there is room left in the vocabulary.
        other_label : str
            Label of the catch-all token.
        readable : bool
            If True, tokens carry a sanitized form of the value
            (``BIC_BANKAAXX``); collisions get a numeric suffix.  If False,
            tokens are frequency ranks (``BIC_0``, ``BIC_1``, ...).
        tail : str
            What values without a head token become (see ``tail.py``).
            ``"other"`` keeps the original single ``OTHER`` token.
        tail_buckets : int
            Bucket count for ``tail="hash"``.
        """
        super().__init__()
        self.prefix = prefix
        self.max_vocab = max_vocab
        self.min_count = min_count
        self.other_label = other_label
        self.readable = readable
        self.stream = stream
        self.tail = tail
        self.tail_buckets = tail_buckets
        self._tail = make_tail(tail, tail_buckets=tail_buckets)

        self._values: List[str] = []
        self._value_to_token: dict = {}
        self._vocab_built = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def other_token(self) -> str:
        return f"{self.prefix}_{self.other_label}"

    @property
    def values(self) -> List[str]:
        """Vocabulary values, most frequent first."""
        return list(self._values)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def build_vocab(self, column_data=None) -> None:
        if column_data is None:
            raise ValueError(
                f"FrequencyVocabTokenizer(prefix={self.prefix}) is data-driven; "
                "build_vocab() requires the training column. Fit the pipeline on "
                "the training split and reload the saved state for other splits."
            )

        counts = column_data.astype("str").value_counts()
        host = counts.to_pandas() if hasattr(counts, "to_pandas") else counts

        # Deterministic order: count desc, then value asc (breaks ties stably).
        frame = host.rename("count").reset_index()
        frame.columns = ["value", "count"]
        frame["value"] = frame["value"].astype(str)
        frame = frame.sort_values(["count", "value"], ascending=[False, True])
        head = frame[frame["count"] >= self.min_count].head(self.max_vocab)

        self._values = head["value"].tolist()
        tail = frame[~frame["value"].isin(set(self._values))]
        self._tail.fit(tail.set_index("value")["count"])
        self._assign_tokens()
        self._vocab_built = True

    def _assign_tokens(self, head_tokens: Optional[List[str]] = None) -> None:
        """Build ``value -> token`` and ``local_index -> token`` maps.

        ``head_tokens`` (a saved state's own tokens) are used as they are, so a
        reload reproduces exactly the tokens the corpus and model were built
        with, whatever the assignment rules below have become since.
        """
        tokens: List[str] = []
        # Reserve the tail's tokens too: a readable head value such as "TH3"
        # must not become BIC_TH3 while the hash tail also emits BIC_TH3.
        used = {self.other_token} | {f"{self.prefix}_{t}" for t in self._tail.labels()}

        for rank, value in enumerate(self._values):
            if head_tokens is not None:
                tokens.append(head_tokens[rank])
                continue
            if self.readable:
                token = f"{self.prefix}_{safe_label(value)}"
                if token in used:  # sanitization collision -- keep both distinct
                    suffix = 2
                    while f"{token}-{suffix}" in used:
                        suffix += 1
                    token = f"{token}-{suffix}"
            else:
                token = f"{self.prefix}_{rank}"
            used.add(token)
            tokens.append(token)

        head_tokens = list(tokens)
        # Tail labels follow the head; with tail="other" that is the single
        # OTHER token at the last local index, exactly as before.
        tokens += [f"{self.prefix}_{label}" for label in self._tail.labels()]
        self._value_to_token = dict(zip(self._values, head_tokens))
        self._value_to_token.update(
            {v: f"{self.prefix}_{label}" for v, label in self._tail.label_seen().items()}
        )
        self._idx_to_token = dict(enumerate(tokens))
        self._vocab = None  # invalidate the lazily built reverse map

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def tokenize(self, column_data) -> cudf.Series:
        if not self._vocab_built:
            raise ValueError(
                f"FrequencyVocabTokenizer(prefix={self.prefix}) is not fitted."
            )
        as_str = column_data.astype("str")
        mapped = as_str.map(self._value_to_token) if self._value_to_token else None
        missing = as_str if mapped is None else as_str[mapped.isna()]
        if len(missing) == 0:
            return as_str if mapped is None else mapped     # empty input: empty output
        # Values with no head token and not seen in the train tail: label the
        # distinct ones once on the host, then map back.
        uniq = missing.unique()
        uniq = uniq.to_pandas() if hasattr(uniq, "to_pandas") else pd.Series(uniq)
        uniq = [str(v) for v in uniq.tolist()]
        unseen = {
            v: f"{self.prefix}_{label}"
            for v, label in zip(uniq, self._tail.label_unseen(uniq))
        }
        filled = as_str.map(unseen)
        return filled if mapped is None else mapped.fillna(filled)

    def __repr__(self) -> str:
        status = f"{len(self._values)} values" if self._vocab_built else "not built"
        return (
            f"FrequencyVocabTokenizer(prefix={self.prefix}, "
            f"max_vocab={self.max_vocab}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "prefix": self.prefix,
            "max_vocab": self.max_vocab,
            "min_count": self.min_count,
            "other_label": self.other_label,
            "readable": self.readable,
            "stream": None,
            "tail": self.tail,
            "tail_buckets": self.tail_buckets,
        }

    def _get_fitted_state(self) -> dict:
        # The token strings themselves are in the vocab state (_idx_to_token);
        # a reload takes them from there rather than re-deriving them.
        return {"values": list(self._values), "_vocab_built": self._vocab_built,
                "tail_state": self._tail.get_state()}

    def _set_fitted_state(self, state: dict) -> None:
        self._values = [str(v) for v in state.get("values", [])]
        self._vocab_built = state.get("_vocab_built", False)
        self._tail.set_state(state.get("tail_state", {}))
        if self._vocab_built:
            # _set_vocab_state ran first: its index -> token map holds the exact
            # head tokens this state was fitted with (keys may be str after JSON).
            saved = self._idx_to_token or {}
            head = [saved.get(i, saved.get(str(i))) for i in range(len(self._values))]
            if None in head:
                raise ValueError(
                    f"FrequencyVocabTokenizer(prefix={self.prefix}): the saved state lacks "
                    "tokens for some head values; re-deriving them could rename tokens the "
                    "corpus and model were built with. Re-fit the tokenizer.")
            self._assign_tokens(head_tokens=head)

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
What a frequency vocabulary does with the values that get no token of their own.

``FrequencyVocabTokenizer`` gives the frequent values ("head") one token each.
Everything else -- values below ``min_count`` or beyond ``max_vocab`` in
train ("tail"), and values never seen in train ("unseen") -- is the tail
strategy's job.

Why it matters: with the original single ``OTHER`` token and caps above the
field's cardinality, ``OTHER`` *never occurs in training*, so its embedding
stays at random initialisation -- and every unseen value at val/test time is
fed through that untrained row. Each strategy below therefore maps unseen
values onto a token that training has seen, provided the tail is non-empty
(``min_count`` > 1 puts rare training values there).

``"other"``      one ``OTHER`` token (the original behaviour; kept for old states).
``"rare_bins"``  ``RARE_<lo>-<hi>``: how rare the value was in train, in
                 geometric bands (1, 2-4, 5-9, 10-19, ...). Unseen -> the rarest
                 band that occurred in train. Keeps "rare / novel" without
                 identity.
``"hash"``       ``TH<k>``: the value hashed into ``tail_buckets`` buckets. Rare
                 values stay mostly distinct; an unseen value whose bucket stayed
                 empty in train is folded onto an occupied (trained) bucket.

New strategy: subclass :class:`TailStrategy`, decorate with
:func:`register_tail_strategy`, pass ``tail="<name>"``.
"""

from typing import Any, Dict, List, Type

import pandas as pd

from ._hash import hash_text

_TAILS: Dict[str, Type["TailStrategy"]] = {}


def register_tail_strategy(cls: Type["TailStrategy"]) -> Type["TailStrategy"]:
    _TAILS[cls.name] = cls
    return cls


def known_tail_strategies() -> List[str]:
    return sorted(_TAILS)


def make_tail(name: str, **params) -> "TailStrategy":
    if name not in _TAILS:
        raise ValueError(f"Unknown tail strategy {name!r}; choose from {known_tail_strategies()}")
    return _TAILS[name](**params)


class TailStrategy:
    """Labels for tail and unseen values. Labels are local (no prefix)."""

    name = ""

    def __init__(self, tail_buckets: int = 16):
        self.tail_buckets = int(tail_buckets)

    def fit(self, tail_counts: pd.Series) -> None:
        """``tail_counts``: train count per tail value (index = value as str)."""

    def labels(self) -> List[str]:
        """Every label, in local-index order (appended after the head)."""
        raise NotImplementedError

    def label_seen(self) -> Dict[str, str]:
        """``{tail value seen in train: label}``; may be empty."""
        return {}

    def label_unseen(self, values: List[str]) -> List[str]:
        """Labels for values with no head token and no entry in label_seen()."""
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        pass


@register_tail_strategy
class OtherTail(TailStrategy):
    name = "other"

    def labels(self):
        return ["OTHER"]

    def label_unseen(self, values):
        return ["OTHER"] * len(values)


def _band_edges(max_count: int) -> List[int]:
    """1, 2, 5, 10, 20, 50, ... up to and including the first edge > max_count."""
    edges, k = [1], 0
    while edges[-1] <= max_count:
        k += 1
        edges.append((1, 2, 5)[k % 3] * 10 ** (k // 3))
    return edges


@register_tail_strategy
class RareBinsTail(TailStrategy):
    name = "rare_bins"

    def __init__(self, tail_buckets: int = 16):
        super().__init__(tail_buckets)
        self.edges: List[int] = [1, 2]
        self.value_band: Dict[str, int] = {}

    def _band(self, count: int) -> int:
        return max(i for i, e in enumerate(self.edges[:-1]) if count >= e)

    def fit(self, tail_counts):
        self.edges = _band_edges(int(tail_counts.max()) if len(tail_counts) else 1)
        self.value_band = {str(v): self._band(int(c)) for v, c in tail_counts.items()}

    def _label(self, band: int) -> str:
        lo, hi = self.edges[band], self.edges[band + 1] - 1
        return f"RARE_{lo}" if lo == hi else f"RARE_{lo}-{hi}"

    def labels(self):
        return [self._label(b) for b in range(len(self.edges) - 1)]

    def label_seen(self):
        return {v: self._label(b) for v, b in self.value_band.items()}

    def label_unseen(self, values):
        # The rarest band that training actually emitted -- an empty band's
        # token would be as untrained as the old OTHER.
        band = min(self.value_band.values()) if self.value_band else 0
        return [self._label(band)] * len(values)

    def get_state(self):
        return {"edges": list(self.edges), "value_band": dict(self.value_band)}

    def set_state(self, state):
        self.edges = [int(e) for e in state.get("edges", [1, 2])]
        self.value_band = {str(k): int(v) for k, v in state.get("value_band", {}).items()}


@register_tail_strategy
class HashTail(TailStrategy):
    name = "hash"

    def __init__(self, tail_buckets: int = 16):
        super().__init__(tail_buckets)
        self.occupied: List[int] = []      # buckets some train tail value fell in

    def fit(self, tail_counts):
        self.occupied = sorted({hash_text(str(v)) % self.tail_buckets for v in tail_counts.index})

    def labels(self):
        return [f"TH{i}" for i in range(self.tail_buckets)]

    def _bucket(self, value: str) -> int:
        h = hash_text(value)
        b = h % self.tail_buckets
        if self.occupied and b not in self.occupied:
            b = self.occupied[h % len(self.occupied)]
        return b

    def label_unseen(self, values):
        return [f"TH{self._bucket(v)}" for v in values]

    def get_state(self):
        return {"occupied": list(self.occupied)}

    def set_state(self, state):
        self.occupied = [int(b) for b in state.get("occupied", [])]

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
Free-text field (e.g. the transfer description) -> a fixed number of tokens.

Every transaction must emit the same number of tokens -- chunking, history
windows and the per-field loss diagnostic all rely on a rigid period -- so the
text gets ``slots`` positions, filled by a **strategy** and padded with
``TXT_PAD``. All slots share one vocabulary: ``TXT_RENT`` is the same token
(and the same embedding row) whichever slot it lands in.

Strategies (``TextTokenizer(strategy=...)``)
--------------------------------------------
``"words"``     the first ``slots`` words, in order; words outside the
                vocabulary become ``TXT_OTHER`` (the position is kept).
``"bag"``       the distinct in-vocabulary words, rarest first (most
                informative), up to ``slots``; order of the text ignored.
``"category"``  the whole cleaned string as one value, top-K + ``TXT_OTHER``.
                Forces ``slots=1``. Right when descriptions are templated.
``"hashed"``    the first ``slots`` words hashed into ``hash_size`` buckets.
                No fitted vocabulary, so unseen words still get a stable token.

Adding one: subclass :class:`TextStrategy`, decorate it with
:func:`register_text_strategy`, and pass its name. Nothing else changes --
the pipeline, corpus, training and embedding code only see token columns.

Vocabulary
----------
Fitted on the training split only, like every data-driven step. A unit (word,
or whole string for ``"category"``) enters the vocabulary only if at least
``min_customers`` distinct customers use it: words private to one customer --
names, personal references -- stay out, which keeps identities out of the
vocabulary and the model from memorising them. ``readable=False`` (default)
names tokens by frequency rank (``TXT_17``), so no word is written into the
state file or the checkpoint's vocabulary.

Cleaning (``clean_text``): lowercase, accents stripped, every run of
non-alphanumerics -> space, words containing a digit dropped (references,
IBANs, dates, amounts), words shorter than ``min_word_len`` dropped.
"""

import re
from typing import Any, Dict, List, Optional, Type

import cudf
import pandas as pd

from ._hash import hash_text
from .base import BaseTokenizer
from .frequency_vocab import safe_label

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_HAS_DIGIT = re.compile(r"\d")


def clean_text(
    texts: pd.Series, min_word_len: int = 2, drop_digit_words: bool = True
) -> pd.Series:
    """pandas Series of raw strings -> Series of word lists."""
    s = texts.fillna("").astype(str).str.lower()
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    s = s.str.replace(_NON_ALNUM, " ", regex=True)

    def _words(text: str) -> List[str]:
        return [
            w for w in text.split()
            if len(w) >= min_word_len and not (drop_digit_words and _HAS_DIGIT.search(w))
        ]

    return s.map(_words)


# ── Strategies ──────────────────────────────────────────────────────────

_STRATEGIES: Dict[str, Type["TextStrategy"]] = {}


def register_text_strategy(cls: Type["TextStrategy"]) -> Type["TextStrategy"]:
    """Make a strategy available by its ``name`` (and reloadable from a state)."""
    _STRATEGIES[cls.name] = cls
    return cls


def known_text_strategies() -> List[str]:
    return sorted(_STRATEGIES)


class TextStrategy:
    """Turns one cleaned text (a word list) into ``slots`` token labels.

    Labels are local: ``"PAD"``, ``"EMPTY"``, ``"OTHER"`` or a vocabulary
    entry's label. :class:`TextTokenizer` adds the prefix and owns the IDs.
    """

    name: str = ""
    fitted: bool = True       # False -> fit() learns a vocabulary from data
    fixed_slots: Optional[int] = None

    def __init__(self, slots: int, max_vocab: int, min_customers: int,
                 hash_size: int, readable: bool):
        self.slots = self.fixed_slots or int(slots)
        self.max_vocab = int(max_vocab)
        self.min_customers = int(min_customers)
        self.hash_size = int(hash_size)
        self.readable = bool(readable)
        #: Vocabulary keys, most customers first. A key is the unit itself when
        #: readable, else its stable hash -- so the state never holds the words.
        self.values: List[str] = []
        self._label: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def key(self, unit: str) -> str:
        return unit if self.readable else str(hash_text(unit))

    # -- vocabulary ------------------------------------------------------
    def units(self, words: List[str]) -> List[str]:
        """Vocabulary candidates found in one text (counted when fitting)."""
        return list(dict.fromkeys(words))

    def fit(self, customers_per_unit: pd.Series) -> None:
        """``customers_per_unit``: distinct customers per unit, any order."""
        frame = customers_per_unit[customers_per_unit >= self.min_customers]
        frame = frame.rename("n").reset_index()
        frame.columns = ["unit", "n"]
        frame = frame.sort_values(["n", "unit"], ascending=[False, True])
        self.set_values([self.key(u) for u in frame["unit"].tolist()[: self.max_vocab]])

    def set_values(self, values: List[str]) -> None:
        self.values = [str(v) for v in values]
        self._label, self._rank = {}, {}
        used = {"PAD", "EMPTY", "OTHER"}
        for rank, value in enumerate(self.values):
            label = safe_label(value, max_chars=32) if self.readable else str(rank)
            if label in used:
                suffix = 2
                while f"{label}-{suffix}" in used:
                    suffix += 1
                label = f"{label}-{suffix}"
            used.add(label)
            self._label[value] = label
            self._rank[value] = rank

    def lookup(self, unit: str) -> Optional[str]:
        """Label of a unit, or None if it is outside the vocabulary."""
        return self._label.get(self.key(unit))

    def labels(self) -> List[str]:
        """Every label this strategy can emit, in local-index order."""
        return ["PAD", "EMPTY", "OTHER"] + [self._label[v] for v in self.values]

    # -- encoding --------------------------------------------------------
    def encode(self, words: List[str]) -> List[str]:
        raise NotImplementedError

    def _pad(self, labels: List[str], words: List[str]) -> List[str]:
        if not words:
            labels = ["EMPTY"]
        labels = labels[: self.slots]
        return labels + ["PAD"] * (self.slots - len(labels))


@register_text_strategy
class WordSlots(TextStrategy):
    name = "words"

    def encode(self, words):
        return self._pad([self.lookup(w) or "OTHER" for w in words], words)


@register_text_strategy
class WordBag(TextStrategy):
    name = "bag"

    def encode(self, words):
        known = {self.key(w) for w in words} & self._label.keys()
        # Rarest first: a frequent word ("payment") carries the least information.
        labels = [self._label[k] for k in sorted(known, key=lambda k: -self._rank[k])]
        if words and not labels:
            labels = ["OTHER"]
        return self._pad(labels, words)


@register_text_strategy
class WholeText(TextStrategy):
    name = "category"
    fixed_slots = 1

    def units(self, words):
        return [" ".join(words)] if words else []

    def encode(self, words):
        return self._pad([self.lookup(" ".join(words)) or "OTHER"], words)


@register_text_strategy
class HashedWords(TextStrategy):
    name = "hashed"
    fitted = False

    def fit(self, customers_per_unit):
        self.set_values([])

    def labels(self):
        return ["PAD", "EMPTY", "OTHER"] + [f"H{i}" for i in range(self.hash_size)]

    def encode(self, words):
        return self._pad([f"H{hash_text(w) % self.hash_size}" for w in words], words)


# ── Pipeline step ───────────────────────────────────────────────────────


class TextTokenizer(BaseTokenizer):
    """A free-text column as ``slots`` tokens per transaction.

    Input columns: ``[text]`` or ``[text, customer_id]``. The customer column
    is only read when fitting, to count distinct customers per word; without
    it rows are counted instead.
    """

    def __init__(
        self,
        prefix: str = "TXT",
        strategy: str = "words",
        slots: int = 4,
        max_vocab: int = 4096,
        min_customers: int = 5,
        hash_size: int = 2048,
        readable: bool = False,
        min_word_len: int = 2,
        drop_digit_words: bool = True,
    ):
        super().__init__()
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown text strategy {strategy!r}; choose from {known_text_strategies()}"
            )
        self.prefix = prefix
        self.strategy_name = strategy
        self.max_vocab = max_vocab
        self.min_customers = min_customers
        self.hash_size = hash_size
        self.readable = readable
        self.min_word_len = min_word_len
        self.drop_digit_words = drop_digit_words
        self.strategy = _STRATEGIES[strategy](
            slots=slots, max_vocab=max_vocab, min_customers=min_customers,
            hash_size=hash_size, readable=readable,
        )
        self.slots = self.strategy.slots
        self._vocab_built = False

    def filler_tokens(self) -> list:
        # Only the padding of unused slots; TXT_EMPTY ("no text") is information.
        return [f"{self.prefix}_PAD"]

    def output_columns(self, step_name: str) -> list:
        return [f"{self.prefix.lower()}_{i}" for i in range(self.slots)]

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _split(column_data):
        if hasattr(column_data, "columns"):
            cols = list(column_data.columns)
            return column_data[cols[0]], (column_data[cols[1]] if len(cols) > 1 else None)
        return column_data, None

    def _clean(self, texts: pd.Series) -> pd.Series:
        return clean_text(texts, self.min_word_len, self.drop_digit_words)

    # -- fit -------------------------------------------------------------
    def build_vocab(self, column_data=None) -> None:
        if column_data is None:
            raise ValueError("TextTokenizer is data-driven; build_vocab() needs the training column.")
        text, customer = self._split(column_data)
        if self.strategy.fitted:
            if customer is not None:
                pairs = cudf.DataFrame({"t": text.astype("str"), "c": customer.astype("str")})
                pairs = pairs.drop_duplicates()
            else:
                pairs = cudf.DataFrame({"t": text.astype("str")})
                pairs["c"] = cudf.Series(range(len(pairs)), index=pairs.index).astype("str")
            pairs = pairs.to_pandas() if hasattr(pairs, "to_pandas") else pairs

            uniq = pd.Series(pd.unique(pairs["t"]))
            units_of = dict(zip(uniq, self._clean(uniq).map(self.strategy.units)))
            exploded = pd.DataFrame({
                "unit": pairs["t"].map(units_of), "c": pairs["c"].values,
            }).explode("unit").dropna(subset=["unit"])
            customers_per_unit = exploded.groupby("unit")["c"].nunique()
            self.strategy.fit(customers_per_unit)
        else:
            self.strategy.fit(pd.Series(dtype="int64"))
        self._assign_tokens()
        self._vocab_built = True

    def _assign_tokens(self) -> None:
        self._idx_to_token = {
            i: f"{self.prefix}_{label}" for i, label in enumerate(self.strategy.labels())
        }
        self._vocab = None

    # -- transform -------------------------------------------------------
    def tokenize(self, column_data):
        if not self._vocab_built:
            raise ValueError("TextTokenizer is not fitted.")
        text, _ = self._split(column_data)
        cat = text.astype("str").astype("category")
        categories = cat.cat.categories
        categories = categories.to_pandas() if hasattr(categories, "to_pandas") else categories
        codes = cat.cat.codes.values

        slots = self._clean(pd.Series(categories)).map(self.strategy.encode)
        names = self.output_columns("")
        out = {}
        for s, name in enumerate(names):
            table = cudf.Series([f"{self.prefix}_{labels[s]}" for labels in slots])
            col = table.take(codes)
            col.index = text.index
            out[name] = col
        return cudf.DataFrame(out, index=text.index)

    def __repr__(self) -> str:
        status = f"{len(self.strategy.values)} values" if self._vocab_built else "not built"
        return (f"TextTokenizer(strategy={self.strategy_name}, slots={self.slots}, "
                f"{status})")

    # -- serialization -----------------------------------------------------
    def _get_init_params(self) -> dict:
        return {
            "prefix": self.prefix,
            "strategy": self.strategy_name,
            "slots": self.slots,
            "max_vocab": self.max_vocab,
            "min_customers": self.min_customers,
            "hash_size": self.hash_size,
            "readable": self.readable,
            "min_word_len": self.min_word_len,
            "drop_digit_words": self.drop_digit_words,
        }

    def _get_fitted_state(self) -> dict:
        # Only the vocabulary values are stored; labels are re-derived.
        return {"values": list(self.strategy.values), "_vocab_built": self._vocab_built}

    def _set_fitted_state(self, state: Dict[str, Any]) -> None:
        self.strategy.set_values(state.get("values", []))
        self._vocab_built = state.get("_vocab_built", False)
        if self._vocab_built:
            self._assign_tokens()

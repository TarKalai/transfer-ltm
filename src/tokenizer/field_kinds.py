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
Field kinds: how a column of a given *type* is cleaned and tokenized.

Adding a column to the token stream is one entry in
``BankTokenizerConfig.extra_fields``::

    extra_fields={
        "CHAN": {"column": "channel", "kind": "categorical"},
        "FEE":  {"column": "fee_eur", "kind": "numeric", "bins": 16},
    }

The key is the token prefix (``CHAN_...``) and the name used in
``field_order`` / ``drop_fields``. Every other key is an option of the kind.

=============  ==========================================  =====================
kind           use it for                                  tokenizer
=============  ==========================================  =====================
categorical    one string/code per row, up to ~thousands   FrequencyVocab + tail
               of distinct values (channel, status, type)  strategy
identifier     IDs with 10^4+ values (accounts, devices,   HybridHash: shared
               merchants): only *recurrence* matters        values get a token,
                                                           the rest hash buckets
numeric        amounts, counts, balances                   QuantileBin (fitted
                                                           on train)
text           a few words of free text                    TextTokenizer
=============  ==========================================  =====================

Not sure which one? :func:`suggest_kind` looks at the data and says why.

New kind: subclass :class:`FieldKind`, decorate with :func:`register_field_kind`.
"""

from typing import Any, Dict, List, Type

from ._hash import stable_hash
from .frequency_vocab import FrequencyVocabTokenizer
from .hybrid_hash import HybridHashTokenizer
from .quantile_numeric import QuantileBinTokenizer
from .text import TextTokenizer

_KINDS: Dict[str, Type["FieldKind"]] = {}


def register_field_kind(cls: Type["FieldKind"]) -> Type["FieldKind"]:
    _KINDS[cls.name] = cls
    return cls


def known_field_kinds() -> List[str]:
    return sorted(_KINDS)


def get_field_kind(name: str) -> "FieldKind":
    if name not in _KINDS:
        raise ValueError(f"Unknown field kind {name!r}; choose from {known_field_kinds()}")
    return _KINDS[name]()


class FieldKind:
    """``prepare`` builds the derived column; ``tokenizer`` builds the step.

    ``pipeline`` gives access to the shared cleaning helpers and the config
    (tail strategy and its threshold apply to every categorical field).
    """

    name = ""
    #: True -> the step also reads the customer column (fit-time statistics).
    needs_customer = False
    #: Options accepted in the extra_fields entry, with their defaults.
    options: Dict[str, Any] = {}

    def prepare(self, series, pipeline):
        raise NotImplementedError

    def tokenizer(self, prefix: str, opts: Dict[str, Any], pipeline):
        raise NotImplementedError

    def resolve(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(spec) - set(self.options) - {"column", "kind"}
        if unknown:
            raise ValueError(
                f"Field kind {self.name!r} has no option(s) {sorted(unknown)}; "
                f"it accepts {sorted(self.options)}"
            )
        return {**self.options, **{k: v for k, v in spec.items() if k in self.options}}


@register_field_kind
class Categorical(FieldKind):
    name = "categorical"
    options = {"max_vocab": 256, "readable": True}

    def prepare(self, series, pipeline):
        return pipeline._clean_str(series)

    def tokenizer(self, prefix, opts, pipeline):
        return pipeline._freq(prefix, opts["max_vocab"], readable=opts["readable"])


@register_field_kind
class Identifier(FieldKind):
    name = "identifier"
    needs_customer = True
    options = {"hash_size": 1024, "top_k": 512, "min_customers": 10}

    def prepare(self, series, pipeline):
        return stable_hash(pipeline._clean_str(series))

    def tokenizer(self, prefix, opts, pipeline):
        return HybridHashTokenizer(
            vocab_limit=opts["hash_size"], special_token=prefix,
            top_k=opts["top_k"], min_customers=opts["min_customers"],
        )


@register_field_kind
class Numeric(FieldKind):
    name = "numeric"
    options = {"bins": 16, "signed_log": False, "clip_quantiles": (0.0005, 0.9995)}

    def prepare(self, series, pipeline):
        return pipeline._to_float(series)

    def tokenizer(self, prefix, opts, pipeline):
        clip = opts["clip_quantiles"]
        return QuantileBinTokenizer(
            prefix=prefix, num_bins=opts["bins"], signed_log=opts["signed_log"],
            clip_quantiles=tuple(clip) if clip is not None else None,
        )


@register_field_kind
class Text(FieldKind):
    name = "text"
    needs_customer = True
    options = {"strategy": "words", "slots": 4, "max_vocab": 4096,
               "min_customers": 5, "hash_size": 2048, "readable": False}

    def prepare(self, series, pipeline):
        return series.fillna("").astype("str")

    def tokenizer(self, prefix, opts, pipeline):
        return TextTokenizer(
            prefix=prefix, strategy=opts["strategy"], slots=opts["slots"],
            max_vocab=opts["max_vocab"], min_customers=opts["min_customers"],
            hash_size=opts["hash_size"], readable=opts["readable"],
        )


def suggest_kind(series, max_categorical: int = 5000) -> str:
    """Suggest a kind for a raw column, with the reason. Reads a sample."""
    import pandas as pd

    s = series.to_pandas() if hasattr(series, "to_pandas") else series
    s = s.dropna()
    if len(s) > 200_000:
        s = s.sample(200_000, random_state=0)
    if s.empty:
        return "empty column -- nothing to tokenize"
    n_distinct = s.nunique()
    if pd.api.types.is_numeric_dtype(s):
        if n_distinct <= 50:
            return f"categorical  ({n_distinct} distinct numbers: codes, not quantities)"
        return f"numeric  ({n_distinct} distinct values)"
    words = s.astype(str).str.split().str.len().mean()
    if words >= 1.5:
        return f"text  ({words:.1f} words per value on average)"
    if n_distinct > max_categorical:
        return f"identifier  ({n_distinct:,} distinct values in the sample)"
    return f"categorical  ({n_distinct:,} distinct values)"

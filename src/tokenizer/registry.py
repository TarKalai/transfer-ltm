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
Tokenizer-step class registry for pipeline (de)serialization.

Steps are resolved by *class name only*, never by fully-qualified module
path: the same class is imported as ``src.tokenizer.fixed_vocab`` from the
notebooks and as ``tokenizer.fixed_vocab`` from ``clm_data.py`` (which NeMo
AutoModel loads by file path), so the module prefix in a saved state file is
not stable.

Register custom steps with :func:`register_tokenizer` before loading a state
file that references them.
"""

from typing import Dict, Type

from .base import BaseTokenizer
from .categorical_hash import CategoricalHashTokenizer
from .fixed_vocab import FixedVocabTokenizer
from .frequency_vocab import FrequencyVocabTokenizer
from .hybrid_hash import HybridHashTokenizer
from .mapping import MappingTokenizer
from .quantile_numeric import QuantileBinTokenizer
from .text import TextTokenizer
from .timedelta import TimeDeltaTokenizer

_REGISTRY: Dict[str, Type[BaseTokenizer]] = {
    cls.__name__: cls
    for cls in (
        CategoricalHashTokenizer,
        FixedVocabTokenizer,
        FrequencyVocabTokenizer,
        HybridHashTokenizer,
        MappingTokenizer,
        QuantileBinTokenizer,
        TextTokenizer,
        TimeDeltaTokenizer,
    )
}


def register_tokenizer(cls: Type[BaseTokenizer]) -> Type[BaseTokenizer]:
    """Register a custom tokenizer step so saved pipelines can reload it."""
    _REGISTRY[cls.__name__] = cls
    return cls


def resolve_tokenizer(class_path: str) -> Type[BaseTokenizer]:
    """Resolve ``"pkg.module.ClassName"`` (or ``"ClassName"``) to a class."""
    name = class_path.rsplit(".", 1)[-1]
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown tokenizer step '{class_path}'. Register it with "
            f"register_tokenizer() before loading. Known steps: "
            f"{sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def known_tokenizers() -> Dict[str, Type[BaseTokenizer]]:
    return dict(_REGISTRY)

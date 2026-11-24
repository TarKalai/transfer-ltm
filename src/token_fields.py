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

"""Which field a token belongs to -- dependency-free, for the trainer and the tokenizer."""

from typing import Iterable, Optional


def field_of_token(token: str, prefixes: Iterable[str]) -> Optional[str]:
    """The field a ``<PREFIX>_<label>`` token belongs to: the longest matching prefix.

    Longest wins, so ``TDIFA_3`` is TDIFA (not TDIF) and a legacy ``CPTY_TYPE_3``
    is CPTY_TYPE (not CPTY). Specials such as ``<sep>`` have no field.
    """
    best = None
    for prefix in prefixes:
        if token.startswith(prefix + "_") and (best is None or len(prefix) > len(best)):
            best = prefix
    return best

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

"""Parsing of the ``finetune`` setting (torch-free, so experiments validate without it)."""

from typing import Optional, Tuple


def parse_finetune(spec: Optional[str]) -> Tuple[str, Optional[int]]:
    """``"top:2"`` -> ("top", 2); ``"full"`` -> ("full", None)."""
    spec = (spec or "none").strip().lower()
    mode, _, value = spec.partition(":")
    if mode in ("none", "full") and not value:
        return mode, None
    if mode in ("top", "lora") and value.isdigit() and int(value) > 0:
        return mode, int(value)
    raise ValueError(f"finetune={spec!r}; use none | top:N | lora:R | full")

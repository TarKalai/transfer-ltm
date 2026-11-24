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

"""Content fingerprints for cache keys.

A path says where an artifact is, not what it is: re-fitting the tokenizer or
retraining to the same checkpoint directory keeps the path and changes the
content, and a path-keyed cache then serves stale results silently. These
fingerprints change exactly when the content does.
"""

import hashlib
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def file_fingerprint(path: PathLike) -> str:
    """SHA-256 of the file's bytes (16 hex chars). For small files: states, configs."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def dir_fingerprint(path: PathLike) -> str:
    """Name, size and modification time of every file under *path* (16 hex chars).

    For checkpoints: hashing gigabytes of weights is slow, and a rewritten
    weight file always changes size or mtime.
    """
    root = Path(path)
    digest = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = f.stat()
        digest.update(f"{f.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:16]

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

"""Stable string hashing for hash-bucket tokenizer steps.

Why not ``cudf.Series.hash_values()``
-------------------------------------
``CategoricalHashTokenizer`` turns an integer hash into a token by taking
``hash % vocab_limit``, so **the hash function is part of the token's
identity**. ``CPTY_1183`` means "the counterparty whose hash lands in bucket
1183 under this particular hash function".

That made the token space depend on a library-internal implementation detail:

* ``cudf.Series.hash_values()`` uses MurmurHash3; ``pandas`` has no equivalent,
  and ``pandas.util.hash_pandas_object`` uses SipHash-2-4. Swapping backends
  silently reassigns every bucket.
* Either library is free to change its hash across versions. A checkpoint
  trained today would then be indexed by a different bucketing tomorrow, with
  no error raised and quietly meaningless embeddings.

So the hash lives here instead, pinned and owned by this project: BLAKE2b
truncated to 8 bytes, over the UTF-8 bytes of the *cleaned* string, with the
sign bit cleared. Identical on CPU and GPU, identical across library versions,
and one line to state in a paper.

Cost
----
Only the **distinct** values are hashed; the result is mapped back over the
column. Cost therefore scales with cardinality, not row count — a few hundred
thousand distinct counterparties hash in well under a second, against a
tokenizer fit that already takes seconds.

Changing ``HASH_ALGO``
----------------------
It renumbers every hash-bucket token, which invalidates any corpus and any
checkpoint trained on one. ``TokenizerPipeline`` records this value in its
state file and refuses to load a state written under a different one, so the
mismatch fails loudly instead of silently degrading results.
"""

import hashlib
from typing import Any

import pandas as pd

#: Identifier recorded in ``tokenizer_state.json``. Bump it whenever the
#: function below changes in any way that moves a value to a different bucket.
HASH_ALGO = "blake2b-8"

#: Clear the sign bit: the result is stored in an int64 column and then used
#: with ``%``, which would otherwise yield negative bucket indices.
_MASK = (1 << 63) - 1

_DIGEST_SIZE = 8


def hash_text(value: Any) -> int:
    """Hash one value to a non-negative int64.

    The scalar form of :func:`stable_hash`, exposed for tests and for hashing
    a single lookup key.
    """
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=_DIGEST_SIZE)
    return int.from_bytes(digest.digest(), "big") & _MASK


def stable_hash(series):
    """Hash a cleaned string column to non-negative int64 values.

    Parameters
    ----------
    series
        A cuDF or pandas Series of strings, already passed through
        ``_clean_str`` (uppercased, trimmed, nulls filled). Hashing the
        *cleaned* strings is what makes the bucket reproducible.

    Returns
    -------
    Series of the same type and index as the input, dtype ``int64``.

    Notes
    -----
    Distinct values are hashed once each and mapped back, so this is cheap even
    on tens of millions of rows.
    """
    # `_clean_str` fills nulls with "NA", but be defensive: a null reaching
    # str() would hash as the literal "nan"/"None" and differ across backends.
    filled = series.fillna("NA").astype("str")

    # Only the distinct values travel to the host and get hashed; the lookup is
    # mapped back where the column lives (on the GPU for cuDF).
    uniq = filled.unique()
    uniq = uniq.to_pandas() if hasattr(uniq, "to_pandas") else pd.Series(uniq)
    table = {value: hash_text(value) for value in uniq.astype("object")}
    return filled.map(table).astype("int64")

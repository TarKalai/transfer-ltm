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

"""Offline checks for src/tokenizer/_hash.py and the hash_algo state guard.

Runs on a machine with no GPU: pandas/numpy are real, the hash is pure Python,
and the state guard is exercised against a hand-built state dict. Requires
pandas -- run it in an environment that has it (the other machine, or any env
with pandas installed).

    python scripts/verify_stable_hash_offline.py
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# src/tokenizer/__init__.py imports every pipeline, which imports cuDF/CuPy.
# Stub them before the first import so this runs on a machine with no GPU.
# Only the attributes touched at module import time are needed.
if "cudf" not in sys.modules:
    stub_cudf = types.ModuleType("cudf")
    stub_cudf.DataFrame = type("DataFrame", (), {})
    stub_cudf.Series = type("Series", (), {})
    stub_cudf.concat = lambda *a, **k: None
    stub_cudf.to_datetime = lambda *a, **k: None
    sys.modules["cudf"] = stub_cudf
if "cupy" not in sys.modules:
    stub_cupy = types.ModuleType("cupy")
    stub_cupy.ndarray = type("ndarray", (), {})
    stub_cupy.cuda = types.SimpleNamespace(Stream=object)
    stub_cupy.log = lambda *a, **k: None
    stub_cupy.linspace = lambda *a, **k: None
    sys.modules["cupy"] = stub_cupy

import pandas as pd  # noqa: E402

from src.tokenizer._hash import HASH_ALGO, hash_text, stable_hash  # noqa: E402

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


# ------------------------------------------------------- 1. hash properties
print("\n1. hash_text properties")
MASK63 = (1 << 63) - 1
samples = ["BE68539007547034", "NA", "", "ABC-DEF", "0", "é", "X" * 200]
values = [hash_text(v) for v in samples]

check("all non-negative", all(v >= 0 for v in values))
check("all fit in int64", all(v <= MASK63 for v in values), f"max {max(values):x}")
check("deterministic across calls", values == [hash_text(v) for v in samples])
check("distinct inputs -> distinct outputs", len(set(values)) == len(samples))
check("sensitive to one-character change",
      hash_text("ACCOUNT-A") != hash_text("ACCOUNT-B"))
check("not order/whitespace insensitive",
      hash_text("A B") != hash_text("B A") != hash_text("AB"))
check("HASH_ALGO is a pinned identifier", HASH_ALGO == "blake2b-8", repr(HASH_ALGO))

# Frozen literals -- the actual regression tripwire. If either of these stops
# matching, the hash has changed, every hash-bucket token has been renumbered,
# and HASH_ALGO must be bumped (which invalidates existing corpora and
# checkpoints). Do not "fix" these values to match new output.
FROZEN = {
    "NA": 6190902687595129163,
    "BE68539007547034": 5784330264588460912,
    # `_clean_str` maps "" to "NA", so this never occurs in production; kept
    # only as a second point pinning the function.
    "": 7252660547403494068,
}
for text, expected in FROZEN.items():
    actual = hash_text(text)
    check(f"frozen reference {text!r}", actual == expected,
          f"got {actual}" if actual != expected else str(actual))
check("digest width is 8 bytes (all values < 2**64)",
      all(v < (1 << 64) for v in FROZEN.values()))

# ------------------------------------------------------ 2. stable_hash column
print("\n2. stable_hash over a pandas column")
s = pd.Series(["AAA", "BBB", "AAA", None, "CCC", "BBB"], name="cpty")
out = stable_hash(s)

check("dtype is int64", str(out.dtype) == "int64", str(out.dtype))
check("index preserved", out.index.equals(s.index))
check("length preserved", len(out) == len(s))
check("equal inputs -> equal hashes", out.iloc[0] == out.iloc[2])
check("distinct inputs -> distinct hashes", out.iloc[0] != out.iloc[1])
check("null becomes the 'NA' bucket", out.iloc[3] == hash_text("NA"),
      f"{out.iloc[3]} vs {hash_text('NA')}")
check("matches scalar hash_text elementwise",
      all(out.iloc[i] == hash_text(v) for i, v in enumerate(["AAA", "BBB", "AAA", "NA", "CCC", "BBB"])))
check("all non-negative (mod gives valid bucket)", bool((out >= 0).all()))

# uniques-only means cost is cardinality-bound, not row-bound
big = pd.Series(["V%d" % (i % 500) for i in range(200_000)])
big_out = stable_hash(big)
check("200k rows / 500 uniques hashes correctly",
      big_out.nunique() == 500 and len(big_out) == 200_000,
      f"{big_out.nunique()} uniques")

# bucketing, the way CategoricalHashTokenizer uses it
buckets = (out % 4096)
check("bucket indices in range", bool(((buckets >= 0) & (buckets < 4096)).all()))

# ------------------------------------------------ 3. hash_algo state guard
print("\n3. hash_algo state guard")


# Exercises the real production method, not a copy.
from src.tokenizer.pipeline import STATE_FORMAT, TokenizerPipeline  # noqa: E402


class HashingPipeline(TokenizerPipeline):
    HASH_ALGO = HASH_ALGO


def state(algo="__omit__"):
    st = {"format": STATE_FORMAT}
    if algo != "__omit__":
        st["hash_algo"] = algo
    return st


def guard_result(pipeline_cls, algo):
    inst = pipeline_cls.__new__(pipeline_cls)
    try:
        TokenizerPipeline._assert_hash_algo(inst, state(algo))
        return None
    except ValueError as exc:
        return str(exc)


check("matching algo passes", guard_result(HashingPipeline, HASH_ALGO) is None)

msg = guard_result(HashingPipeline, "cudf-hash_values")
check("different algo raises", msg is not None)
check("  message names both algos",
      msg is not None and "cudf-hash_values" in msg and HASH_ALGO in msg)
check("  message says how to recover",
      msg is not None and "regenerate" in msg.lower())

msg_missing = guard_result(HashingPipeline, "__omit__")
check("pre-guard state (no hash_algo key) raises", msg_missing is not None)
check("  message explains it predates the guard",
      msg_missing is not None and "predates" in msg_missing)

check("non-hashing pipeline accepts any state (vocab-only view)",
      guard_result(TokenizerPipeline, HASH_ALGO) is None
      and guard_result(TokenizerPipeline, "cudf-hash_values") is None
      and guard_result(TokenizerPipeline, "__omit__") is None)

# ------------------------------------------------- 4. declared HASH_ALGO values
print("\n4. declared HASH_ALGO per pipeline")
from src.tokenizer.bank_pipeline import BankTransferTokenizerPipeline  # noqa: E402

check("base TokenizerPipeline declares None", TokenizerPipeline.HASH_ALGO is None)
check("bank pipeline declares blake2b-8",
      BankTransferTokenizerPipeline.HASH_ALGO == "blake2b-8",
      repr(BankTransferTokenizerPipeline.HASH_ALGO))

# ------------------------------------------- 5. rename_map / normalize_name
print("\n5. BankSchema.rename_map -- bookkeeping columns must survive")
from src.tokenizer.bank_pipeline import BankSchema, normalize_name  # noqa: E402

check("normalize_name strips underscores (why the bug existed)",
      (normalize_name("__row_id__"), normalize_name("_target")) == ("row_id", "target"))

schema = BankSchema()
cols = [
    schema.customer_id, schema.account_number, schema.datetime, schema.amount,
    schema.counterparty_account, schema.counterparty_bic, schema.counterparty_country,
    schema.employee, schema.comment_structure, schema.operations_status,
    schema.movement_code, schema.movement_sign, schema.dc_flag,
    schema.month, schema.dow, schema.hour,
    "__row_id__", "_target",
]
mapping = schema.rename_map(cols)

# preprocess() promises to keep every original column; notebooks 04/05 look
# these two up by name, so a rename silently broke row alignment and labels.
check("__row_id__ is left alone", "__row_id__" not in mapping,
      f"-> {mapping.get('__row_id__')!r}")
check("_target is left alone", "_target" not in mapping,
      f"-> {mapping.get('_target')!r}")
check("canonical columns are untouched (already normalized)",
      all(c not in mapping for c in cols if c not in ("__row_id__", "_target")),
      str({k: v for k, v in mapping.items()}))

# The branch still has to do its job for messy source names.
messy = schema.rename_map(["Counterparty BIC", "DC Flag", "__row_id__", "_target"])
check("'Counterparty BIC' -> counterparty_bic",
      messy.get("Counterparty BIC") == "counterparty_bic", repr(messy.get("Counterparty BIC")))
check("'DC Flag' -> dc_flag", messy.get("DC Flag") == "dc_flag", repr(messy.get("DC Flag")))
check("bookkeeping columns still left alone in the messy case",
      "__row_id__" not in messy and "_target" not in messy)

alias = schema.rename_map(["cust_id", "__row_id__"])
check("alias 'cust_id' -> customer_id still resolves",
      alias.get("cust_id") == "customer_id", repr(alias.get("cust_id")))

# ------------------------------------ 6. field-position bucketing arithmetic
print("\n6. per-field-position loss bucketing")
T = 16
PERIOD, OFFSET = T + 1, 1


def field_of(position):
    """Mirror of accumulate_field_loss: field index of a target position."""
    return (position - OFFSET) % PERIOD


# Layout: <bos> [T fields] <sep> [T fields] <sep> ... [T fields] <eos>
check("position 1 is field 0", field_of(1) == 0)
check("position 16 is the last field", field_of(T) == T - 1)
check("position 17 is the <sep> slot", field_of(T + 1) == PERIOD - 1, str(field_of(T + 1)))
check("second transaction realigns", field_of(T + 2) == 0)
check("field indices stay in range",
      all(0 <= field_of(p) < PERIOD for p in range(1, 4096)))
check("each field appears once per period",
      sorted(field_of(p) for p in range(1, PERIOD + 1)) == list(range(PERIOD)))
check("<eos> lands in the <sep> slot",
      field_of(1 + 3 * PERIOD - 1) == PERIOD - 1)
check("period matches tokens_per_transaction + 1", PERIOD == T + 1)

print(f"\n{'=' * 60}\n{len(ok)} passed, {len(fail)} failed")
if fail:
    for name in fail:
        print(f"  FAILED: {name}")
raise SystemExit(1 if fail else 0)

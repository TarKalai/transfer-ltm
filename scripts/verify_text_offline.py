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

"""Offline checks for the text field and multi-token steps:

1. clean_text: lowercase, accents, digit words, short words.
2. Each text strategy: vocabulary rules, encoding, slots, state round-trip;
   readable=False keeps the words out of the state; custom strategies plug in.
3. The bank pipeline with a text field: token count, field names, corpus
   lines, history windows, save/load, and the error when the column is absent.

pandas stands in for cuDF and numpy for CuPy, as in verify_tier2_offline.py.

    python scripts/verify_text_offline.py
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ stubs
# cuDF -> pandas. The pipeline calls .to_pandas() on cuDF objects.
pd.Series.to_pandas = lambda self: self
pd.DataFrame.to_pandas = lambda self: self
cudf = types.ModuleType("cudf")
cudf.Series, cudf.DataFrame = pd.Series, pd.DataFrame
cudf.concat, cudf.to_datetime = pd.concat, pd.to_datetime
sys.modules["cudf"] = cudf

# CuPy -> numpy. ndarray is a distinct dummy type so code that calls .get() on
# CuPy arrays (state serialization) takes its numpy branch instead.
cupy = types.ModuleType("cupy")
for _name in dir(np):
    if not _name.startswith("_"):
        setattr(cupy, _name, getattr(np, _name))
cupy.ndarray = type("ndarray", (), {})
cupy.cuda = types.SimpleNamespace(
    Stream=object, null=types.SimpleNamespace(synchronize=lambda: None)
)
sys.modules["cupy"] = cupy

from src.tokenizer import (  # noqa: E402
    BankSchema, BankTokenizerConfig, BankTransferTokenizerPipeline,
    HistoryWindowEncoder, TextTokenizer, known_text_strategies, register_text_strategy,
)
from src.tokenizer.text import TextStrategy, clean_text  # noqa: E402

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc as e:  # noqa: F841
        return str(e)
    return None


# ------------------------------------------------------- 1. clean_text
print("1. clean_text")
cleaned = clean_text(pd.Series(["Loyer Février REF 2030-01 BE68539007547034", None, "a b  CAFÉ!"]))
check("lowercase, accents stripped, digit words dropped",
      cleaned[0] == ["loyer", "fevrier", "ref"], str(cleaned[0]))
check("missing text -> no words", cleaned[1] == [])
check("one-letter words dropped", cleaned[2] == ["cafe"], str(cleaned[2]))

# ------------------------------------------------------- 2. strategies
print("\n2. strategies")
# Distinct customers per word: payment 15, rent 13, gym 10, membership 10,
# monthly 9. "janssens" (a name) is used by one customer only, many times.
rows = []
for c in range(10):
    rows += [("Rent payment", f"K{c}"), ("Gym membership payment", f"K{c}")]
rows += [("Janssens payment", "K0")] * 20
rows += [("Monthly rent", f"K{c}") for c in list(range(6)) + [10, 11, 12]]
rows += [("Payment", f"K{c}") for c in range(10, 15)]
fit_df = pd.DataFrame(rows, columns=["txt", "cust"])
probe = pd.Series(["rent payment", "janssens gym rent extra", "", "unknownword"])

check("registered strategies",
      known_text_strategies() == ["bag", "category", "hashed", "words"],
      str(known_text_strategies()))
check("unknown strategy rejected", raises(lambda: TextTokenizer(strategy="nope")) is not None)

words = TextTokenizer(strategy="words", slots=3, min_customers=5, readable=True)
words.build_vocab(fit_df)
vocab_words = list(words._idx_to_token.values())
check("word used by 1 customer (a name) is not in the vocabulary",
      "TXT_JANSSENS" not in vocab_words and "TXT_RENT" in vocab_words, str(vocab_words))
check("vocabulary ranked by distinct customers", vocab_words[3] == "TXT_PAYMENT",
      str(vocab_words[:6]))
out = words.tokenize(probe)
check("output columns txt_0..txt_2", list(out.columns) == ["txt_0", "txt_1", "txt_2"])
check("words: order kept, unknown -> OTHER, padded",
      out.iloc[0].tolist() == ["TXT_RENT", "TXT_PAYMENT", "TXT_PAD"]
      and out.iloc[1].tolist() == ["TXT_OTHER", "TXT_GYM", "TXT_RENT"],
      str(out.values.tolist()[:2]))
check("empty text -> EMPTY then PAD", out.iloc[2].tolist() == ["TXT_EMPTY", "TXT_PAD", "TXT_PAD"])
check("index preserved", (out.index == probe.index).all())

bag = TextTokenizer(strategy="bag", slots=2, min_customers=5, readable=True)
bag.build_vocab(fit_df)
bout = bag.tokenize(probe)
check("bag: known words only, rarest first",
      bout.iloc[1].tolist() == ["TXT_GYM", "TXT_RENT"], str(bout.iloc[1].tolist()))
check("bag: text with no known word -> OTHER", bout.iloc[3].tolist() == ["TXT_OTHER", "TXT_PAD"])

cat = TextTokenizer(strategy="category", slots=7, min_customers=5, readable=True)
cat.build_vocab(fit_df)
cout = cat.tokenize(pd.Series(["RENT payment", "rent  payment!", "something else"]))
check("category forces one slot", cat.slots == 1 and list(cout.columns) == ["txt_0"])
check("category: same cleaned string -> same token; unseen -> OTHER",
      cout["txt_0"].tolist() == ["TXT_RENT-PAYMENT"] * 2 + ["TXT_OTHER"],
      str(cout["txt_0"].tolist()))

hashed = TextTokenizer(strategy="hashed", slots=2, hash_size=16)
hashed.build_vocab(fit_df)
hout = hashed.tokenize(pd.Series(["zzz never seen", "zzz"]))
check("hashed: vocab = PAD/EMPTY/OTHER + buckets", hashed.vocab_size == 3 + 16)
check("hashed: unseen word still gets a stable bucket",
      hout.iloc[0, 0] == hout.iloc[1, 0] and hout.iloc[0, 0].startswith("TXT_H"))

private = TextTokenizer(strategy="words", slots=3, min_customers=5, readable=False)
private.build_vocab(fit_df)
state = json.loads(json.dumps(private.get_state()))
blob = json.dumps(state).lower()
check("readable=False: no word in the state file",
      not any(w in blob for w in ("rent", "payment", "gym")))
restored = TextTokenizer.from_state(state)
check("state round-trip reproduces the tokens",
      restored.tokenize(probe).equals(private.tokenize(probe)))


@register_text_strategy
class FirstWordOnly(TextStrategy):
    name = "first_word"
    fixed_slots = 1

    def encode(self, words):
        return self._pad([self.lookup(words[0]) or "OTHER"] if words else [], words)


custom = TextTokenizer(strategy="first_word", min_customers=5, readable=True)
custom.build_vocab(fit_df)
check("a new strategy plugs in without touching anything else",
      custom.tokenize(pd.Series(["gym rent"]))["txt_0"].tolist() == ["TXT_GYM"])

# ------------------------------------------------- 3. pipeline integration
print("\n3. BankTransferTokenizerPipeline with a text field")
rng = np.random.default_rng(0)
n = 400
raw = pd.DataFrame({
    "customer_id": rng.choice([f"K{i}" for i in range(8)], n),
    "account_number": rng.choice(["A1", "A2", "A3"], n),
    "datetime": pd.Timestamp("2023-01-01")
    + pd.to_timedelta(rng.integers(0, 90 * 86400, n), unit="s"),
    "counterparty_account": rng.choice([f"BE{i:04d}" for i in range(30)], n),
    "counterparty_bic": rng.choice(["BANKAAXX", "BANKBBXX"], n),
    "counterparty_country": rng.choice(["BE", "NL"], n),
    "employee": rng.choice(["e1", "e2"], n),
    "comment_structure": rng.integers(0, 3, n),
    "operations_status": rng.choice(["OK", "KO"], n),
    "movement_code": rng.integers(100, 110, n),
    "movement_sign": rng.choice(["+", "-"], n),
    "dc_flag": rng.choice(["D", "C"], n),
    "amt_euro": rng.lognormal(4, 1, n).round(2),
    "Description": rng.choice(["Rent May", "Groceries Delhaize", "Salary", None, "Gym 42"], n),
})
base_cfg = dict(include_account_time_delta=True, counterparty_hash_size=64,
                counterparty_min_customers=3, drop_fields=("EMP",))
schema = BankSchema(description="description")

plain = BankTransferTokenizerPipeline(schema=schema, config=BankTokenizerConfig(**base_cfg))
pproc = plain.preprocess(raw.copy())
plain.fit(pproc)

pipe = BankTransferTokenizerPipeline(schema=schema, config=BankTokenizerConfig(
    **base_cfg, text_strategy="words", text_slots=3, text_min_customers=2))
proc = pipe.preprocess(raw.copy())
check("description column found after name normalisation", "txt_raw" in proc.columns)
pipe.fit(proc)
toks = pipe.transform(proc)
T = pipe.tokens_per_transaction
check("tokens per transaction = without text + slots", T == plain.tokens_per_transaction + 3,
      f"{plain.tokens_per_transaction} -> {T}")
start = pipe.field_names.index("txt_0")
check("field_names list each slot, matching the transform's columns",
      pipe.field_names == list(toks.columns)
      and pipe.field_names[start:start + 3] == ["txt_0", "txt_1", "txt_2"],
      " ".join(pipe.field_names))
check("TXT sits after CMT and before CTRY",
      pipe.field_names.index("cmt_val") < start < pipe.field_names.index("ctry_clean"))
check("no token ID collisions", not pipe.id_collisions())
check("every emitted token is in the vocabulary",
      set(np.unique(toks.values.astype(str))) <= set(pipe.vocab))
check("chunk size accounts for the extra slots",
      pipe.recommended_chunk_size(4096) == 4094 // (T + 1))

lines = pipe.to_corpus_lines(toks, proc, pipe.group_cols, chunk_size=50)
per_txn = {
    len(seg.split())
    for line in lines
    for seg in line.replace("<bos>", "").replace("<eos>", "").split("<sep>")
}
check("every transaction in the corpus has T tokens", per_txn == {T}, str(per_txn))

enc = HistoryWindowEncoder.from_token_frame(toks, proc, pipe.group_cols, pipe.vocab, window=4)
check("history windows are T tokens per transaction wide", enc.tokens_per_txn == T)

state = json.loads(json.dumps(pipe.get_state()))
check("state records field_names and field_vocab",
      state["field_names"] == pipe.field_names and state["field_vocab"]["txt_1"] > 3)
re_pipe = BankTransferTokenizerPipeline.from_state(state)
check("state round-trip reproduces the tokens",
      re_pipe.transform(re_pipe.preprocess(raw.copy())).equals(toks))
check("a state without the text options loads with no text field",
      BankTransferTokenizerPipeline.from_state(json.loads(json.dumps(plain.get_state())))
      .config.text_strategy is None)

no_desc = BankTransferTokenizerPipeline(config=BankTokenizerConfig(text_strategy="words"))
msg = raises(lambda: no_desc.preprocess(raw.drop(columns=["Description"])))
check("text_strategy without a description column fails loudly",
      msg is not None and "description" in msg)

# ------------------------------------------------------------------ summary
print("\n" + "=" * 60)
print(f"{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:")
    for name in fail:
        print("  -", name)
    sys.exit(1)

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

"""Offline checks for tail strategies and kind-declared extra fields:

1. FrequencyVocabTokenizer tails: "other" unchanged; "rare_bins" and "hash"
   map unseen values onto tokens that training emitted; state round-trip.
2. The bank pipeline: default tail, legacy states, and the key property --
   no token emitted on new data is one the model never saw in training.
3. extra_fields: each kind, placement before AMT, drop_fields, validation,
   save/load, suggest_kind.

pandas stands in for cuDF and numpy for CuPy, as in verify_tier2_offline.py.

    python scripts/verify_fields_offline.py
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
    BankTokenizerConfig, BankTransferTokenizerPipeline, FrequencyVocabTokenizer,
)
from src.tokenizer.field_kinds import suggest_kind  # noqa: E402
from src.tokenizer.pipeline import _int_keys  # noqa: E402  (what the pipeline's load does)

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


# ------------------------------------------------------- 1. tail strategies
print("1. FrequencyVocabTokenizer tails")
train = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 2 + ["D"])
new = pd.Series(["A", "C", "D", "E", "F"])          # E, F never seen in train

legacy = FrequencyVocabTokenizer(prefix="X", max_vocab=8)
legacy.build_vocab(train)
check("'other' (default): vocabulary = head + OTHER last",
      list(legacy._idx_to_token.values()) == ["X_A", "X_B", "X_C", "X_D", "X_OTHER"])
check("'other': unseen -> OTHER", legacy.tokenize(new).tolist()[-2:] == ["X_OTHER"] * 2)
check("'other' with caps above cardinality: OTHER never emitted in train",
      "X_OTHER" not in set(legacy.tokenize(train)))

bins = FrequencyVocabTokenizer(prefix="X", max_vocab=8, min_count=3, tail="rare_bins")
bins.build_vocab(train)
check("rare_bins: head is values seen >= min_count", bins._values == ["A", "B"])
out = bins.tokenize(new).tolist()
check("rare_bins: tail value -> its band, unseen -> rarest band",
      out == ["X_A", "X_RARE_2-4", "X_RARE_1", "X_RARE_1", "X_RARE_1"], str(out))
check("rare_bins: every token on new data was emitted in train",
      set(out) <= set(bins.tokenize(train)))
state = json.loads(json.dumps(bins.get_state()))
check("rare_bins: state round-trip",
      FrequencyVocabTokenizer.from_state(state).tokenize(new).tolist() == out)

hashed = FrequencyVocabTokenizer(prefix="X", max_vocab=8, min_count=3, tail="hash",
                                 tail_buckets=4)
hashed.build_vocab(train)
hout = hashed.tokenize(new).tolist()
check("hash: tail and unseen values -> TH buckets",
      hout[0] == "X_A" and all(t.startswith("X_TH") for t in hout[1:]), str(hout))
check("hash: same value, same bucket across calls",
      hashed.tokenize(pd.Series(["E"])).tolist() == [hout[3]])
check("hash: vocabulary = head + buckets", hashed.vocab_size == 2 + 4)
wide = FrequencyVocabTokenizer(prefix="X", max_vocab=8, min_count=3, tail="hash",
                               tail_buckets=64)
wide.build_vocab(train)
trained = set(wide.tokenize(train))
check("hash: unseen values only land in buckets that training filled",
      set(wide.tokenize(pd.Series([f"new{i}" for i in range(200)]))) <= trained)
empty = FrequencyVocabTokenizer(prefix="X", max_vocab=8, min_count=99, tail="hash")
empty.build_vocab(train)
out_empty = empty.tokenize(pd.Series([], dtype=str))
check("empty input -> empty Series (not None)", out_empty is not None and len(out_empty) == 0)
clash = FrequencyVocabTokenizer(prefix="X", max_vocab=8, tail="hash", tail_buckets=4)
clash.build_vocab(pd.Series(["TH3"] * 5 + ["A"] * 5 + ["Z"]))
toks = list(clash._idx_to_token.values())
check("a head value named like a tail token gets a distinct token",
      len(toks) == len(set(toks)) and "X_TH3" in toks and "X_TH3-2" in toks, str(toks))
clash_state = json.loads(json.dumps(clash.get_state()))
reloaded = FrequencyVocabTokenizer.from_state(_int_keys(clash_state))
check("tokens are saved with the state and reloaded exactly",
      list(reloaded._idx_to_token.values()) == toks
      and reloaded.tokenize(pd.Series(["TH3", "A"])).tolist()
      == clash.tokenize(pd.Series(["TH3", "A"])).tolist())
head_idx = clash._values.index("TH3")
clash_state["vocab_state"]["_idx_to_token"][str(head_idx)] = "X_TH3-OLD"   # an older rule
old_style = FrequencyVocabTokenizer.from_state(_int_keys(clash_state))
check("reload takes the saved tokens verbatim, whatever the rules have become",
      old_style._value_to_token["TH3"] == "X_TH3-OLD")
del clash_state["vocab_state"]["_idx_to_token"][str(head_idx)]
check("a saved state missing some head tokens fails loudly instead of re-deriving",
      "Re-fit" in (raises(lambda: FrequencyVocabTokenizer.from_state(_int_keys(clash_state))) or ""))
check("unknown tail rejected",
      raises(lambda: FrequencyVocabTokenizer(prefix="X", tail="nope")) is not None)

# ------------------------------------------------- 2. pipeline defaults
print("\n2. bank pipeline")
rng = np.random.default_rng(0)
n = 600


def frame(bics):
    return pd.DataFrame({
        "customer_id": rng.choice([f"K{i}" for i in range(12)], n),
        "account_number": rng.choice(["A1", "A2"], n),
        "datetime": pd.Timestamp("2023-01-01")
        + pd.to_timedelta(rng.integers(0, 90 * 86400, n), unit="s"),
        "counterparty_account": rng.choice([f"BE{i:04d}" for i in range(40)], n),
        "counterparty_bic": rng.choice(bics, n),
        "counterparty_country": rng.choice(["BE", "NL", "FR"], n, p=[0.8, 0.15, 0.05]),
        "employee": rng.choice(["e1", "e2"], n),
        "comment_structure": rng.integers(0, 3, n),
        "operations_status": rng.choice(["OK", "KO"], n),
        "movement_code": rng.integers(100, 110, n),
        "movement_sign": rng.choice(["+", "-"], n),
        "dc_flag": rng.choice(["D", "C"], n),
        "amt_euro": rng.lognormal(4, 1, n).round(2),
        "channel": rng.choice(["WEB", "APP", "BRANCH"], n),
        "fee_eur": rng.exponential(2, n).round(2),
        "device_id": rng.choice([f"dev{i}" for i in range(300)], n),
        "note": rng.choice(["rent", "gift for mum", "", "loan repayment"], n),
    })


# Train: two common BICs plus many rare ones; new data: BICs never seen.
train_raw = frame(["BANKAAXX", "BANKBBXX"] * 20 + [f"RARE{i}" for i in range(30)])
new_raw = frame(["BANKAAXX", "UNSEEN1", "UNSEEN2"])

pipe = BankTransferTokenizerPipeline(config=BankTokenizerConfig(counterparty_hash_size=64))
check("default tail is rare_bins, min_count 10",
      pipe.config.tail_strategy == "rare_bins" and pipe.config.tail_min_count == 10)
proc = pipe.preprocess(train_raw.copy())
pipe.fit(proc)
train_toks = pipe.transform(proc)
new_toks = pipe.transform(pipe.preprocess(new_raw.copy()))
# Frequency-vocabulary fields only: a time-gap bin absent from this tiny
# train sample is not a tail question.
freq_cols = [c for c in pipe.tokenizer_order
             if type(pipe.steps[c]).__name__ == "FrequencyVocabTokenizer"]
seen = set(np.unique(train_toks[freq_cols].values.astype(str)))
never = sorted(set(np.unique(new_toks[freq_cols].values.astype(str))) - seen)
check("no categorical token on new data that training never emitted", not never,
      str(never[:5]))
check("the rarest band is empty here, so unseen BICs go to the rarest occupied one",
      "BIC_RARE_1" not in seen and set(new_toks["bic_clean"]) <= seen)

old = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
    counterparty_hash_size=64, **BankTokenizerConfig.LEGACY_DEFAULTS))
oproc = old.preprocess(train_raw.copy())
old.fit(oproc)
o_new = old.transform(old.preprocess(new_raw.copy()))
check("legacy config: unseen BIC -> BIC_OTHER, never seen in training",
      "BIC_OTHER" in set(o_new["bic_clean"])
      and "BIC_OTHER" not in set(old.transform(oproc)["bic_clean"]))
ostate = json.loads(json.dumps(old.get_state()))
for key in ("tail_strategy", "tail_min_count"):
    ostate["metadata"]["config"].pop(key)
check("a state from before tail strategies reloads with OTHER",
      BankTransferTokenizerPipeline.from_state(ostate).config.tail_strategy == "other")

cfg_obj = BankTokenizerConfig(counterparty_hash_size=64)
BankTransferTokenizerPipeline(config=cfg_obj)                  # a preview, as in notebook 02
cfg_obj.drop_fields = ("emp",)                                 # edited afterwards
check("a config edited after a first pipeline is normalised again (emp -> EMP dropped)",
      "emp_clean" not in BankTransferTokenizerPipeline(config=cfg_obj).tokenizer_order)
vocab_dir = Path(__import__("tempfile").mkdtemp())
pipe.save_vocab(vocab_dir / "vocab.json")
saved_vocab = json.loads((vocab_dir / "vocab.json").read_text())
check("vocab.json records every field's token prefix",
      saved_vocab["field_prefixes"] == pipe.field_prefixes() and "AMT" in saved_vocab["field_prefixes"])
from src.token_fields import field_of_token  # noqa: E402
check("field of a token: longest prefix (TDIFA_3 -> TDIFA, CPTY_TYPE_3 -> CPTY_TYPE)",
      field_of_token("TDIFA_3", ["TDIF", "TDIFA"]) == "TDIFA"
      and field_of_token("CPTY_TYPE_3", ["CPTY", "CPTY_TYPE"]) == "CPTY_TYPE"
      and field_of_token("<sep>", ["CPTY"]) is None)

# ------------------------------------------------- 3. extra fields
print("\n3. extra_fields")
extra = {
    "CHAN": {"column": "channel", "kind": "categorical"},
    "FEE": {"column": "fee_eur", "kind": "numeric", "bins": 4},
    "DEV": {"column": "device_id", "kind": "identifier", "hash_size": 32, "top_k": 4,
            "min_customers": 2},
    "NOTE": {"column": "note", "kind": "text", "slots": 2, "min_customers": 2},
}
xp = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
    counterparty_hash_size=64, extra_fields=extra))
xproc = xp.preprocess(train_raw.copy())
xp.fit(xproc)
xtoks = xp.transform(xproc)
check("one token per field, text adds its slots",
      xp.tokens_per_transaction == pipe.tokens_per_transaction + 1 + 1 + 1 + 2,
      f"{pipe.tokens_per_transaction} -> {xp.tokens_per_transaction}")
names = xp.field_names
check("extra fields placed just before AMT, AMT still last",
      names[-1] == "amt_val"
      and names[-6:-1] == ["x_chan", "x_fee", "x_dev_customer_id", "note_0", "note_1"],
      " ".join(names[-7:]))
check("tokens carry the field prefix",
      xtoks["x_chan"].str.startswith("CHAN_").all()
      and xtoks["x_fee"].str.startswith("FEE_").all()
      and xtoks["x_dev_customer_id"].str.startswith("DEV_").all())
check("no token ID collisions", not xp.id_collisions())
re_xp = BankTransferTokenizerPipeline.from_state(json.loads(json.dumps(xp.get_state())))
check("state round-trip reproduces the tokens",
      re_xp.transform(re_xp.preprocess(train_raw.copy())).equals(xtoks))

dropped = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
    counterparty_hash_size=64, extra_fields=extra, drop_fields=("CHAN",)))
check("drop_fields works on extra fields", "x_chan" not in dropped.tokenizer_order)
check("unknown kind rejected", raises(lambda: BankTokenizerConfig(
    extra_fields={"Z": {"column": "c", "kind": "nope"}}).normalized()) is not None)
check("unknown option rejected", "bins" in (raises(lambda: BankTokenizerConfig(
    extra_fields={"Z": {"column": "c", "kind": "categorical", "bins": 3}}).normalized()) or ""))
def fit_with_extra(name):
    p = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
        counterparty_hash_size=64, extra_fields={name: {"column": "channel", "kind": "categorical"}}))
    return p.fit(p.preprocess(train_raw.copy()))


check("non-ASCII extra field names rejected when fitting",
      raises(lambda: fit_with_extra("CANAL\u00c9")) is not None)
check("lower-case extra field names are upper-cased (tokens and weights agree)",
      "CPTYTYPE" in BankTokenizerConfig(
          extra_fields={"cptytype": {"column": "c", "kind": "categorical"}}).normalized()
      .extra_fields)
legacy_named = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
    counterparty_hash_size=64, extra_fields={"CHAN": {"column": "channel", "kind": "categorical"}}))
legacy_named.fit(legacy_named.preprocess(train_raw.copy()))
legacy_state = json.loads(json.dumps(legacy_named.get_state()))
legacy_state["metadata"]["config"]["extra_fields"] = {
    "CHAN_X": legacy_state["metadata"]["config"]["extra_fields"]["CHAN"]}
check("a saved state is loaded as fitted, even if a rule for new configs has tightened",
      raises(lambda: BankTransferTokenizerPipeline.from_state(legacy_state)) is None
      or "letters and digits" not in (raises(lambda: BankTransferTokenizerPipeline.from_state(
          legacy_state)) or ""))
check("extra field names with '_' rejected when fitting (token prefixes must be unambiguous)",
      "letters and digits" in (raises(lambda: fit_with_extra("CPTY_TYPE")) or ""))
check("name clash with a built-in rejected", raises(lambda: BankTokenizerConfig(
    extra_fields={"AMT": {"column": "c", "kind": "numeric"}}).normalized()) is not None)
missing = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
    extra_fields={"Z": {"column": "absent", "kind": "categorical"}}))
check("missing column fails loudly",
      "absent" in (raises(lambda: missing.preprocess(train_raw.copy())) or ""))

check("suggest_kind: strings, few values -> categorical",
      suggest_kind(train_raw["channel"]).startswith("categorical"))
check("suggest_kind: many-valued floats -> numeric",
      suggest_kind(train_raw["fee_eur"]).startswith("numeric"))
check("suggest_kind: several words -> text",
      suggest_kind(pd.Series(["gift for mum", "loan repayment x"] * 50)).startswith("text"))
check("suggest_kind: thousands of IDs -> identifier",
      suggest_kind(pd.Series([f"id{i}" for i in range(6000)])).startswith("identifier"))
check("suggest_kind: small integer codes -> categorical",
      suggest_kind(pd.Series(np.arange(600) % 7)).startswith("categorical"))

# ------------------------------------------------- 4. relative amount
print("\n4. relative amount (AMTREL)")
from src.tokenizer import QuantileBinTokenizer  # noqa: E402

q = QuantileBinTokenizer(prefix="Q", num_bins=4, missing_label="FIRST")
q.build_vocab(pd.Series([1.0, 2.0, 3.0, 4.0, None]))
qt = q.tokenize(pd.Series([None, 1.0, 4.0])).tolist()
check("missing_label: missing values get their own token, numbers are binned",
      qt[0] == "Q_FIRST" and qt[1] != "Q_FIRST" and q.vocab_size == 5, str(qt))
check("missing_label: state round-trip",
      QuantileBinTokenizer.from_state(_int_keys(json.loads(json.dumps(q.get_state()))))
      .tokenize(pd.Series([None, 1.0, 4.0])).tolist() == qt)
legacy_q = QuantileBinTokenizer(prefix="Q", num_bins=4)
legacy_q.build_vocab(pd.Series([1.0, 2.0, 3.0, 4.0]))
check("default (no missing_label): missing binned like 0, as before",
      legacy_q.tokenize(pd.Series([None, 0.0])).tolist()[0]
      == legacy_q.tokenize(pd.Series([0.0])).tolist()[0] and legacy_q.vocab_size == 4)

seq = train_raw.copy()
seq.loc[:, "customer_id"] = "K0"
seq.loc[:2, "customer_id"] = "SOLO"
seq.loc[:2, "amt_euro"] = [100.0, 100.0, 1000.0]
seq.loc[:2, "datetime"] = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
rp = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
    counterparty_hash_size=64, include_relative_amount=True))
rproc = rp.preprocess(seq)
solo = rproc[rproc["customer_id"] == "SOLO"]
expect = np.log1p(1000.0) - np.log1p(100.0)
check("relative amount: first transfer has none; later = log-amount minus mean of earlier",
      np.isnan(solo["amt_rel"].iloc[0]) and abs(solo["amt_rel"].iloc[1]) < 1e-12
      and abs(solo["amt_rel"].iloc[2] - expect) < 1e-9, str(solo["amt_rel"].tolist()))
rp.fit(rproc)
rtoks = rp.transform(rproc)
check("AMTREL token: FIRST for no history, placed just before AMT",
      rtoks.loc[solo.index[0], "amt_rel"] == "AMTREL_FIRST"
      and rp.field_names[-2:] == ["amt_rel", "amt_val"])
later = seq.copy()
later.loc[:2, "amt_euro"] = [100.0, 100.0, 5.0]       # change only the last amount
check("causal: changing a later amount leaves earlier values unchanged",
      rp.preprocess(later).query("customer_id == 'SOLO'")["amt_rel"].iloc[:2].fillna(-9).tolist()
      == solo["amt_rel"].iloc[:2].fillna(-9).tolist())
check("off by default", "amt_rel" not in BankTransferTokenizerPipeline(
    config=BankTokenizerConfig()).tokenizer_order)

# ------------------------------------------------- 5. overlapping chunks
print("\n5. overlapping pretraining chunks (chunk_copies)")
from src.tokenizer.pipeline import TokenizerPipeline  # noqa: E402

tok_df = pd.DataFrame({"f": [f"T{i}" for i in range(10)] + ["U0", "U1"]})
meta = pd.DataFrame({"c": ["A"] * 10 + ["B"] * 2})
plain_lines = TokenizerPipeline().to_corpus_lines(tok_df, meta, ["c"], chunk_size=4)
lines = TokenizerPipeline().to_corpus_lines(tok_df, meta, ["c"], chunk_size=4, chunk_copies=2)


def txns(line):
    return line.replace("<bos> ", "").replace(" <eos>", "").split(" <sep> ")


check("chunk_copies=1: unchanged consecutive chunks",
      [txns(l) for l in plain_lines] == [["T0", "T1", "T2", "T3"], ["T4", "T5", "T6", "T7"],
                                         ["T8", "T9"], ["U0", "U1"]])
got = [txns(l) for l in lines]
check("chunk_copies=2: a chunk every half-chunk, in time order, redundant tail dropped",
      got == [["T0", "T1", "T2", "T3"], ["T2", "T3", "T4", "T5"], ["T4", "T5", "T6", "T7"],
              ["T6", "T7", "T8", "T9"], ["U0", "U1"]], str(got))
from collections import Counter  # noqa: E402
counts = Counter(t for g in got for t in g)
check("every transaction except the first/last stretch appears twice",
      all(counts[f"T{i}"] == 2 for i in range(2, 8)) and counts["U0"] == 1)

odd = pd.DataFrame({"f": [f"T{i}" for i in range(23)]})
odd_lines = TokenizerPipeline().to_corpus_lines(odd, pd.DataFrame({"c": ["A"] * 23}), ["c"],
                                                chunk_size=5, chunk_copies=2)
odd_counts = Counter(t for line in odd_lines for t in txns(line))
check("chunk size not divisible by copies: exactly 2 copies except the first and the last "
      "block of the sequence; chunks never longer than chunk_size",
      all(odd_counts[f"T{i}"] == 2 for i in range(2, 22))
      and odd_counts["T0"] == odd_counts["T1"] == odd_counts["T22"] == 1
      and max(len(txns(line)) for line in odd_lines) <= 5, str(dict(odd_counts)))

check("chunk_copies larger than the chunk is refused (would overrun the context)",
      raises(lambda: TokenizerPipeline().to_corpus_lines(
          odd, pd.DataFrame({"c": ["A"] * 23}), ["c"], chunk_size=3, chunk_copies=4)) is not None)

print("\n6. hash guard with legacy states")
bank = object.__new__(BankTransferTokenizerPipeline)
check("an unkeyed bank state is refused by the bank pipeline (hash unknown)",
      raises(lambda: bank._assert_hash_algo({"pipeline_class": "BankTransferTokenizerPipeline"}))
      is not None)

# ------------------------------------------------------------------ summary
print("\n" + "=" * 60)
print(f"{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:")
    for name in fail:
        print("  -", name)
    sys.exit(1)

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

"""Offline checks for the tokenizer/training changes of the "tier 2" round:

1. TimeDeltaTokenizer: zero-based bins (TDIF_0 reachable, horizon not merged
   into the last real bin) and TDIF_FIRST for a series' first transaction.
2. HybridHashTokenizer: counterparties shared by many *customers* get their own
   token; everything else keeps the old CPTY_<hash % size> token.
3. BankTokenizerConfig.field_order: emission order, validation.
4. Old tokenizer states reload with the old behaviour (LEGACY_DEFAULTS).
5. <sep> excluded from the loss but still attended to.
6. Embedding extraction skips the LM head.

No GPU needed: the real tokenizer code runs with pandas standing in for cuDF
and numpy for CuPy (the APIs used here coincide), and torch is stubbed. This
checks the logic, not cuDF itself -- notebook 02 on the run machine is still
the end-to-end test.

    python scripts/verify_tier2_offline.py
"""

import json
import sys
import tempfile
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

# torch: only what clm_data / train_decoder / decoder_inference touch at import.
torch = types.ModuleType("torch")
torch.from_numpy = lambda a: a
torch.Tensor = np.ndarray
torch.no_grad = lambda: (lambda fn: fn)
torch.device = lambda *a, **k: "cpu"
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.bfloat16 = "bfloat16"
torch.manual_seed = lambda s: None
_data = types.ModuleType("torch.utils.data")
_data.Dataset, _data.DataLoader, _data.DistributedSampler = object, object, object
torch.utils = types.SimpleNamespace(data=_data)
torch.optim = types.SimpleNamespace(AdamW=object)
torch.nn = types.SimpleNamespace(Module=object)
sys.modules["torch"] = torch
sys.modules["torch.utils"] = torch.utils
sys.modules["torch.utils.data"] = _data

from src.tokenizer import (  # noqa: E402
    BankSchema, BankTokenizerConfig, BankTransferTokenizerPipeline,
    CategoricalHashTokenizer, HybridHashTokenizer, TimeDeltaTokenizer,
)
from src.tokenizer._hash import stable_hash  # noqa: E402
from src.tokenizer.bank_pipeline import DEFAULT_FIELD_ORDER  # noqa: E402
from src.tokenizer.pipeline import _int_keys  # noqa: E402

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


# ------------------------------------------------- 1. TimeDeltaTokenizer
print("\n1. TimeDeltaTokenizer")
N, YEARS = 48, 2.5
legacy = TimeDeltaTokenizer(num_bins=N, max_years=YEARS)
legacy.build_vocab()
fixed = TimeDeltaTokenizer(num_bins=N, max_years=YEARS, zero_based_bins=True, first_token=True)
fixed.build_vocab()
H = fixed.max_horizon

gaps = pd.Series([0.0, 1.0, 60.0, 3600.0, 86400.0, H / 2, float(H), H * 10.0])
old_ids = np.clip(
    np.digitize(np.log(np.clip(gaps.values, 0, H) + 1.0), np.linspace(0, np.log(H + 1.0), N + 1)),
    0, N - 1,
)
check("legacy mode reproduces the original formula exactly",
      legacy.tokenize(gaps).tolist() == [f"TDIF_{i}" for i in old_ids])
check("legacy vocab has no FIRST token", legacy.vocab_size == N, str(legacy.vocab_size))

dense = pd.Series(np.unique(np.round(np.geomspace(1, H, 400_000))).tolist() + [0.0])
reach_old = set(legacy.tokenize(dense))
reach_new = set(fixed.tokenize(dense))
check("legacy: TDIF_0 is unreachable (the bug)", "TDIF_0" not in reach_old)
check("fixed: all bins reachable", reach_new == {f"TDIF_{i}" for i in range(N)},
      f"{len(reach_new)}/{N}")
new = fixed.tokenize(gaps).tolist()
check("fixed: zero gap -> TDIF_0", new[0] == "TDIF_0", new[0])
check("fixed: each bin is the old one shifted down by one (below the horizon)",
      all(int(a.split('_')[1]) == b - 1 for a, b in zip(new[1:6], old_ids[1:6])))
check("fixed: gap at the horizon -> last bin", new[6] == f"TDIF_{N - 1}", new[6])
# One gap in each of the top two log intervals [e46, e47) and [e47, e48).
edges = np.linspace(0, np.log(H + 1.0), N + 1)
top_two = pd.Series([np.exp((edges[N - 2] + edges[N - 1]) / 2) - 1,
                     np.exp((edges[N - 1] + edges[N]) / 2) - 1])
lt, ft = legacy.tokenize(top_two).tolist(), fixed.tokenize(top_two).tolist()
check("legacy: the top two intervals share one token (the bug)", lt[0] == lt[1], str(lt))
check("fixed: the top two intervals get their own tokens",
      ft == [f"TDIF_{N - 2}", f"TDIF_{N - 1}"], str(ft))
first = fixed.tokenize(pd.Series([np.nan, 0.0, 5.0]))
check("fixed: missing gap -> TDIF_FIRST", first.iloc[0] == "TDIF_FIRST", first.iloc[0])
check("fixed: FIRST differs from a true zero gap", first.iloc[0] != first.iloc[1])
check("fixed: vocab = bins + FIRST", fixed.vocab_size == N + 1, str(fixed.vocab_size))
check("legacy mode fills a missing gap as 0 (original behaviour)",
      legacy.tokenize(pd.Series([np.nan])).iloc[0] == legacy.tokenize(pd.Series([0.0])).iloc[0])

def json_state(step):
    """A step's state after a JSON round-trip, keys restored as the pipeline does."""
    s = json.loads(json.dumps(step.get_state(), default=lambda o: np.asarray(o).tolist()))
    s["vocab_state"]["_idx_to_token"] = _int_keys(s["vocab_state"]["_idx_to_token"])
    return s


state = json_state(fixed)
back = TimeDeltaTokenizer.from_state(state)
check("state round-trip keeps both options",
      back.zero_based_bins and back.first_token
      and back.tokenize(gaps).tolist() == new)
old_state = json_state(legacy)
for k in ("zero_based_bins", "first_token"):
    old_state["init_params"].pop(k)
check("a state without the new keys loads in legacy mode",
      TimeDeltaTokenizer.from_state(old_state).tokenize(gaps).tolist()
      == legacy.tokenize(gaps).tolist())

# ------------------------------------------------- 2. HybridHashTokenizer
print("\n2. HybridHashTokenizer")
rows = []
for c in range(15):                       # shared by 15 customers, once each
    rows.append(("BE00SHARED", f"C{c}"))
for _ in range(500):                      # one customer, 500 times
    rows.append(("BE00HEAVY", "C0"))
for c in range(12):                       # 12 customers -- also qualifies
    rows.append(("BE00UTILITY", f"C{c}"))
for c in range(9):                        # 9 customers -- below the floor
    rows.append(("BE00NINE", f"C{c}"))
for i in range(300):                      # long tail of private accounts
    rows.append((f"BE{i:06d}", f"C{i % 40}"))
frame = pd.DataFrame(rows, columns=["cp", "cust"])
frame["h"] = stable_hash(frame["cp"])
V = 64

hyb = HybridHashTokenizer(vocab_limit=V, top_k=8, min_customers=10)
hyb.build_vocab(frame[["h", "cust"]])
tok = hyb.tokenize(frame[["h", "cust"]])
by_cp = dict(zip(frame["cp"], tok))
check("counterparty of 15 customers gets a dedicated token",
      by_cp["BE00SHARED"].startswith("CPTY_T"), by_cp["BE00SHARED"])
check("ranked by distinct customers: SHARED (15) before UTILITY (12)",
      by_cp["BE00SHARED"] == "CPTY_T0" and by_cp["BE00UTILITY"] == "CPTY_T1")
check("500 rows from ONE customer does not qualify (rows are not customers)",
      not by_cp["BE00HEAVY"].startswith("CPTY_T"), by_cp["BE00HEAVY"])
check("below min_customers does not qualify", not by_cp["BE00NINE"].startswith("CPTY_T"))

plain = CategoricalHashTokenizer(vocab_limit=V, special_token="CPTY")
plain.build_vocab()
old_tok = plain.tokenize(frame["h"])
tail = ~tok.str.startswith("CPTY_T")
check("every other counterparty keeps exactly its old CPTY_<bucket> token",
      (tok[tail] == old_tok[tail]).all())
check("vocab = buckets + dedicated", hyb.vocab_size == V + 2, str(hyb.vocab_size))
check("local index == bucket for tail tokens",
      all(hyb._idx_to_token[i] == f"CPTY_{i}" for i in range(V)))
check("row coverage reported", abs(hyb.row_coverage - 27 / len(frame)) < 1e-9,
      f"{hyb.row_coverage:.3%}")

hstate = json.loads(json.dumps(hyb.get_state()))
dumped = json.dumps(hstate)
check("state stores hashes, never the account strings",
      "BE00SHARED" not in dumped and "BE00UTILITY" not in dumped)
hback = HybridHashTokenizer.from_state(hstate)
check("state round-trip reproduces the tokens",
      (hback.tokenize(frame[["h", "cust"]]) == tok).all())
check("unseen counterparty falls back to its bucket",
      hback.tokenize(pd.Series([stable_hash(pd.Series(["BE99NEW"])).iloc[0]])).iloc[0]
      .startswith("CPTY_") and "T" not in hback.tokenize(
          pd.Series([stable_hash(pd.Series(["BE99NEW"])).iloc[0]])).iloc[0].split("_")[1][:1])
none = HybridHashTokenizer(vocab_limit=V, top_k=0)
none.build_vocab(frame[["h", "cust"]])
check("top_k=0 behaves exactly like CategoricalHashTokenizer",
      (none.tokenize(frame[["h", "cust"]]) == old_tok).all())

# --------------------------------------------- 3. pipeline, end to end
print("\n3. BankTransferTokenizerPipeline (pandas standing in for cuDF)")
rng = np.random.default_rng(0)
n = 400
start = pd.Timestamp("2023-01-01")
raw = pd.DataFrame({
    "customer_id": rng.choice([f"K{i}" for i in range(8)], n),
    "account_number": rng.choice(["A1", "A2", "A3"], n),
    "datetime": start + pd.to_timedelta(rng.integers(0, 90 * 86400, n), unit="s"),
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
})
# Two transactions of one customer in the same second -> a true zero gap.
raw.loc[1, ["customer_id", "account_number"]] = raw.loc[0, ["customer_id", "account_number"]].values
raw.loc[1, "datetime"] = raw.loc[0, "datetime"]

cfg = BankTokenizerConfig(include_account_time_delta=True, counterparty_hash_size=64,
                          counterparty_min_customers=3)
pipe = BankTransferTokenizerPipeline(config=cfg)
proc = pipe.preprocess(raw.copy())
pipe.fit(proc)
toks = pipe.transform(proc)

names = [pipe.steps[t].__class__.__name__ for t in pipe.tokenizer_order]
prefixes = []
for tok_id in pipe.tokenizer_order:
    first_token = next(iter(pipe.steps[tok_id]._idx_to_token.values()))
    prefixes.append(first_token.split("_")[0])
expected = [p for p in DEFAULT_FIELD_ORDER if p in set(prefixes)]
check("emission order follows DEFAULT_FIELD_ORDER", prefixes == expected, " ".join(prefixes))
check("context first, amount last", prefixes[0] == "ACCTH" and prefixes[-1] == "AMT")
check("counterparty step is the hybrid one, reading [hash, customer]",
      "HybridHashTokenizer" in names
      and pipe.column_specs[[t for t in pipe.tokenizer_order if t.startswith("cpty")][0]]
      == ["cpty_hash", "customer_id"])

first_rows = proc.groupby("customer_id").cumcount() == 0
check("first transaction of each customer -> TDIF_FIRST",
      (toks.loc[first_rows, "time_delta_s"] == "TDIF_FIRST").all())
check("no other row gets TDIF_FIRST",
      not (toks.loc[~first_rows, "time_delta_s"] == "TDIF_FIRST").any())
acct_first = proc.groupby(["customer_id", "account_number"]).cumcount() == 0
check("first transaction of each account -> TDIFA_FIRST",
      (toks.loc[acct_first, "time_delta_acct_s"] == "TDIFA_FIRST").all())
same_second = (proc["time_delta_s"] == 0)
check("same-second transaction -> TDIF_0, not FIRST",
      same_second.any() and (toks.loc[same_second, "time_delta_s"] == "TDIF_0").all())
check("no token ID collisions", not pipe.id_collisions())
check("tokens per transaction unchanged (16 with TDIFA)", pipe.tokens_per_transaction == 16)
lines = pipe.to_corpus_lines(toks, proc, pipe.group_cols, chunk_size=50)
check("corpus lines start with ACCTH after <bos>", all(l.split()[1].startswith("ACCTH_") for l in lines))

state = json.loads(json.dumps(pipe.get_state()))
reloaded = BankTransferTokenizerPipeline.from_state(state)
check("saved state reloads with the new options",
      reloaded.tokenizer_order == pipe.tokenizer_order
      and reloaded.transform(reloaded.preprocess(raw.copy())).equals(toks))

# ------------------------------------------- 4. old states stay old
print("\n4. A state written before these options reloads unchanged")
legacy_cfg = BankTokenizerConfig(include_account_time_delta=True, counterparty_hash_size=64,
                                 **BankTokenizerConfig.LEGACY_DEFAULTS)
lpipe = BankTransferTokenizerPipeline(config=legacy_cfg)
lproc = lpipe.preprocess(raw.copy())
lpipe.fit(lproc)
ltoks = lpipe.transform(lproc)
check("legacy order is the original one",
      lpipe.tokenizer_order[:3] == ["amt_val", "time_delta_s", "time_delta_acct_s"]
      and lpipe.tokenizer_order[-1] == "acct_hash", " ".join(lpipe.tokenizer_order))
check("legacy: first transaction gets the zero-gap token, as before",
      (ltoks.loc[first_rows, "time_delta_s"] == ltoks.loc[same_second, "time_delta_s"].iloc[0]).all())
check("legacy: counterparty is plain hash buckets",
      ltoks["cpty_hash"].str.match(r"^CPTY_\d+$").all())

old = json.loads(json.dumps(lpipe.get_state()))
for key in BankTokenizerConfig.LEGACY_DEFAULTS:            # what an old file lacks
    old["metadata"]["config"].pop(key)
for step in old["steps"].values():
    for key in ("zero_based_bins", "first_token"):
        step["init_params"].pop(key, None)
restored = BankTransferTokenizerPipeline.from_state(old)
check("old state -> legacy config, not today's defaults",
      restored.config.field_order is None and restored.config.counterparty_top_k == 0
      and not restored.config.time_delta_first_token)
check("old state -> identical tokens to before",
      restored.transform(restored.preprocess(raw.copy())).equals(ltoks))

# ------------------------------------------------ 5. field_order validation
print("\n5. field_order validation")
msg = raises(lambda: BankTransferTokenizerPipeline(config=BankTokenizerConfig(field_order=("AMT", "NOPE"))))
check("unknown field rejected", msg is not None and "NOPE" in msg)
msg = raises(lambda: BankTransferTokenizerPipeline(config=BankTokenizerConfig(field_order=("AMT",))))
check("an enabled field left out is rejected", msg is not None and "does not place" in msg)
msg = raises(lambda: BankTransferTokenizerPipeline(
    config=BankTokenizerConfig(field_order=DEFAULT_FIELD_ORDER + ("AMT",))))
check("duplicate field rejected", msg is not None and "twice" in msg)
lower = BankTransferTokenizerPipeline(config=BankTokenizerConfig(
    field_order=tuple(p.lower() for p in DEFAULT_FIELD_ORDER)))
check("names are case-insensitive", lower.tokenizer_order[0] == "acct_hash")

# ------------------------------------------------ 5b. drop_fields
print("\n5b. drop_fields")
dcfg = BankTokenizerConfig(include_account_time_delta=True, counterparty_hash_size=64,
                           counterparty_min_customers=3, drop_fields=("emp",))
dpipe = BankTransferTokenizerPipeline(config=dcfg)
dproc = dpipe.preprocess(raw.copy())
dpipe.fit(dproc)
dtoks = dpipe.transform(dproc)
check("dropped field has no step", "emp_clean" not in dpipe.tokenizer_order,
      " ".join(dpipe.tokenizer_order))
check("one token fewer per transaction",
      dpipe.tokens_per_transaction == pipe.tokens_per_transaction - 1,
      str(dpipe.tokens_per_transaction))
check("chunk size follows", dpipe.recommended_chunk_size(4096) == 4094 // 16,
      str(dpipe.recommended_chunk_size(4096)))
check("no EMP token in the vocabulary", not any(str(t).startswith("EMP_") for t in dpipe.vocab))
check("other fields tokenize exactly as before",
      dtoks.equals(toks[[c for c in toks.columns if c != "emp_clean"]]),
      f"{list(dtoks.columns)}")
dstate = json.loads(json.dumps(dpipe.get_state()))
check("drop_fields survives a save/load",
      BankTransferTokenizerPipeline.from_state(dstate).tokenizer_order == dpipe.tokenizer_order)
msg = raises(lambda: BankTransferTokenizerPipeline(config=BankTokenizerConfig(drop_fields=("NOPE",))))
check("unknown dropped field rejected", msg is not None and "NOPE" in msg)
check("a state without drop_fields keeps every field",
      "emp_clean" in restored.tokenizer_order)

# ------------------------------------------- 6. <sep> excluded from the loss
print("\n6. loss_ignore_tokens")
from src.clm_data import build_bank_clm_dataset  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    pipe.save_vocab(tmp / "vocab.json")
    (tmp / "corpus.txt").write_text("\n".join(lines[:3]) + "\n")
    vocab = json.loads((tmp / "vocab.json").read_text())["vocab"]
    sep, pad = vocab["<sep>"], vocab["<pad>"]

    ds = build_bank_clm_dataset(str(tmp / "corpus.txt"), str(tmp / "vocab.json"),
                                seq_length=1024, loss_ignore_tokens=["<sep>"])
    item = ds[0]
    ids, labels, mask = item["input_ids"], item["labels"], item["attention_mask"]
    real = ids != pad
    check("<sep> stays in the input", int((ids == sep).sum()) > 0)
    check("<sep> is never a target", not (labels[ids == sep] != -100).any())
    check("every other real token is still a target",
          (labels[real & (ids != sep)] == ids[real & (ids != sep)]).all())
    check("attention mask covers <sep> (it is still attended to)",
          (mask[ids == sep] == 1).all() and (mask == real.astype(int)).all())
    base = build_bank_clm_dataset(str(tmp / "corpus.txt"), str(tmp / "vocab.json"),
                                  seq_length=1024)[0]
    check("without the option labels are unchanged (old behaviour)",
          (base["labels"][real] == ids[real]).all())
    msg = raises(lambda: build_bank_clm_dataset(str(tmp / "corpus.txt"), str(tmp / "vocab.json"),
                                                loss_ignore_tokens=["<nope>"]))
    check("an unknown token is an error, not silently ignored", msg is not None)

from src.train_decoder import batch_attention_mask  # noqa: E402


class _Labels(np.ndarray):
    device = "cpu"

    def long(self):
        return self


class _Mask(np.ndarray):
    def to(self, *a, **k):
        return self


lab = np.array([[5, -100, 7, -100]]).view(_Labels)
given = np.array([[1, 1, 1, 0]]).view(_Mask)
check("trainer uses the batch's attention mask when present",
      (batch_attention_mask({"attention_mask": given}, lab) == given).all())
check("trainer falls back to labels != -100 for old datasets",
      (batch_attention_mask({}, lab) == np.array([[True, False, True, False]])).all())

cfg_text = (ROOT / "configs" / "pretrain_bank_decoder.yaml").read_text(encoding="utf-8")
sections = {}
for block in ("\ndataset:\n", "\nvalidation_dataset:\n"):
    body = cfg_text.split(block, 1)[1].split("\n\n", 1)[0]
    sections[block.strip()] = body
check("YAML: training dataset ignores <sep>",
      'loss_ignore_tokens: ["<sep>"]' in sections["dataset:"])
check("YAML: validation dataset ignores <sep> too",
      'loss_ignore_tokens: ["<sep>"]' in sections["validation_dataset:"])

# --------------------------------------- 7. embeddings skip the LM head
print("\n7. Embedding extraction runs the backbone only")
from src.decoder_inference import HuggingFaceDecoderInference  # noqa: E402


class _Backbone:
    def __call__(self, **kw):
        assert kw.get("output_hidden_states")
        return types.SimpleNamespace(hidden_states=("layer0", "last"))


class _CausalLM:
    base_model_prefix = "model"

    def __init__(self):
        self.model = _Backbone()

    def __call__(self, **kw):
        raise AssertionError("the LM head (vocabulary logits) was computed")


class _Ids:
    def __ne__(self, other):
        return types.SimpleNamespace(long=lambda: "mask")


inf = object.__new__(HuggingFaceDecoderInference)
inf.model = _CausalLM()
inf.tokenizer = types.SimpleNamespace(pad_token_id=0)
try:
    hidden, mask = inf._get_hidden_states(_Ids())
    good = hidden == "last" and mask == "mask"
except AssertionError as e:
    good = False
check("hidden_states[-1] taken from model.model, LM head never called", good)
inf.model = types.SimpleNamespace(base_model_prefix="", __call__=None)
check("model without a base prefix falls back to the model itself",
      inf._backbone is inf.model)

# ------------------------------------------- 8. account number per customer
print("\n8. account_token=order: ACCT_<n>, the account's number within its customer")
from src.tokenizer.bank_pipeline import (  # noqa: E402
    ACCOUNT_ORDER_COL, BankSchema, account_order_table, add_account_order,
)

acc = pd.DataFrame({
    "customer_id": ["c1", "c1", "c1", "c1", "c2", "c2", "c3"],
    "account_number": ["B", "A", "B", "C", "A", "Z", "Q"],
    "datetime": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-01",
                                "2024-05-01", "2024-04-01", "2024-01-01"]),
})
table = account_order_table(acc, BankSchema())
got = {(r.customer_id, r.account_number): r.account_order for r in table.itertuples()}
check("numbered by first transfer, ties by account number, per customer",
      got == {("c1", "B"): 1, ("c1", "C"): 2, ("c1", "A"): 3,
              ("c2", "Z"): 1, ("c2", "A"): 2, ("c3", "Q"): 1}, str(got))
shuffled = acc.sample(frac=1.0, random_state=0).reset_index(drop=True)
with_order = add_account_order(shuffled, BankSchema(), table)
check("add_account_order keeps the rows in their order",
      with_order.drop(columns=[ACCOUNT_ORDER_COL]).equals(shuffled))
check("the same account gets the same number in every frame (from the full history)",
      add_account_order(acc.iloc[2:3], BankSchema(), table)[ACCOUNT_ORDER_COL].tolist() == [1])

ord_raw = add_account_order(raw.copy(), BankSchema())
ord_raw.loc[ord_raw.index[:3], ACCOUNT_ORDER_COL] = 40          # beyond the cap
ocfg = BankTokenizerConfig(include_account_time_delta=True, counterparty_hash_size=64,
                           counterparty_min_customers=3, account_token="order")
opipe = BankTransferTokenizerPipeline(config=ocfg)
oproc = opipe.preprocess(ord_raw.copy())
opipe.fit(oproc)
otoks = opipe.transform(oproc)
oprefixes = [next(iter(opipe.steps[t]._idx_to_token.values())).split("_")[0]
             for t in opipe.tokenizer_order]
check("ACCT replaces ACCTH, first in the transfer", oprefixes[0] == "ACCT"
      and "ACCTH" not in oprefixes, " ".join(oprefixes))
check("same number of tokens per transfer", opipe.tokens_per_transaction == pipe.tokens_per_transaction)
acct_col = [c for c in otoks.columns if otoks[c].astype(str).str.startswith("ACCT_").all()][0]
expected = ["ACCT_%02d" % min(v, 16) for v in oproc[ACCOUNT_ORDER_COL]]
check("ACCT_<n> = account_order, capped at account_order_max",
      otoks[acct_col].tolist() == expected and "ACCT_16" in expected)
check("ACCT vocabulary has account_order_max tokens", opipe.steps[opipe.tokenizer_order[0]].vocab_size == 16)
check("no token ID collisions (ACCT)", not opipe.id_collisions())
oreloaded = BankTransferTokenizerPipeline.from_state(json.loads(json.dumps(opipe.get_state())))
check("ACCT state reloads", oreloaded.transform(oreloaded.preprocess(ord_raw.copy())).equals(otoks))
try:
    opipe.preprocess(raw.copy())
    good = False
except ValueError as e:
    good = "add_account_order.py" in str(e)
check("missing account_order column -> clear error", good)
try:
    BankTokenizerConfig(account_token="rank").normalized()
    good = False
except ValueError:
    good = True
check("unknown account_token rejected", good)
hstate = json.loads(json.dumps(pipe.get_state()))
hstate["metadata"]["config"].pop("account_token", None)
check("a state without account_token reloads as hash",
      BankTransferTokenizerPipeline.from_state(hstate).config.account_token == "hash")

# ------------------------------------------------------------------ summary
print("\n" + "=" * 60)
print(f"{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:")
    for name in fail:
        print("  -", name)
    sys.exit(1)

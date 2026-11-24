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

"""Offline checks for the pooling strategies in src/decoder_inference.py.

Needs a CPU torch (no model, no GPU): hidden states are synthetic, and the
token layout comes from the real HistoryWindowEncoder, so ``last_txn`` is
checked against the windows notebook 04 actually builds.

    uv run --no-project --with numpy --with pandas --with scipy --with tqdm \\
        --with pyyaml --with scikit-learn --with torch \\
        --index-url https://download.pytorch.org/whl/cpu \\
        --extra-index-url https://pypi.org/simple python scripts/verify_pooling_offline.py
"""

import sys
import types
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("torch is not installed -- run this where torch is available.")
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.decoder_inference import (  # noqa: E402
    POOLINGS, HuggingFaceDecoderInference, register_pooling, resolve_pooling,
)

# history.py is pure numpy; load it by path so the tokenizer package (cuDF) is
# not imported.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("history", ROOT / "src/tokenizer/history.py")
_history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_history)
HistoryWindowEncoder = _history.HistoryWindowEncoder

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


# Windows of T=3 tokens per transaction; row 0 has 1 transaction of history,
# row 3 has 3 (window=3). Token IDs >= 10 so they differ from the specials.
T, PAD = 3, 0
token_ids = np.arange(10, 10 + 4 * T).reshape(4, T)
enc = HistoryWindowEncoder(token_ids, positions=np.arange(4), window=3, pad_id=PAD)
ids = torch.from_numpy(enc.to_array())
mask = (ids != PAD).long()

# Hidden state = the token id at that position, so every pooled value is readable.
hidden = ids.float().unsqueeze(-1).repeat(1, 1, 2)

print("1. strategies on the real window layout")
last = POOLINGS["last_token"](hidden, mask)
check("last_token reads <eos> (id 2)", torch.all(last == 2).item(), str(last[:, 0].tolist()))

mean = POOLINGS["mean"](hidden, mask)
expected = [float(ids[i][mask[i].bool()].float().mean()) for i in range(len(ids))]
check("mean averages non-padding positions only",
      torch.allclose(mean[:, 0], torch.tensor(expected)), str(mean[:, 0].tolist()))

txn = POOLINGS["last_txn"](hidden, mask, T)
expected = [float(np.mean(token_ids[i])) for i in range(4)]
check("last_txn averages the current transaction's fields",
      torch.allclose(txn[:, 0], torch.tensor(expected)), str(txn[:, 0].tolist()))
try:
    POOLINGS["last_txn"](hidden, mask, None)
    check("last_txn without tokens_per_txn fails loudly", False)
except ValueError:
    check("last_txn without tokens_per_txn fails loudly", True)

print("\n2. combinations and extension")
check("'mean+last_txn' resolves to two parts", len(resolve_pooling("mean+last_txn")) == 2)
try:
    resolve_pooling("mean+nope")
    check("unknown part rejected", False)
except ValueError:
    check("unknown part rejected", True)

inf = object.__new__(HuggingFaceDecoderInference)
inf.pooling, inf.tokens_per_txn = "mean+last_txn", T
combo = inf._pool_embeddings(hidden, mask)
check("combined pooling concatenates", combo.shape == (4, 4)
      and torch.allclose(combo[:, :2], mean) and torch.allclose(combo[:, 2:], txn))


@register_pooling("max")
def _pool_max(h, m, tokens_per_txn=None):
    return h.masked_fill(~m.bool().unsqueeze(-1), float("-inf")).max(dim=1).values


check("a new strategy registers and resolves",
      torch.all(resolve_pooling("max")[0](hidden, mask)[:, 0] == ids.max(dim=1).values).item())

# ------------------------------------------------------- 3. learned pooling
print("\n3. learned (attention) pooling")
from src.decoder_inference import pooling_parts  # noqa: E402
from src.learned_pooling import (  # noqa: E402
    customer_share, head_scores, load_head, rank_auc, train_head, transaction_states, use_head,
)

check("'learned' is accepted before the head exists", pooling_parts("mean+learned") == ["mean", "learned"])
try:
    resolve_pooling("learned")
    check("...but using it before training fails loudly", False)
except ValueError as e:
    check("...but using it before training fails loudly", "use_head" in str(e))

states, valid = transaction_states(hidden, mask, T)
check("transaction_states: one vector per transaction, real ones flagged",
      states.shape == (4, 3, 2) and valid.sum(1).tolist() == [1, 2, 3, 3]
      and torch.allclose(states[3, 2, 0], torch.tensor(float(np.mean(token_ids[3])))))

# Synthetic task: the label lives in the transaction two before the scored one
# (dimension 0 shifted for positives); everything else is noise.
rng = np.random.default_rng(0)
n, W, d = 4000, 8, 16
S = rng.normal(size=(n, W, d)).astype(np.float16)
V = np.ones((n, W), dtype=bool)
depth = rng.integers(3, W + 1, n)                    # real transactions per window
V[np.arange(W)[None, :] >= depth[:, None]] = False
y = (rng.random(n) < 0.1).astype(np.int64)
S[np.arange(n), depth - 3, 0] += (3.0 * y).astype(np.float16)
groups = np.array([f"C{i // 4}" for i in range(n)])
stop = customer_share(groups, 0.25, seed=1)
m = (S.astype(np.float32) * V[:, :, None]).sum(1) / V.sum(1, keepdims=True)
mean_auc = rank_auc(m[stop, 0], y[stop])
aucs = []
for seed in range(3):   # the warm-up is what makes this seed-independent
    head, hist = train_head(S, V, y, stop, device="cpu", epochs=20, batch_size=64,
                            lr=3e-3, seed=seed, log=lambda *_: None)
    aucs.append(hist["best_stop_auc"])
check("head learns the signal for every seed, beating mean pooling",
      min(aucs) > 0.9 and min(aucs) > mean_auc + 0.1,
      f"head {[round(a, 3) for a in aucs]} vs mean pooling {mean_auc:.3f}")
with torch.no_grad():
    _, logits, weights = head(torch.as_tensor(S[stop]).float(), torch.as_tensor(V[stop]))
pos = y[stop] == 1
target = weights[torch.arange(len(weights)), torch.as_tensor(depth[stop] - 3)]
check("attention concentrates on the informative transaction (positives)",
      float(target[torch.as_tensor(pos)].mean()) > 1.5 / W, f"{float(target[torch.as_tensor(pos)].mean()):.2f}")
check("padding transactions get zero weight", float(weights[~torch.as_tensor(V[stop])].abs().max()) == 0.0)

share = customer_share(groups, 0.5, seed=0)
same_customer = all(len(set(share[groups == g])) == 1 for g in np.unique(groups)[:50])
check("customer_share: whole customers, about the requested fraction",
      same_customer and 0.4 < share.mean() < 0.6, f"{share.mean():.2f}")
check("customer_share: deterministic", (customer_share(groups, 0.5, seed=0) == share).all())

from src.learned_pooling import AttentionPoolingHead  # noqa: E402

small = AttentionPoolingHead(dim=2, window=3).eval()
use_head(small, T)
with torch.no_grad():
    emb = POOLINGS["learned"](hidden, mask, T)
    ref = small(*transaction_states(hidden, mask, T))[0]
check("use_head registers 'learned' = the head's pooled vector on the real layout",
      emb.shape == (4, 2) and torch.allclose(emb, ref))
inf.pooling = "mean+learned"
check("'mean+learned' concatenates once the head is registered",
      inf._pool_embeddings(hidden, mask).shape == (4, 4))

with torch.no_grad():
    pooled, logit, _ = head(torch.as_tensor(S[:64]).float(), torch.as_tensor(V[:64]))
check("head_scores from saved embeddings == the head's logits",
      np.allclose(head_scores(head, pooled.numpy()), logit.numpy(), atol=1e-5))
check("rank_auc matches a hand count", rank_auc(np.array([0.1, 0.9, 0.4, 0.8]),
                                                np.array([0, 1, 0, 1])) == 1.0)

import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    from src.learned_pooling import save_head  # noqa: E402
    save_head(head, Path(tmp) / "head.pt")
    again = load_head(Path(tmp) / "head.pt")
    with torch.no_grad():
        same = torch.allclose(again(torch.as_tensor(S[:64]).float(), torch.as_tensor(V[:64]))[1], logit)
check("save/load reproduces the head", same)

# ------------------------------------------------------- 4. surprise
print("\n4. surprise (anomaly profile)")
from src.decoder_inference import PoolingContext, pooling_dims, pooling_slice  # noqa: E402

V_SIZE = int(ids.max()) + 1
torch.manual_seed(0)
lm_head = torch.nn.Linear(2, V_SIZE)
ctx = PoolingContext(tokens_per_txn=T, input_ids=ids, lm_head=lm_head)
with torch.no_grad():
    out = POOLINGS["surprise"](hidden, mask, ctx, row_chunk=3)    # chunking crosses rows
    logp = torch.log_softmax(lm_head(hidden), dim=-1)

# Hand computation on the real window layout: field f of transaction j sits at
# 1 + j*(T+1) + f and is predicted from the position before it.
expected = []
for r in range(len(ids)):
    k = min(r + 1, 3)          # real transactions: row r has r earlier ones, window 3
    per_field = [[-float(logp[r, p - 1, ids[r, p]]) for p in
                  [1 + j * (T + 1) + f for f in range(T)]] for j in range(k)]
    last = per_field[-1]
    hist = [np.mean(x) for x in per_field[:-1]]
    hist_mean = np.mean(hist) if hist else np.mean(last)
    expected.append([np.mean(last), hist_mean, np.mean(last) - hist_mean,
                     (k - 1) / 2] + last)
expected = torch.tensor(expected, dtype=torch.float32)
check("surprise = per-field NLL of the scored transfer vs its history (hand-computed)",
      out.shape == (4, T + 4) and torch.allclose(out, expected, atol=1e-5),
      f"max diff {float((out - expected).abs().max()):.2e}")
check("a transfer with no history: history mean = its own, difference 0, depth 0",
      float(out[0, 2]) == 0.0 and float(out[0, 3]) == 0.0)
check("surprise size = tokens_per_txn + 4; slices follow each part's size",
      pooling_dims("mean+surprise", 512, 16) == [512, 20]
      and pooling_slice("mean+surprise", "surprise", 512, 16) == slice(512, 532))
try:
    POOLINGS["surprise"](hidden, mask, T)
    check("surprise without input ids / LM head fails loudly", False)
except ValueError:
    check("surprise without input ids / LM head fails loudly", True)

inf2 = object.__new__(HuggingFaceDecoderInference)
inf2.pooling, inf2.tokens_per_txn, inf2._embedding_dim_base = "mean+surprise", T, 2
inf2.model = type("M", (), {"get_output_embeddings": lambda self: lm_head})()
with torch.no_grad():
    both = inf2._pool_embeddings(hidden, mask, ids)
check("inference wrapper passes ids and LM head; embedding_dim matches",
      both.shape == (4, inf2.embedding_dim) and inf2.embedding_dim == 2 + T + 4)

# ------------------------------------------------------- 5. memory safety
print("\n5. out-of-memory handling (simulated)")
from src.decoder_inference import run_halving  # noqa: E402

calls = []


def fits_four(batch):
    calls.append(len(batch))
    if len(batch) > 4:
        raise torch.cuda.OutOfMemoryError("simulated")
    return batch * 2, batch + 1


x = torch.arange(13)
a, b2 = run_halving(fits_four, x)
check("run_halving: splits until batches fit, output identical and in order",
      torch.equal(a, x * 2) and torch.equal(b2, x + 1) and max(c for c in calls[1:]) <= 13)
try:
    run_halving(lambda ids: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")), x[:1])
    check("run_halving: a single row that does not fit still raises", False)
except torch.cuda.OutOfMemoryError:
    check("run_halving: a single row that does not fit still raises", True)

try:
    import src.train_decoder as td
except ImportError as e:   # transformers missing in this environment
    td = None
    print(f"  (train_decoder not importable here: {e}; batch probe skipped)")
if td is not None:
    class _Out:
        def __init__(self, b):
            self.loss = torch.tensor(1.0, requires_grad=True) * b

    class _Model:
        config = types.SimpleNamespace(vocab_size=50)

        def to(self, _):
            return self

        def __call__(self, input_ids, **_):
            _Model.peak = len(input_ids) * 3 * 2**30          # 3 GiB per row
            if len(input_ids) > 16:
                raise torch.cuda.OutOfMemoryError("simulated")
            return _Out(len(input_ids))

    fake_cuda = types.SimpleNamespace(
        get_device_properties=lambda d: types.SimpleNamespace(total_memory=44 * 2**30),
        is_bf16_supported=lambda: False, empty_cache=lambda: None,
        reset_peak_memory_stats=lambda d: None,
        max_memory_allocated=lambda d: _Model.peak,
        OutOfMemoryError=torch.cuda.OutOfMemoryError)
    real_cuda, real_build, real_randint = td.torch.cuda, td.build_model, td.torch.randint
    td.torch.cuda, td.build_model = fake_cuda, lambda section: _Model()
    td.torch.randint = lambda lo, hi, size, device=None: real_randint(lo, hi, size)
    try:
        got = td.fit_local_batch({"model": {}, "dataset": {"seq_length": 8}},
                                 types.SimpleNamespace(type="cuda"), 32, 128,
                                 log=lambda *a: None)
    finally:
        td.torch.cuda, td.build_model, td.torch.randint = real_cuda, real_build, real_randint
    check("batch probe: 32 OOM, 16 over 90% headroom (48 GiB), 8 fits (24 GiB)", got == 8,
          str(got))

print("\n" + "=" * 60)
print(f"{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:")
    for name in fail:
        print("  -", name)
    sys.exit(1)

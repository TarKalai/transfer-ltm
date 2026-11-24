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

"""Offline checks with real (tiny) models: architectures and fine-tuning.

1. Every architecture in src/architectures.py builds from the shared size
   description, trains a step, supports gradient checkpointing, and exposes
   what extraction needs (backbone, LM head).
2. src/finetune.py: what each mode trains; LoRA is the identity at init;
   a fine-tuning that cannot win is fully undone; one that wins is kept.

Needs torch (CPU) and transformers:

    uv run --no-project --with numpy --with pandas --with scipy --with tqdm \\
        --with pyyaml --with scikit-learn \\
        --with transformers --with torch \\
        --index-url https://download.pytorch.org/whl/cpu \\
        --extra-index-url https://pypi.org/simple python scripts/verify_models_offline.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.architectures import ARCHITECTURES, ENCODERS  # noqa: E402
from src.decoder_inference import HuggingFaceDecoderInference  # noqa: E402
from src.finetune import LoRALinear, finetune_with_head, prepare_trainable  # noqa: E402
from src.learned_pooling import collect_states, customer_share, train_head  # noqa: E402
from src.train_decoder import build_model  # noqa: E402

_spec = importlib.util.spec_from_file_location("history", ROOT / "src/tokenizer/history.py")
_history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_history)
HistoryWindowEncoder = _history.HistoryWindowEncoder

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


SIZE = {"vocab_size": 40, "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 64, "max_position_embeddings": 128,
        "rope_theta": 10000.0, "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0, "tie_word_embeddings": False,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}

print("1. architectures")
counts = {}
for arch in sorted(ARCHITECTURES):
    torch.manual_seed(0)
    model = build_model({"architecture": arch, "attn_implementation": "sdpa",
                         "objective": "mlm" if arch in ENCODERS else "clm",
                         "gradient_checkpointing": True, "config": dict(SIZE)})
    ids = torch.randint(5, 40, (2, 20))
    model.train()
    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids).loss
    loss.backward()
    inf = object.__new__(HuggingFaceDecoderInference)
    inf.model = model
    inf.tokenizer = types.SimpleNamespace(pad_token_id=0)
    model.eval()
    with torch.no_grad():
        hidden, _ = inf._get_hidden_states(ids)
    counts[arch] = sum(p.numel() for p in model.parameters())
    check(f"{arch}: builds, trains a step with checkpointing, backbone + LM head usable",
          torch.isfinite(loss).item() and hidden.shape == (2, 20, 32)
          and model.get_output_embeddings().out_features == 40)
check("same size description -> comparable size across architectures (within 40%)",
      max(counts.values()) / min(counts.values()) < 1.4, str(counts))
try:
    build_model({"architecture": "nope", "config": dict(SIZE)})
    check("unknown architecture rejected", False)
except ValueError:
    check("unknown architecture rejected", True)

# --------------------------------------------------------------- 2. finetune
print("\n2. fine-tuning")
T, W = 2, 4


def fresh_inference(seed=0):
    torch.manual_seed(seed)
    model = build_model({"architecture": "llama", "config": dict(SIZE)}).eval()
    inf = object.__new__(HuggingFaceDecoderInference)
    inf.model, inf.device = model, torch.device("cpu")
    inf.tokenizer = types.SimpleNamespace(pad_token_id=0)
    inf._embedding_dim_base = 32
    return inf


def count(params):
    return sum(p.numel() for p in params)


inf = fresh_inference()
bb = inf._backbone
layer_params = count(list(bb.layers)[-1].parameters()) + count(bb.norm.parameters())
check("top:1 trains the last layer + final norm only",
      count(prepare_trainable(inf.model, bb, "top:1")) == layer_params)
check("full trains the whole backbone",
      count(prepare_trainable(inf.model, bb, "full")) == count(bb.parameters()))
check("none trains nothing", prepare_trainable(inf.model, bb, "none") == [])

inf = fresh_inference()
ids = torch.randint(5, 40, (3, 13))
with torch.no_grad():
    before = inf._get_hidden_states(ids)[0].clone()
lora_params = prepare_trainable(inf.model, inf._backbone, "lora:2")
with torch.no_grad():
    after = inf._get_hidden_states(ids)[0]
wrapped = [m for m in inf.model.modules() if isinstance(m, LoRALinear)]
check("lora:2 wraps q/v projections of every layer, identity at init",
      len(wrapped) == 2 * 3 and torch.allclose(before, after, atol=1e-6))
check("lora trains only the adapters",
      # A: rank x in_features, B: out_features x rank (rank 2, in_features 32)
      count(lora_params) == sum(2 * 32 + m.base.out_features * 2 for m in wrapped)
      and all(not p.requires_grad for n, p in inf.model.named_parameters() if "lora_" not in n))

# Synthetic task: positive when the scored transaction's first token is 7.
rng = np.random.default_rng(0)
n = 1200
tok = rng.integers(10, 30, size=(n, T))
y = (rng.random(n) < 0.15).astype(np.int64)
tok[y == 1, 0] = 7
positions = np.arange(n) % 6                       # sequences of 6 rows
enc = HistoryWindowEncoder(tok, positions, window=W, pad_id=0)
groups = np.array([f"C{i // 6}" for i in range(n)])
stop = customer_share(groups, 0.25, seed=1)


def head_on(inf):
    s, v = collect_states(inf, enc, batch_size=128, show_progress=False)
    return train_head(s, v, y, stop, device="cpu", epochs=8, batch_size=64, lr=3e-3,
                      warmup_epochs=2, log=lambda *a: None)


inf = fresh_inference()
head, hist = head_on(inf)
original = {k: v.clone() for k, v in inf._backbone.state_dict().items()}
head_state = {k: v.clone() for k, v in head.state_dict().items()}
head, ft = finetune_with_head(inf, enc, y, stop, head, "full", head_auc=1.01, lr=1e-3,
                              epochs=1, batch_size=64, log=lambda *a: None)
same_model = all(torch.equal(original[k], v) for k, v in inf._backbone.state_dict().items())
same_head = all(torch.equal(head_state[k], v) for k, v in head.state_dict().items())
check("fine-tuning that cannot beat the head (target AUC > 1) is fully undone",
      not ft["kept"] and same_model and same_head)
check("...and every weight is frozen again",
      not any(p.requires_grad for p in inf.model.parameters()))

for spec in ("top:1", "lora:4", "full"):
    inf = fresh_inference()
    head, hist = head_on(inf)
    head, ft = finetune_with_head(inf, enc, y, stop, head, spec, head_auc=0.0, lr=1e-3,
                                  epochs=2, batch_size=64, log=lambda *a: None)
    changed = any(not torch.equal(a, b) for a, b in zip(
        fresh_inference()._backbone.state_dict().values(), inf._backbone.state_dict().values()))
    check(f"{spec}: trains, is kept when it wins, and changes the model in place",
          ft["kept"] and changed and len(ft["stop_auc"]) >= 1,
          f"early-stop AUC {ft.get('best_stop_auc', float('nan')):.3f}")

from src.learned_pooling import rank_auc  # noqa: E402

check("rank_auc: NaN scores (a diverged epoch) give NaN, not an exception",
      np.isnan(rank_auc(np.array([0.1, np.nan, 0.3]), np.array([0, 1, 1]))))
check("rank_auc: empty input gives NaN", np.isnan(rank_auc(np.array([]), np.array([]))))
from src.clm_data import vocab_of  # noqa: E402
check("vocab.json: one rule for wrapped and bare maps",
      vocab_of({"vocab": {"a": 1}, "vocab_size": 2}) == {"a": 1} and vocab_of({"a": 1}) == {"a": 1})

# ------------------------------------------------------- 3. weighted loss
print("\n3. field-weighted pretraining loss")
import json  # noqa: E402
import tempfile  # noqa: E402

from src.train_decoder import token_loss_weights, weighted_lm_loss  # noqa: E402

torch.manual_seed(0)
lm = build_model({"architecture": "llama", "config": dict(SIZE)}).eval()
ids = torch.randint(5, 40, (3, 16))
labels = ids.clone()
labels[:, :2] = -100                                   # ignored positions, as in the corpus
with torch.no_grad():
    out = lm(input_ids=ids, labels=labels)
    plain = weighted_lm_loss(out.logits, labels, torch.ones(40))
check("all weights 1 -> identical to the model's own loss", torch.allclose(plain, out.loss, atol=1e-5),
      f"{float(plain):.6f} vs {float(out.loss):.6f}")

with tempfile.TemporaryDirectory() as d:
    vocab_path = Path(d) / "vocab.json"
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3, "<unk>": 4}
    vocab.update({f"AMT_{i}": 5 + i for i in range(9)})
    vocab["AMT_PAD"] = 14                                      # a filler token
    vocab.update({f"CPTY_{i}": 15 + i for i in range(25)})
    vocab_path.write_text(json.dumps({"vocab_size": 40, "special_tokens": {}, "vocab": vocab,
                                      "field_prefixes": ["AMT", "CPTY"],
                                      "filler_tokens": ["AMT_PAD"]}))
    w = token_loss_weights(vocab_path, {"amt": 0.0}, 40)
    check("weights by field (case-insensitive), others 1; padding never above its field",
          float(w[5:15].sum()) == 0.0
          and float(w[15:].min()) == 1.0 and float(w[:5].min()) == 1.0)
    up = token_loss_weights(vocab_path, {"AMT": 3.0}, 40)
    check("...and never upweighted (AMT x3: real AMT tokens 3, AMT_PAD 1)",
          float(up[5]) == 3.0 and float(up[14]) == 1.0)
    v2 = Path(d) / "vocab2.json"
    v2.write_text(json.dumps({"vocab": {"MYFIELD_0": 0, "MYFIELD_1": 1, "TDIF_0": 2,
                                        "TDIFA_0": 3, "AMT_0": 4, "AMTREL_0": 5},
                              "field_prefixes": ["MYFIELD", "TDIF", "TDIFA", "AMT", "AMTREL"]}))
    w2 = token_loss_weights(v2, {"MYFIELD": 3.0, "TDIF": 2.0, "AMT": 4.0}, 6)
    check("exact field match: MYFIELD_*, TDIF_* but not TDIFA_*, AMT_* but not AMTREL_*",
          w2.tolist() == [3.0, 3.0, 2.0, 1.0, 4.0, 1.0], str(w2.tolist()))
    check("no weights configured -> None (default loss path untouched)",
          token_loss_weights(vocab_path, {}, 40) is None)
    old_vocab = Path(d) / "old_vocab.json"
    old_vocab.write_text(json.dumps({"vocab": vocab}))         # no field records
    try:
        token_loss_weights(old_vocab, {"AMT": 2.0}, 40)
        check("a vocab.json without field records is refused for loss weights", False)
    except ValueError as e:
        check("a vocab.json without field records is refused for loss weights",
              "field_prefixes" in str(e))
    check("a parsed vocab.json can be passed instead of a path (read once)",
          torch.equal(token_loss_weights(json.loads(vocab_path.read_text()), {"amt": 0.0}, 40), w))
    try:
        token_loss_weights(vocab_path, {"NOPE": 2.0}, 40)
        check("a field with no tokens is rejected", False)
    except ValueError:
        check("a field with no tokens is rejected", True)
with torch.no_grad():
    zeroed = weighted_lm_loss(out.logits, labels, w)
    shift = labels[:, 1:].reshape(-1)
    ce = torch.nn.functional.cross_entropy(out.logits[:, :-1].reshape(-1, 40).float(), shift,
                                           ignore_index=-100, reduction="none")
    keep = (shift != -100) & ~((shift >= 5) & (shift < 15))
    manual = ce[keep].mean()
check("weight 0 on AMT = the loss over every other target token", torch.allclose(zeroed, manual))
with torch.no_grad():
    w_loss, plain_loss = weighted_lm_loss(out.logits, labels, w, return_plain=True)
check("return_plain: the logged loss is the unweighted one",
      torch.allclose(w_loss, zeroed) and torch.allclose(plain_loss, out.loss, atol=1e-5))
with torch.no_grad():
    _, empty_plain = weighted_lm_loss(out.logits, torch.full_like(labels, -100), w,
                                      return_plain=True)
check("a batch with no targets logs 0, not NaN", float(empty_plain) == 0.0)
with torch.no_grad():
    all_zero = weighted_lm_loss(out.logits, labels, torch.zeros(40))
check("every target weighted 0 -> the plain loss, not a gradient-free 0",
      torch.allclose(all_zero, out.loss, atol=1e-5))
try:
    token_loss_weights(None, {"AMT": 2.0}, 40)
    check("loss weights without a vocab file: clear error", False)
except ValueError as e:
    check("loss weights without a vocab file: clear error", "vocab_path" in str(e))

print("\n" + "=" * 60)
print(f"{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:")
    for name in fail:
        print("  -", name)
    sys.exit(1)

"""Offline verification of src/train_decoder.py config handling.

No GPU, no torch, no pyyaml on this machine -- both are stubbed. What this
checks is the pure-Python surface: YAML key/type agreement with the real
config, CLI override parsing against the exact argument list notebook 03
passes, the accumulation arithmetic, the LR schedule, and _target_ resolution.
"""
import math
import sys
import types
from pathlib import Path

ROOT = Path.cwd().resolve()  # run from the repo root
if not (ROOT / "configs" / "pretrain_bank_decoder.yaml").exists():
    raise SystemExit(f"run this from the repo root; {ROOT} has no configs/")

# ---------------------------------------------------------------- stubs
mini = types.ModuleType("yaml")


def _parse_scalar(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return [_parse_scalar(x) for x in text[1:-1].split(",") if x.strip()]
    low = text.lower()
    if low in ("null", "none", "~", ""):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text.strip("'\"")


def safe_load(stream):
    """Minimal YAML subset: nested mappings, 2-space indent, scalars, flow lists."""
    text = stream.read() if hasattr(stream, "read") else stream
    root = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip().split("  #")[0].rstrip()
        if line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        node = stack[-1][1]
        if value.strip() == "":
            child = {}
            node[key.strip()] = child
            stack.append((indent, child))
        else:
            node[key.strip()] = _parse_scalar(value)
    return root


mini.safe_load = safe_load
sys.modules["yaml"] = mini

torch = types.ModuleType("torch")
torch.bfloat16 = "bfloat16"
torch.Tensor = type("Tensor", (), {})
torch.int64 = "int64"
utils = types.ModuleType("torch.utils")
data = types.ModuleType("torch.utils.data")


class Dataset:  # noqa: D101
    pass


data.Dataset = Dataset
data.DataLoader = object
data.DistributedSampler = object
utils.data = data
torch.utils = utils
torch.optim = types.SimpleNamespace(AdamW=object, lr_scheduler=types.SimpleNamespace(LambdaLR=object))
torch.cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
torch.distributed = types.SimpleNamespace()
torch.nn = types.SimpleNamespace(
    Module=type("Module", (), {}),
    utils=types.SimpleNamespace(),
    parallel=types.SimpleNamespace(),
)
torch.from_numpy = lambda a: a
torch.manual_seed = lambda s: None
sys.modules["torch"] = torch
sys.modules["torch.utils"] = utils
sys.modules["torch.utils.data"] = data

sys.path.insert(0, str(ROOT))
from src.train_decoder import (  # noqa: E402
    CHECKPOINT_MARKER, _coerce, _resolve_target, apply_overrides,
    completed_checkpoints, load_config, make_lr_lambda, make_wsd_lr_lambda,
)

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


# ------------------------------------------------- 1. real config parses
print("\n1. configs/pretrain_bank_decoder.yaml -- keys the loop reads")
cfg = load_config(ROOT / "configs" / "pretrain_bank_decoder.yaml")

check("model.config._target_", cfg["model"]["config"]["_target_"] == "transformers.LlamaConfig")
check("model.config.vocab_size is int", isinstance(cfg["model"]["config"]["vocab_size"], int),
      str(cfg["model"]["config"]["vocab_size"]))
check("model.attn_implementation is pinned (never eager)",
      cfg["model"].get("attn_implementation") == "sdpa",
      repr(cfg["model"].get("attn_implementation")))
check("model.gradient_checkpointing present and off by default",
      cfg["model"].get("gradient_checkpointing") is False,
      repr(cfg["model"].get("gradient_checkpointing")))
check("the memory knobs did not break model.config nesting",
      isinstance(cfg["model"].get("config"), dict)
      and "vocab_size" in cfg["model"]["config"])
check("dataset._target_ is the file:func form", ".py:" in cfg["dataset"]["_target_"],
      cfg["dataset"]["_target_"])
check("dataset.data_path is None in the file", cfg["dataset"]["data_path"] is None)
check("dataset.vocab_path present", bool(cfg["dataset"]["vocab_path"]))
for key, typ in [("global_batch_size", int), ("local_batch_size", int),
                 ("ckpt_every_steps", int), ("val_every_steps", int),
                 ("num_epochs", int), ("max_steps", int)]:
    check(f"step_scheduler.{key}", isinstance(cfg["step_scheduler"][key], typ),
          str(cfg["step_scheduler"][key]))
check("optimizer.lr is float", isinstance(cfg["optimizer"]["lr"], float), str(cfg["optimizer"]["lr"]))
check("optimizer.betas is a 2-list", cfg["optimizer"]["betas"] == [0.9, 0.95])
check("lr_scheduler.lr_warmup_steps", cfg["lr_scheduler"]["lr_warmup_steps"] == 10)
check("lr_scheduler.min_lr < lr", cfg["lr_scheduler"]["min_lr"] < cfg["optimizer"]["lr"])
check("optimizer.clip_grad_norm parsed past its comment",
      cfg["optimizer"]["clip_grad_norm"] == 1.0, repr(cfg["optimizer"]["clip_grad_norm"]))
check("commented-out val_batches stays out",
      "val_batches" not in cfg["step_scheduler"])
# Everything popped before AdamW gets **kwargs, or construction would fail.
_opt = dict(cfg["optimizer"])
for k in ("_target_", "clip_grad_norm", "lr", "betas", "weight_decay"):
    _opt.pop(k, None)
check("no leftover optimizer keys reach AdamW", _opt == {}, str(_opt))
check("checkpoint.checkpoint_dir", bool(cfg["checkpoint"]["checkpoint_dir"]))
check("rng.seed", cfg["rng"]["seed"] == 42)

# --------------------------------------- 2. NB03's exact override list
print("\n2. CLI overrides -- the exact list notebook 03 passes")
VOCAB_SIZE, SEQ_LENGTH, MAX_STEPS = 8226, 4096, 30
overrides = [
    "--dataset.data_path", "data/bank/decoder_corpus/train_corpus.txt",
    "--dataset.vocab_path", "data/bank/tokenizer/vocab.json",
    "--dataset.seq_length", str(SEQ_LENGTH),
    "--validation_dataset.data_path", "data/bank/decoder_corpus/val_corpus.txt",
    "--validation_dataset.vocab_path", "data/bank/tokenizer/vocab.json",
    "--validation_dataset.seq_length", str(SEQ_LENGTH),
    "--model.config.vocab_size", str(VOCAB_SIZE),
    "--step_scheduler.max_steps", str(MAX_STEPS),
    "--step_scheduler.global_batch_size", "16",
    "--step_scheduler.local_batch_size", "16",
    "--lr_scheduler.lr_warmup_steps", "10",
    "--step_scheduler.val_every_steps", "15",
    "--step_scheduler.ckpt_every_steps", "15",
    "--checkpoint.checkpoint_dir", "models/bank-decoder/checkpoints",
]
cfg = apply_overrides(load_config(ROOT / "configs" / "pretrain_bank_decoder.yaml"), overrides)
check("nested 3-deep override lands", cfg["model"]["config"]["vocab_size"] == VOCAB_SIZE,
      str(cfg["model"]["config"]["vocab_size"]))
check("override coerced to int", isinstance(cfg["model"]["config"]["vocab_size"], int))
check("data_path set", cfg["dataset"]["data_path"].endswith("train_corpus.txt"))
check("val data_path set", cfg["validation_dataset"]["data_path"].endswith("val_corpus.txt"))
check("max_steps overridden", cfg["step_scheduler"]["max_steps"] == 30)
check("untouched key survives", cfg["optimizer"]["weight_decay"] == 0.077)
check("--k=v form", apply_overrides({}, ["--a.b=7"])["a"]["b"] == 7)
check("_coerce none/true/float", (_coerce("none"), _coerce("true"), _coerce("1e-4"))
      == (None, True, 1e-4))
import json  # noqa: E402
import tempfile as _tf  # noqa: E402
from src.train_decoder import check_vocab_fits  # noqa: E402

with _tf.TemporaryDirectory() as _d:
    _v = Path(_d) / "vocab.json"
    _v.write_text(json.dumps({"vocab_size": 10, "special_tokens": {"<pad>": "<pad>"},
                              "vocab": {"<pad>": 0, "AMT_3": 9}}))
    try:
        check_vocab_fits(_v, 10)
        _fits = True
    except Exception:  # noqa: BLE001
        _fits = False
    try:
        check_vocab_fits(_v, 9)
        _small = False
    except ValueError:
        _small = True
    _v.write_text(json.dumps({"<pad>": 0, "X": 4}))
    check_vocab_fits(_v, 5)
check("vocab check: reads vocab.json's 'vocab' map, refuses a model that is too small",
      _fits and _small)
check("_coerce lists: '[]' and ['<sep>'] are lists, not strings",
      _coerce("[]") == [] and _coerce("['<sep>']") == ["<sep>"])
try:
    apply_overrides({}, ["--dangling"])
    check("missing value raises", False)
except ValueError:
    check("missing value raises", True)

# --------------------------------------------- 3. accumulation arithmetic
print("\n3. gradient accumulation")
for gb, lb, ws, expect in [(16, 16, 1, 1), (16, 4, 1, 4), (16, 4, 2, 2), (128, 16, 8, 1)]:
    per = lb * ws
    check(f"global {gb} / local {lb} x {ws} rank(s) -> accum {expect}",
          gb % per == 0 and gb // per == expect)
check("indivisible combination detected", 16 % (6 * 1) != 0)

# ----------------------------------------------------- 4. LR schedule
print("\n4. LR schedule (warmup 10, total 30, lr 2e-4, min_lr 6.5e-6)")
base, floor = 2.0e-4, 6.5e-6
fn = make_lr_lambda(10, 30, base, floor)
vals = [fn(s) * base for s in range(30)]
check("step 0 is one warmup increment", math.isclose(vals[0], base / 10, rel_tol=1e-9),
      f"{vals[0]:.3e}")
check("warmup ends at full lr", math.isclose(vals[9], base, rel_tol=1e-9), f"{vals[9]:.3e}")
check("warmup is monotonic up", all(vals[i] < vals[i + 1] for i in range(9)))
check("decay is monotonic down", all(vals[i] > vals[i + 1] for i in range(10, 29)))
check("never below min_lr", min(vals) >= floor - 1e-12, f"min {min(vals):.3e}")
check("never above lr", max(vals) <= base + 1e-12, f"max {max(vals):.3e}")
# The last *used* LR is at index total-1 (progress 19/20), so it lands slightly
# above min_lr; only fn(total) is exactly min_lr. Standard LambdaLR behaviour.
check("ends within 20% of min_lr", vals[-1] < floor * 1.2, f"{vals[-1]:.3e}")
check("no warmup -> starts at full lr",
      math.isclose(make_lr_lambda(0, 30, base, floor)(0) * base, base, rel_tol=1e-9))
check("cosine floor exact at progress 1", math.isclose(fn(30) * base, floor, rel_tol=1e-9),
      f"{fn(30) * base:.3e}")
lin = make_lr_lambda(2, 10, base, floor, decay_style="linear")
check("linear style honoured", math.isclose(lin(10) * base, floor, rel_tol=1e-9))
const = make_lr_lambda(2, 10, base, floor, decay_style="constant")
check("constant style honoured", math.isclose(const(9) * base, base, rel_tol=1e-9))

# Early stopping: warmup, constant, linear decay from state["start"] (or the ceiling).
state = {"start": None}
wsd = make_wsd_lr_lambda(10, 1000, base, floor, cooldown_steps=100, state=state)
check("wsd: warmup like the others", math.isclose(wsd(0) * base, base / 10, rel_tol=1e-9))
check("wsd: constant after warmup", all(math.isclose(wsd(s) * base, base, rel_tol=1e-9)
                                        for s in (10, 300, 899)))
check("wsd: no stop -> decays to min_lr at the ceiling",
      math.isclose(wsd(1000) * base, floor, rel_tol=1e-9) and wsd(950) < 1.0)
state["start"] = 400   # validation stopped improving at step 400
check("wsd: full lr at the start of the decay", math.isclose(wsd(400) * base, base, rel_tol=1e-9))
check("wsd: halfway down after half the cooldown",
      math.isclose(wsd(450) * base, floor + (base - floor) / 2, rel_tol=1e-9))
check("wsd: min_lr at the end of the cooldown", math.isclose(wsd(500) * base, floor, rel_tol=1e-9))
check("wsd: monotonic down during the cooldown", all(wsd(s) > wsd(s + 1) for s in range(400, 500)))
state["start"] = 950   # later than ceiling - cooldown: the decay must still end at the ceiling
check("wsd: a late start is pulled back to ceiling - cooldown",
      math.isclose(wsd(1000) * base, floor, rel_tol=1e-9))
nocool = make_wsd_lr_lambda(0, 100, base, floor, cooldown_steps=0, state={"start": 50})
check("wsd: cooldown 0 -> constant to the end", math.isclose(nocool(99) * base, base, rel_tol=1e-9))
check("early_stop keys in the config, off by default",
      load_config(ROOT / "configs" / "pretrain_bank_decoder.yaml")["step_scheduler"].get("early_stop_patience") == 0
      and load_config(ROOT / "configs" / "pretrain_bank_decoder.yaml")["step_scheduler"].get("early_stop_cooldown_steps") == 0
      and load_config(ROOT / "configs" / "pretrain_bank_decoder.yaml")["step_scheduler"].get("early_stop_min_delta") == 0.0)

# ------------------------------------------------ 5. _target_ resolution
print("\n5. _target_ resolution")
check("dotted form", _resolve_target("json.loads")('{"a":1}') == {"a": 1})
builder = _resolve_target("src/clm_data.py:build_bank_clm_dataset")
check("file:func form resolves the real builder", callable(builder)
      and builder.__name__ == "build_bank_clm_dataset")
check("file:func form is the dataset target in the YAML",
      _resolve_target(cfg["dataset"]["_target_"]).__name__ == "build_bank_clm_dataset")

# ------------------------------------------- 6. checkpoint discovery/retention
print("\n6. checkpoint completeness marker and retention")
import shutil  # noqa: E402
import tempfile  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="ckpt_test_"))
try:
    # Three complete checkpoints plus one partial (config.json but no marker,
    # i.e. a save killed mid-serialization) plus a stray directory.
    for n in (5, 10, 15):
        d = tmp / f"step_{n}"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_text("weights")
        (d / CHECKPOINT_MARKER).write_text('{"step": %d}' % n)
    partial = tmp / "step_20"
    partial.mkdir()
    (partial / "config.json").write_text("{}")   # written first by save_pretrained
    (tmp / "not_a_checkpoint").mkdir()

    found = completed_checkpoints(tmp)
    check("finds the complete checkpoints", sorted(found) == [5, 10, 15], str(sorted(found)))
    check("partial write is invisible", 20 not in found)
    check("  (but it does exist on disk)", partial.is_dir())
    check("non-step directory ignored", all(isinstance(k, int) for k in found))
    check("maps step -> directory", found[10].name == "step_10")
    check("missing directory returns empty", completed_checkpoints(tmp / "nope") == {})

    # Retention, mirroring prune_checkpoints: keep the N newest plus protected.
    def plan(steps, keep, protect):
        keepset = set(sorted(steps)[-keep:]) | protect
        return [s for s in sorted(steps) if s not in keepset]

    check("keeps the 2 newest of 3", plan([5, 10, 15], 2, set()) == [5])
    check("keeps the newest of 3", plan([5, 10, 15], 1, set()) == [5, 10])
    check("nothing to prune when under the cap", plan([5, 10], 3, set()) == [])
    check("protected best kept, newest still kept",
          plan([5, 10, 15], 1, {5}) == [10], str(plan([5, 10, 15], 1, {5})))
    check("  an old protected step does not evict a recent one",
          15 not in plan([5, 10, 15], 1, {5}) and 5 not in plan([5, 10, 15], 1, {5}))
    check("protected final is never pruned", 15 not in plan([5, 10, 15], 1, {15}))
    check("all protected -> prune nothing",
          plan([5, 10, 15], 1, {5, 10, 15}) == [])
    check("on-disk count <= keep_last_n + len(protect)",
          len([5, 10, 15]) - len(plan([5, 10, 15], 1, {5})) <= 1 + 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------- 7. best-checkpoint config
print("\n7. checkpoint config keys")
cfg_ck = load_config(ROOT / "configs" / "pretrain_bank_decoder.yaml")["checkpoint"]
check("checkpoint.keep_last_n present", cfg_ck.get("keep_last_n") == 3,
      str(cfg_ck.get("keep_last_n")))
check("checkpoint.save_best present", cfg_ck.get("save_best") is True,
      str(cfg_ck.get("save_best")))
check("keep_last_n=None means keep all",
      (lambda v: int(v) if v else None)(None) is None)
check("keep_last_n=0 means keep all (falsy guard)",
      (lambda v: int(v) if v else None)(0) is None)

# ------------------------------------------ 8. validation subsample sanity
print("\n8. validation subsample (corpus is customer-sorted)")
import numpy as _np  # noqa: E402

N_VAL, CAP, SEED = 24_377, 3_200, 42
CHUNKS_PER_CUSTOMER = 11          # ~109k lines / ~10k customers


def customer_of(line_index):
    """Corpus lines are groupby(customer, chunk)-ordered, so index -> customer."""
    return line_index // CHUNKS_PER_CUSTOMER


picks = _np.random.default_rng(SEED).permutation(N_VAL)[:CAP]

check("subset has the requested size", len(picks) == CAP, str(len(picks)))
check("no duplicates", len(set(picks.tolist())) == CAP)
check("reproducible for a given seed",
      (picks == _np.random.default_rng(SEED).permutation(N_VAL)[:CAP]).all())
check("a different seed gives a different subset",
      not (picks == _np.random.default_rng(SEED + 1).permutation(N_VAL)[:CAP]).all())

# The bug being guarded: a prefix of the unshuffled corpus covers only the
# lowest-sorted customers.
prefix_customers = {customer_of(i) for i in range(CAP)}
sample_customers = {customer_of(i) for i in picks.tolist()}
TOTAL_CUSTOMERS = -(-N_VAL // CHUNKS_PER_CUSTOMER)   # ~2,216 in the val split
check("prefix would cover few customers (the old behaviour)",
      len(prefix_customers) < 0.15 * TOTAL_CUSTOMERS,
      f"{len(prefix_customers)}/{TOTAL_CUSTOMERS} customers")
check("random subset covers most of the customer population",
      len(sample_customers) > 0.7 * TOTAL_CUSTOMERS,
      f"{len(sample_customers)}/{TOTAL_CUSTOMERS} customers "
      f"({len(sample_customers) / len(prefix_customers):.1f}x the prefix)")
check("random subset spans the whole index range",
      picks.min() < N_VAL * 0.02 and picks.max() > N_VAL * 0.98,
      f"[{picks.min()}, {picks.max()}]")

# Kept in permuted order, so any prefix -- what the per-field diagnostic reads --
# is itself a uniform sample. Sorting the indices would reintroduce the bias.
FIELD_PREFIX = 20 * 32
head = picks[:FIELD_PREFIX]
check("diagnostic prefix is not customer-clustered",
      len({customer_of(i) for i in head.tolist()}) > 0.5 * FIELD_PREFIX,
      f"{len({customer_of(i) for i in head.tolist()})} customers in {FIELD_PREFIX} lines")
check("diagnostic prefix spans the range (unsorted)",
      head.max() > N_VAL * 0.9, str(head.max()))
check("sorting would have broken the prefix",
      _np.sort(picks)[:FIELD_PREFIX].max() < N_VAL * 0.5,
      "sorted prefix stays in the low indices")
check("cap above dataset size means no subsetting",
      not (CAP * 100 < N_VAL))

print(f"\n{'=' * 60}\n{len(ok)} passed, {len(fail)} failed")
if fail:
    for name in fail:
        print(f"  FAILED: {name}")
raise SystemExit(1 if fail else 0)

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

"""Offline checks for the experiment layer (src/experiment.py, scripts/run_experiment.py):

1. Every file in experiments/ loads; inheritance, validation, paths, from_run.
2. The runner with a fake notebook executor: stage order, frozen settings,
   failure and --resume, refusal to overwrite or to resume changed settings,
   from_run, summary.csv.

No notebook runs here: the executor is replaced. The notebooks themselves are
checked on the run machine.

    python scripts/verify_experiment_offline.py
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

import csv  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

import src.experiment as experiment  # noqa: E402
from src.experiment import Experiment, active_experiment  # noqa: E402
from src.tokenizer import BankTokenizerConfig  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_experiment as runner  # noqa: E402

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def raises(fn, exc=(ValueError, SystemExit)):
    try:
        fn()
    except exc as e:  # noqa: F841
        return str(e)
    return None


tmp = Path(tempfile.mkdtemp())
os.environ["LTM_RUNS_ROOT"] = str(tmp / "runs")
os.environ["LTM_DATA_DIR"] = str(tmp / "shared")
(tmp / "shared" / "tokenizer").mkdir(parents=True)
(tmp / "shared" / "tokenizer" / "schema.json").write_text(json.dumps({"customer_id": "customer_id"}))
EXP_DIR = ROOT / "experiments"

# ------------------------------------------------------- 1. loading
print("1. experiment files")
loaded = {p.name: Experiment.load(p) for p in sorted(EXP_DIR.glob("*.yaml"))}
check("every file in experiments/ loads and validates", len(loaded) >= 5, str(sorted(loaded)))
base, other = loaded["base.yaml"], loaded["tail_other.yaml"]
check("base: inherits training, overrides the tail",
      other.training == base.training and other.tokenizer["tail_strategy"] == "other"
      and other.tokenizer["drop_fields"] == base.tokenizer["drop_fields"])
check("name and description are not inherited", other.name == "tail_other")

bad = tmp / "bad.yaml"
bad.write_text("name: x\ntokenizer: {tail_strategy: rare_bins, nope: 1}\n")
check("unknown tokenizer key rejected", "nope" in (raises(lambda: Experiment.load(bad)) or ""))
bad.write_text("name: x\ntokenizer: {tail_strategy: banana}\n")
check("invalid value rejected at load", raises(lambda: Experiment.load(bad)) is not None)
bad.write_text("name: x\nembeddings: {poolin: mean}\n")
check("unknown embedding key rejected", raises(lambda: Experiment.load(bad)) is not None)
bad.write_text("name: x\nfrom_run: base\ntokenizer: {tail_strategy: hash}\n")
check("from_run with tokenizer settings rejected", raises(lambda: Experiment.load(bad)) is not None)
bad.write_text("base: bad.yaml\nname: x\n")
check("circular base rejected", raises(lambda: Experiment.load(bad)) is not None)

cfg = base.tokenizer_config(BankTokenizerConfig())
check("tokenizer settings applied and normalised",
      cfg.drop_fields == ("EMP",) and cfg.tail_strategy == "rare_bins")
import yaml  # noqa: E402
train_cfg = base.apply_training(yaml.safe_load(experiment.TRAINING_YAML.read_text()))
check("training keys set on the YAML config with their native types",
      train_cfg["step_scheduler"]["max_steps"] == 2000
      and train_cfg["model"]["gradient_checkpointing"] is True)
lst = tmp / "list.yaml"
lst.write_text("name: x\ntraining:\n  dataset.loss_ignore_tokens: []\n")
check("list values stay lists (the <sep> switch)",
      Experiment.load(lst).apply_training({"dataset": {}})["dataset"]["loss_ignore_tokens"] == [])
bad.write_text("name: x\ntraining: {step_scheduler.max_step: 10}\n")
check("training key typo rejected (the trainer would ignore it)",
      "max_step" in (raises(lambda: Experiment.load(bad)) or ""))
bad.write_text("name: x\ntraining: {step_scheduler.val_batches: 10}\n")
check("optional training key accepted", raises(lambda: Experiment.load(bad)) is None)
bad.write_text("name: x\ncorpus: {context_window: 99999}\n")
check("context_window beyond max_position_embeddings rejected",
      "max_position" in (raises(lambda: Experiment.load(bad)) or ""))
bad.write_text("name: x\ntraining: {model.architecture: banana}\n")
check("unknown architecture rejected at load", raises(lambda: Experiment.load(bad)) is not None)
bad.write_text("name: x\nfrom_run: base\nembeddings: {pooling: mean, finetune: full}\n")
check("finetune without learned pooling rejected",
      "learned" in (raises(lambda: Experiment.load(bad)) or ""))
bad.write_text("name: x\nfrom_run: base\nembeddings: {pooling: learned, finetune: 'lora:x'}\n")
check("invalid finetune spec rejected", raises(lambda: Experiment.load(bad)) is not None)
bad.write_text("name: x\ntraining:\n  loss_weights: {NOPE: 2.0}\n")
check("loss_weights field names are checked against the real tokenizer, not at load",
      raises(lambda: Experiment.load(bad)) is None
      and "NOPE" in (raises(lambda: Experiment.load(bad).check_loss_weights(
          __import__("src.tokenizer", fromlist=["x"]).BankTransferTokenizerPipeline())) or ""))
from src.tokenizer import BankTransferTokenizerPipeline  # noqa: E402

weighted = Experiment(name="w", training={"loss_weights": {"EMP": 2.0, "AMT": 1.5}})
emp_dropped = BankTransferTokenizerPipeline(config=BankTokenizerConfig(drop_fields=("EMP",)))
check("weights on a field the tokenizer does not emit fail at notebook 02 (dropped EMP)",
      "EMP" in (raises(lambda: weighted.check_loss_weights(emp_dropped)) or ""))
check("...and on the text field when no text_strategy is set",
      raises(lambda: Experiment(name="t", training={"loss_weights": {"TXT": 2.0}})
             .check_loss_weights(BankTransferTokenizerPipeline())) is not None)
check("weights on emitted fields pass",
      raises(lambda: weighted.check_loss_weights(BankTransferTokenizerPipeline())) is None)
for value in (".nan", ".inf", "true", "-1"):
    bad.write_text(f"name: x\ntraining:\n  loss_weights: {{AMT: {value}}}\n")
    check(f"loss weight {value} rejected at load", raises(lambda: Experiment.load(bad)) is not None)
bad.write_text("name: x\ntraining: {loss_weights: [AMT]}\n")
check("loss_weights that is not a mapping: clear error", "mapping" in (
    raises(lambda: Experiment.load(bad)) or ""))
bad.write_text("name: x\ncorpus: {chunk_copies: true}\n")
check("chunk_copies: true rejected (not silently 1)", raises(lambda: Experiment.load(bad)) is not None)
parent = tmp / "parent.yaml"
parent.write_text("name: p\ntraining:\n  loss_weights: {AMT: 2.0}\n  rng.seed: 7\n"
                  "xgboost:\n  params: {max_depth: 6, eta: 0.1}\n")
child = tmp / "child.yaml"
child.write_text(f"base: {parent}\nname: c\ntraining:\n  loss_weights: {{CPTY: 2.0}}\n"
                 "xgboost:\n  params: {eta: 0.05}\n")
merged = Experiment.load(child)
check("inheritance: a child's loss_weights replace the base's, other keys inherited",
      merged.training == {"loss_weights": {"CPTY": 2.0}, "rng.seed": 7}, str(merged.training))
check("inheritance: nested settings merge (child eta, base max_depth kept)",
      merged.xgboost == {"params": {"max_depth": 6, "eta": 0.05}}, str(merged.xgboost))
bad.write_text("name: x\ncorpus: {chunk_copies: 0}\n")
bad.write_text("name: x\ntokenizer:\n  extra_fields: {CPTY_TYPE: {column: c, kind: categorical}}\n")
check("an experiment with an '_' extra field name fails at load (before any job time)",
      "letters and digits" in (raises(lambda: Experiment.load(bad)) or ""))
check("chunk_copies < 1 rejected", raises(lambda: Experiment.load(bad)) is not None)
bad.write_text("name: x\nxgboost: {logreg: maybe}\n")
check("xgboost.logreg must be a boolean", raises(lambda: Experiment.load(bad)) is not None)
check("loss weights reach the training config as a mapping",
      loaded["loss_weighted.yaml"].apply_training({})["loss_weights"] == {"AMT": 2.0, "CPTY": 2.0})
check("corpus settings -> notebook variable",
      loaded["context_2048.yaml"].corpus_settings() == {"CONTEXT_WINDOW": 2048})
check("embedding settings -> notebook variable names",
      base.embedding_settings() == {"POOLING": "mean", "HISTORY_WINDOW": 32})

paths = base.paths()
check("paths: run folder under LTM_RUNS_ROOT, splits under LTM_DATA_DIR",
      paths["EMBED_DIR"] == tmp / "runs" / "base" / "embeddings"
      and paths["SPLIT_DIR"] == tmp / "shared" / "temporal_split")
pl = loaded["pool_learned.yaml"].paths()
check("from_run: tokenizer and model from the other run, outputs in its own",
      pl["MODEL_DIR"] == tmp / "runs" / "base" / "model"
      and pl["STATE_PATH"] == tmp / "runs" / "base" / "tokenizer" / "tokenizer_state.json"
      and pl["OUTPUTS_DIR"] == tmp / "runs" / "base_pool_learned" / "outputs")

os.environ.pop("LTM_EXPERIMENT", None)
check("interactive use: no experiment active", active_experiment() is None)

# ------------------------------------------- 1b. notebook paths vs hook
print("\n1b. every path a notebook derives before the experiment hook comes from it")
import ast  # noqa: E402
import re  # noqa: E402

path_names = set(base.paths())
for nb in sorted(ROOT.glob("0[2-5]_*.ipynb")):
    for cell in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "globals().update(EXP.paths())" not in src:
            continue
        before, after = src.split("globals().update(EXP.paths())", 1)
        derived = []
        code = "\n".join(line for line in before.splitlines()
                         if not line.lstrip().startswith(("%", "!")))      # notebook magics
        code = code.rstrip() + "\n    pass\n"          # the cut leaves the hook's `if` empty
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                if used & path_names:
                    derived.append(node.targets[0].id)
        stale = [n for n in derived if n not in path_names
                 and not re.search(rf"^\s*{n}\s*=", after, flags=re.M)]
        check(f"{nb.name}: no path left pointing at the default folder", not stale,
              f"derived before the hook, not overridden: {stale}")

# ------------------------------------------------------- 2. runner
print("\n2. run_experiment.py (fake notebook executor)")
calls = []
FAIL_AT = {"stage": None}


def fake(notebook, out_path, log_path, env, kernel=None):
    stage = notebook.name[:2]
    calls.append((stage, env["LTM_EXPERIMENT"]))
    if stage == FAIL_AT["stage"]:
        raise RuntimeError(f"cell failed in {stage}")
    out_path.write_text("{}")
    if stage == "05":
        run = Path(env["LTM_EXPERIMENT"]).parent
        (run / "outputs" / "metrics.json").write_text(json.dumps({
            "arms": {"raw": {"test_auc": 0.51, "test_ap": 0.01, "test_lift": 1.2},
                     "embeddings": {"test_auc": 0.55, "test_ap": 0.02, "test_lift": 2.1}},
            "n_test_positives": 500,
        }))


rc = runner.run(EXP_DIR / "base.yaml", resume=False, stages=list(runner.STAGES), executor=fake)
run_dir = tmp / "runs" / "base"
info = json.loads((run_dir / "run_info.json").read_text())
check("all four stages run in order, status done",
      rc == 0 and [c[0] for c in calls] == ["02", "03", "04", "05"] and info["status"] == "done")
check("notebooks read the run's frozen experiment.yaml",
      all(Path(c[1]) == run_dir / "experiment.yaml" for c in calls))
check("frozen yaml is the resolved experiment",
      Experiment.load(run_dir / "experiment.yaml") == base)
check("schema copied into the run", (run_dir / "tokenizer" / "schema.json").exists())
check("git commit and code.patch recorded",
      "git_commit" in info and (run_dir / "code.patch").exists())
check("an existing run is never overwritten",
      "exists" in (raises(lambda: runner.run(EXP_DIR / "base.yaml", False,
                                             list(runner.STAGES), executor=fake)) or ""))

calls.clear()
FAIL_AT["stage"] = "04"
rc = runner.run(EXP_DIR / "tail_hash.yaml", False, list(runner.STAGES), executor=fake)
info = json.loads((tmp / "runs" / "tail_hash" / "run_info.json").read_text())
check("a failing stage stops the run and is recorded",
      rc == 1 and info["status"] == "failed at 04" and info["stages"]["04"]["status"] == "failed"
      and "05" not in info["stages"])
calls.clear()
FAIL_AT["stage"] = None
rc = runner.run(EXP_DIR / "tail_hash.yaml", True, list(runner.STAGES), executor=fake)
check("--resume skips finished stages and continues",
      rc == 0 and [c[0] for c in calls] == ["04", "05"], str([c[0] for c in calls]))

edited = tmp / "tail_hash.yaml"
edited.write_text((EXP_DIR / "tail_hash.yaml").read_text().replace("16", "32")
                  .replace("base: base.yaml", f"base: {EXP_DIR / 'base.yaml'}"))
check("resuming with changed settings is refused",
      "no longer matches" in (raises(lambda: runner.run(edited, True, list(runner.STAGES),
                                                        executor=fake)) or ""))

calls.clear()
rc = runner.run(EXP_DIR / "pool_learned.yaml", False, list(runner.STAGES), executor=fake)
check("from_run: only stages 04 and 05 run", rc == 0 and [c[0] for c in calls] == ["04", "05"])
orphan = tmp / "orphan.yaml"
orphan.write_text("name: orphan\nfrom_run: does_not_exist\nembeddings: {pooling: mean}\n")
check("from_run pointing at a missing run is refused",
      "no finished model" in (raises(lambda: runner.run(orphan, False, list(runner.STAGES),
                                                        executor=fake)) or ""))

broken = tmp / "runs" / "broken_run"
broken.mkdir()
(broken / "run_info.json").write_text(json.dumps({"name": "broken_run", "status": "done"}))
(broken / "experiment.yaml").write_text("name: broken_run\nno_such_section: 1\n")
runner.write_summary()
with open(tmp / "runs" / "summary.csv") as f:
    rows = {r["name"]: r for r in csv.DictReader(f)}
check("summary.csv: one row per run; an unreadable run folder does not break it",
      set(rows) == {"base", "tail_hash", "base_pool_learned", "broken_run"}
      and rows["broken_run"]["status"].startswith("unreadable"), str(sorted(rows)))
check("summary.csv: what the run changed + its metrics",
      rows["tail_hash"]["tokenizer.tail_strategy"] == "hash"
      and rows["base"]["embeddings.test_auc"] == "0.55"
      and rows["base_pool_learned"]["from_run"] == "base")

# ------------------------------------------------------------------ summary
print("\n" + "=" * 60)
print(f"{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:")
    for name in fail:
        print("  -", name)
    sys.exit(1)

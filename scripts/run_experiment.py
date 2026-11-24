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

"""Run one experiment end to end, unattended -- the command a Domino job runs.

    python scripts/run_experiment.py experiments/base.yaml
    python scripts/run_experiment.py experiments/base.yaml --resume   # after a crash
    python scripts/run_experiment.py --prepare                         # notebook 01, once
    python scripts/run_experiment.py --summary                         # rebuild the table

Executes notebooks 02 (tokenize) -> 03 (pretrain) -> 04 (embed) -> 05 (XGBoost)
headless, with ``LTM_EXPERIMENT`` pointing at the run's frozen YAML, and keeps
everything in ``<runs root>/<name>/``::

    experiment.yaml       resolved settings (bases merged) -- what the run used
    run_info.json         status per stage, timings, git commit, host, versions
    code.patch            uncommitted changes at launch (empty if clean)
    notebooks/            executed copies, with every output and plot
    logs/                 nbconvert output per stage
    tokenizer/ decoder_corpus/ model/ embeddings/ outputs/
    outputs/metrics.json  test/val metrics per arm (written by notebook 05)

When every stage is done, the files that can be rebuilt (token corpus, embeddings,
unused checkpoints) are deleted to save disk space -- see scripts/clean_runs.py;
``--keep-files`` keeps them. ``--resume`` skips stages already done. A run folder is never overwritten:
change the name for a new run. ``<runs root>/summary.csv`` gets one row per
finished run: what it changed and its metrics.
"""

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment import (  # noqa: E402
    ENV_EXPERIMENT, Experiment, runs_root, shared_data_dir,
)

STAGES = {
    "02": "02_seq_preproc_tokenization.ipynb",
    "03": "03_foundation_model_training.ipynb",
    "04": "04_inference_embedding_extraction.ipynb",
    "05": "05_xgboost_fraud_detection.ipynb",
}
PREPARE = "01_dataset_baseline.ipynb"


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def git(*args) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.decode("utf-8", errors="replace")   # not the console codepage


RUN_KERNEL = "ltm-run"


def own_kernel(where: Path, env: dict) -> str:
    """A kernel spec running *this* interpreter, made visible through JUPYTER_PATH.

    The notebooks ask for "python3", which on a Domino image can resolve to a
    different Python than the one holding torch and cuDF. Executing with the
    runner's own interpreter removes that ambiguity.
    """
    spec = where / "kernels" / RUN_KERNEL
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"LTM run ({sys.executable})",
        "language": "python",
    }))
    env["JUPYTER_PATH"] = os.pathsep.join(filter(None, [str(where), env.get("JUPYTER_PATH")]))
    return RUN_KERNEL


def execute_notebook(notebook: Path, out_path: Path, log_path: Path, env: dict,
                     kernel: str = None) -> None:
    """Run *notebook* top to bottom; the executed copy goes to *out_path*.

    On the first failing cell, prints the end of the log (where the actual
    error is) and raises RuntimeError with it. The executed copy up to that
    cell is still written, so the full traceback is in the notebook too.
    """
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
           str(notebook), "--output", out_path.name, "--output-dir", str(out_path.parent),
           "--ExecutePreprocessor.timeout=-1"]   # errors stop the run (nbconvert default)
    if kernel:
        cmd.append(f"--ExecutePreprocessor.kernel_name={kernel}")
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        message = (f"{notebook.name} failed. End of {log_path}:\n{tail}\n"
                   f"(full traceback in {out_path})")
        print(message, file=sys.stderr)
        raise RuntimeError(message)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def prepare_run(exp: Experiment, run_dir: Path) -> dict:
    for sub in ("notebooks", "logs", "embeddings", "outputs", "tokenizer",
                "decoder_corpus", "model"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.yaml").write_text(exp.to_yaml())

    if not exp.from_run:
        # The schema notebook 01 wrote, plus this experiment's schema settings.
        shared_schema = shared_data_dir() / "tokenizer" / "schema.json"
        if not shared_schema.exists():
            raise SystemExit(f"{shared_schema} not found: run `--prepare` (notebook 01) first.")
        schema = json.loads(shared_schema.read_text())
        schema.update(exp.schema)
        write_json(run_dir / "tokenizer" / "schema.json", schema)

    (run_dir / "code.patch").write_text(git("diff", "HEAD"), encoding="utf-8")
    info = {
        "name": exp.name,
        "status": "running",
        "created": now(),
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain").strip()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "data_dir": str(shared_data_dir()),
        "from_run": exp.from_run,
        "stages": {},
    }
    write_json(run_dir / "run_info.json", info)
    return info


def summary_row(run_dir: Path) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text())
    # Not re-validated: a finished run is a record, even if the current code
    # would reject some of its settings.
    exp = Experiment.load(run_dir / "experiment.yaml", validate=False)
    row = {"name": info["name"], "status": info["status"], "finished": info.get("finished", ""),
           "git_commit": info.get("git_commit", "")[:10], "git_dirty": info.get("git_dirty")}
    row.update({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in exp.summary_fields().items()})
    metrics_path = run_dir / "outputs" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        for arm, m in metrics.get("arms", {}).items():
            for key in ("test_auc", "test_auc_se", "test_ap", "test_lift",
                        "test_recall_at_1pct_fpr", "test_precision_at_top_1pct",
                        "test_value_recall_at_top_1pct", "val_auc"):
                if key in m:
                    row[f"{arm}.{key}"] = round(float(m[key]), 5)
        if "head_test_auc" in metrics:
            row["head.test_auc"] = round(float(metrics["head_test_auc"]), 5)
        row["test_positives"] = metrics.get("n_test_positives")
    return row


def write_summary() -> Path:
    root = runs_root()
    rows = []
    for d in sorted(root.iterdir()):
        if (d / "run_info.json").exists() and (d / "experiment.yaml").exists():
            try:
                rows.append(summary_row(d))
            except Exception as exc:  # noqa: BLE001 -- one bad folder must not hide the rest
                rows.append({"name": d.name, "status": f"unreadable: {exc}"[:200]})
                print(f"WARNING: {d.name}: could not be summarised ({exc!r})", file=sys.stderr)
    columns = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    out = root / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


def run(exp_path: Path, resume: bool, stages, kernel=None, executor=execute_notebook,
        keep_files: bool = False) -> int:
    exp = Experiment.load(exp_path)
    run_dir = exp.run_dir
    if run_dir.exists() and not resume:
        raise SystemExit(f"{run_dir} exists. Use --resume to continue it, or rename the "
                         "experiment for a new run (run folders are never overwritten).")
    if exp.from_run:
        src_info = runs_root() / exp.from_run / "run_info.json"
        done = json.loads(src_info.read_text())["stages"] if src_info.exists() else {}
        if done.get("03", {}).get("status") != "done":
            raise SystemExit(f"from_run={exp.from_run}: that run has no finished model "
                             f"(stage 03) under {runs_root()}")
        stages = [s for s in stages if s not in ("02", "03")]

    if run_dir.exists():
        info = json.loads((run_dir / "run_info.json").read_text())
        frozen = Experiment.load(run_dir / "experiment.yaml")
        if frozen != exp:
            raise SystemExit(f"{exp_path} no longer matches the run's frozen experiment.yaml; "
                             "a resumed run must keep its settings. Rename for a new run.")
    else:
        info = prepare_run(exp, run_dir)

    env = dict(os.environ, **{ENV_EXPERIMENT: str(run_dir / "experiment.yaml")})
    kernel = kernel or own_kernel(run_dir / ".jupyter", env)
    info["status"] = "running"
    for stage in stages:
        if info["stages"].get(stage, {}).get("status") == "done":
            print(f"[{stage}] already done -- skipped")
            continue
        notebook = STAGES[stage]
        print(f"[{stage}] {notebook} ...", flush=True)
        info["stages"][stage] = {"status": "running", "started": now()}
        write_json(run_dir / "run_info.json", info)
        t0 = time.time()
        try:
            executor(ROOT / notebook, run_dir / "notebooks" / notebook,
                     run_dir / "logs" / f"{stage}.log", env, kernel)
        except Exception as exc:          # noqa: BLE001 -- record any failure
            info["stages"][stage].update(status="failed", error=str(exc)[-2000:],
                                         seconds=round(time.time() - t0))
            info["status"] = f"failed at {stage}"
            write_json(run_dir / "run_info.json", info)
            write_summary()
            print(f"[{stage}] FAILED -- see {run_dir / 'notebooks' / notebook} and "
                  f"{run_dir / 'logs' / (stage + '.log')}")
            return 1
        info["stages"][stage].update(status="done", finished=now(),
                                     seconds=round(time.time() - t0))
        write_json(run_dir / "run_info.json", info)
        print(f"[{stage}] done in {info['stages'][stage]['seconds']}s", flush=True)

    info["status"] = "done"
    info["finished"] = now()
    write_json(run_dir / "run_info.json", info)
    if not keep_files:
        from scripts.clean_runs import clean_run
        freed, _ = clean_run(run_dir, delete=True)
        print(f"Cleaned up : {freed / 1e9:.1f} GB (corpus, embeddings, unused checkpoints; "
              "--keep-files keeps them)")
    print(f"\nRun finished: {run_dir}\nSummary    : {write_summary()}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", nargs="?", type=Path)
    parser.add_argument("--resume", action="store_true", help="continue an existing run")
    parser.add_argument("--stages", default=",".join(STAGES),
                        help="comma-separated subset, e.g. 04,05 (default: all)")
    parser.add_argument("--kernel", help="Jupyter kernel name (default: this interpreter)")
    parser.add_argument("--prepare", action="store_true",
                        help="run notebook 01 into the shared data dir (once per split)")
    parser.add_argument("--summary", action="store_true", help="rebuild summary.csv only")
    parser.add_argument("--keep-files", action="store_true",
                        help="keep corpus, embeddings and every checkpoint after the run")
    args = parser.parse_args(argv)

    if args.summary:
        print(write_summary())
        return 0
    if args.prepare:
        out = shared_data_dir() / "notebooks"
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        kernel = args.kernel or own_kernel(shared_data_dir() / ".jupyter", env)
        try:
            execute_notebook(ROOT / PREPARE, out / PREPARE, out / "01.log", env, kernel)
        except RuntimeError:
            return 1                      # the error was printed above
        print(f"Notebook 01 done -> {shared_data_dir()}")
        return 0
    if not args.experiment:
        parser.error("give an experiment YAML, or --prepare / --summary")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {list(STAGES)}")
    return run(args.experiment, args.resume, stages, args.kernel, keep_files=args.keep_files)


if __name__ == "__main__":
    sys.exit(main())

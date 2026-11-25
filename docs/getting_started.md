# Running the code, step by step

Everything below is typed in a terminal on the run machine (Linux, one GPU), from the
code folder. Do the steps in order; steps 1–4 are done once.

| Step | What | How often |
|---|---|---|
| 1 | Get the code | once (then `git pull`, or `codesync.sh pull`, to update) |
| 2 | Create the Python environment with uv | once |
| 3 | Set the paths in `configs/paths.yaml` | once |
| 4 | Prepare the data | once |
| 5 | Check the label | once |
| 6 | Run an experiment | per experiment |
| 7 | Look at the results | per experiment |

---

## Step 1 — Get the code

```bash
git clone <this repository's URL> ltm
cd ltm
```

(Or on GitHub: **Code → Download ZIP**, upload it, unzip it and `cd` into the folder.
With a zip, runs cannot record the code version.)

**Run machine without GitHub access:** move the code as one file with
`scripts/codesync.sh` (needs only `git` on both sides).

```bash
# where you edit (committed code only: no data, no untracked files)
bash scripts/codesync.sh pack                 # -> ../transfer-ltm.bundle

# upload transfer-ltm.bundle to the run machine, then, in the code folder there:
bash scripts/codesync.sh pull /path/to/transfer-ltm.bundle
```

`pull` makes the folder exactly the bundle's version: changed files are replaced, files
deleted from the repository are deleted, and everything else (data, runs, `.venv`) is left
alone. If files were edited on the run machine it stops and lists them; `--force` saves
them to `.codesync_backup/` first, then replaces them. The first time, upload
`scripts/codesync.sh` by hand too (it comes with the bundle afterwards); that first pull
turns the folder into a git copy, so runs record the code version they used.

Every command below is run from this folder.

## Step 2 — Create the Python environment with uv

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==5.8.1
uv pip install "cudf-cu12==25.6.*" --extra-index-url https://pypi.nvidia.com
```

The last three are installed separately because they come from their own download
servers (PyTorch's and NVIDIA's). These are the versions the earlier runs used; if your
previous workspace had another cuDF version (`uv pip list | grep cudf` there), use that one.

Check that it worked:

```bash
python -c "import torch, cudf, transformers, xgboost; print('GPU:', torch.cuda.is_available())"
```

It must print `GPU: True`.

**Every new terminal:** run `source .venv/bin/activate` first.

> **Do not run `uv sync`.** `uv sync` makes the environment match `uv.lock` exactly, and
> torch, transformers and cuDF are not in it (they come from separate servers), so it
> would uninstall them. To add or update packages, use `uv pip install`.

## Step 3 — Set the paths (one file: `configs/paths.yaml`)

Open **`configs/paths.yaml`**. It is the only place where paths are set: every notebook
and script reads it. It has three lines:

```yaml
raw_file: data/bank/raw/transfers.parquet   # the raw extract (.parquet or .csv)
data_dir: data/bank                         # prepared data (written in step 4)
runs_dir: data/runs                         # results: one folder per experiment
```

Replace them with your locations, for example:

```yaml
raw_file: /mnt/data/my-dataset/transfers.parquet
data_dir: /mnt/data/my-dataset/bank
runs_dir: /mnt/data/my-dataset/runs
```

Use persistent storage (in Domino, a project dataset: find it with `ls /mnt/data`). Paths
that do not start with `/` are relative to the code folder.

**Column names.** If the extract's columns are not named as in the defaults, edit the
`SCHEMA = BankSchema(...)` block in the first code cell of `01_dataset_baseline.ipynb`
(each line maps a role to your column name, e.g. `amount="amt_euro"`).

## Step 4 — Prepare the data

```bash
python scripts/run_experiment.py --prepare
```

This runs notebook 01: it splits the extract by date into train / val / test and writes
them, with the column schema, into `data_dir`. Takes minutes. When it finishes, open
`<data_dir>/notebooks/01_dataset_baseline.ipynb` to see the split dates and the number
of frauds per split.

Run this again only if the extract changes.

## Step 5 — Check the label

```bash
python scripts/diagnose_label.py
python scripts/diagnose_label_level.py --show 5
```

The first reports whether the fraud rate is stable across the three periods; the
second whether the label marks single transfers, or whole accounts (from some date on).
If the second says *account status from a date*, rethink the target before running
experiments.

## Step 6 — Run an experiment

One command per experiment; the first is `base`:

```bash
python scripts/run_experiment.py experiments/base.yaml
```

It runs the four stages one after the other and prints each as it finishes:

| Stage | Notebook | What it does | Time (base) |
|---|---|---|---|
| 02 | `02_seq_preproc_tokenization.ipynb` | turns transfers into tokens | minutes |
| 03 | `03_foundation_model_training.ipynb` | pretrains the model | most of the ~4 h of a full run |
| 04 | `04_inference_embedding_extraction.ipynb` | computes the embeddings | minutes–1 h |
| 05 | `05_xgboost_fraud_detection.ipynb` | trains and evaluates the fraud models | minutes |

Everything it produces goes to `<runs_dir>/base/`.

**If it stops** (error, or the session ends), fix the cause if there is one and run the
same command with `--resume`; finished stages are skipped:

```bash
python scripts/run_experiment.py experiments/base.yaml --resume
```

**To go stage by stage** (e.g. to look at the results of one before starting the next):

```bash
python scripts/run_experiment.py experiments/base.yaml --stages 02
python scripts/run_experiment.py experiments/base.yaml --stages 03 --resume
python scripts/run_experiment.py experiments/base.yaml --stages 04 --resume
python scripts/run_experiment.py experiments/base.yaml --stages 05 --resume
```

**Next experiments**: the same command with another file from `experiments/`. Start
with the two seed repeats:

```bash
python scripts/run_experiment.py experiments/seed_1.yaml
python scripts/run_experiment.py experiments/seed_2.yaml
```

Files named `pool_*`, `window_*`, `ft_*` and `xgb_pca` reuse the model of `base` and
only run stages 04–05: run them after `base` has finished.

For a long run, the workspace must stay open until it ends (or use a Domino job:
`docs/experiments.md`, section 5). A run is never lost: `--resume` continues it.

## Step 7 — Look at the results

The easiest way: open **`06_results.ipynb`**, set `RUN = "base"` (or another run) in its
first cell, and run it cell by cell. It shows the run's status, the language model's
loss curves, every fraud model with its noise band, the results per month, and all
finished runs side by side. It only reads the run folders. Clear its outputs before
committing it (the numbers are private).

The same results as files:

```bash
python scripts/run_experiment.py --summary
```

This writes `<runs_dir>/summary.csv`: one row per experiment, with what it changed and
its test results for every model. Per experiment, `<runs_dir>/<name>/` contains:

| Where | What |
|---|---|
| `notebooks/` | the four executed notebooks, with all outputs and plots |
| `outputs/metrics.json` | every metric, per model and split |
| `outputs/metrics_by_month.csv` | test results per month |
| `outputs/predictions/` | every score of every model (to compute any other metric later) |
| `run_info.json` | status of each stage, timing, code version |
| `logs/` | the log of each stage |

Keep exact results out of the repository: the data is private.

---

## Running the notebooks yourself instead

You can also open the notebooks in VS Code / Jupyter and run them cell by cell, in order
01 → 05, with the **`.venv`** kernel. They read `configs/paths.yaml` like everything else.
Two differences with the runner: the settings are the ones written in each notebook
(not an experiment file), and notebook 03 then trains only **30 steps** (a quick test —
`MAX_STEPS` in its training cell). For real runs, use step 6.

## When something goes wrong

| Message | What to do |
|---|---|
| `Raw extract not found` | `raw_file` in `configs/paths.yaml` does not point at the file. |
| `schema.json not found: run --prepare` | Do step 4 first (or `data_dir` changed since). |
| `Missing required columns` | Column names: end of step 3. |
| `<run> exists. Use --resume ...` | Add `--resume` to continue it, or change `name:` in the YAML for a new run. |
| `... no longer matches the run's frozen experiment.yaml` | The YAML was edited after the run started: give it a new `name:`. |
| `from_run=base: that run has no finished model` | Finish `base` first. |
| `The tokenizer changed since this model was trained` | That run's stages 02–03 must be redone: run it under a new name. |
| A stage `FAILED` | `<runs_dir>/<name>/notebooks/` shows the failing cell and its error; fix, then `--resume`. |
| `No module named ...` | `source .venv/bin/activate`, then `uv pip install <module>`. |
| `couldn't connect to 'huggingface.co'` | The machine has no internet. Only stage 02's GPT-2 comparison used it, and it now skips itself: update the notebook, then `--resume`. |

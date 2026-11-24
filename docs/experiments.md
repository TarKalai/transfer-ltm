# Experiments: what can vary, and how to run it

Every modelling choice in the bank pipeline is a **named switch**. An experiment is a
YAML file that sets some of them; a run executes notebooks 02–05 unattended and keeps
everything it produced in its own folder. Runs are never overwritten, so any result can
be traced back to the exact settings and code that produced it.

- [1. Running an experiment](#1-running-an-experiment)
- [2. What a run keeps](#2-what-a-run-keeps)
- [How the data is used](#how-the-data-is-used)
- [3. The switches](#3-the-switches)
- [4. Adding a new strategy](#4-adding-a-new-strategy)
- [5. Running on Domino as a job](#5-running-on-domino-as-a-job)
- [6. Reading the results](#6-reading-the-results)
- [GPU memory](#gpu-memory)

Step by step from the code to a first run: [getting_started.md](getting_started.md).

---

## 1. Running an experiment

```bash
python scripts/run_experiment.py --prepare                  # notebook 01, once per split design
python scripts/run_experiment.py experiments/base.yaml      # notebooks 02 -> 05
python scripts/run_experiment.py experiments/base.yaml --resume       # after a crash
python scripts/run_experiment.py experiments/base.yaml --stages 04,05 --resume
python scripts/run_experiment.py --summary                  # rebuild summary.csv
```

An experiment file sets only what differs from the defaults:

```yaml
base: base.yaml               # inherit everything, then override
name: tail_hash               # the run folder; must be new
description: base with hash-bucket tail
tokenizer:
  tail_strategy: hash
```

Rules the runner enforces:

| Rule | Why |
|---|---|
| A run folder is never overwritten; a new run needs a new `name`. | Results stay attributable to their settings. |
| `--resume` refuses if the YAML changed since the run started. | A run is one set of settings. |
| Every key is validated before anything runs (unknown key, bad value, training-key typo). | A typo would otherwise leave the default in force, silently. |
| `from_run: <name>` reuses that run's tokenizer and model; only notebooks 04–05 run. It may not set `tokenizer`, `training` or `schema`. | Pooling and XGBoost variants cost minutes, not a 2–3 h retrain. |

Interactive use is unchanged: with `LTM_EXPERIMENT` unset, the notebooks use their own
settings and the `data/bank` layout. To debug a run interactively, set
`LTM_EXPERIMENT=<runs root>/<name>/experiment.yaml` before starting the kernel.

## 2. What a run keeps

```
<runs root>/<name>/
  experiment.yaml        settings actually used (bases merged)
  run_info.json          status per stage, timings, git commit, dirty flag, host
  code.patch             uncommitted code changes at launch (empty if clean)
  notebooks/             executed notebooks with every output and plot
  logs/                  execution log per stage
  tokenizer/             tokenizer_state.json, vocab.json, schema.json
  decoder_corpus/        corpora + corpus_report.json
  model/                 checkpoints/, latest_checkpoint.json
  embeddings/            {split}_embeddings/labels/features, metadata.json,
                         attention_head.pt + {split}_head_scores.npy (learned pooling)
  outputs/               metrics.json, xgb_models/{raw,embeddings,combined}.json, plots
    metrics_by_month.csv          test metrics per month and model (drift)
    predictions/{split}.parquet   __row_id__, label, score_raw, score_embeddings,
                                  score_combined (+ score_head): every score, so any
                                  metric can be added later without retraining
<runs root>/summary.csv  one row per run: what it changed + its metrics
```

**Disk space.** When all stages are done, `run_experiment.py` deletes what can be rebuilt:
`decoder_corpus/*.txt`, the embeddings (`*.npy`, `*_features.parquet`) and every checkpoint
except the one `latest_checkpoint.json` names (`--keep-files` keeps them). Results, the
tokenizer and the model in use stay, so notebooks 06 and 07 work as before; rerunning 05
alone then needs 04 again (`--stages 04,05 --resume`). For older runs:
`python scripts/clean_runs.py` (dry run) then `--delete`.

Where everything lives is set in **`configs/paths.yaml`**, the only place: the raw
extract (`raw_file`), the prepared data from notebook 01 (`data_dir`, default `data/bank`)
and the run folders (`runs_dir`, default `data/runs`). The defaults are under
`data/`, which git ignores.

Within a run, stale artifacts cannot be reused by accident: notebook 04 keys its cache
on the *contents* of the tokenizer state and checkpoint (plus pooling and window), and
refuses a checkpoint trained under a different tokenizer.

## How the data is used

```
 time ─────────────────────────────────────────────────────────────────────►
 │            TRAIN period (80%)             │  VAL (10%)  │  TEST (10%)  │
 │                                            │             │              │
 │ pretraining: ALL transactions, no labels   │ val loss    │  never seen  │
 │ XGBoost training: every positive +         │ XGBoost     │  every       │
 │   NEG_PER_POS negatives each (labelled)    │ early stop  │  reported    │
 │ (learned pooling: the head's own share of  │             │  number      │
 │   customers, left out of XGBoost training) │             │              │
```

- **One time split, three periods** (notebook 01). Cutoffs are dates, so everything
  in val and test happens after everything in train.
- **The train period is used twice, for two different jobs.** Pretraining reads every
  transaction *without* labels (it learns what normal sequences look like); the
  supervised step then takes labelled rows of the *same* period. That is not leakage:
  the language model never saw a label, and the test period is untouched by both.
  Exception: `corpus.pretrain_on_test: true` adds the test period to pretraining (still
  no labels) -- fine for notebook 07, but notebook 05's test scores are then on
  transfers the model has seen.
- **The supervised training set** keeps every positive of the train period and samples
  `NEG_PER_POS` (50) negatives per positive, within `MAX_TRAIN_ROWS` (2M; above it,
  negatives are cut first, down to one per positive). Fewer negatives make it affordable; scores
  are therefore not calibrated probabilities, which ranking metrics do not need.
- **Val and test are used whole** (`EVAL_SAMPLES = None`), at their true fraud rate:
  as many positives as the data has, hence the smallest possible noise on every metric.
  A number instead keeps that many rows, still at the true rate.
- **Embeddings** of a labelled row are computed from its own history (the previous 31
  transfers of the same customer), reaching back across the split cutoffs: that is past
  data, available in production too.

These are notebook 01 settings (`NEG_PER_POS`, `MAX_TRAIN_ROWS`, `EVAL_SAMPLES`), shared by all runs:
changing them means `--prepare` again, and comparing only runs made after it.

## 3. The switches

**Cost** says which stages must re-run when the switch changes: *02–05* means
re-tokenize and retrain (hours); *04–05* means re-embed only (use `from_run`).
**Original** is the value that reproduces the NVIDIA/TabFormer-era behaviour, for
ablations.

### Tokenizer — `tokenizer:` (`BankTokenizerConfig`, cost 02–05)

| Key | Values (default first) | Original | What it does |
|---|---|---|---|
| `sequence_scope` | `customer`, `account` | — | One sequence per customer (accounts interleaved) or per account. |
| `include_account_time_delta` | `false`, `true` | `false` | Extra gap token measured within the account (customer scope). base sets `true`. |
| `drop_fields` | `[]`, e.g. `[EMP]` | `[]` | Token prefixes left out of each transaction. |
| `extra_fields` | `{}` | `{}` | New columns by kind, see below. |
| `field_order` | coarse-to-fine, `AMT` last | `null` | Emission order of the fields. |
| `tail_strategy` | `rare_bins`, `hash`, `other` | `other` | What rare/unseen values of frequency fields become (`tail.py`). |
| `tail_min_count` | `10` | `1` | Values seen fewer times in train go to the tail (this trains the tail tokens). |
| `tail_buckets` | `16` | — | Buckets for `tail_strategy: hash`. |
| `text_strategy` | `null`, `words`, `bag`, `category`, `hashed` | `null` | Free-text field (`text.py`); needs `schema.description`. |
| `text_slots` | `4` | — | Tokens per transaction for the text (`category` forces 1). |
| `text_min_customers` | `5` | — | A word needs this many distinct customers (keeps names out). |
| `text_max_vocab`, `text_hash_size`, `text_readable` | `4096`, `2048`, `false` | — | Text vocabulary size, buckets for `hashed`, readable tokens. |
| `counterparty_top_k` | `2048` | `0` | Counterparties shared by many customers get their own token. |
| `counterparty_min_customers` | `10` | — | Floor for a dedicated counterparty token. |
| `time_delta_first_token` | `true` | `false` | `TDIF_FIRST` for a sequence's first transaction (was gap 0). |
| `time_delta_zero_based_bins` | `true` | `false` | `TDIF_0` reachable; top two intervals kept apart. |
| `time_delta_bins`, `time_delta_max_years` | `48`, `10.0` | — | Gap resolution and horizon. |
| `amount_bins`, `amount_signed_log` | `32`, `false` | — | Amount quantile bins. |
| `include_relative_amount`, `relative_amount_bins` | `false`, `16` | `false` | `AMTREL` token: amount vs the mean of the customer's earlier amounts (log scale); `AMTREL_FIRST` without history. |
| `bic_max_vocab`, `country_max_vocab`, `movement_code_max_vocab`, … | see `bank_pipeline.py` | — | Vocabulary caps per field. |
| `include_account_hash`, `include_customer_hash` | scope-dependent, `false` | — | Identity tokens. |
| `account_token`, `account_order_max` | `hash`, `order`; `16` | `hash` | Account token. `hash`: `ACCTH_<bucket>`, a hash of the account number (a bucket mixes unrelated accounts). `order`: `ACCT_01`, `ACCT_02`, … the account's number within its customer, by first transfer; accounts from `account_order_max` on share the last token. Needs the `account_order` column in the split files: `python scripts/add_account_order.py` once (notebook 01 writes it for new splits). |

`extra_fields` — a new column in one entry (`field_kinds.py`):

```yaml
tokenizer:
  extra_fields:
    CHAN: {column: channel, kind: categorical}
    FEE:  {column: fee_eur, kind: numeric, bins: 16}
```

| Kind | For | Options (default) |
|---|---|---|
| `categorical` | one string/code per row | `max_vocab` (256), `readable` (true) |
| `identifier` | 10^4+ IDs: only recurrence matters | `hash_size` (1024), `top_k` (512), `min_customers` (10) |
| `numeric` | amounts, counts | `bins` (16), `signed_log` (false), `clip_quantiles` |
| `text` | a few words | `strategy` (words), `slots` (4), `min_customers` (5), `max_vocab`, `hash_size`, `readable` |

Names are letters and digits only (tokens are `<NAME>_<value>`, so `_` would make a token's
field ambiguous). Unsure of the kind: `from src.tokenizer import suggest_kind; suggest_kind(df["col"])`.
Extra fields are placed just before `AMT`, can be dropped with `drop_fields`, and
notebook 05 gives the raw arm the same columns, encoded by kind (text not yet).

### Schema — `schema:` (`BankSchema`, cost 02–05)

Column names of the extract. Needed for text: `schema: {description: <column>}`.

### Corpus — `corpus:` (cost 02–05)

| Key | Default | What it does |
|---|---|---|
| `chunk_copies` | `1` | `k > 1`: overlapping training chunks -- a chunk starts every `chunk_size // k` transactions and holds `k` such blocks, so every transaction (except in a sequence's first and last k-1 blocks) is seen exactly k times, with different amounts of history before it. Train corpus only (validation unchanged, so val loss stays comparable); k x the training data. |
| `context_window` | `4096` | Pretraining sequence length in tokens (each customer's history is cut into chunks of this length). Must not exceed `model.config.max_position_embeddings` (8192). Embedding windows (`embeddings.history_window`) may not be longer than this: notebook 04 refuses, since the model never saw such positions. |
| `pretrain_on_test` | `false` | `true`: the test period's sequences are added to the train corpus (pretraining uses no labels). Validation, and so the early stop, stays on the val period. The model has then seen the test period: fine for notebook 07 (customer task); notebook 05's test scores are no longer on unseen transfers, so compare them only with runs that do the same. Run: `pretrain_on_test`. |

### Training — `training:` (cost 03–05)

Dotted keys of `configs/pretrain_bank_decoder.yaml`, applied with their YAML types.
Unknown keys are rejected (the trainer itself would silently ignore them).

| Key | base | Note |
|---|---|---|
| `step_scheduler.max_steps` | 2000 | |
| `step_scheduler.early_stop_patience`, `early_stop_min_delta`, `early_stop_cooldown_steps` | 0 (off), 0.0, 0 | Early stopping: stop once the validation loss has not improved by more than `min_delta` for `patience` validations in a row, after `cooldown_steps` of linear lr decay to `min_lr`. `max_steps` becomes a ceiling; the lr is constant after warmup (`lr_decay_style` ignored). The step where it triggered is printed at the end of training (`early_stop_step` in the training history). Runs: `early_stop`, `early_stop_31m`, `early_stop_large`. |
| `step_scheduler.global_batch_size` / `local_batch_size` | 128 / 32 | local × accumulation = global |
| `model.gradient_checkpointing` | true | needed for local batch 32 at 4096 tokens |
| `lr_scheduler.lr_warmup_steps` | 100 | |
| `step_scheduler.val_every_steps`, `val_batches`, `ckpt_every_steps` | 100, 100, 250 | |
| `checkpoint.keep_last_n` | 3 | bounds disk use |
| `loss_weights` | `{}` | Weight per field on the **training** loss, e.g. `{AMT: 2.0, CPTY: 2.0}` (others 1). Names and values are checked at load; that each field is actually emitted is checked when notebook 02 builds the tokenizer (before any tokenizing). The logged training loss and the validation loss stay unweighted, so curves remain comparable. |
| `rng.seed` | `42` | Pretraining seed (initialisation, data order). The validation subset is drawn with its own fixed seed and size, so runs with different seeds or model sizes are validated on the same sequences. |
| `model.architecture` | `llama` | `llama`, `qwen2`, `mistral`, `gpt2`, `modernbert` (`src/architectures.py`). The size in `model.config.*` is written in Llama's terms and translated, so every architecture gets the same size. Trained from scratch on our tokens: no pretrained text weights. `modernbert` is an encoder: only with `model.objective: mlm`. |
| `model.objective` | `clm` | `clm`: next-token prediction (a decoder). `mlm`: masked-token prediction with `model.architecture: modernbert` -- the model sees both sides of a hidden token. The `<mask>` token is added at training (one id after the tokenizer's vocabulary); the tokenizer does not change. Notebook 04 is unchanged: an embedding window ends at the scored transfer, so nothing after it is seen. Not with `loss_weights`, pooling `surprise` or `finetune` (refused). Its val loss is not comparable with `clm` runs: compare in notebooks 05 and 07. Runs: `mlm`, `mlm_txn`. |
| `mlm.mask_unit`, `mlm.mask_prob` | `token`, `0.15` | `mlm` only. `token`: each field token hidden with probability `mask_prob` (BERT). `transaction`: each transfer hidden whole (all its fields), so it must be rebuilt from the transfers around it, not copied from its own other fields. A hidden position is `<mask>` 80% of the time, a random field token 10%, unchanged 10%. Validation uses the same masks at every validation (`mlm.val_mask_seed`). |
| `model.config.hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size` | 512, 8, 8, 2, 1408 | Model size |
| `dataset.loss_ignore_tokens` + `validation_dataset.loss_ignore_tokens` | `["<sep>"]` | `[]` = original; set both |

### Embeddings — `embeddings:` (cost 04–05)

| Key | Values (default) | What it does |
|---|---|---|
| `pooling` | `mean`, `last_token`, `last_txn`, `learned`, `surprise`, or `+`-joined, e.g. `mean+surprise` | How a window becomes one vector (`decoder_inference.py`, `learned_pooling.py`). `surprise` is an anomaly profile instead: how unexpected the scored transfer is to the model given its history (per-field negative log-likelihood, vs. the window's earlier transfers), `tokens_per_txn + 4` values. |
| `history_window` | `32` | Transactions per window (1 = the original single-transaction input). |
| `history_context` | `true` | Windows reach back across the split boundary. |
| `head_fraction` | `0.5` | Learned pooling: share of train customers fitting the head; XGBoost trains on the rest (every arm). |
| `head_max_negatives`, `head_seed` | `100000`, `0` | Learned pooling: negatives sampled for the head; seed. |
| `finetune` | `none`, `top:N`, `lora:R`, `full` | With `learned` pooling only: part of the model trains on the label together with the head (`src/finetune.py`) -- last N layers, rank-R LoRA adapters on attention, or everything. The head is trained first on the frozen model; fine-tuning is kept only if it beats that on the early-stop slice, otherwise the original weights are restored. |
| `finetune_lr`, `finetune_epochs`, `finetune_batch` | `1e-5`, `3`, `32` | Model learning rate (the head gets 10x), epochs, windows per step. |
| `batch_size`, `restrict_to_sequences` | `256`, `true` | Throughput / scope of extraction. |

### XGBoost — `xgboost:` (cost 05)

`use_pca` (false: all embedding dimensions; true: PCA to `pca_dim`, 64),
`random_state` (42), `params:` merged into `XGB_PARAMS` (one config for all three arms),
`logreg` (true): also fit **logistic regression** on the same three feature sets
(`src/downstream.py`, imputed + standardised, C chosen on val) as a sanity check next to
XGBoost; a fit that hits `max_iter` is flagged (`converged: false`). It is fitted on at most
300,000 training rows (every positive kept) and scored on every row, to bound its cost. Its arms appear as `logreg_raw`, `logreg_embeddings`, `logreg_combined` in the
metrics, predictions and `summary.csv`.

## 4. Adding a new strategy

Each family is a registry: write one small class or function, register it under a
name, then use that name in an experiment. Nothing else changes.

| Family | Base class / decorator | File | Then use |
|---|---|---|---|
| Text strategy | `TextStrategy`, `@register_text_strategy` | `src/tokenizer/text.py` | `text_strategy: <name>` |
| Tail strategy | `TailStrategy`, `@register_tail_strategy` | `src/tokenizer/tail.py` | `tail_strategy: <name>` |
| Field kind | `FieldKind`, `@register_field_kind` | `src/tokenizer/field_kinds.py` | `extra_fields: {X: {kind: <name>}}` |
| Pooling | `@register_pooling("<name>")` | `src/decoder_inference.py` | `pooling: <name>` |
| Tokenizer step (new class) | `BaseTokenizer`, `register_tokenizer` | `src/tokenizer/registry.py` | via a field kind |

A tokenizer-side strategy must also reload from a saved state (store only what
`get_state()` needs), and old states must keep loading: give new config options a
value in `BankTokenizerConfig.LEGACY_DEFAULTS` that reproduces the old behaviour.
Add an offline check next to `scripts/verify_*_offline.py`.

## 5. Running on Domino as a job

A workspace is interactive and ends with the session; a **job** runs one command on
a fresh machine and stops. `scripts/domino_job.sh` is that command.

**Before the first job (once):**

1. **Environment.** A job starts from the project's *compute environment* image,
   not from your workspace: packages installed in a workspace with `uv pip install`
   are not there. Either ask for (or build) an environment with torch 2.8 (cu128),
   transformers, cuDF, xgboost, scikit-learn, pyyaml, nbconvert and ipykernel, or keep
   a venv on the dataset and set `PYTHON=/mnt/data/.../venv/bin/python` for the job.
2. **GPU hardware tier for jobs.** In an earlier session no GPU tier was available for
   jobs, which is why runs happened in a workspace. Check the hardware-tier list on the
   Jobs page; if no GPU tier appears, ask the Domino admin.
3. **Storage.** The job's disk is discarded at the end. Put data and runs on a
   project dataset: in a workspace, find its path (`ls /mnt/data` or
   `ls /domino/datasets/local`) and point the three paths of `configs/paths.yaml` there
   (commit and push it: a job runs the committed code).
4. **Code.** In a Git-based project the job checks out the repository (at
   `/mnt/code`): push before launching. Each run records the commit and any
   uncommitted diff.

**Each experiment:**

1. Jobs → **Run**. File/command: `bash scripts/domino_job.sh experiments/<name>.yaml`
   (first ever: `bash scripts/domino_job.sh --prepare`).
2. Pick the GPU hardware tier and the environment; start.
3. Follow the job's log; at the end the run folder is on the dataset and
   `summary.csv` has a new row.
4. If it fails, the run folder says which stage and why
   (`run_info.json`, `notebooks/`, `logs/`). Fix, push, then re-run with `--resume`.

Pooling and XGBoost variants (`from_run`) only need the base run to have finished;
several can run as separate jobs.

The same launch is possible without the UI: the Domino CLI (`domino run --tier <tier> bash
scripts/domino_job.sh experiments/<name>.yaml`) or the `python-domino` library
(`job_start(...)`). See Domino's documentation for the version your organisation runs.

## 6. Reading the results

Per split and per model (raw, embeddings, combined, and the learned head when used),
`outputs/metrics.json` holds (`src/evaluation.py`):

| Metric | Meaning |
|---|---|
| `auc`, `auc_se` | ROC-AUC and its standard error (Hanley–McNeil) |
| `ap`, `lift` | Average precision; AP / fraud rate (1 = chance) |
| `recall_at_0_1pct_fpr`, `recall_at_1pct_fpr` | Share of frauds caught when 0.1% / 1% of legitimate transfers are flagged |
| `precision_at_top_0_1pct`, `precision_at_top_1pct` | Share of frauds among the 0.1% / 1% highest scores |
| `value_recall_at_top_0_1pct`, `value_recall_at_top_1pct` | Share of the fraud **amount** among the 0.1% / 1% highest scores (val and test; shares only, never totals) |
| `recall_at_top_0_1pct`, `recall_at_top_1pct` | Share of the fraud **count** at the same alert budgets, for comparison |
| `n`, `n_pos`, `prevalence` | Rows, positives, fraud rate |

`summary.csv` has, per run: name, status, commit, every setting the experiment changed,
and the test AUC (+ SE), AP, lift, recall at 1% FPR, precision and fraud value caught in the
top 1% per arm,
plus the learned head's own test AUC. Any other metric:
`metrics_from_predictions("<run>/outputs/predictions/test.parquet")`, or any tool that
reads the predictions files.

**Drift.** `outputs/metrics_by_month.csv` gives every model's test metrics per month. A
metric falling across the test period means performance decays with time since training.
Both drift and value can be redone for runs made before they existed, or for another
split or granularity:
`python scripts/analyze_predictions.py drift|value <run folder> [--split val] [--freq W]`.

**Seeds.** `auc_se` covers the randomness of the test sample only. How much a result moves
when the model is retrained is measured by `seed_1` / `seed_2` against `base`: a strategy
beats `base` only if its gap exceeds both.

The standard error of a test AUC near 0.5 is `sqrt((n_pos + n_neg + 1) / (12 n_pos n_neg))`
(about 0.011 at 750 positives): a difference smaller than two to three of these is noise. Compare a variant to `base`
only on that scale, and prefer changing one thing per experiment.

## GPU memory

The run machine has one L40S (44.4 GiB usable). What can run out, and what protects it:

| Stage | Grows with | Protection |
|---|---|---|
| Pretraining (03) | local batch x 4096 tokens x layers (activations); x vocabulary (logits) | **Automatic**: before training, one real step at full length; the per-step batch is halved until the peak stays under 90% of the GPU. The global batch is unchanged (more accumulation). `step_scheduler.auto_fit_batch: false` disables it. |
| Embedding extraction (04) | batch x window length (x vocabulary for `surprise`) | **Automatic**: a batch that runs out of memory is split in half and retried. `surprise` builds vocabulary logits in chunks of ~1 GiB. |
| Learned head (04) | fixed small batches | Transaction states are held in host RAM (~32 KB per row, `HEAD_MAX_NEGATIVES` bounds it). |
| Fine-tuning (04) | `finetune_batch` x window length, backward through the trained part | **Automatic**: a step that runs out of memory is redone as smaller micro-batches with gradient accumulation (same optimisation). |
| XGBoost (05) | training rows x feature count | `MAX_TRAIN_ROWS` (notebook 01) bounds the training set. |

Measured on an L40S, 4096 tokens, vocabulary ~8k: local batch 32 without
gradient checkpointing runs out of memory; 32 with checkpointing ~12–14 GB; 8 without ~10–12 GB.
Text slots and the tail strategies grow the vocabulary (~10–14k): the logits grow with it, and
the probe adjusts.

**Host RAM** is not protected automatically. Notebook 05 holds train + val + test embeddings
(~2 KB per row at 512 dims, float32) and the combined matrices: with whole val/test (millions of rows
each) and 2M training rows, plan for tens of GB. If the workspace has less, set `EVAL_SAMPLES`.


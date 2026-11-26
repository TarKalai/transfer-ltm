# transfer-ltm: transaction language models for bank transfers

Research code. A language model is pretrained, without labels, on each customer's
sequence of bank transfers; its embeddings are then tested on fraud detection, per
transfer and per customer, against a model on the raw columns.

This is exploratory work. **The data is private and not included, and no results are
reported here.** The code runs on any extract with the same kind of columns
(see [The schema](#the-schema)).

Based on NVIDIA's [Transaction Foundation Model](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model)
developer example (Apache 2.0), which pretrains on public card transactions. This
repository changes it substantially: a fitted tokenizer for bank transfers, a plain
PyTorch training loop, next-token, masked-token, NPPR and CoLES pretraining, history windows for
embeddings, and an experiment runner. Files taken from the original keep their NVIDIA
headers; see [License](#license).

## Pipeline

| Notebook | What it does |
|---|---|
| `01_dataset_baseline.ipynb` | Declare the column names, split by time (train / val / test), build the labelled sets, XGBoost on the raw columns |
| `02_seq_preproc_tokenization.ipynb` | Fit the tokenizer on train only, write one token corpus per split |
| `03_foundation_model_training.ipynb` | Pretrain the model (next-token, masked-token, NPPR or CoLES), with early stopping |
| `04_inference_embedding_extraction.ipynb` | One embedding per labelled transfer, from a window of the customer's previous transfers |
| `05_xgboost_fraud_detection.ipynb` | XGBoost and logistic regression on raw columns, embeddings, and both |
| `06_results.ipynb` | Read one finished run (curves, metrics, checks) |
| `07_customer_task.ipynb` | Customer-level task: one label per customer, cross-validated over customers |
| `08_data_exploration.ipynb` | Checks on the data and on the customer-level labels |

Every modelling choice is a named switch, and an experiment is a YAML file in
`experiments/` that changes one thing from a reference run. `scripts/run_experiment.py`
runs notebooks 02–05 for it, unattended, into its own folder:

```bash
python scripts/run_experiment.py --prepare                  # notebook 01, once
python scripts/run_experiment.py experiments/early_stop.yaml
```

- **Step by step, from the code to a first run:** [docs/getting_started.md](docs/getting_started.md)
- **Every switch, how runs are stored, how to read the results:** [docs/experiments.md](docs/experiments.md)

## Requirements

Linux, one NVIDIA GPU, CUDA 12, Python 3.12. cuDF (RAPIDS) does the preprocessing and
tokenization; PyTorch and Hugging Face Transformers the models; XGBoost and scikit-learn
the downstream models. The environment is built with uv; torch, transformers and cuDF are
installed from their own package servers ([docs/getting_started.md](docs/getting_started.md),
step 2). Neither NeMo nor cuML is needed.

## The schema

The column names are declared once (`BankSchema`, notebook 01) and travel with the
saved tokenizer. Each field becomes one token per transfer:

| Column | Tokenizer | Token |
|---|---|---|
| `amt_euro` | `QuantileBinTokenizer`: quantile bins fitted on train | `AMT_0..31` |
| `datetime` | `TimeDeltaTokenizer`: log-binned gap since the previous transfer | `TDIF_*` (+ `TDIFA_*` within the account) |
| `dc_flag`, `movement_sign`, `movement_code`, `operations_status`, `comment_structure`, `counterparty_country`, `counterparty_bic` | `FrequencyVocabTokenizer`: frequent values keep their own token, rare and unseen ones share tail tokens | `DC_*`, `SIGN_*`, `MVC_*`, `OPST_*`, `CMT_*`, `CTRY_*`, `BIC_*` |
| `counterparty_account` | `HybridHashTokenizer`: counterparties shared by many customers get their own token, the rest are hashed | `CPTY_T*`, `CPTY_*` |
| `account_number` | which of the customer's accounts: hashed, or its number within the customer (`account_token: order`) | `ACCTH_*` or `ACCT_01..16` |
| `employee` | `FrequencyVocabTokenizer`, ordinal (no names) | `EMP_*` (dropped by default) |
| `month`, `dow`, `hour` | `FixedVocabTokenizer` | `MONTH_*`, `DOW_*`, `HOUR_*` |
| free text (optional) | `TextTokenizer`, four strategies | `TXT_*` |
| `customer_id` | no token: it is the sequence key | — |

A sequence is one customer's transfers over all their accounts, in time order
(`sequence_scope: account` gives one per account). New fields can be added by kind
(categorical, identifier, numeric, text) without code: `extra_fields` in
[docs/experiments.md](docs/experiments.md).

Design points worth knowing:

- **The tokenizer is fitted once, on the training period**, and saved. Refitting per split
  would give the same value a different token in each split.
- **The string hash is pinned** (`src/tokenizer/_hash.py`, BLAKE2b, 8 bytes): a hash
  bucket is part of a token's meaning, so a library's hash changing between versions
  would silently renumber them. A saved tokenizer records its hash and refuses another.
- **Embeddings come from a history window** (the scored transfer and the customer's
  previous ones), never from a transfer alone, and the window never reaches past the
  scored transfer.

## Models

Built from scratch on the transfer tokens (no pretrained text weights), in
`src/train_decoder.py`:

- `model.objective: clm`: next-token prediction with a decoder (`llama` by default;
  `qwen2`, `mistral`, `gpt2`).
- `model.objective: mlm`: masked-token prediction with an encoder (`modernbert`), hiding
  single tokens or whole transfers.
- `model.objective: nppr`: next transfer prediction and past reconstruction
  ([Skalski et al. 2023](https://arxiv.org/abs/2401.01641)), on the state at the end of
  each transfer.
- `model.objective: coles`: contrastive learning on slices of each customer's history
  ([Babaev et al. 2022](https://arxiv.org/abs/2002.08232)).

All four train the same transformer; `src/event_objectives.py` holds the last two.

Model sizes, early stopping, pooling, fine-tuning and the rest are switches in
`configs/pretrain_bank_decoder.yaml` and [docs/experiments.md](docs/experiments.md).

## Layout

```
0*_*.ipynb       the pipeline (above)
configs/         paths.yaml (where data and runs live), pretrain_bank_decoder.yaml
experiments/     one YAML per experiment
src/             tokenizer, corpus, training loop, inference, evaluation, experiment runner
scripts/         run_experiment.py, data checks, helpers
docs/            getting started, every switch
```

## License

Apache License 2.0 ([LICENSE](LICENSE)). Derived from NVIDIA's Transaction Foundation
Model developer example, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES; the files
that come from it carry its header and have been modified.

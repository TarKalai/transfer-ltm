"""Labelled rows without the sticky-label leak ("first flag" labels).

In the extract, once a customer has a flagged transfer, every later transfer of that
customer is flagged too. A model then only has to recognise customers already flagged.
Here each customer counts as fraud only at their FIRST flagged transfer; their later
transfers are left out of the labelled rows (they stay in the full splits, so they are
still in the language model's data, which has no labels).

Writes next to notebook 01's files in <data_dir>/temporal_split/:
    train_balanced_first_flag.parquet   every first flag + NEG_PER_POS normal transfers each
    val_eval_first_flag.parquet         every kept transfer of the period
    test_eval_first_flag.parquet        every kept transfer of the period
Notebook 04 reads them when the experiment sets ``embeddings: {labels: first_flag}``.

    python scripts/first_flag_subsets.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import shared_data_dir  # noqa: E402

NEG_PER_POS = 50            # as notebook 01
MAX_TRAIN_ROWS = 2_000_000  # as notebook 01
SEED = 42

split_dir = shared_data_dir() / "temporal_split"
schema = json.loads((shared_data_dir() / "tokenizer" / "schema.json").read_text())
cust, acc, time = schema["customer_id"], schema["account_number"], schema["datetime"]
cols = ["__row_id__", cust, acc, time, "_target"]

splits = {s: pd.read_parquet(split_dir / f"{s}.parquet", columns=cols)
          for s in ("train", "val", "test")}
everything = pd.concat(splits.values())
first = everything[everything["_target"] == 1].groupby(cust)[time].min()
print(f"Customers with a flag: {len(first):,}")
print(f"Share of transfers flagged on or after the first flag: "
      f"{everything.loc[everything[time] >= everything[cust].map(first), '_target'].mean():.1%}")
del everything

for name, df in splits.items():
    keep = df[cust].map(first).isna() | (df[time] <= df[cust].map(first))
    df = df[keep]
    if name == "train":
        pos = df[df["_target"] == 1]
        neg = df[df["_target"] == 0]
        n_neg = min(len(neg), len(pos) * NEG_PER_POS, MAX_TRAIN_ROWS - len(pos))
        df = pd.concat([pos, neg.sample(n=n_neg, random_state=SEED)])
        df = df.sample(frac=1.0, random_state=SEED)
        out = split_dir / "train_balanced_first_flag.parquet"
    else:
        out = split_dir / f"{name}_eval_first_flag.parquet"
    df.reset_index(drop=True).to_parquet(out, index=False)
    print(f"{name:<5} {len(df):>12,} rows  {int(df['_target'].sum()):>7,} frauds "
          f"({df[cust][df['_target'] == 1].nunique():,} customers)  -> {out.name}")

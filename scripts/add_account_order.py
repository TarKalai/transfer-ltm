"""Add the ``account_order`` and ``counterparty_order`` columns to the split files.

``account_order`` is each account's number within its customer: 1 for the customer's
first account, 2 for the next one, ... in order of first transfer, over the FULL
history (train + val + test). The tokenizer option ``account_token: order`` turns it
into the ACCT_01, ACCT_02, ... tokens.

``counterparty_order`` is the same number for the counterparty account when it is one
of the customer's own accounts, 0 otherwise. The option ``own_account_counterparty``
puts that account's ACCT token in the counterparty slot. Notebook 01 writes both
columns for new splits.

Rewrites every parquet file in <data_dir>/temporal_split/ that has the customer and
account columns, adding (or with --force, recomputing) the columns. Rows and the other
columns are unchanged, so finished runs are not affected. Each file is written to a
temporary file first and then swapped in, so a run reading it meanwhile never sees a
half-written file.

    python scripts/add_account_order.py           # files that miss a column
    python scripts/add_account_order.py --force   # every file, recomputed
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import shared_data_dir  # noqa: E402
from src.tokenizer.bank_pipeline import ACCOUNT_ORDER_COL, BankSchema  # noqa: E402
from src.tokenizer.bank_pipeline import COUNTERPARTY_ORDER_COL  # noqa: E402
from src.tokenizer.bank_pipeline import account_order_table, add_account_order  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--force", action="store_true", help="recompute files that have them")
args = parser.parse_args()

split_dir = shared_data_dir() / "temporal_split"
fields = json.loads((shared_data_dir() / "tokenizer" / "schema.json").read_text())
fields.pop("REQUIRED_FIELDS", None)
schema = BankSchema(**fields)
cust, acct, time = schema.customer_id, schema.account_number, schema.datetime
cpty = schema.counterparty_account
added = (ACCOUNT_ORDER_COL, COUNTERPARTY_ORDER_COL)

history = pd.concat([pd.read_parquet(split_dir / f"{s}.parquet", columns=[cust, acct, time])
                     for s in ("train", "val", "test")], ignore_index=True)
history[time] = pd.to_datetime(history[time])
table = account_order_table(history, schema)
del history
n = table.groupby(cust)[ACCOUNT_ORDER_COL].max()
print(f"{len(table):,} accounts of {len(n):,} customers")
print("Accounts per customer (share of customers):")
print(n.clip(upper=17).value_counts(normalize=True).sort_index()
      .rename(index={17: "17+"}).map("{:.2%}".format).to_string())

for path in sorted(split_dir.glob("*.parquet")):
    names = pq.read_schema(path).names
    if cust not in names or acct not in names:
        print(f"skip  {path.name} (no customer/account columns)")
        continue
    if cpty not in names:
        print(f"note  {path.name}: no {cpty!r} column, so no {COUNTERPARTY_ORDER_COL}")
    want = [c for c in added if c != COUNTERPARTY_ORDER_COL or cpty in names]
    if all(c in names for c in want) and not args.force:
        print(f"keep  {path.name} (has {', '.join(want)}; --force recomputes)")
        continue
    data = pq.read_table(path)
    data = data.drop([c for c in added if c in names])
    keys = data.select([c for c in (cust, acct, cpty) if c in names]).to_pandas()
    keyed = add_account_order(keys, schema, table)                  # keeps row order
    missing = int(keyed[ACCOUNT_ORDER_COL].isna().sum())
    if missing:
        raise SystemExit(f"{path.name}: {missing:,} rows with an account not in train/val/test")
    for col in want:
        data = data.append_column(col, pa.array(keyed[col].to_numpy("int32")))
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(data, tmp)
    os.replace(tmp, path)
    own = ""
    if COUNTERPARTY_ORDER_COL in keyed:
        own = f"; counterparty is an own account in {(keyed[COUNTERPARTY_ORDER_COL] > 0).mean():.1%}"
    print(f"wrote {path.name} ({data.num_rows:,} rows{own})")

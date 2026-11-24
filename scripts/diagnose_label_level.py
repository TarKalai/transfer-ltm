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

"""At what level is the label set: transaction, account, or customer?

The label sits on every row, but that does not make it a transaction label.
Three regimes leave different traces in an account's history, read in time order:

*Transaction.*  Only the fraudulent transfers are positive.  Positives come in
short bursts, are interleaved with negatives -- often on the same day -- and
lean heavily towards one direction (outgoing payments).

*Account from a date.*  Once an account is flagged, everything from then on is
positive.  One run of positives that lasts until the account's last row.

*Whole account (or customer).*  Every transaction of a flagged account is
positive, from the first row on; at customer level, every account of the
customer is, from the same day.

Which one holds decides what a per-transaction model can learn, and whether
customer- or account-level sequences fit the label.  This reads the full split
files (all three, concatenated, because an episode can cross a split boundary)::

    python scripts/diagnose_label_level.py --show 5

Prints counts only: no identifiers, amounts or dates leave the data.  The
``--show`` timelines give days relative to each account's first positive.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")

def _configured_data_dir() -> Path:
    """data_dir from configs/paths.yaml (the one place for locations)."""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from src.experiment import shared_data_dir
    return shared_data_dir()


#: Verdict thresholds. Heuristics, printed next to the numbers they judge.
WHOLE_SHARE = 0.95        # median share of positives in a flagged account
SUFFIX_SHARE = 0.80       # flagged accounts whose positives run to the end
MIXED_DAY_SHARE = 0.50    # positive account-days that also hold a negative


def _load_backend():
    """cuDF if it is importable, else pandas.  Either works here."""
    try:
        import cudf

        return cudf, "cudf"
    except ImportError:
        return pd, "pandas"


def _to_pandas(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj


def _schema_columns(data_dir):
    """Resolve column names from the schema notebook 01 writes."""
    names = {"customer_id": "customer_id", "account_number": "account_number",
             "datetime": "datetime", "dc_flag": "dc_flag"}
    schema_path = Path(data_dir) / "tokenizer" / "schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        names.update({k: schema.get(k, v) for k, v in names.items()})
    return names


def _pct(x):
    return f"{x:6.1%}"


def _describe(series, fmt="{:.2f}"):
    q = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return "  ".join(f"p{int(k * 100)}=" + fmt.format(v) for k, v in q.items())


def _rle(labels):
    """[0,0,0,1,1,0] -> 'neg x3 | POS x2 | neg x1'."""
    out, prev, count = [], None, 0
    for y in labels:
        if y == prev:
            count += 1
            continue
        if prev is not None:
            out.append(f"{'POS' if prev else 'neg'} x{count}")
        prev, count = y, 1
    out.append(f"{'POS' if prev else 'neg'} x{count}")
    return " | ".join(out)


def load_flagged(df_lib, split_dir, cols):
    """Every row of every account that has at least one positive, all splits.

    The filter runs on the backend (cuDF on the run machine), so only the
    flagged accounts' rows -- a small fraction -- are brought into pandas.
    Also returns accounts per customer, for the customer-level check.
    """
    import pyarrow.parquet as pq

    cust, acct, dt, dc = (cols[k] for k in ("customer_id", "account_number", "datetime", "dc_flag"))
    paths = [str(split_dir / f"{split}.parquet") for split in SPLITS]

    # Pass 1: flagged accounts over ALL splits, so an account first flagged in
    # val keeps its train history in pass 2.
    key_frames, acct_counts, total_rows, total_pos = [], [], 0, 0
    for path in paths:
        gdf = df_lib.read_parquet(path, columns=[cust, acct, "_target"])
        total_rows += len(gdf)
        total_pos += int(gdf["_target"].sum())
        acct_counts.append(_to_pandas(gdf[[cust, acct]].drop_duplicates()))
        key_frames.append(gdf[gdf["_target"] == 1][[cust, acct]].drop_duplicates())
        del gdf
    keys = df_lib.concat(key_frames, ignore_index=True).drop_duplicates()

    # Pass 2: every row of those accounts.
    frames = []
    for path in paths:
        read = [c for c in (cust, acct, dt, "_target", dc)
                if c in pq.read_schema(path).names]
        gdf = df_lib.read_parquet(path, columns=read)
        frames.append(_to_pandas(gdf.merge(keys, on=[cust, acct], how="inner")))
        del gdf

    df = pd.concat(frames, ignore_index=True)
    rename = {cust: "c", acct: "a", dt: "t", "_target": "y"}
    if dc in df.columns:
        rename[dc] = "d"
    df = df.rename(columns=rename)
    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values(["c", "a", "t"], kind="mergesort").reset_index(drop=True)
    df["k"] = df.groupby(["c", "a"], sort=False).ngroup()

    accounts = pd.concat(acct_counts, ignore_index=True).drop_duplicates()
    accts_per_cust = accounts.groupby(cust).size().rename("n_accounts")
    accts_per_cust.index.name = "c"
    return df, accts_per_cust, total_rows, total_pos


def per_account(df):
    """One row per flagged account: size, positives, runs, where they sit."""
    g = df.groupby("k", sort=False)
    df["i"] = g.cumcount()
    prev = g["y"].shift(fill_value=0)
    df["run_start"] = (df["y"] == 1) & (prev == 0)

    pos = df[df["y"] == 1].groupby("k")
    acc = pd.DataFrame({
        "n_tx": g.size(),
        "n_pos": g["y"].sum(),
        "n_runs": g["run_start"].sum(),
        "first_i": pos["i"].min(),
        "last_i": pos["i"].max(),
        "first_t": pos["t"].min(),
        "last_t": pos["t"].max(),
        "c": g["c"].first(),
    })
    acc["share"] = acc["n_pos"] / acc["n_tx"]
    acc["density"] = acc["n_pos"] / (acc["last_i"] - acc["first_i"] + 1)
    acc["share_after_first"] = acc["n_pos"] / (acc["n_tx"] - acc["first_i"])
    acc["whole"] = acc["n_pos"] == acc["n_tx"]
    acc["suffix"] = (acc["last_i"] == acc["n_tx"] - 1) & (acc["n_runs"] == 1)
    acc["span_days"] = (acc["last_t"] - acc["first_t"]).dt.total_seconds() / 86400
    return acc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check whether the fraud label is set per transaction, account or customer."
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory holding temporal_split/ "
                             "(default: data_dir in configs/paths.yaml)")
    parser.add_argument("--show", type=int, default=5,
                        help="Flagged accounts to print as timelines (default: 5)")
    parser.add_argument("--context", type=int, default=8,
                        help="Transactions shown before/after the first positive (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir or _configured_data_dir()

    split_dir = args.data_dir / "temporal_split"
    absent = [s for s in SPLITS if not (split_dir / f"{s}.parquet").exists()]
    if absent:
        parser.error(f"Missing splits {absent} under {split_dir}. Run notebook 01 first.")

    df_lib, backend = _load_backend()
    cols = _schema_columns(args.data_dir)
    print(f"Backend : {backend}")
    print(f"Columns : {cols}\n")

    df, accts_per_cust, total_rows, total_pos = load_flagged(df_lib, split_dir, cols)
    acc = per_account(df)
    n_acc = len(acc)
    print("=" * 72)
    print(f"{total_pos:,} positives in {total_rows:,} rows; "
          f"{n_acc:,} accounts of {acc['c'].nunique():,} customers hold them")
    print("=" * 72)

    # ---------------------------------------------------------- 1. accounts
    print("\n1. INSIDE A FLAGGED ACCOUNT")
    print(f"  transactions per account     {_describe(acc['n_tx'], '{:.0f}')}")
    print(f"  positives per account        {_describe(acc['n_pos'], '{:.0f}')}")
    print(f"  share of account positive    {_describe(acc['share'])}")
    print(f"  runs of positives            {_describe(acc['n_runs'], '{:.0f}')}")
    print(f"  density between 1st and last {_describe(acc['density'])}")
    print(f"  days from 1st to last pos.   {_describe(acc['span_days'], '{:.1f}')}")
    print(f"  all rows positive (whole)        {_pct(acc['whole'].mean())}")
    print(f"  one run to the end (from date)   {_pct(acc['suffix'].mean())}")
    print(f"  exactly one positive             {_pct((acc['n_pos'] == 1).mean())}")

    # ------------------------------------------------------- 2. same day
    print("\n2. SAME-DAY MIXING (account-days holding a positive)")
    df["day"] = df["t"].dt.floor("D")
    days = df.groupby(["k", "day"])["y"].agg(["sum", "count"])
    pos_days = days[days["sum"] > 0]
    busy = pos_days[pos_days["count"] >= 2]
    mixed = (busy["sum"] < busy["count"]).mean() if len(busy) else float("nan")
    print(f"  positive account-days                  {len(pos_days):,}")
    print(f"  ... with 2+ transactions               {len(busy):,}")
    print(f"  ... of which also hold a negative      {_pct(mixed)}")

    # ------------------------------------------------------- 3. direction
    if "d" in df.columns:
        print("\n3. DIRECTION (flagged accounts only)")
        table = pd.crosstab(df["y"].map({0: "negative", 1: "positive"}), df["d"],
                            normalize="index")
        print(textwrap.indent(table.map(lambda v: f"{v:.1%}").to_string(), "  "))
        print("  Transaction labels lean to one direction; account labels mirror the\n"
              "  account's own mix.")

    # ------------------------------------------------------- 4. customers
    print("\n4. ACROSS A CUSTOMER'S ACCOUNTS")
    per_cust = acc.groupby("c").agg(flagged=("n_tx", "size"), onset_min=("first_t", "min"),
                                    onset_max=("first_t", "max"))
    per_cust = per_cust.join(accts_per_cust, how="left")
    multi = per_cust[per_cust["n_accounts"] >= 2]
    print(f"  flagged customers                       {len(per_cust):,}")
    print(f"  ... with 2+ accounts                    {len(multi):,}")
    if len(multi):
        all_flagged = (multi["flagged"] == multi["n_accounts"]).mean()
        share = multi["flagged"] / multi["n_accounts"]
        print(f"  ... every account flagged               {_pct(all_flagged)}")
        print(f"  share of accounts flagged     {_describe(share)}")
        several = multi[multi["flagged"] >= 2]
        if len(several):
            same_day = ((several["onset_max"] - several["onset_min"])
                        <= pd.Timedelta(days=1)).mean()
            print(f"  2+ flagged accounts, onset within 1 day {_pct(same_day)}  "
                  f"(of {len(several):,})")

    # ------------------------------------------------------- 5. timelines
    if args.show:
        print(f"\n5. TIMELINES ({args.show} random flagged accounts; days relative "
              "to the first positive)")
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(acc.index.to_numpy(), size=min(args.show, n_acc), replace=False)
        for n, k in enumerate(picks, 1):
            rows = df[df["k"] == k]
            a = acc.loc[k]
            print(f"\n  account #{n}: {int(a['n_tx'])} transactions, {int(a['n_pos'])} positive")
            print(f"    labels in order: {_rle(rows['y'].tolist())}")
            lo = max(0, int(a["first_i"]) - args.context)
            hi = int(a["first_i"]) + args.context + 1
            window = rows.iloc[lo:hi]
            rel = (window["t"] - a["first_t"]).dt.total_seconds() / 86400
            for (_, r), d in zip(window.iterrows(), rel):
                direction = f"  {r['d']}" if "d" in window.columns else ""
                print(f"    {d:+9.2f} d{direction}  {'POS' if r['y'] else '-'}")

    # ------------------------------------------------------- verdict
    print("\n" + "=" * 72)
    print("VERDICT (heuristic)")
    print("=" * 72)
    whole = acc["share"].median() >= WHOLE_SHARE
    suffix = acc["suffix"].mean() >= SUFFIX_SHARE
    transaction = mixed >= MIXED_DAY_SHARE and acc["density"].median() < 1
    if whole:
        print(f"  Median share of a flagged account that is positive >= {WHOLE_SHARE:.0%}:\n"
              "  the label marks ACCOUNTS (or customers -- see section 4), not transfers.\n"
              "  A per-transaction target is then a proxy; score accounts instead.")
    elif suffix:
        print(f"  >= {SUFFIX_SHARE:.0%} of flagged accounts are positive from one point to the\n"
              "  end: the label is an ACCOUNT STATUS set from a date on (e.g. after\n"
              "  detection). Rows after that date are not fraud events; the useful target\n"
              "  is the onset, and later rows leak the status into the history.")
    elif transaction:
        print(f"  Positives come in short bursts, and >= {MIXED_DAY_SHARE:.0%} of positive days\n"
              "  also hold negatives on the same account: a TRANSACTION-level label.")
    else:
        print("  No regime dominates. Read the timelines above: some accounts may be\n"
              "  labelled per transfer and others per account -- ask the data owner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

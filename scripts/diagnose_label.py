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

"""What is actually in the label?  Run this before tuning anything.

Every downstream model here is fitted on ``train`` and scored on ``test``.  That
is only meaningful if the label means the same thing on both sides of the
cutoff, and if the supervised step is being shown a representative sample of it.
Two things can break that, and the notebook counts alone cannot tell them apart:

*Truncation.*  Notebook 01's former ``balanced_sample`` capped positives at
``BALANCED_TRAIN_SIZE * MAX_POSITIVE_SHARE``.  When ``train_balanced`` holds
exactly that many positives, the cap bound and an unknown number of positives
were dropped.  The current notebook 01
keeps every positive (``NEG_PER_POS`` negatives each); this check still tells
which kind of ``train_balanced`` is on disk.

*Drift.*  ``val_eval`` and ``test_eval`` come from ``stratified_sample``, which
preserves prevalence, so their rates are the true rates of those
periods.  If the train split's true rate is far below that, the model is fitted
on one labelling regime and scored on another, which produces chance -- or
slightly worse, when the learned ranking inverts -- whatever the features are.

Only the full split files answer either question, so that is what this reads::

    python scripts/diagnose_label.py

Prints rows / positives / rate per split and per month, how concentrated the
positives are across customers, what the supervised subsets kept of each split,
and a verdict.  Exit codes: 2 the label is not stationary, 3 positives were
truncated out of the training sample, 0 neither.
"""

import argparse
import json
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")

def _configured_data_dir() -> Path:
    """data_dir from configs/paths.yaml (the one place for locations)."""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from src.experiment import shared_data_dir
    return shared_data_dir()


#: The supervised subsets notebook 01 writes beside each split. train_balanced
#: is capped, val_eval / test_eval are stratified.
SUBSETS = {
    "train": "train_balanced.parquet",
    "val": "val_eval.parquet",
    "test": "test_eval.parquet",
}

#: Ratio of evaluation-period to training-period positive rate above which the
#: label cannot be treated as one phenomenon.
DRIFT_FATAL = 3.0
DRIFT_WARN = 1.5


def _load_backend():
    """cuDF if it is importable, else pandas.  Either works here."""
    try:
        import cudf

        return cudf, "cudf"
    except ImportError:
        import pandas

        return pandas, "pandas"


def _to_pandas(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else obj


def _schema_columns(data_dir):
    """Resolve the customer / datetime column names written by notebook 01."""
    schema_path = Path(data_dir) / "tokenizer" / "schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        return (
            schema.get("customer_id", "customer_id"),
            schema.get("datetime", "datetime"),
        )
    return "customer_id", "datetime"


def monthly_table(df_lib, gdf, dt_col):
    """Rows / positives / rate per calendar month, oldest first."""
    dt = gdf[dt_col]
    ym = (dt.dt.year * 100 + dt.dt.month).astype("int32")
    table = df_lib.DataFrame({"ym": ym, "y": gdf["_target"]})
    table = table.groupby("ym").agg({"y": ["count", "sum"]})
    table = _to_pandas(table).sort_index()
    table.columns = ["rows", "positives"]
    table["rate_pct"] = table["positives"] / table["rows"] * 100
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check whether the fraud label is stationary across the splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory holding temporal_split/ (default: data_dir in configs/paths.yaml)",
    )
    parser.add_argument(
        "--top-customers",
        type=int,
        default=10,
        help="Customers to report in the concentration check (default: 10)",
    )
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir or _configured_data_dir()

    split_dir = args.data_dir / "temporal_split"
    absent = [s for s in SPLITS if not (split_dir / f"{s}.parquet").exists()]
    if absent:
        parser.error(f"Missing splits {absent} under {split_dir}. Run notebook 01 first.")

    df_lib, backend = _load_backend()
    cust_col, dt_col = _schema_columns(args.data_dir)
    print(f"Backend      : {backend}")
    print(f"Splits       : {split_dir}")
    print(f"Customer col : {cust_col}")
    print(f"Datetime col : {dt_col}\n")

    rates = {}
    full_pos = {}
    total_rows = total_pos = 0

    for split in SPLITS:
        gdf = df_lib.read_parquet(
            str(split_dir / f"{split}.parquet"),
            columns=[dt_col, "_target", cust_col],
        )
        n = len(gdf)
        pos = int(gdf["_target"].sum())
        rates[split] = pos / max(n, 1)
        full_pos[split] = pos
        total_rows += n
        total_pos += pos

        print("=" * 72)
        print(f"{split.upper():<6}{n:>15,} rows{pos:>10,} positives{rates[split]:>11.4%}")
        print("=" * 72)

        table = monthly_table(df_lib, gdf, dt_col)
        print(f"{'month':>8}{'rows':>13}{'positives':>12}{'rate %':>11}")
        for ym_value, row in table.iterrows():
            print(
                f"{int(ym_value):>8}{int(row['rows']):>13,}"
                f"{int(row['positives']):>12,}{row['rate_pct']:>11.4f}"
            )

        # Concentration.  A label injected or sampled to a target count tends to
        # sit in a handful of customers, which no per-transaction model can
        # generalise from; a genuine fraud label is usually spread thin.
        if pos:
            per_customer = _to_pandas(gdf[gdf["_target"] == 1].groupby(cust_col).size())
            top = per_customer.nlargest(min(args.top_customers, len(per_customer)))
            print(
                f"\n  positives spread over {len(per_customer):,} customers "
                f"(median {int(per_customer.median())} each); "
                f"top {len(top)} hold {top.sum() / pos:.1%}"
            )
        else:
            print("\n  no positives in this split")
        print()

        del gdf

    # What the supervised step is actually handed. A train_balanced written by
    # the former notebook 01 could be capped; compare against the split itself.
    print("=" * 72)
    print("SUPERVISED SUBSETS (what notebooks 04/05 actually train and score on)")
    print("=" * 72)
    print(f"  {'file':<24}{'rows':>12}{'positives':>11}{'share':>9}"
          f"{'kept of split':>16}")
    truncated = {}
    for split in SPLITS:
        path = split_dir / SUBSETS[split]
        if not path.exists():
            print(f"  {SUBSETS[split]:<24}{'-- not written --':>48}")
            continue
        sub = df_lib.read_parquet(str(path), columns=["_target"])
        n_sub, pos_sub = len(sub), int(sub["_target"].sum())
        kept = pos_sub / max(full_pos[split], 1)
        print(f"  {SUBSETS[split]:<24}{n_sub:>12,}{pos_sub:>11,}"
              f"{pos_sub / max(n_sub, 1):>9.3%}{kept:>15.1%}")
        # Only train is meant to keep every positive. val/test are stratified
        # down to EVAL_SAMPLES on purpose, preserving prevalence -- that is by
        # design, not a defect, though it does cost statistical power.
        if split == "train" and pos_sub < full_pos[split]:
            truncated[split] = (pos_sub, full_pos[split])
        del sub

    if "train" in truncated:
        kept_n, had = truncated["train"]
        print(
            f"\n  ** {SUBSETS['train']} holds {kept_n:,} of the split's {had:,} "
            f"positives; {had - kept_n:,} were discarded.\n"
            "  This train_balanced was written by the former notebook 01, whose\n"
            "  MAX_POSITIVE_SHARE ceiling dropped positives. Re-run notebook 01\n"
            "  (--prepare): it now keeps every positive."
        )

    n_eval_pos = min(full_pos["test"], 1)
    if (split_dir / SUBSETS["test"]).exists():
        sub = df_lib.read_parquet(str(split_dir / SUBSETS["test"]), columns=["_target"])
        n_eval_pos = max(int(sub["_target"].sum()), 1)
        del sub
    # Standard error of AUROC under the null, ~sqrt((n1+n0+1)/(12*n1*n0)); with
    # balanced-ish n0 >> n1 it is dominated by the positive count. Printed so a
    # difference between arms is not read as a result when it is inside noise.
    se_auc = (1.0 / (12.0 * n_eval_pos)) ** 0.5
    print(f"\n  Test AUROC noise floor: {n_eval_pos:,} positives -> SE ~ {se_auc:.3f}, "
          f"so chance spans ~{0.5 - 2 * se_auc:.2f}-{0.5 + 2 * se_auc:.2f}.")
    print("  Arms inside that band are indistinguishable from each other.")
    print()

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  overall  {total_pos:,} positives / {total_rows:,} rows "
          f"({total_pos / max(total_rows, 1):.4%})")
    for split in SPLITS:
        print(f"  {split:<8} {rates[split]:.4%}")

    base = rates["train"]
    if base <= 0:
        print("\n  The train split has NO positives. Nothing downstream can work.")
        return 2

    val_ratio, test_ratio = rates["val"] / base, rates["test"] / base
    print(f"\n  val / train  rate ratio: {val_ratio:.1f}x")
    print(f"  test / train rate ratio: {test_ratio:.1f}x")

    worst = max(val_ratio, test_ratio)
    if worst >= DRIFT_FATAL:
        print(
            "\n  ** The label is NOT stationary. **\n"
            "  The training period and the evaluation period carry the label at very\n"
            "  different rates, so every model here is fitted on one labelling regime\n"
            "  and scored on another. Chance-level -- or slightly worse than chance --\n"
            "  test metrics are the expected outcome, and no feature set, embedding or\n"
            "  hyperparameter will change that.\n\n"
            "  Take the monthly table above to the data owner and ask when and how\n"
            "  `label` was populated. Then restrict every split to the period where the\n"
            "  label is stable, and re-split inside that period."
        )
        return 2

    if worst >= DRIFT_WARN:
        print(
            f"\n  The label rate drifts across the splits ({worst:.1f}x) but not\n"
            "  sharply. Check the monthly table for a step change before trusting\n"
            "  the test metrics."
        )
    else:
        print(
            "\n  Label rates are comparable across the splits. Non-stationarity is\n"
            "  not the explanation -- look at the sampling above, then the features."
        )

    if "train" in truncated:
        kept_n, had = truncated["train"]
        print(
            f"\n  ** But the training sample holds only {kept_n:,} of {had:,}\n"
            f"  positives ({kept_n / had:.1%}). Fix that before concluding anything\n"
            "  about the features -- the supervised step is being shown a fraction\n"
            "  of the signal that exists."
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

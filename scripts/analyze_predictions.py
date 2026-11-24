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

"""Analyses of a finished run's saved predictions -- no retraining.

Notebook 05 already writes both for new runs; this redoes them for any run, or
for another split or period::

    python scripts/analyze_predictions.py drift <runs root>/base [--split val] [--freq W]
    python scripts/analyze_predictions.py value <runs root>/base [--split val]

``drift``  metrics per month (or week): a falling curve means performance decays
           with time since training. Months with few positives are noisy (auc_se).
``value``  share of the fraud *amount* caught when the top 0.1% / 1% of scores are
           flagged, next to the share of the fraud *count* (shares only).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import metrics_by_period, value_metrics  # noqa: E402


def schema_column(run_dir: Path, field: str, default: str) -> str:
    """An extract column name from the run's schema.

    A ``from_run`` run has no tokenizer of its own: its source run's schema is used.
    """
    candidates = [run_dir / "tokenizer" / "schema.json"]
    exp_file = run_dir / "experiment.yaml"
    if exp_file.exists():
        import yaml
        source = (yaml.safe_load(exp_file.read_text()) or {}).get("from_run")
        if source:
            candidates.append(run_dir.parent / source / "tokenizer" / "schema.json")
    for schema in candidates:
        if schema.exists():
            return json.loads(schema.read_text()).get(field, default)
    return default


def load_scored(run_dir: Path, split: str, column: str) -> pd.DataFrame:
    """Predictions of *split* joined (by row id) to one feature column."""
    preds = pd.read_parquet(run_dir / "outputs" / "predictions" / f"{split}.parquet")
    feats = pd.read_parquet(run_dir / "embeddings" / f"{split}_features.parquet")
    if column not in feats.columns:
        raise SystemExit(f"{column!r} not in {split}_features.parquet: {list(feats.columns)}")
    frame = preds.merge(feats[["__row_id__", column]], on="__row_id__", how="left",
                        indicator=True)
    unmatched = int((frame.pop("_merge") != "both").sum())
    if unmatched or len(frame) != len(preds):
        raise SystemExit(f"{unmatched} of {len(preds)} predictions have no row in "
                         f"{split}_features.parquet: were they written by another notebook 04 run?")
    return frame


def _scores(frame: pd.DataFrame):
    return {c[len("score_"):]: frame[c].to_numpy() for c in frame if c.startswith("score_")}


def datetime_format(run_dir: Path):
    """The extract's datetime format from the tokenizer state, if recorded."""
    import yaml
    candidates = [run_dir / "tokenizer" / "tokenizer_state.json"]
    exp_file = run_dir / "experiment.yaml"
    if exp_file.exists():
        source = (yaml.safe_load(exp_file.read_text()) or {}).get("from_run")
        if source:
            candidates.append(run_dir.parent / source / "tokenizer" / "tokenizer_state.json")
    for state in candidates:
        if state.exists():
            return (json.loads(state.read_text()).get("metadata", {})
                    .get("config", {}).get("datetime_format"))
    return None


def drift(run_dir: Path, split: str = "test", freq: str = "M") -> pd.DataFrame:
    col = schema_column(run_dir, "datetime", "datetime")
    frame = load_scored(run_dir, split, col)
    return metrics_by_period(frame["label"].to_numpy(), _scores(frame), frame[col], freq,
                             date_format=datetime_format(run_dir))


def value(run_dir: Path, split: str = "test") -> pd.DataFrame:
    col = schema_column(run_dir, "amount", "amt_euro")
    frame = load_scored(run_dir, split, col)
    amounts = pd.to_numeric(frame[col], errors="coerce").to_numpy()
    rows = [{"model": m, **value_metrics(frame["label"].to_numpy(), s, amounts)}
            for m, s in _scores(frame).items()]
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyses of saved predictions.")
    sub = parser.add_subparsers(dest="analysis", required=True)
    d = sub.add_parser("drift", help="metrics per period")
    d.add_argument("run_dir", type=Path)
    d.add_argument("--split", default="test", choices=("train", "val", "test"))
    d.add_argument("--freq", default="M", help="pandas period: M (month), W (week)")
    v = sub.add_parser("value", help="fraud value caught at fixed alert budgets")
    v.add_argument("run_dir", type=Path)
    v.add_argument("--split", default="test", choices=("train", "val", "test"))
    args = parser.parse_args(argv)

    if args.analysis == "drift":
        table = drift(args.run_dir, args.split, args.freq)
        out = args.run_dir / "outputs" / f"metrics_by_period_{args.split}_{args.freq}.csv"
        shown = table.pivot(index="period", columns="model", values="auc").round(3)
    else:
        table = value(args.run_dir, args.split)
        out = args.run_dir / "outputs" / f"value_caught_{args.split}.csv"
        shown = table.set_index("model").round(3)
    table.to_csv(out, index=False)
    print(shown.to_string())
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

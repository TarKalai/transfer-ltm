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

"""Offline checks for src/evaluation.py: metrics and saved predictions.

1. ranking_metrics on cases with known answers; the AUC standard error
   against a bootstrap.
2. save_predictions -> metrics_from_predictions reproduces the metrics.

Needs scikit-learn, and pyarrow for the parquet round trip (skipped without it).

    python scripts/verify_evaluation_offline.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import (  # noqa: E402
    auc_standard_error, head_metrics, metrics_from_predictions, ranking_metrics,
    save_predictions, value_metrics,
)

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


print("1. ranking_metrics")
y = np.array([0] * 990 + [1] * 10)
perfect = ranking_metrics(y, y.astype(float))
check("perfect ranking: AUC 1, AP 1, lift = 1/prevalence",
      perfect["auc"] == 1.0 and perfect["ap"] == 1.0 and abs(perfect["lift"] - 100) < 1e-9)
check("perfect ranking: all frauds in the top 1%, recall 1 at 0.1% FPR",
      perfect["precision_at_top_1pct"] == 1.0 and perfect["recall_at_0_1pct_fpr"] == 1.0)
inverted = ranking_metrics(y, -y.astype(float))
check("inverted ranking: AUC 0", inverted["auc"] == 0.0)
check("counts and prevalence recorded",
      perfect["n"] == 1000 and perfect["n_pos"] == 10 and perfect["prevalence"] == 0.01)

rng = np.random.default_rng(0)
n, n_pos = 100_000, 750
yy = np.zeros(n, int)
yy[:n_pos] = 1
scores = rng.normal(size=n) + 0.3 * yy
m = ranking_metrics(yy, scores)
boot = []
for _ in range(200):
    idx = rng.integers(0, n, n)
    boot.append(ranking_metrics(yy[idx], scores[idx])["auc"])
check("AUC standard error matches a bootstrap (within 25%)",
      abs(m["auc_se"] - np.std(boot)) < 0.25 * np.std(boot),
      f"formula {m['auc_se']:.4f} vs bootstrap {np.std(boot):.4f}")
check("at 750 positives and AUC 0.5 the SE is ~0.011 (the noise level in the docs)",
      abs(auc_standard_error(0.5, 750, 99_250) - 0.011) < 0.001,
      f"{auc_standard_error(0.5, 750, 99_250):.4f}")
check("no positives: no ranking metrics, no crash",
      "auc" not in ranking_metrics(np.zeros(10), np.arange(10)))

print("\n2. saved predictions")
try:
    import pyarrow  # noqa: F401
    have_parquet = True
except ImportError:
    have_parquet = False
    print("  (pyarrow not installed -- parquet round trip skipped)")

if have_parquet:
    with tempfile.TemporaryDirectory() as tmp:
        ids = {"test": {"__row_id__": np.arange(n), "label": yy}}
        other = rng.normal(size=n)
        files = save_predictions(Path(tmp), ids, {"raw": {"test": scores},
                                                  "embeddings": {"test": other}})
        again = metrics_from_predictions(files["test"])
        check("one score column per model, metrics recomputed identically",
              set(again) == {"raw", "embeddings"} and again["raw"]["auc"] == m["auc"]
              and again["raw"]["ap"] == m["ap"])
        try:
            save_predictions(Path(tmp), ids, {"raw": {"test": scores[:-1]}})
            check("length mismatch rejected", False)
        except ValueError:
            check("length mismatch rejected", True)

print("\n3. logistic regression (sanity check next to XGBoost)")
from src.downstream import fit_logreg  # noqa: E402

Xs = rng.normal(size=(6000, 5))
Xs[rng.random(Xs.shape) < 0.01] = np.nan                    # imputation is exercised
ys = (Xs[:, 0] + rng.normal(scale=1.0, size=6000) > 2.0).astype(int)
_, sc, lm = fit_logreg(Xs[:4000], ys[:4000], Xs[4000:5000], ys[4000:5000],
                       Xs[5000:], ys[5000:])
check("logreg: learns a linear signal, C picked on val, same metric keys as XGBoost",
      lm["test_auc"] > 0.85 and lm["C"] in (0.01, 0.1, 1.0)
      and {"test_auc", "test_ap", "val_auc", "train_auc_se"} <= set(lm),
      f"test AUC {lm['test_auc']:.3f}")
check("logreg: one score per row per split", len(sc["test"]) == 1000 and len(sc["train"]) == 4000)
check("logreg: convergence recorded", lm["converged"] is True)
_, sc_cap, lm_cap = fit_logreg(Xs[:4000], ys[:4000], Xs[4000:5000], ys[4000:5000],
                               Xs[5000:], ys[5000:], max_fit_rows=1000)
check("logreg: fitted on a bounded sample (all positives kept), scored on every row",
      lm_cap["fit_rows"] == max(1000, int(ys[:4000].sum())) and len(sc_cap["train"]) == 4000,
      str(lm_cap["fit_rows"]))
_, _, lm1 = fit_logreg(Xs[:4000], ys[:4000], Xs[4000:5000], ys[4000:5000],
                       Xs[5000:], ys[5000:], max_iter=1)
check("logreg: a fit stopped at max_iter is flagged", lm1["converged"] is False)

print("\n4. metrics per period")
from src.evaluation import metrics_by_period  # noqa: E402

dates = np.repeat(np.array(["2030-01-15", "2030-02-15", "2030-03-15"], dtype="datetime64[ns]"),
                  [40000, 30000, 30000])
table = metrics_by_period(yy, {"raw": scores}, dates)
first = ranking_metrics(yy[:40000], scores[:40000])
check("one row per month per model, values = metrics on that month's rows",
      list(table["period"]) == ["2030-01", "2030-02", "2030-03"]
      and abs(table.iloc[0]["auc"] - first["auc"]) < 1e-12 and table["n"].sum() == n)

if have_parquet:
    import json  # noqa: E402
    import pandas as pd  # noqa: E402
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_predictions import drift, value  # noqa: E402
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        (run / "outputs" / "predictions").mkdir(parents=True)
        (run / "embeddings").mkdir()
        (run / "tokenizer").mkdir()
        save_predictions(run / "outputs" / "predictions",
                         {"test": {"__row_id__": np.arange(n), "label": yy}},
                         {"raw": {"test": scores}})
        order = rng.permutation(n)            # features in another order: joined by row id
        amounts = rng.lognormal(4, 1, n)
        pd.DataFrame({"__row_id__": np.arange(n)[order], "booked_at": dates[order],
                      "amt": amounts[order]}).to_parquet(run / "embeddings" / "test_features.parquet")
        (run / "tokenizer" / "schema.json").write_text(
            json.dumps({"datetime": "booked_at", "amount": "amt"}))
        feats_path = run / "embeddings" / "test_features.parquet"
        full = pd.read_parquet(feats_path)
        full.iloc[: n // 2].to_parquet(run / "embeddings" / "half.parquet")
        (run / "embeddings" / "test_features.parquet").rename(run / "embeddings" / "full.parquet")
        (run / "embeddings" / "half.parquet").rename(feats_path)
        try:
            drift(run)
            check("predictions without a matching feature row are refused", False)
        except SystemExit as e:
            check("predictions without a matching feature row are refused", "no row" in str(e))
        feats_path.unlink()
        (run / "embeddings" / "full.parquet").rename(feats_path)
        check("analyze_predictions drift: joins predictions to dates by row id (schema column)",
              drift(run).equals(table))
        v = value(run)
        check("analyze_predictions value: same as value_metrics on the joined amounts",
              abs(v.iloc[0]["value_recall_at_top_1pct"]
                  - value_metrics(yy, scores, amounts)["value_recall_at_top_1pct"]) < 1e-12)

print("\n5. fraud value caught")

yv = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
amt = np.array([900.0, 10.0, 5, 5, 5, 5, 5, 5, 5, 90.0])
sv = np.array([0.95, 0.1, 0.9, 0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.05])  # big fraud ranked 1st
vm = value_metrics(yv, sv, amt, top_fractions=(0.1, 0.2))
check("value recall counts amounts, count recall counts frauds (top 10%: the 900 fraud)",
      abs(vm["value_recall_at_top_10pct"] - 0.9) < 1e-12
      and abs(vm["recall_at_top_10pct"] - 1 / 3) < 1e-12, str(vm))
check("negative amounts (debits) count by size; no frauds -> nothing",
      value_metrics(yv, sv, -amt, (0.1,)) == value_metrics(yv, sv, amt, (0.1,))
      and value_metrics(np.zeros(10), sv, amt) == {})

# Ties at the cut: XGBoost-like discrete scores, metrics must not depend on row order.
disc = np.round(scores, 0)                       # a handful of distinct values -> big ties
perm = rng.permutation(n)
m1, m2 = ranking_metrics(yy, disc), ranking_metrics(yy[perm], disc[perm])
v1 = value_metrics(yy, disc, amounts_v := rng.lognormal(4, 1, n))
v2 = value_metrics(yy[perm], disc[perm], amounts_v[perm])
check("tied scores at the cut: precision and value metrics independent of row order",
      abs(m1["precision_at_top_1pct"] - m2["precision_at_top_1pct"]) < 1e-12
      and all(abs(v1[k] - v2[k]) < 1e-12 for k in v1))
flat = value_metrics(yv, np.zeros(10), amt, top_fractions=(0.1,))
check("all scores tied: flagged share = budget (expected value of a random pick)",
      abs(flat["recall_at_top_10pct"] - 0.1) < 1e-12
      and abs(flat["value_recall_at_top_10pct"] - 0.1) < 1e-12, str(flat))
with_nan = disc.astype(float).copy()
with_nan[:3] = np.nan
wn = value_metrics(yy, with_nan, amounts_v)
check("NaN scores rank last: budget respected, no weight above 1",
      all(0 <= wn[k] <= 1 for k in wn) and np.isfinite(list(wn.values())).all())
dayfirst = metrics_by_period(yy[:4], {"m": scores[:4]}, ["03/04/2024"] * 4, date_format="%d/%m/%Y")
check("date format honoured: 03/04/2024 day-first is April", list(dayfirst["period"]) == ["2024-04"])

hm = head_metrics({"test": scores}, {"test": yy})
check("head_metrics: per split", hm["test"]["auc"] == m["auc"])

print("\n" + "=" * 60)
print(f"{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:")
    for name in fail:
        print("  -", name)
    sys.exit(1)

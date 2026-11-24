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

"""Metrics and saved predictions for the downstream fraud models.

Every run saves, per split, one row per scored transaction with its label and
the score of every model (``outputs/predictions/{split}.parquet``). Any metric
can therefore be computed later without retraining; :func:`ranking_metrics`
is what notebook 05 reports and ``summary.csv`` collects.

All metrics are *ranking* metrics: the training sample undersamples negatives,
so scores are not calibrated probabilities, but rankings are unaffected.
"""

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

#: False-positive rates at which recall is reported (alert-budget view).
FPR_POINTS = (0.001, 0.01)
#: Top fractions of scores at which precision is reported.
TOP_FRACTIONS = (0.001, 0.01)


def _pct(x: float) -> str:
    return f"{x * 100:g}pct".replace(".", "_")


def _top_weights(scores: np.ndarray, fraction: float) -> np.ndarray:
    """How much each row counts as flagged when the top ``fraction`` is flagged.

    1 above the cut, 0 below. Rows *tied* at the cut share the remaining
    places: each counts ``places / tied`` -- the average over every random
    tie-break, so the result does not depend on row order (XGBoost scores are
    discrete, and ties straddling the cut are common). O(n): no full sort.
    """
    scores = np.where(np.isnan(scores), -np.inf, scores)            # NaN ranks last
    k = max(1, int(round(len(scores) * fraction)))
    cut = np.partition(scores, len(scores) - k)[len(scores) - k]      # k-th largest
    above = scores > cut
    tied = scores == cut
    w = above.astype(np.float64)
    w[tied] = (k - above.sum()) / tied.sum()
    return w


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of a ROC-AUC."""
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def safe_auc(y, scores) -> float:
    """ROC-AUC, or NaN -- never an exception -- for empty, one-class or non-finite input.

    The one AUC used for model *selection* (early stopping, C choice), where a
    diverged epoch must count as "no improvement", not crash the run.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or not np.isfinite(s).all() or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def ranking_metrics(y, scores) -> Dict[str, float]:
    """ROC-AUC (with its standard error), AP, lift and alert-budget metrics.

    ``auc``                     ROC-AUC
    ``auc_se``                  its standard error; two runs differ beyond noise only
                                if their gap is well above ~2x this
    ``ap``                      average precision (area under precision-recall)
    ``lift``                    AP / prevalence (1.0 = chance)
    ``recall_at_{x}pct_fpr``    share of frauds caught when x% of legit ones are flagged
    ``precision_at_top_{x}pct`` share of frauds among the x% highest scores
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n = int(y.sum()), len(y)
    prevalence = n_pos / max(n, 1)
    out: Dict[str, float] = {"n": n, "n_pos": n_pos, "prevalence": prevalence}
    if n_pos == 0 or n_pos == n:
        return out
    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    out.update(auc=auc, auc_se=auc_standard_error(auc, n_pos, n - n_pos), ap=ap,
               lift=ap / prevalence)
    fpr, tpr, _ = roc_curve(y, s)
    for x in FPR_POINTS:
        out[f"recall_at_{_pct(x)}_fpr"] = float(np.interp(x, fpr, tpr))
    for x in TOP_FRACTIONS:
        w = _top_weights(s, x)
        out[f"precision_at_top_{_pct(x)}"] = float((w * y).sum() / w.sum())
    return out


def save_predictions(out_dir, splits: Mapping[str, Mapping[str, np.ndarray]],
                     scores: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Path]:
    """Write ``{split}.parquet``: identifiers + label + one ``score_<model>`` column each.

    ``splits[split]`` holds the row identifiers (e.g. ``__row_id__``) and
    ``label``; ``scores[model][split]`` the model's scores in the same order.
    A model missing for a split is simply not a column there.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, ids in splits.items():
        frame = pd.DataFrame({k: np.asarray(v) for k, v in ids.items()})
        for model, by_split in scores.items():
            if split in by_split:
                values = np.asarray(by_split[split])
                if len(values) != len(frame):
                    raise ValueError(f"{model}/{split}: {len(values)} scores for "
                                     f"{len(frame)} rows")
                frame[f"score_{model}"] = values
        path = out_dir / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        written[split] = path
    return written


def metrics_from_predictions(path, label: str = "label") -> Dict[str, Dict[str, float]]:
    """Recompute every model's metrics from a saved predictions file."""
    frame = pd.read_parquet(path)
    return {
        col[len("score_"):]: ranking_metrics(frame[label], frame[col])
        for col in frame.columns if col.startswith("score_")
    }


def value_metrics(y, scores, amounts, top_fractions=TOP_FRACTIONS) -> Dict[str, float]:
    """How much of the fraud *value* the highest scores catch.

    At an alert budget of the top x% of scores:

    ``value_recall_at_top_{x}pct``  share of the total fraud amount among flagged rows
    ``recall_at_top_{x}pct``        share of the fraud count among flagged rows

    A model can catch many small frauds and miss the large ones (value recall
    below count recall) or the reverse. Only shares are returned -- never euro
    totals -- so the results stay shareable as ranges.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    a = np.abs(np.nan_to_num(np.asarray(amounts, dtype=np.float64)))
    fraud_value, n_fraud = float((a * y).sum()), int(y.sum())
    out: Dict[str, float] = {}
    if n_fraud == 0 or fraud_value <= 0:
        return out
    for x in top_fractions:
        w = _top_weights(s, x)
        out[f"value_recall_at_top_{_pct(x)}"] = float((w * a * y).sum() / fraud_value)
        out[f"recall_at_top_{_pct(x)}"] = float((w * y).sum() / n_fraud)
    return out


def metrics_by_period(labels, scores: Mapping[str, np.ndarray], dates,
                      freq: str = "M", date_format: Optional[str] = None) -> pd.DataFrame:
    """One row per (period, model): drift view of a split, e.g. test AUC per month.

    A falling curve across the test period means results decay with time since
    training -- a single test number then overstates what a deployed model keeps.
    """
    labels = np.asarray(labels).astype(int)
    # The extract's own format when it is known (BankTokenizerConfig.datetime_format):
    # day-first strings would otherwise be read month-first.
    period = (pd.to_datetime(pd.Series(np.asarray(dates)), format=date_format)
              .dt.to_period(freq).astype(str).to_numpy())
    rows = []
    for p in sorted(set(period)):
        m = period == p
        for model, s in scores.items():
            r = ranking_metrics(labels[m], np.asarray(s)[m])
            rows.append({"period": p, "model": model,
                         **{k: r.get(k) for k in ("n", "n_pos", "prevalence", "auc",
                                                  "auc_se", "ap", "lift")}})
    return pd.DataFrame(rows)


def head_metrics(scores_by_split: Mapping[str, np.ndarray],
                 labels_by_split: Mapping[str, np.ndarray]) -> Optional[Dict[str, Dict]]:
    """Metrics of a model that scores without XGBoost (e.g. the attention head)."""
    if not scores_by_split:
        return None
    return {split: ranking_metrics(labels_by_split[split], s)
            for split, s in scores_by_split.items() if split in labels_by_split}

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

"""Downstream classifiers besides XGBoost.

Logistic regression is the sanity check next to XGBoost, on the same three
feature sets (raw, embeddings, combined):

* If logistic regression on the *embeddings* is close to XGBoost on them, the
  signal is (near-)linearly readable from the representation -- what a good
  embedding should give.
* If XGBoost wins clearly on raw features but not on embeddings, the trees are
  finding interactions the embedding does not expose.
* If logistic regression beats XGBoost anywhere, XGBoost is overfitting or
  badly tuned for that feature set.

Features are median-imputed and standardised (fitted once on train); the
regularisation strength is chosen on validation (ROC-AUC) from a small grid,
walked with warm starts; classes are balanced by weight, as the training set
undersamples negatives. A fit that reaches ``max_iter`` is flagged
(``converged: False``).
"""

import copy
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import ranking_metrics, safe_auc

C_GRID: Tuple[float, ...] = (0.01, 0.1, 1.0)


#: Rows used to *fit* the sanity-check model: every positive, plus negatives up
#: to this total. Plenty for a linear model, and it bounds the cost of a check
#: that runs in every experiment. Scores are still computed on every row.
MAX_FIT_ROWS = 300_000


def fit_logreg(X_tr, y_tr, X_va, y_va, X_te, y_te, c_grid: Sequence[float] = C_GRID,
               max_iter: int = 300, seed: int = 0,
               max_fit_rows: int = MAX_FIT_ROWS) -> Tuple[object, Dict[str, np.ndarray], Dict]:
    """Fit on train, pick C on val ROC-AUC; return (model, {split: scores}, metrics).

    ``metrics`` uses the same keys as notebook 05's XGBoost arms
    (``{split}_auc``, ``{split}_ap``, ...), plus ``C``, ``converged`` and
    ``fit_rows``.
    """
    y_tr = np.asarray(y_tr)
    fit_idx = np.arange(len(y_tr))
    if max_fit_rows and len(y_tr) > max_fit_rows:
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
        n_neg = max(0, min(len(neg), max_fit_rows - len(pos)))
        fit_idx = np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
    X_fit = X_tr[fit_idx] if isinstance(X_tr, np.ndarray) else X_tr.iloc[fit_idx]
    # Imputation and scaling are fitted once (on train) and shared by every C;
    # the C grid is walked in increasing order with warm starts, each fit
    # starting from the previous solution.
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X_fit)
    Z_fit, Z_va = prep.transform(X_fit), prep.transform(X_va)
    best = None
    lr = LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=seed,
                            warm_start=True)
    for c in sorted(c_grid):
        lr.set_params(C=c).fit(Z_fit, y_tr[fit_idx])
        val_auc = safe_auc(y_va, lr.decision_function(Z_va))
        rank = val_auc if np.isfinite(val_auc) else -np.inf      # NaN never wins
        if best is None or rank > best[0]:
            best = (rank, c, copy.deepcopy(lr))
    _, c, chosen = best
    converged = bool(chosen.n_iter_.max() < max_iter)
    model = make_pipeline(*[step for _, step in prep.steps], chosen)
    scores = {"train": chosen.decision_function(prep.transform(X_tr)),
              "val": chosen.decision_function(Z_va),
              "test": chosen.decision_function(prep.transform(X_te))}
    truth = {"train": y_tr, "val": y_va, "test": y_te}
    metrics = {"n_features": int(np.shape(X_tr)[1]), "C": c, "converged": converged,
               "fit_rows": int(len(fit_idx))}
    for split in ("train", "val", "test"):
        for key, value in ranking_metrics(truth[split], scores[split]).items():
            metrics[f"{split}_{key}"] = value
    return model, scores, metrics

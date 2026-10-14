"""P6 (PROGRESS.md, + amendment 1; claim C3): secondary CatBoost variants on top of CTE (cte/3). Engine p6/2.

catboost_native_text : structured features + raw text columns as CatBoost text_features (no embeddings, no experts).
cte_plus_native      : CTE features (same caches as cte) + the same CatBoost text_features.
cte_nb               : CTE; for classification the lexical expert is NB log-count-ratio weighted TF-IDF + Ridge (NBSVM,
                       Wang & Manning 2012), one-vs-rest; the NB ratios come from each fit's training rows only.
cte_residual         : CTE experts trained on residuals of a structured-only CatBoost, nested cross-fitted.
The learner side of the native text (raw columns, text_features) lives in learners.OursTextMixin.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from multabench.ours.experts import ExpertConfig, SEED, expert_folds, rank_gauss

P6_VERSION = "p6/2"  # p6/1 = before P6 amendment 1 (never run on Colab)
P6_METHODS = {  # method -> (CTE base method whose experts/mode/numeracy it uses, lexical kind, targets, native text)
    "catboost_native_text": (None, None, None, True),
    "cte_plus_native": ("cte", "tfidf", "y", True),
    "cte_nb": ("cte", "nb", "y", False),
    "cte_residual": ("cte", "tfidf", "residual", False),
}
NATIVE_SUFFIX = "__native_text"  # raw text column name the learner adds for CatBoost text_features


def native_feature_calcers(is_cls: bool, n_classes: int) -> list[str]:
    """CatBoost supports NaiveBayes and BM25 for classification only; regression gets BoW (tested). P6 amendment 1:
    BoW only for C > 10 classes (NaiveBayes / BM25 are computed per class: hours per job on Wine, E23)."""
    return ["BoW", "NaiveBayes", "BM25"] if is_cls and n_classes <= 10 else ["BoW"]


@dataclass(frozen=True)
class P6Config:
    struct_iterations: int = 300
    struct_depth: int = 6
    struct_learning_rate: float = 0.1
    inner_folds: int = 3  # nested OOF of the structured model inside each expert fold's training rows
    nb_smoothing: float = 1.0  # p = 1 + sum of class rows, q = 1 + sum of other rows (Wang & Manning)
    seed: int = SEED

    def key(self) -> str:
        return hashlib.sha1(json.dumps([P6_VERSION, asdict(self)], sort_keys=True).encode()).hexdigest()[:10]


P6_CONFIG = P6Config()


# ---------------------------------------------------------------------------------------- cte_nb

def nb_ridge_scores(X_tr, y_tr: np.ndarray, X_tes: list, n_classes: int, alpha: float,
                    c: P6Config = P6_CONFIG) -> list:
    """NB-weighted TF-IDF + one-vs-rest Ridge (classification). Returns full class scores (n, C) per X in X_tes;
    binary: one model for class 1 and score_0 = 1 - score_1. r_c uses only the rows of this fit (X_tr, y_tr)."""
    from scipy.sparse import diags
    from sklearn.linear_model import Ridge
    outs = [np.zeros((X.shape[0], n_classes)) for X in X_tes]
    if X_tr.shape[1] == 0:  # no vocabulary: class prior
        prior = np.bincount(y_tr, minlength=n_classes) / len(y_tr)
        return [np.tile(prior, (o.shape[0], 1)) for o in outs]
    for k in ([1] if n_classes == 2 else range(n_classes)):
        ind = (y_tr == k).astype(float)
        p = c.nb_smoothing + np.asarray(X_tr[ind == 1].sum(axis=0)).ravel()
        q = c.nb_smoothing + np.asarray(X_tr[ind == 0].sum(axis=0)).ravel()
        R = diags(np.log((p / p.sum()) / (q / q.sum())))
        m = Ridge(alpha=alpha, solver="sparse_cg").fit(X_tr @ R, ind)
        for o, X in zip(outs, X_tes):
            o[:, k] = m.predict(X @ R)
    if n_classes == 2:
        for o in outs:
            o[:, 0] = 1.0 - o[:, 1]
    return outs


# ---------------------------------------------------------------------------------------- cte_residual

def struct_frame(x: DataFrame, struct_cols: list[str], cat_cols: list[str]) -> DataFrame:
    """Structured-only CatBoost input: categorical columns as strings, everything else numeric (NaN allowed)."""
    out = pd.DataFrame(index=x.index)
    for col in struct_cols:
        if col in cat_cols:
            out[col] = x[col].astype(str).fillna("nan")
        else:
            out[col] = pd.to_numeric(x[col], errors="coerce").astype(float)
    return out


class StructuredResidual:
    """Residual targets of a structured-only CatBoost S for the experts (P6 cte_residual).

    regression     : residual = rank-Gaussian y - S's prediction (S fitted on the rank-Gaussian y of its own rows)
    classification : residual = one-hot(y) - S's class probabilities (negative log-loss gradient at S's logits)
    train_targets(T): residuals for the expert-training rows T, with S's predictions OOF *within T* (inner folds), so
                      the labels of rows outside T never reach them (nested cross-fitting).
    full_targets()  : residuals of all pool rows from S's OOF over the expert folds (for the val/test experts).
    """

    def __init__(self, xs: DataFrame, cat_cols: list[str], y: np.ndarray, is_cls: bool, n_classes: int, folds,
                 ecfg: ExpertConfig, c: P6Config = P6_CONFIG):
        self.xs, self.cat_cols, self.y, self.is_cls, self.n_classes = xs.reset_index(drop=True), cat_cols, y, is_cls, n_classes
        self.folds, self.ecfg, self.c = folds, ecfg, c
        self._memo: dict = {}
        self.n_struct_fits = 0

    def _target(self, y_part: np.ndarray) -> np.ndarray:
        if self.is_cls:
            return np.eye(self.n_classes)[y_part]
        return rank_gauss(y_part, self.ecfg)[:, None]

    def _fit_predict(self, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        """S fitted on pool positions tr, predicted for te: (len(te), C) probabilities or (len(te), 1)."""
        y_tr = self.y[tr]
        if self.xs.shape[1] == 0 or (self.is_cls and len(np.unique(y_tr)) < 2):
            return np.tile(self._target(y_tr).mean(axis=0), (len(te), 1))  # constant model
        from catboost import CatBoostClassifier, CatBoostRegressor
        threads = int(os.environ.get("OURS_LEARNER_THREADS", "0") or 0) or -1
        kw = dict(iterations=self.c.struct_iterations, depth=self.c.struct_depth,
                  learning_rate=self.c.struct_learning_rate, random_seed=self.c.seed, thread_count=threads,
                  verbose=0, allow_writing_files=False)
        cats = [col for col in self.xs.columns if col in self.cat_cols]
        X_tr, X_te = self.xs.iloc[tr], self.xs.iloc[te]
        self.n_struct_fits += 1
        if self.is_cls:
            m = CatBoostClassifier(**kw).fit(X_tr, y_tr, cat_features=cats)
            P = np.zeros((len(te), self.n_classes))
            P[:, np.asarray(m.classes_, dtype=int)] = m.predict_proba(X_te)
            return P
        m = CatBoostRegressor(**kw).fit(X_tr, rank_gauss(y_tr, self.ecfg), cat_features=cats)
        return m.predict(X_te)[:, None]

    def _oof(self, positions: np.ndarray, folds) -> np.ndarray:
        """S's OOF predictions for `positions` (pool positions), folds given as indices into `positions`."""
        P = None
        for tr, te in folds:
            p = self._fit_predict(positions[tr], positions[te])
            P = np.zeros((len(positions), p.shape[1])) if P is None else P
            P[te] = p
        return P

    def train_targets(self, tr: np.ndarray) -> np.ndarray:
        key = ("train", hashlib.sha1(np.asarray(tr, dtype=np.int64).tobytes()).hexdigest())
        if key not in self._memo:
            inner = expert_folds(len(tr), self.y[tr], self.is_cls,
                                 ExpertConfig(n_folds=self.c.inner_folds, seed=self.c.seed))
            self._memo[key] = self._target(self.y[tr]) - self._oof(np.asarray(tr), inner)
        return self._memo[key]

    def offset_full(self) -> np.ndarray:
        if "offset" not in self._memo:
            self._memo["offset"] = self._oof(np.arange(len(self.y)), self.folds)
        return self._memo["offset"]

    def full_targets(self) -> np.ndarray:
        if "full" not in self._memo:
            self._memo["full"] = self._target(self.y) - self.offset_full()
        return self._memo["full"]

    def features(self, R: np.ndarray, top: Optional[np.ndarray]) -> np.ndarray:
        """Predicted residual columns: regression 1; binary class 1; 3..10 classes C-1; C > 10 the top-10 class
        residuals + the max and min residual over all classes."""
        if not self.is_cls:
            return R
        if self.n_classes == 2:
            return R[:, 1:2]
        if top is None:
            return R[:, 1:]
        return np.hstack([R[:, top], R.max(axis=1, keepdims=True), R.min(axis=1, keepdims=True)])

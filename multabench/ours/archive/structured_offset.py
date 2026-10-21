"""Structured-only model used by the offset / prefix / context variants (docs/method_spec.md, sections 1 and 7).

5-fold out-of-fold CatBoost on the structured columns of the E5 train + val pool (the rows TAR fine-tunes on):
- OOF predictions (class probabilities / regression values) for every pool row, each from a model that never saw
  that row's label — these are the offsets used in training;
- one full fit on the whole pool, used only for feature importances and for diagnostic predictions on
  learner-val / test rows (never as encoder inputs, never fitted on those rows).
With no structured columns the "model" is the class prior / the mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold

N_FOLDS = 5
SEED = 42
CATBOOST_PARAMS = {"iterations": 500, "depth": 6, "random_seed": SEED, "verbose": 0, "allow_writing_files": False}
PROB_FLOOR = 1e-3


def catboost_factory(is_cls: bool, cat_cols: list[str]):
    from catboost import CatBoostClassifier, CatBoostRegressor
    cls = CatBoostClassifier if is_cls else CatBoostRegressor
    return cls(**CATBOOST_PARAMS, cat_features=cat_cols)


def oof_folds(n: int, strata: Optional[np.ndarray], seed: int = SEED, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train positions, predict positions) per fold; predict sets are disjoint and cover all n rows."""
    if strata is not None and pd.Series(strata).value_counts().min() >= n_folds:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in splitter.split(np.zeros(n), strata)]
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.zeros(n))]


@dataclass
class StructuredOffset:
    is_cls: bool
    classes: list[str]                 # TAR's E5 classes (classification); [] for regression
    feature_cols: list[str]
    cat_cols: list[str]
    pool_index: np.ndarray
    oof: np.ndarray                    # (n_pool, n_classes) probabilities or (n_pool,) predictions
    importances: dict = field(default_factory=dict)
    full_model: object = None
    prior: Optional[np.ndarray] = None  # class prior / mean (used when there are no structured features)

    @property
    def has_features(self) -> bool:
        return bool(self.feature_cols)

    def top_features(self, k: int = 8) -> list[str]:
        return [c for c, _ in sorted(self.importances.items(), key=lambda kv: (-kv[1], kv[0]))][:k]

    def oof_for(self, rows) -> np.ndarray:
        pos = pd.Index(self.pool_index).get_indexer(np.asarray(rows))
        assert (pos >= 0).all(), "oof_for: rows outside the pool"
        return self.oof[pos]

    def predict_full(self, x: DataFrame) -> np.ndarray:
        """Diagnostics only: full-fit predictions for rows outside the pool (learner-val / test)."""
        if not self.has_features:
            return np.tile(self.prior, (len(x), 1)) if self.is_cls else np.full(len(x), float(self.prior))
        X = _prepare(x, self.feature_cols, self.cat_cols)
        if self.is_cls:
            return _align_probs(self.full_model.predict_proba(X), self.full_model.classes_, self.classes)
        return np.asarray(self.full_model.predict(X), dtype=float)

    def log_offset(self, probs: np.ndarray) -> np.ndarray:
        return np.log(np.clip(probs, PROB_FLOOR, 1.0))


def _prepare(x: DataFrame, feature_cols: list[str], cat_cols: list[str]) -> DataFrame:
    X = x[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    return X


def _align_probs(probs: np.ndarray, model_classes, classes: list[str]) -> np.ndarray:
    """Map a model's predict_proba columns onto TAR's class list; classes the model never saw get 0."""
    out = np.zeros((len(probs), len(classes)))
    pos = {str(c): i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        if str(c) in pos:
            out[:, pos[str(c)]] = probs[:, j]
    return out


def fit_structured_offset(x_pool: DataFrame, y_pool: Series, feature_cols: list[str], cat_cols: list[str],
                          is_cls: bool, classes: list[str], strata: Optional[np.ndarray] = None,
                          model_factory: Optional[Callable] = None) -> StructuredOffset:
    """OOF + full fit on the pool. y_pool: class labels (cls) or numeric targets (reg), indexed like x_pool."""
    assert x_pool.index.equals(y_pool.index)
    factory = model_factory or catboost_factory
    feature_cols = list(feature_cols)
    cat_cols = [c for c in cat_cols if c in feature_cols]
    y = y_pool.astype(str).to_numpy() if is_cls else pd.to_numeric(y_pool).to_numpy(dtype=float)
    n = len(y)
    if is_cls:
        prior = np.array([(y == c).mean() for c in classes])
    else:
        prior = np.array(y.mean())
    if not feature_cols:
        oof = np.tile(prior, (n, 1)) if is_cls else np.full(n, float(prior))
        return StructuredOffset(is_cls, list(classes), [], [], np.asarray(x_pool.index), oof, {}, None, prior)

    X = _prepare(x_pool, feature_cols, cat_cols)
    oof = np.zeros((n, len(classes))) if is_cls else np.zeros(n)
    for tr, te in oof_folds(n, strata if strata is not None else (y if is_cls else None)):
        if is_cls and len(np.unique(y[tr])) < 2:  # degenerate fold: only one class to learn from
            oof[te] = _align_probs(np.ones((len(te), 1)), [y[tr][0]], classes)
            continue
        m = factory(is_cls, cat_cols)
        m.fit(X.iloc[tr], y[tr])
        if is_cls:
            oof[te] = _align_probs(m.predict_proba(X.iloc[te]), m.classes_, classes)
        else:
            oof[te] = m.predict(X.iloc[te])
    full = factory(is_cls, cat_cols)
    full.fit(X, y)
    try:
        imp = dict(zip(feature_cols, (float(v) for v in full.get_feature_importance())))
    except AttributeError:  # test doubles without importances
        imp = {c: 0.0 for c in feature_cols}
    return StructuredOffset(is_cls, list(classes), feature_cols, cat_cols, np.asarray(x_pool.index), oof, imp, full,
                            prior)

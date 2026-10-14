"""P5 (PROGRESS.md): cheap label-aware text "experts" whose out-of-fold predictions are stacked as learner features.

LEGACY (P5 v1, engine experts/1; superseded by P5 v2 = multabench/ours/cte.py, never run as a claim): LexicalExpert,
NeighbourExpert and compute_expert_features below implement P5 v1 only (used by stacking.run_experts). CTE (cte/3)
and P6 use only the shared helpers from this module: ExpertConfig, expert_folds, rank_gauss, cached_expert_features,
ALL_COLS and SEED.

Experts, per text column AND on all text columns concatenated ("__all__"):
- lexical:   TF-IDF word 1-2-grams + char_wb 3-5-grams (sublinear_tf, min_df=2, capped vocabularies), then RidgeCV
             (regression) or LogisticRegression whose C is chosen by an inner 3-fold CV on the training rows (cls);
- neighbour: k = 20 cosine neighbours in frozen E5 space; similarity-weighted mean label (reg) / class distribution
             (cls). For "__all__" the row vector is the L2-normalised mean of its per-column frozen embeddings.
Cross-fitting: 5-fold out-of-fold on the learner's training rows (the pool); learner-val / test rows get the
predictions of experts fit on all pool rows. The "insample" mode (ablation only) fits on all pool rows and predicts
them too. Output per expert: cls -> P(class) for classes 1.. (binary: 1 column), reg -> 1 column.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from os.path import exists, join
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold

EXPERTS_VERSION = "experts/1"
ALL_COLS = "__all__"
SEED = 42


@dataclass(frozen=True)
class ExpertConfig:
    n_folds: int = 5
    word_ngrams: tuple = (1, 2)
    char_ngrams: tuple = (3, 5)
    min_df: int = 2
    max_word_features: int = 100_000
    max_char_features: int = 100_000
    ridge_alphas: tuple = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    logreg_Cs: tuple = (0.25, 1.0, 4.0, 16.0)
    logreg_inner_folds: int = 3
    k_neighbors: int = 20
    seed: int = SEED
    neighbour_weights: str = "similarity"  # P5 v1: max(1 - d, 0); CTE (P5 v2): "inverse_distance" = 1 / (d + 1e-6)
    reg_target: str = "raw"                # P5 v1: raw y; CTE: "rank_gauss" (QuantileTransformer, normal output)

    def key(self) -> str:
        return hashlib.sha1(json.dumps([EXPERTS_VERSION, asdict(self)], sort_keys=True).encode()).hexdigest()[:10]


def expert_folds(n: int, y: np.ndarray, is_cls: bool, cfg: ExpertConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic (train positions, predict positions); stratified for classification when possible."""
    if is_cls and pd.Series(y).value_counts().min() >= cfg.n_folds:
        sp = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(np.zeros(n), y)
    else:
        sp = KFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(np.zeros(n))
    return [(tr, te) for tr, te in sp]


def rank_gauss(y_tr: np.ndarray, cfg: ExpertConfig) -> np.ndarray:
    """Regression target for an expert: raw, or rank-Gaussian fit on exactly these (training) rows."""
    if cfg.reg_target != "rank_gauss":
        return y_tr
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=min(1000, len(y_tr)), output_distribution="normal", random_state=cfg.seed)
    return qt.fit_transform(np.asarray(y_tr, dtype=float)[:, None])[:, 0]


def _cls_output(p_full: np.ndarray, n_classes: int) -> np.ndarray:
    return p_full[:, 1:2] if n_classes == 2 else p_full[:, 1:]


# ---------------------------------------------------------------------------------------- experts

class LexicalExpert:
    """LEGACY P5 v1 (experts/1). CTE uses cte.tfidf_matrices / cte.ridge_fit instead."""
    def __init__(self, is_cls: bool, n_classes: int, cfg: ExpertConfig):
        self.is_cls, self.n_classes, self.cfg = is_cls, n_classes, cfg
        self.C = None  # chosen once on the full pool (inner CV), then reused for every out-of-fold fit

    def _vectorize(self, tr: list[str], te: list[str]):
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        c = self.cfg
        mats_tr, mats_te = [], []
        for kw in (dict(analyzer="word", ngram_range=c.word_ngrams, max_features=c.max_word_features),
                   dict(analyzer="char_wb", ngram_range=c.char_ngrams, max_features=c.max_char_features)):
            v = TfidfVectorizer(sublinear_tf=True, min_df=c.min_df, dtype=np.float32, **kw)
            try:
                mats_tr.append(v.fit_transform(tr))
                mats_te.append(v.transform(te))
            except ValueError:  # empty vocabulary (e.g. all texts identical / too few) -> skip this analyzer
                continue
        if not mats_tr:
            return None, None
        return hstack(mats_tr).tocsr(), hstack(mats_te).tocsr()

    def choose_C(self, texts: list[str], y: np.ndarray) -> None:
        if not self.is_cls:
            return
        from sklearn.linear_model import LogisticRegression
        best, best_score = self.cfg.logreg_Cs[0], -np.inf
        folds = expert_folds(len(texts), y, True, ExpertConfig(n_folds=self.cfg.logreg_inner_folds, seed=self.cfg.seed))
        scores = {C: [] for C in self.cfg.logreg_Cs}
        for tr, te in folds:
            Xtr, Xte = self._vectorize([texts[i] for i in tr], [texts[i] for i in te])
            if Xtr is None:
                return
            for C in self.cfg.logreg_Cs:
                m = LogisticRegression(C=C, max_iter=2000)
                m.fit(Xtr, y[tr])
                p = np.full((len(te), self.n_classes), 1e-9)
                p[:, m.classes_] = np.clip(m.predict_proba(Xte), 1e-9, 1)
                scores[C].append(np.mean(np.log(p[np.arange(len(te)), y[te]])))  # mean log-likelihood
        for C in self.cfg.logreg_Cs:
            if np.mean(scores[C]) > best_score:
                best, best_score = C, np.mean(scores[C])
        self.C = best

    def fit_predict(self, tr_texts: list[str], y_tr: np.ndarray, te_texts: list[str]) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression, RidgeCV
        if not self.is_cls:
            y_tr = rank_gauss(y_tr, self.cfg)
        Xtr, Xte = self._vectorize(tr_texts, te_texts)
        if Xtr is None:  # no usable vocabulary: constant prediction
            if self.is_cls:
                prior = np.bincount(y_tr, minlength=self.n_classes) / len(y_tr)
                return _cls_output(np.tile(prior, (len(te_texts), 1)), self.n_classes)
            return np.full((len(te_texts), 1), float(np.mean(y_tr)))
        if self.is_cls:
            m = LogisticRegression(C=self.C or 1.0, max_iter=2000)
            m.fit(Xtr, y_tr)
            p = np.zeros((Xte.shape[0], self.n_classes))
            p[:, m.classes_] = m.predict_proba(Xte)
            return _cls_output(p, self.n_classes)
        m = RidgeCV(alphas=np.asarray(self.cfg.ridge_alphas, dtype=float))
        m.fit(Xtr, y_tr)
        return m.predict(Xte)[:, None]


class NeighbourExpert:
    """LEGACY P5 v1 (experts/1). CTE uses cte.knn_topk / cte.neighbour_average instead."""
    def __init__(self, is_cls: bool, n_classes: int, cfg: ExpertConfig):
        self.is_cls, self.n_classes, self.cfg = is_cls, n_classes, cfg

    def fit_predict(self, e_tr: np.ndarray, y_tr: np.ndarray, e_te: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors
        k = min(self.cfg.k_neighbors + int(exclude_self), len(e_tr))
        dist, idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(e_tr).kneighbors(e_te)
        if exclude_self:
            dist, idx = dist[:, 1:], idx[:, 1:]
        if self.cfg.neighbour_weights == "inverse_distance":
            w = 1.0 / (dist + 1e-6)
        else:
            w = np.clip(1.0 - dist, 0.0, None) + 1e-6  # cosine similarity, non-negative
        if not self.is_cls:
            y_tr = rank_gauss(y_tr, self.cfg)
        w = w / w.sum(axis=1, keepdims=True)
        if self.is_cls:
            Y = np.eye(self.n_classes)[y_tr]
            return _cls_output(np.einsum("nk,nkc->nc", w, Y[idx]), self.n_classes)
        return (w * y_tr[idx]).sum(axis=1)[:, None]


# ---------------------------------------------------------------------------------------- features

def _cross_fit(fit_predict: Callable, pool_inputs, y: np.ndarray, extra_inputs: dict, folds, mode: str,
               take) -> tuple[np.ndarray, dict]:
    """OOF predictions for the pool (mode 'oof') or in-sample (mode 'insample'); extras from the full-pool fit."""
    full_extra = {k: fit_predict(pool_inputs, y, v) for k, v in extra_inputs.items()}
    if mode == "insample":
        return fit_predict(pool_inputs, y, pool_inputs), full_extra
    oof = None
    for tr, te in folds:
        p = fit_predict(take(pool_inputs, tr), y[tr], take(pool_inputs, te))
        oof = np.zeros((len(y), p.shape[1])) if oof is None else oof
        oof[te] = p
    return oof, full_extra


def compute_expert_features(texts: Dict[str, list[str]], embs: Optional[Dict[str, np.ndarray]], y: np.ndarray,
                            is_cls: bool, n_classes: int, extra_texts: Dict[str, Dict[str, list[str]]],
                            extra_embs: Optional[Dict[str, Dict[str, np.ndarray]]], experts: tuple[str, ...],
                            mode: str, cfg: ExpertConfig, n_jobs: int = 1) -> Dict[str, Dict[str, np.ndarray]]:
    """LEGACY P5 v1 (experts/1); CTE uses cte.compute_cte_features.
    {split: {feature_name: (n_rows, k)}} for split in {"pool", *extras}. texts / embs per text column; the
    concatenated "__all__" view is built here. y: int class ids (cls) or floats (reg) for the pool rows.
    n_jobs > 1 runs the lexical views in parallel processes (independent fits, identical results)."""
    assert mode in ("oof", "insample")
    cols = sorted(texts)
    n = len(y)
    folds = expert_folds(n, y, is_cls, cfg)
    concat = lambda d: [" | ".join(f"{c}: {d[c][i]}" for c in cols) for i in range(len(d[cols[0]]))]
    views_t = {**{c: texts[c] for c in cols}, ALL_COLS: concat(texts)}
    extra_views_t = {k: {**{c: v[c] for c in cols}, ALL_COLS: concat(v)} for k, v in extra_texts.items()}
    out = {"pool": {}, **{k: {} for k in extra_texts}}
    take_list = lambda xs, idx: [xs[i] for i in idx]
    if "lex" in experts:
        def lex_view(view):
            tx = views_t[view]
            ex = LexicalExpert(is_cls, n_classes, cfg)
            ex.choose_C(tx, y)
            return _cross_fit(ex.fit_predict, tx, y, {k: v[view] for k, v in extra_views_t.items()}, folds, mode,
                              take_list)
        views = list(views_t)
        if n_jobs > 1 and len(views) > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=min(n_jobs, len(views)))(delayed(lex_view)(v) for v in views)
        else:
            res = [lex_view(v) for v in views]
        for view, (oof, ext) in zip(views, res):
            out["pool"][f"lex:{view}"] = oof
            for k, v in ext.items():
                out[k][f"lex:{view}"] = v
    if "nbr" in experts:
        def all_view(d):
            m = np.mean([d[c] for c in cols], axis=0)
            return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
        views_e = {**{c: embs[c] for c in cols}, ALL_COLS: all_view(embs)}
        extra_views_e = {k: {**{c: v[c] for c in cols}, ALL_COLS: all_view(v)} for k, v in extra_embs.items()}
        ex = NeighbourExpert(is_cls, n_classes, cfg)
        for view, E in views_e.items():
            if mode == "insample":  # ablation-only path; neighbours of a row include the row itself
                oof = ex.fit_predict(E, y, E)
                ext = {k: ex.fit_predict(E, y, v[view]) for k, v in extra_views_e.items()}
            else:
                oof, ext = _cross_fit(ex.fit_predict, E, y, {k: v[view] for k, v in extra_views_e.items()}, folds,
                                      "oof", lambda a, idx: a[idx])
            out["pool"][f"nbr:{view}"] = oof
            for k, v in ext.items():
                out[k][f"nbr:{view}"] = v
    return out


def cached_expert_features(cache_root: Optional[str], key: str, compute: Callable[[], dict]) -> tuple[dict, bool]:
    """Expert features are learner- and variant-independent for the same rows: compute once per (dataset, fold)."""
    path = join(cache_root, "experts", key + ".npz") if cache_root else None
    if path and exists(path):
        z = np.load(path, allow_pickle=False)
        out = {}
        for name in z.files:
            split, feat = name.split("||", 1)
            out.setdefault(split, {})[feat] = z[name]
        return out, True
    out = compute()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp.npz"  # unique per process: parallel jobs may share a cache
        np.savez_compressed(tmp, **{f"{s}||{f}": a for s, d in out.items() for f, a in d.items()})
        os.replace(tmp, path)
    return out, False

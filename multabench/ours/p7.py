"""P7 (PROGRESS.md): development round on top of CTE. Engine p7/1. cte/3 is untouched: label-aware variants call
cte.run_cte("cte") (same caches) and append extra feature groups; label-free variants start from frozen PCA only.

Extra feature groups (all fitted on the learner's training rows only; val/test rows are only transformed):
  num    : numeracy / form features per text column (label-free; same as cte's)
  lsa    : 30 LSA components of TF-IDF (word 1-2 + char_wb 3-5) of all text columns concatenated (label-free)
  row    : PCA-30 of a frozen e5 embedding of the verbalized full row, structured columns first then text
           ("col: val | ...", label-free)
  probe  : cross-fitted linear probe: Ridge on the full 384-d frozen embeddings of every text column (label-aware)
  rowlex : cross-fitted lexical expert on the verbalized full row: structured tokens (numeric deciles, categorical
           values) + the text (label-aware)
Label-aware groups use cte's machinery: 5-fold OOF on training rows, alpha from (0.3, 1, 3, 10) by the training-row OOF
metric, one-hot / rank-Gaussian targets, the same column budget.
"""
from __future__ import annotations

import re
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.ours import cte
from multabench.ours import tar_engine as te
from multabench.ours.experts import ALL_COLS, expert_folds

P7_VERSION = "p7/1"
P7_METHODS = {  # method -> (start from cte?, extra groups)
    "frozen0": (False, ()),
    "tf_num": (False, ("num",)),
    "tf_lsa": (False, ("lsa",)),
    "tf_row": (False, ("row",)),
    "tf_num_lsa": (False, ("num", "lsa")),
    "tf_all": (False, ("num", "lsa", "row")),
    "cte_probe": (True, ("probe",)),
    "cte_rowlex": (True, ("rowlex",)),
    "cte_lsa_row": (True, ("lsa", "row")),
    "cte_probe_rowlex": (True, ("probe", "rowlex")),
    "cte_all": (True, ("probe", "rowlex", "lsa", "row")),
}
N_COMPONENTS = 30
SEED = 42


# ---------------------------------------------------------------------------------------- verbalization

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def verbalize_rows(x: DataFrame, text_cols: list[str], struct_cols: list[str]) -> list[str]:
    """'col: val | ...' with the (short) structured columns first, so truncation cuts long text, not structure."""
    cols = list(struct_cols) + list(text_cols)
    vals = {c: x[c].tolist() for c in cols}
    return [" | ".join(f"{c}: {_fmt(vals[c][i])}" for c in cols) for i in range(len(x))]


def _token(s: str) -> str:
    return re.sub(r"\W+", "_", str(s)).strip("_")[:40] or "empty"


class RowTokenizer:
    """Structured columns as lexical tokens: numeric -> '<col>__q<decile>' (deciles of the training rows),
    categorical -> '<col>__<value>'; then the text of all text columns. Label-free; fitted on training rows."""

    def __init__(self, x_pool: DataFrame, text_cols: list[str], struct_cols: list[str], cat_cols: list[str]):
        self.text_cols, self.cat_cols = list(text_cols), set(cat_cols)
        self.num_cols = [c for c in struct_cols if c not in self.cat_cols]
        self.cats = [c for c in struct_cols if c in self.cat_cols]
        self.edges = {}
        for c in self.num_cols:
            v = pd.to_numeric(x_pool[c], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            self.edges[c] = np.unique(np.quantile(v, np.linspace(0.1, 0.9, 9))) if len(v) else np.array([])

    def transform(self, x: DataFrame) -> list[str]:
        parts = []
        for c in self.num_cols:
            v = pd.to_numeric(x[c], errors="coerce").to_numpy(dtype=float)
            b = np.searchsorted(self.edges[c], v, side="right")
            parts.append([f"{_token(c)}__{'missing' if not np.isfinite(vi) else f'q{bi}'}" for vi, bi in zip(v, b)])
        for c in self.cats:
            parts.append([f"{_token(c)}__{_token(s)}" for s in x[c].astype(str).tolist()])
        texts = [te.encoding_texts(x, c) for c in self.text_cols]
        return [" ".join(p[i] for p in parts) + " | " + " | ".join(f"{c}: {t[i]}" for c, t in zip(self.text_cols, texts))
                for i in range(len(x))]


# ---------------------------------------------------------------------------------------- cross-fitted linear expert

def xfit_linear(X, X_extra: Dict[str, object], y: np.ndarray, is_cls: bool, n_classes: int, c=cte.CTE_CONFIG):
    """cte's lexical machinery on any design matrix: OOF on training rows for each alpha, alpha by the OOF metric,
    the chosen alpha's OOF predictions for training rows, the all-training-rows fit for val/test."""
    folds = expert_folds(len(y), y, is_cls, c.expert_config())
    top = cte.top_classes(y, n_classes, c) if is_cls else None
    targets = lambda tr: cte._targets(y[tr], is_cls, n_classes, c)
    oofs, metrics = {}, []
    for a in c.ridge_alphas:
        P = None
        for tr, te_ in folds:
            p = cte.ridge_fit(X[tr], targets(tr), [X[te_]], a)[0]
            P = np.zeros((len(y), p.shape[1])) if P is None else P
            P[te_] = p
        oofs[a] = P
        metrics.append(cte.oof_metric(P, y, is_cls, n_classes))
    alpha = c.ridge_alphas[int(np.argmax(metrics))]
    names = list(X_extra)
    full = cte.ridge_fit(X, targets(np.arange(len(y))), [X_extra[k] for k in names], alpha)
    feat = lambda P: cte.class_features(P, is_cls, n_classes, top)
    return feat(oofs[alpha]), {k: feat(P) for k, P in zip(names, full)}, {"alpha": alpha, "oof_metric": metrics}


# ---------------------------------------------------------------------------------------- main entry

def run_p7(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], struct_cols: list[str],
           cat_cols: list[str], is_cls: bool, extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig,
           device: torch.device, cache_root: Optional[str], dataset: str, fold: int):
    from sklearn.decomposition import PCA, TruncatedSVD
    from multabench.ours.stacking import StackResult
    use_cte, groups = P7_METHODS[method]
    text_cols, struct_cols = sorted(text_cols), sorted(struct_cols or [])
    extra = {k: v for k, v in extra.items() if v is not None}
    splits = {"pool": x_fit, **{k: x for k, (x, _) in extra.items()}}
    rows = {k: np.asarray(x.index) for k, x in splits.items()}
    all_rows = np.sort(np.concatenate(list(rows.values())))
    t0 = time.time()
    diag = {"engine_version": P7_VERSION, "method": method, "groups": list(groups), "seconds": {}}
    if use_cte:
        base = cte.run_cte("cte", x_fit, y_fit, text_cols, is_cls, extra, cfg, device, cache_root, dataset, fold)
        frozen, features = base.embedding_result, {c: base.features[c].copy() for c in text_cols}
        diag["cte"] = {k: base.diagnostics.get(k) for k in ("lex_alpha", "expert_cache_hits", "seconds")}
    else:
        frozen, stats = cte.frozen_embeddings_cached([x[text_cols] for x in splits.values()], text_cols, cfg, device,
                                                     cache_root, dataset)
        features = {c: DataFrame(index=all_rows) for c in text_cols}
        diag["frozen_embedding_cache"] = stats
    diag["seconds"]["base"] = round(time.time() - t0, 1)
    if is_cls:
        classes = sorted(y_fit.astype(str).unique())
        y = y_fit.astype(str).map({k: i for i, k in enumerate(classes)}).to_numpy()
    else:
        classes, y = [], pd.to_numeric(y_fit).to_numpy(dtype=float)
    n_classes = len(classes) if is_cls else 1
    texts = {k: {c: te.encoding_texts(x, c) for c in text_cols} for k, x in splits.items()}
    concat = {k: [" | ".join(f"{c}: {t[c][i]}" for c in text_cols) for i in range(len(splits[k]))] for k, t in texts.items()}
    other = [k for k in splits if k != "pool"]
    new: Dict[str, tuple[str, Dict[str, np.ndarray], list]] = {}  # name -> (owner col, {split: array}, col names)

    for g in groups:
        t1 = time.time()
        if g == "num":
            for c in text_cols:
                new[f"num:{c}"] = (c, {k: cte.numeracy_matrix(texts[k][c]) for k in splits}, list(cte.NUMERACY_NAMES))
        elif g == "lsa":
            X, Xe = cte.tfidf_matrices(concat["pool"], {k: concat[k] for k in other}, cte.CTE_CONFIG)
            k_ = max(1, min(N_COMPONENTS, X.shape[1] - 1, X.shape[0] - 1))
            svd = TruncatedSVD(n_components=k_, random_state=SEED).fit(X)
            new[f"lsa:{ALL_COLS}"] = (text_cols[0], {"pool": svd.transform(X), **{k: svd.transform(Xe[k]) for k in other}},
                                      list(range(k_)))
        elif g == "row":
            verb = {k: DataFrame({"row": verbalize_rows(x, text_cols, struct_cols)}, index=x.index)
                    for k, x in splits.items()}
            remb, st = cte.frozen_embeddings_cached(list(verb.values()), ["row"], cfg, device, cache_root,
                                                    f"{dataset}__p7row")
            E = {k: remb.row_embeddings("row", rows[k]) for k in splits}
            pca = PCA(n_components=min(N_COMPONENTS, len(rows["pool"]) - 1), random_state=SEED).fit(E["pool"])
            new[f"row:{ALL_COLS}"] = (text_cols[0], {k: pca.transform(E[k]) for k in splits}, list(range(pca.n_components_)))
            diag["row_embedding_cache"] = st
        elif g == "probe":
            E = {k: np.hstack([frozen.row_embeddings(c, rows[k]) for c in text_cols]) for k in splits}
            oof, ext, meta = xfit_linear(E["pool"], {k: E[k] for k in other}, y, is_cls, n_classes)
            new[f"probe:{ALL_COLS}"] = (text_cols[0], {"pool": oof, **ext}, list(range(oof.shape[1])))
            diag["probe"] = meta
        elif g == "rowlex":
            tok = RowTokenizer(x_fit, text_cols, struct_cols, cat_cols)
            X, Xe = cte.tfidf_matrices(tok.transform(x_fit), {k: tok.transform(splits[k]) for k in other}, cte.CTE_CONFIG)
            oof, ext, meta = xfit_linear(X, Xe, y, is_cls, n_classes)
            new[f"rowlex:{ALL_COLS}"] = (text_cols[0], {"pool": oof, **ext}, list(range(oof.shape[1])))
            diag["rowlex"] = meta
        diag["seconds"][g] = round(time.time() - t1, 1)

    for name, (owner, arrs, cols) in new.items():
        block = pd.concat([DataFrame(arrs[k], index=rows[k]) for k in splits]).sort_index()
        block.columns = [f"{name}#{c}" for c in cols]
        features[owner] = pd.concat([features[owner], block.loc[features[owner].index]], axis=1)
    diag["n_extra_columns"] = {c: int(features[c].shape[1]) for c in text_cols}
    print(f"  [STACK:{method}] p7 groups {list(groups)} on {'cte' if use_cte else 'frozen'}; seconds {diag['seconds']}",
          flush=True)
    return StackResult(method=method, embedding_result=frozen, base_result=None, features=features, diagnostics=diag)

"""P9 (PROGRESS.md): development round 2 on top of cte_probe. Engine p9/1. cte/3, p6 and p7 are untouched.

One configurable builder of the whole cte_probe feature set (lexical expert on all text columns, neighbour expert
per view, cross-fitted linear probe on the full frozen embeddings, numeracy), so every candidate differs from the
baseline `p9_base` (= cte_probe, checked by tests/test_p9.py) in exactly one switch:
  test_mode "full" : val/test predictions from experts refit on all training rows (as cte / cte_probe)
            "avg"  : val/test predictions = mean of the fold models that made the training rows' OOF predictions
  repeats   r      : OOF predictions averaged over r fold splits (fold seeds 42, 43, ...)
  prompt           : an extra frozen-e5 view "passage: predict <target name> | <col>: <val>" (the target's name only),
                     with a cross-fitted probe and kNN on it
  rff              : a nonlinear probe: random Fourier features (RBF) of the frozen embeddings -> cross-fitted Ridge
  nb               : classification lexical expert on NB-weighted TF-IDF (as P6 cte_nb)
All label-aware parts use training rows only: OOF on the training rows; val/test rows are only predicted.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.ours import cte
from multabench.ours import tar_engine as te
from multabench.ours.experts import ALL_COLS, ExpertConfig, expert_folds

P9_VERSION = "p9/1"


@dataclasses.dataclass(frozen=True)
class P9Spec:
    test_mode: str = "full"
    repeats: int = 1
    prompt: bool = False
    rff: bool = False
    nb: bool = False
    query: bool = False  # e5 "query: " prefix (e5 model card: use it for embeddings as features / linear probing)
    percol: bool = False  # P12: per-column lexical experts in addition to the all-columns one
    k10: bool = False     # P12: 10-fold cross-fitting for every expert
    maxpool: bool = False  # P12: probe + kNN on max-pooled frozen e5 token states
    tok: bool = False     # P12: TabSTAR-style target tokens (similarity to frozen embeddings of the class names)


P9_METHODS = {
    "p9_base": P9Spec(),
    "p9_avg": P9Spec(test_mode="avg"),
    "p9_rep3": P9Spec(repeats=3),
    "p9_rep3_avg": P9Spec(test_mode="avg", repeats=3),
    "p9_prompt": P9Spec(prompt=True),
    "p9_rff": P9Spec(rff=True),
    "p9_nb": P9Spec(nb=True),
    "p9_query": P9Spec(query=True),
    "p9_percol": P9Spec(percol=True),
    "p9_k10": P9Spec(k10=True),
    "p9_maxpool": P9Spec(maxpool=True),
    "p9_tok": P9Spec(tok=True),
}
RFF_COMPONENTS = 2048
SEED = 42


# ---------------------------------------------------------------------------------------- generic cross-fitting

def _folds_list(y, is_cls, c, repeats: int):
    return [expert_folds(len(y), y, is_cls, ExpertConfig(n_folds=c.n_folds, seed=c.seed + r)) for r in range(repeats)]


def _oof_and_ext(fit_predict: Callable, Xp, Xe: Dict[str, object], y, folds_list, take):
    """Mean OOF over the fold splits, and the mean of all fold models' predictions on each extra split."""
    names = list(Xe)
    oof_sum, ext_sum, n_models = None, {k: None for k in names}, 0
    for folds in folds_list:
        P = None
        for tr, te_ in folds:
            outs = fit_predict(take(Xp, tr), tr, [take(Xp, te_)] + [Xe[k] for k in names])
            P = np.zeros((len(y), outs[0].shape[1])) if P is None else P
            P[te_] = outs[0]
            for k, o in zip(names, outs[1:]):
                ext_sum[k] = o if ext_sum[k] is None else ext_sum[k] + o
            n_models += 1
        oof_sum = P if oof_sum is None else oof_sum + P
    return oof_sum / len(folds_list), {k: v / n_models for k, v in ext_sum.items()}


def linear_expert(Xp, Xe, y, is_cls, n_classes, c, spec: P9Spec, fit: Optional[Callable] = None):
    """cte's lexical machinery (alpha by the training-row OOF metric) with P9's test_mode / repeats switches."""
    take = lambda A, idx: A[idx]
    fit = fit or (lambda A, ytr, Bs, a: cte.ridge_fit(A, cte._targets(ytr, is_cls, n_classes, c), Bs, a))
    folds_list = _folds_list(y, is_cls, c, spec.repeats)
    oofs, exts, metrics = {}, {}, []
    for a in c.ridge_alphas:
        oofs[a], exts[a] = _oof_and_ext(lambda A, tr, Bs: fit(A, y[tr], Bs, a), Xp, Xe, y, folds_list, take)
        metrics.append(cte.oof_metric(oofs[a], y, is_cls, n_classes))
    alpha = c.ridge_alphas[int(np.argmax(metrics))]
    if spec.test_mode == "full":
        names = list(Xe)
        ext = dict(zip(names, fit(Xp, y, [Xe[k] for k in names], alpha)))
    else:
        ext = exts[alpha]
    return oofs[alpha], ext, {"alpha": alpha, "oof_metric": [round(float(m), 6) for m in metrics]}


def knn_expert(Ep, Ee, y, is_cls, n_classes, c, spec: P9Spec, device):
    take = lambda A, idx: A[idx]
    fp = lambda A, tr, Bs: [cte.neighbour_average(A, cte._targets(y[tr], is_cls, n_classes, c), B, c, device) for B in Bs]
    oof, ext_avg = _oof_and_ext(fp, Ep, Ee, y, _folds_list(y, is_cls, c, spec.repeats), take)
    if spec.test_mode == "avg":
        return oof, ext_avg
    names = list(Ee)
    return oof, dict(zip(names, fp(Ep, np.arange(len(y)), [Ee[k] for k in names])))


# ---------------------------------------------------------------------------------------- main entry

def run_p9(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], struct_cols: list[str],
           cat_cols: list[str], is_cls: bool, extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig,
           device: torch.device, cache_root: Optional[str], dataset: str, fold: int):
    from multabench.ours.stacking import StackResult
    spec, c = P9_METHODS[method], cte.CTE_CONFIG
    if spec.k10:
        c = dataclasses.replace(c, n_folds=10)
    text_cols = sorted(text_cols)
    extra = {k: v for k, v in extra.items() if v is not None}
    splits = {"pool": x_fit, **{k: x for k, (x, _) in extra.items()}}
    other = [k for k in splits if k != "pool"]
    rows = {k: np.asarray(x.index) for k, x in splits.items()}
    all_rows = np.sort(np.concatenate(list(rows.values())))
    t0 = time.time()
    frozen, stats = cte.frozen_embeddings_cached([x[text_cols] for x in splits.values()], text_cols, cfg, device,
                                                 cache_root, dataset, prefix="query: " if spec.query else "passage: ")
    diag = {"engine_version": P9_VERSION, "method": method, "spec": dataclasses.asdict(spec),
            "frozen_embedding_cache": stats, "seconds": {"embeddings": round(time.time() - t0, 1)}}
    if is_cls:
        classes = sorted(y_fit.astype(str).unique())
        y = y_fit.astype(str).map({k: i for i, k in enumerate(classes)}).to_numpy()
    else:
        classes, y = [], pd.to_numeric(y_fit).to_numpy(dtype=float)
    n_classes = len(classes) if is_cls else 1
    top = cte.top_classes(y, n_classes, c) if is_cls else None
    feat = lambda P: cte.class_features(P, is_cls, n_classes, top)
    texts = {k: {col: te.encoding_texts(x, col) for col in text_cols} for k, x in splits.items()}
    concat = {k: [" | ".join(f"{col}: {t[col][i]}" for col in text_cols) for i in range(len(splits[k]))]
              for k, t in texts.items()}
    E = {k: {col: frozen.row_embeddings(col, rows[k]) for col in text_cols} for k in splits}
    new: Dict[str, tuple[str, Dict[str, np.ndarray], list]] = {}

    def put(name, owner, oof, ext):
        arrs = {"pool": feat(oof), **{k: feat(v) for k, v in ext.items()}}
        new[name] = (owner, arrs, list(range(arrs["pool"].shape[1])))

    # lexical expert (all text columns concatenated); NB weighting for classification if spec.nb
    t1 = time.time()
    X, Xe = cte.tfidf_matrices(concat["pool"], {k: concat[k] for k in other}, c)
    nb_fit = None
    if spec.nb and is_cls:
        from multabench.ours.p6 import nb_ridge_scores
        nb_fit = lambda A, ytr, Bs, a: nb_ridge_scores(A, ytr, Bs, n_classes, a)
    oof, ext, diag["lex"] = linear_expert(X, Xe, y, is_cls, n_classes, c, spec, nb_fit)
    put(f"lex:{ALL_COLS}", text_cols[0], oof, ext)
    diag["seconds"]["lex"] = round(time.time() - t1, 1)

    if spec.percol and len(text_cols) > 1:  # P12: one lexical expert per text column too
        t1 = time.time()
        for col in text_cols:
            Xc, Xce = cte.tfidf_matrices(texts["pool"][col], {k: texts[k][col] for k in other}, c)
            oof, ext, diag[f"lex_{col}"] = linear_expert(Xc, Xce, y, is_cls, n_classes, c, spec)
            put(f"lex:{col}", col, oof, ext)
        diag["seconds"]["percol"] = round(time.time() - t1, 1)

    # neighbour expert per view (multiclass: concatenation only), as cte
    t1 = time.time()
    views = {k: {ALL_COLS: cte._normalise(np.mean([E[k][col] for col in text_cols], axis=0))} for k in splits}
    if not (is_cls and n_classes > 2):
        for k in splits:
            views[k] = {**{col: E[k][col] for col in text_cols}, **views[k]}
    for view in views["pool"]:
        oof, ext = knn_expert(views["pool"][view], {k: views[k][view] for k in other}, y, is_cls, n_classes, c, spec,
                              device)
        put(f"nbr:{view}", text_cols[0] if view == ALL_COLS else view, oof, ext)
    diag["seconds"]["nbr"] = round(time.time() - t1, 1)

    # linear probe on the full frozen embeddings (as p7 cte_probe)
    t1 = time.time()
    Ecat = {k: np.hstack([E[k][col] for col in text_cols]) for k in splits}
    oof, ext, diag["probe"] = linear_expert(Ecat["pool"], {k: Ecat[k] for k in other}, y, is_cls, n_classes, c, spec)
    put(f"probe:{ALL_COLS}", text_cols[0], oof, ext)
    diag["seconds"]["probe"] = round(time.time() - t1, 1)

    if spec.rff:  # nonlinear probe: random Fourier features of the (label-free) embeddings
        from sklearn.kernel_approximation import RBFSampler
        t1 = time.time()
        rng = np.random.default_rng(SEED)
        sub = Ecat["pool"][rng.choice(len(y), size=min(len(y), 1000), replace=False)].astype(np.float64)
        sq = (sub ** 2).sum(axis=1)  # pairwise squared distances without an (n, n, d) array (E30: 9 GB, swapped)
        d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * sub @ sub.T, 0.0, None)
        gamma = 1.0 / float(np.median(d2[np.triu_indices(len(sub), 1)]))
        rff = RBFSampler(gamma=gamma, n_components=RFF_COMPONENTS, random_state=SEED).fit(Ecat["pool"])
        R = {k: rff.transform(Ecat[k]) for k in splits}
        oof, ext, diag["rff"] = linear_expert(R["pool"], {k: R[k] for k in other}, y, is_cls, n_classes, c, spec)
        diag["rff"]["gamma"] = gamma
        put(f"rff:{ALL_COLS}", text_cols[0], oof, ext)
        diag["seconds"]["rff"] = round(time.time() - t1, 1)

    if spec.prompt:  # task-aware prompting of the frozen encoder: the target's NAME in the passage
        t1 = time.time()
        target = str(y_fit.name or "target")
        pcols = {col: f"predict {target} | {col}" for col in text_cols}
        pframes = [DataFrame({pcols[col]: x[col].to_numpy() for col in text_cols}, index=x.index)
                   for x in splits.values()]
        pres, pst = cte.frozen_embeddings_cached(pframes, list(pcols.values()), cfg, device, cache_root, dataset)
        P_ = {k: {col: pres.row_embeddings(pcols[col], rows[k]) for col in text_cols} for k in splits}
        Pcat = {k: np.hstack([P_[k][col] for col in text_cols]) for k in splits}
        oof, ext, diag["pprobe"] = linear_expert(Pcat["pool"], {k: Pcat[k] for k in other}, y, is_cls, n_classes, c, spec)
        put(f"pprobe:{ALL_COLS}", text_cols[0], oof, ext)
        pall = {k: cte._normalise(np.mean([P_[k][col] for col in text_cols], axis=0)) for k in splits}
        oof, ext = knn_expert(pall["pool"], {k: pall[k] for k in other}, y, is_cls, n_classes, c, spec, device)
        put(f"pnbr:{ALL_COLS}", text_cols[0], oof, ext)
        diag["prompt_embedding_cache"] = pst
        diag["seconds"]["prompt"] = round(time.time() - t1, 1)

    if spec.maxpool:  # P12: max-pooled token states of the same frozen e5 ("passage: col: val")
        t1 = time.time()
        M = maxpool_embeddings({k: x[text_cols] for k, x in splits.items()}, text_cols, cfg, device, cache_root, dataset)
        Mcat = {k: np.hstack([M[k][col] for col in text_cols]) for k in splits}
        oof, ext, diag["mprobe"] = linear_expert(Mcat["pool"], {k: Mcat[k] for k in other}, y, is_cls, n_classes, c, spec)
        put(f"mprobe:{ALL_COLS}", text_cols[0], oof, ext)
        mall = {k: cte._normalise(np.mean([M[k][col] for col in text_cols], axis=0)) for k in splits}
        oof, ext = knn_expert(mall["pool"], {k: mall[k] for k in other}, y, is_cls, n_classes, c, spec, device)
        put(f"mnbr:{ALL_COLS}", text_cols[0], oof, ext)
        diag["seconds"]["maxpool"] = round(time.time() - t1, 1)

    if spec.tok and is_cls:  # P12: TabSTAR-style target tokens: label NAMES only, no row labels
        target = str(y_fit.name or "target")
        cres, _ = cte.frozen_embeddings_cached([DataFrame({target: classes}, index=pd.RangeIndex(len(classes)))], [target],
                                              cfg, device, cache_root, f"{dataset}__tok")
        CE = cres.row_embeddings(target, np.arange(len(classes)))
        keep = top if top is not None else np.arange(n_classes)
        arrs = {k: (cte._normalise(np.mean([E[k][col] for col in text_cols], axis=0)) @ CE.T)[:, keep] for k in splits}
        new[f"tok:{ALL_COLS}"] = (text_cols[0], arrs, list(range(len(keep))))

    for col in text_cols:  # numeracy (label-free), as cte
        new[f"num:{col}"] = (col, {k: cte.numeracy_matrix(texts[k][col]) for k in splits}, list(cte.NUMERACY_NAMES))

    features = {col: DataFrame(index=all_rows) for col in text_cols}
    for name in sorted(new):
        owner, arrs, cols = new[name]
        block = pd.concat([DataFrame(arrs[k], index=rows[k]) for k in splits]).sort_index()
        block.columns = [f"{name}#{j}" for j in cols]
        features[owner] = pd.concat([features[owner], block.loc[all_rows]], axis=1)
    print(f"  [STACK:{method}] p9 {dataclasses.asdict(spec)}; seconds {diag['seconds']}", flush=True)
    return StackResult(method=method, embedding_result=frozen, base_result=None, features=features, diagnostics=diag)


def maxpool_embeddings(frames: Dict[str, DataFrame], text_cols: list[str], cfg: te.TarConfig, device: torch.device,
                       cache_root: Optional[str], dataset: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Max over (unpadded) last-layer token states of the frozen e5, L2-normalised, for "passage: col: val" (the same
    input strings as the mean-pooled embeddings). Cached per dataset and column by text hash. Label-free."""
    import hashlib
    import os
    from os.path import exists, join
    from transformers import AutoModel, AutoTokenizer
    from multabench.e5.constants import format_e5_passage
    tok = model = None
    out = {k: {} for k in frames}
    for col in text_cols:
        texts = {k: te.encoding_texts(f, col) for k, f in frames.items()}
        uniq = sorted({t for v in texts.values() for t in v})
        hashes = te._text_hash(uniq)
        path = (join(cache_root, "frozen_maxpool", dataset, hashlib.sha1(col.encode()).hexdigest()[:12] + ".npz")
                if cache_root else None)
        store = {}
        if path and exists(path):
            z = np.load(path)
            store = dict(zip(z["hash"].tolist(), z["emb"]))
        missing = [i for i, h in enumerate(hashes) if int(h) not in store]
        if missing:
            if model is None:
                tok = AutoTokenizer.from_pretrained(cfg.model_name)
                model = AutoModel.from_pretrained(cfg.model_name).to(device).eval()
            strings = [format_e5_passage(col, uniq[i]) for i in missing]
            embs = []
            with torch.no_grad():
                for s0 in range(0, len(strings), 64):
                    b = tok(strings[s0:s0 + 64], padding=True, truncation=True, max_length=cfg.max_length,
                            return_tensors="pt").to(device)
                    h = model(**b).last_hidden_state.masked_fill(~b["attention_mask"].bool()[..., None], -1e4)
                    embs.append(torch.nn.functional.normalize(h.max(dim=1).values, dim=1).cpu().numpy())
            for i, e in zip(missing, np.concatenate(embs)):
                store[int(hashes[i])] = e.astype(np.float32)
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp.npz"
                np.savez(tmp, hash=np.array(list(store), dtype=np.uint64), emb=np.stack(list(store.values())))
                os.replace(tmp, path)
        pos = {t: store[int(h)] for t, h in zip(uniq, hashes)}
        for k, v in texts.items():
            out[k][col] = np.stack([pos[t] for t in v])
    return out

"""P3 (PROGRESS.md): TAR head predictions as extra features for the learner ("stacking").

Methods:
- insample_pred     tar_repro embeddings + the full-fit tar_repro head's predictions (training rows in-sample; ablation)
- xfit_pred         tar_repro embeddings + K=3 cross-fitted tar_repro head predictions
- frozen_xfit_pred  Joint Frozen E5 embeddings + the same cross-fitted predictions as xfit_pred
- xfit_offset_pred  tar_repro embeddings + K=3 cross-fitted OFFSET head predictions (pool-level OOF structured offsets)

Cross-fitting: the E5 pool (the learner's rows given to the text-encoder step) is split into K folds; for fold k the
model is the *unchanged* tar_repro (or offset) code run on the other K−1 folds, and it predicts the held-out fold,
learner-val and test rows. Pool rows keep the prediction of the one model that never saw them; learner-val / test
rows get the mean over the K models. tar_repro itself is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold

from multabench.ours import tar_engine as te
import hashlib

from multabench.ours.variants import _metric, run_variant
from multabench.ours.structured_offset import fit_structured_offset

STACKING_ENGINE_VERSION = "stacking/1"
K_FOLDS = 3
SEED = 42
# P5 (PROGRESS.md): (base embeddings, experts, mode). C2 = tar_experts, C3 = frozen_experts, the rest are ablations.
EXPERT_METHODS = {
    "tar_experts": ("tar", ("lex", "nbr"), "oof"),
    "frozen_experts": ("frozen", ("lex", "nbr"), "oof"),
    "tar_experts_lex": ("tar", ("lex",), "oof"),
    "tar_experts_nbr": ("tar", ("nbr",), "oof"),
    "tar_experts_lex_insample": ("tar", ("lex",), "insample"),
}
from multabench.ours.cte import CTE_METHODS  # P5 v2
from multabench.ours.p6 import P6_METHODS  # P6 (exploratory); catboost_native_text has no stacked features
P6_CTE_METHODS = {m for m, v in P6_METHODS.items() if v[0] is not None}
from multabench.ours.p7 import P7_METHODS  # P7 development round (exploratory until pre-registered)
from multabench.ours.p9 import P9_METHODS  # P9 development round 2
from multabench.ours.depth import DEPTH_METHODS  # P11 depth probes (cte_probe + every-layer probes)
STACK_METHODS = ({"insample_pred", "xfit_pred", "frozen_xfit_pred", "xfit_offset_pred"} | set(EXPERT_METHODS)
                 | set(CTE_METHODS) | P6_CTE_METHODS | set(P7_METHODS) | set(P9_METHODS) | set(DEPTH_METHODS))
COMPANIONS = {"xfit_pred": ["frozen_xfit_pred"]}  # written by the same job (shared fold fine-tunes)


def xfit_folds(pool_index: pd.Index, y: Series, is_cls: bool, k: int = K_FOLDS, seed: int = SEED) -> list[np.ndarray]:
    """K disjoint held-out index sets covering the pool; deterministic for the same pool order and labels."""
    n = len(pool_index)
    labels = y.loc[pool_index].astype(str).to_numpy() if is_cls else None
    if is_cls and pd.Series(labels).value_counts().min() >= k:
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n), labels)
    else:
        splits = KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n))
    return [np.asarray(pool_index)[te_idx] for _, te_idx in splits]


# ---------------------------------------------------------------------------------------- prediction features

def _bin_midpoints(classes: list[str], edges: np.ndarray) -> np.ndarray:
    """Midpoint of each "Quantile a-b%" class (upstream discretize_numerical labels), in class order."""
    n = len(edges) - 1
    step = 100 / n
    label_to_bin = {f"Quantile {round(i * step)}-{round((i + 1) * step)}%": i for i in range(n)}
    mids = (edges[:-1] + edges[1:]) / 2
    return np.array([mids[label_to_bin[c]] for c in classes])


def _y_edges(y: Series, n_bins: int) -> np.ndarray:
    _, edges = pd.qcut(pd.to_numeric(y), q=n_bins, labels=False, duplicates="drop", retbins=True)
    return np.asarray(edges, dtype=float)


def head_features(result: te.TarResult, rows: np.ndarray, is_cls: bool, all_classes: Optional[list[str]],
                  midpoints: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Per text column: (n_rows, f) features from the head's softmax of that column's cell (offset NOT added).
    cls: probabilities on all_classes (binary -> P(classes[1]); multiclass -> drop the first class);
    reg: [expected value over bin midpoints, std of that distribution]."""
    out = {}
    for col in result.text_cols:
        c = result.columns[col]
        uid = np.array([c.row_uid[int(r)] for r in rows])
        p = torch.softmax(torch.tensor(c.logits[uid]), dim=-1).numpy()
        if is_cls:
            full = np.zeros((len(rows), len(all_classes)))
            pos = {cl: i for i, cl in enumerate(all_classes)}
            for j, cl in enumerate(result.head["classes"]):
                full[:, pos[cl]] = p[:, j]
            out[col] = full[:, 1:2] if len(all_classes) == 2 else full[:, 1:]
        else:
            ev = p @ midpoints
            sd = np.sqrt(np.clip(p @ (midpoints ** 2) - ev ** 2, 0, None))
            out[col] = np.stack([ev, sd], axis=1)
    return out


@dataclass
class StackResult:
    method: str
    embedding_result: te.TarResult           # embeddings fed to PCA (tar_repro's, or frozen E5)
    base_result: Optional[te.TarResult]      # the full-pool tar_repro fine-tune (JSON diagnostics); None = skipped
    features: Dict[str, DataFrame]           # per text column: rows (pool + extras) x prediction features
    diagnostics: dict = field(default_factory=dict)


def _features_frame(per_col_rows: list[tuple[np.ndarray, Dict[str, np.ndarray]]], text_cols: list[str]) -> Dict[str, DataFrame]:
    out = {}
    for col in text_cols:
        idx = np.concatenate([r for r, _ in per_col_rows])
        vals = np.concatenate([f[col] for _, f in per_col_rows])
        out[col] = pd.DataFrame(vals, index=idx).sort_index()
    return out


def _quality(features: Dict[str, DataFrame], y: Series, rows: np.ndarray, is_cls: bool, classes: list[str],
             residual_of: Optional[np.ndarray] = None) -> Optional[float]:
    """Quality of the prediction features themselves (mean over text columns of the head output) on these rows."""
    if len(rows) == 0:
        return None
    f = np.mean([features[c].loc[rows].to_numpy() for c in features], axis=0)
    if is_cls:
        probs = np.hstack([1 - f.sum(axis=1, keepdims=True), f]) if f.shape[1] == len(classes) - 1 else f
        return _metric(y.loc[rows].to_numpy(), probs, True, classes)
    target = pd.to_numeric(y.loc[rows]).to_numpy(dtype=float) - (0 if residual_of is None else residual_of)
    return _metric(target, f[:, 0], False, [])


# ---------------------------------------------------------------------------------------- main entry

def _tar_inputs(x: DataFrame, text_cols: list[str], extra: dict) -> tuple[DataFrame, dict]:
    return x[text_cols], {k: (v[0][text_cols], v[1]) for k, v in extra.items()}


def frozen_embeddings(x_fit: DataFrame, extra: dict, text_cols: list[str], cfg: te.TarConfig,
                      device: torch.device) -> te.TarResult:
    """Joint Frozen embeddings (vanilla e5-small-v2, "passage: col: val", L2-normalized mean pool) for all rows —
    numerically equal to upstream's encode_texts_with_e5 (tests/test_tar_engine.py), via the fast path."""
    from transformers import AutoTokenizer
    from multabench.e5.constants import format_e5_passage
    import itertools
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    frames = [x_fit] + [x for x, _ in extra.values()]
    te.set_seed(cfg.seed)
    model = te.build_model(cfg, d_output=2, device=device)  # fresh LoRA (B = 0) = vanilla E5; head unused
    columns = {}
    for col in sorted(text_cols):
        rows = np.concatenate([np.asarray(f.index) for f in frames])
        raw = list(itertools.chain.from_iterable(te.encoding_texts(f, col) for f in frames))
        uniq, inverse = np.unique(np.array(raw, dtype=object), return_inverse=True)
        uniq = uniq.tolist()
        strings = [format_e5_passage(col, t) for t in uniq]
        tokens = te.TokenStore(strings, tok, cfg.max_length)
        emb, logits = te.encode(model, te.make_source(model, tokens, cfg, device), tokens.uids(strings), cfg, device)
        columns[col] = te.ColumnOutputs(emb=emb, logits=logits, text_hash=te._text_hash(uniq),
                                        row_uid=dict(zip(rows.tolist(), inverse.tolist())))
    del model
    return te.TarResult(variant="frozen", cache_key="", cache_hit=False, config={"model_name": cfg.model_name},
                        text_cols=sorted(text_cols), splits={}, columns=columns, head={"classes": [], "d_output": 0},
                        diagnostics={})


def tar_repro_cache_key(x_fit: DataFrame, extra: dict, cfg: te.TarConfig, device: torch.device, dataset: str,
                        fold: int) -> str:
    """The cache key tar_engine.run_tar would use for these inputs (same formula; tests/test_stacking.py checks it)."""
    extra = {k: v for k, v in extra.items() if v is not None}
    return (f"{dataset}/fold{fold}/tar_repro/seed{cfg.seed}/{te.config_hash(cfg, 'tar_repro', device)}_"
            f"{te._rows_hash(x_fit.index, *[x.index for x, _ in extra.values()])}")


def cached_tar_repro(x_fit: DataFrame, extra: dict, cfg: te.TarConfig, device: torch.device, cache_root: Optional[str],
                     dataset: str, fold: int) -> Optional[te.TarResult]:
    """The full-pool tar_repro result if it is already cached, else None (never trains)."""
    if not cache_root:
        return None
    from os.path import exists, join
    path = join(cache_root, tar_repro_cache_key(x_fit, extra, cfg, device, dataset, fold))
    return te._load(path) if exists(join(path, "meta.json")) else None


def run_stacking(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], struct_cols: list[str],
                 cat_cols: list[str], is_cls: bool, extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig,
                 device: torch.device, cache_root: Optional[str], dataset: str, fold: int) -> StackResult:
    assert method in STACK_METHODS, method
    if method in EXPERT_METHODS:
        return run_experts(method, x_fit, y_fit, text_cols, is_cls, extra, cfg, device, cache_root, dataset, fold)
    if method in DEPTH_METHODS:
        from multabench.ours.depth import run_depth
        return run_depth(method, x_fit, y_fit, text_cols, struct_cols, cat_cols, is_cls, extra, cfg, device, cache_root,
                         dataset, fold)
    if method in P9_METHODS:
        from multabench.ours.p9 import run_p9
        return run_p9(method, x_fit, y_fit, text_cols, struct_cols, cat_cols, is_cls, extra, cfg, device, cache_root,
                      dataset, fold)
    if method in P7_METHODS:
        from multabench.ours.p7 import run_p7
        return run_p7(method, x_fit, y_fit, text_cols, struct_cols, cat_cols, is_cls, extra, cfg, device, cache_root,
                      dataset, fold)
    if method in CTE_METHODS or method in P6_CTE_METHODS:
        from multabench.ours.cte import run_cte
        return run_cte(method, x_fit, y_fit, text_cols, is_cls, extra, cfg, device, cache_root, dataset, fold,
                       struct_cols=struct_cols, cat_cols=cat_cols)
    text_cols = sorted(text_cols)
    extra = {k: v for k, v in extra.items() if v is not None}
    pool = pd.Index(x_fit.index)
    x_t, extra_t = _tar_inputs(x_fit, text_cols, extra)
    # The full-pool tar_repro fine-tune: tar_repro embeddings + the in-sample head. frozen_xfit_pred needs it only for
    # the JSON's TAR diagnostics, so it loads it if cached (full run: tar_repro CatBoost runs first) and otherwise
    # skips it (e.g. the TabPFN rows) instead of spending a fine-tune on diagnostics (P4).
    if method == "frozen_xfit_pred":
        base = cached_tar_repro(x_t, extra_t, cfg, device, cache_root, dataset, fold)
        print(f"  [STACK:{method}] base tar_repro diagnostics: {'cache hit' if base else 'skipped (not cached)'}",
              flush=True)
    else:
        base = te.run_tar(x_t, y_fit, text_cols, is_cls, extra_t, cfg, device, cache_root, dataset, fold,
                          variant="tar_repro")
    classes = sorted(y_fit.astype(str).unique()) if is_cls else []
    extra_rows = {k: np.asarray(x.index) for k, (x, _) in extra.items()}
    diag = {"engine_version": STACKING_ENGINE_VERSION, "k_folds": K_FOLDS, "method": method}

    if method == "insample_pred":
        mids = None if is_cls else _bin_midpoints(base.head["classes"], _y_edges(y_fit, cfg.n_bins))
        all_rows = np.concatenate([np.asarray(pool)] + list(extra_rows.values()))
        feats = _features_frame([(all_rows, head_features(base, all_rows, is_cls, classes, mids))], text_cols)
        diag["fold_models"] = []
        emb = base
        residual_of_pool = residual_of_test = None
    else:
        offset = method == "xfit_offset_pred"
        so = None
        if offset:
            struct_cols_ = [c for c in struct_cols if c not in text_cols]
            d = te.prepare_e5_data(x_t, y_fit, text_cols, is_cls, cfg.n_bins)  # TAR's split -> its strata
            strata = pd.concat([d["y_tr"], d["y_va"]]).loc[pool].astype(str).to_numpy()
            so = fit_structured_offset(x_fit.loc[pool], y_fit.loc[pool], struct_cols_, [c for c in cat_cols if c in struct_cols_],
                                       is_cls, d["classes"] if is_cls else [], strata=strata) if struct_cols_ else None
        folds = xfit_folds(pool, y_fit, is_cls)
        pieces, test_like = [], {k: [] for k in extra_rows}
        diag["fold_models"] = []
        for k, held in enumerate(folds):
            train_rows = pool.difference(pd.Index(held), sort=False)
            sub_extra = {"heldout": (x_fit.loc[held], y_fit.loc[held]), **extra}
            if offset and so is not None:
                res = run_variant("offset", x_fit.loc[train_rows], y_fit.loc[train_rows], text_cols, so.feature_cols,
                                  so.cat_cols, is_cls, sub_extra, cfg, device, cache_root, dataset, fold, structured=so)
                mids = None if is_cls else _bin_midpoints(res.head["classes"], np.asarray(res.head["bin_edges"]))
            else:  # plain tar_repro on the fold's rows (also offset with no structured columns = tar_repro, spec 7.6)
                sx, sextra = _tar_inputs(x_fit.loc[train_rows], text_cols, sub_extra)
                res = te.run_tar(sx, y_fit.loc[train_rows], text_cols, is_cls, sextra, cfg, device, cache_root, dataset,
                                 fold, variant="tar_repro")
                mids = None if is_cls else _bin_midpoints(res.head["classes"], _y_edges(y_fit.loc[train_rows], cfg.n_bins))
            pieces.append((np.asarray(held), head_features(res, np.asarray(held), is_cls, classes, mids)))
            for name, rows in extra_rows.items():
                test_like[name].append(head_features(res, rows, is_cls, classes, mids))
            diag["fold_models"].append({"fold": k, "n_train_rows": int(len(train_rows)), "n_heldout": int(len(held)),
                                        "cache_key": res.cache_key, "cache_hit": res.cache_hit,
                                        "best_epoch": res.diagnostics.get("train", {}).get("best_epoch"),
                                        "train_rows_hash": te._rows_hash(pd.Index(train_rows)),
                                        "heldout_rows_hash": te._rows_hash(pd.Index(held))})
        for name, rows in extra_rows.items():  # mean over the K fold models
            pieces.append((rows, {c: np.mean([f[c] for f in test_like[name]], axis=0) for c in text_cols}))
        feats = _features_frame(pieces, text_cols)
        emb = frozen_embeddings(x_t, extra_t, text_cols, cfg, device) if method == "frozen_xfit_pred" else base
        diag["folds_cover_pool"] = bool(np.array_equal(np.sort(np.concatenate(folds)), np.sort(np.asarray(pool))))
        residual_of_pool = residual_of_test = None
        if offset and so is not None and not is_cls:
            residual_of_pool = so.oof_for(pool)
            residual_of_test = so.predict_full(extra["test"][0]) if "test" in extra else None

    # Quality of the prediction features: training rows (OOF for xfit, in-sample for insample) vs test.
    q = {"train_rows": _quality(feats, y_fit, np.asarray(pool), is_cls, classes, residual_of_pool)}
    if "test" in extra:
        q["test"] = _quality(feats, extra["test"][1], np.asarray(extra["test"][0].index), is_cls, classes,
                             residual_of_test)
    if "learner_val" in extra:
        q["learner_val"] = _quality(feats, extra["learner_val"][1], np.asarray(extra["learner_val"][0].index),
                                    is_cls, classes, None if residual_of_test is None else
                                    so.predict_full(extra["learner_val"][0]))
    q["gap_train_minus_test"] = (q["train_rows"] - q["test"]) if q.get("train_rows") is not None and q.get("test") is not None else None
    diag["pred_feature_quality"] = q
    diag["n_pred_features_per_col"] = {c: int(feats[c].shape[1]) for c in text_cols}
    print(f"  [STACK:{method}] pred-feature quality: " +
          ", ".join(f"{k}={v:.4f}" for k, v in q.items() if v is not None), flush=True)
    return StackResult(method=method, embedding_result=emb, base_result=base, features=feats, diagnostics=diag)


# ---------------------------------------------------------------------------------------- P5 experts

def run_experts(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], is_cls: bool,
                extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig, device: torch.device,
                cache_root: Optional[str], dataset: str, fold: int) -> StackResult:
    """P5: base embeddings (tar_repro's full-pool fine-tune, or frozen E5) + cross-fitted expert predictions.
    One tar_repro fine-tune per (dataset, fold), cached and shared by every tar_* method and learner; expert features
    cached per (dataset, fold, expert, mode) and shared too."""
    import time
    from multabench.ours.experts import ALL_COLS, ExpertConfig, cached_expert_features, compute_expert_features
    base_kind, experts, mode = EXPERT_METHODS[method]
    text_cols = sorted(text_cols)
    extra = {k: v for k, v in extra.items() if v is not None}
    x_t, extra_t = _tar_inputs(x_fit, text_cols, extra)
    t0 = time.time()
    if base_kind == "tar":
        base = te.run_tar(x_t, y_fit, text_cols, is_cls, extra_t, cfg, device, cache_root, dataset, fold,
                          variant="tar_repro")
    else:  # frozen_experts: the tar_repro result only for JSON diagnostics, if it is already cached
        base = cached_tar_repro(x_t, extra_t, cfg, device, cache_root, dataset, fold)
    t_base = time.time() - t0
    frozen = None

    def get_frozen():
        nonlocal frozen
        if frozen is None:
            frozen = frozen_embeddings(x_t, extra_t, text_cols, cfg, device)
        return frozen

    pool_rows = np.asarray(x_fit.index)
    if is_cls:
        classes = sorted(y_fit.astype(str).unique())
        pos = {c: i for i, c in enumerate(classes)}
        y = y_fit.astype(str).map(pos).to_numpy()
    else:
        classes, y = [], pd.to_numeric(y_fit).to_numpy(dtype=float)
    ecfg = ExpertConfig()
    texts = {c: te.encoding_texts(x_t, c) for c in text_cols}
    extra_texts = {k: {c: te.encoding_texts(x, c) for c in text_cols} for k, (x, _) in extra_t.items()}
    rows_key = te._rows_hash(x_fit.index, *[x.index for x, _ in extra.values()])
    texts_key = hashlib.sha1("\x00".join(t for c in text_cols for t in texts[c]).encode()).hexdigest()[:10]
    t1 = time.time()
    feats_by_split, hits = {}, {}
    for kind in experts:
        key = f"{dataset}/fold{fold}/{kind}_{mode}_{ecfg.key()}_{rows_key}_{texts_key}"

        def compute(kind=kind):
            embs = extra_embs = None
            if kind == "nbr":
                fr = get_frozen()
                embs = {c: fr.row_embeddings(c, pool_rows) for c in text_cols}
                extra_embs = {k: {c: fr.row_embeddings(c, np.asarray(x.index)) for c in text_cols}
                              for k, (x, _) in extra_t.items()}
            return compute_expert_features(texts, embs, y, is_cls, len(classes), extra_texts, extra_embs, (kind,),
                                           mode, ecfg)
        out, hits[kind] = cached_expert_features(cache_root, key, compute)
        for split, d in out.items():
            feats_by_split.setdefault(split, {}).update(d)
    t_experts = time.time() - t1

    # Per text column: its own experts; the "__all__" experts go with the first text column.
    owner = {c: c for c in text_cols} | {ALL_COLS: text_cols[0]}
    split_rows = {"pool": pool_rows, **{k: np.asarray(x.index) for k, (x, _) in extra.items()}}
    features = {}
    for col in text_cols:
        names = sorted(f for f in feats_by_split["pool"] if owner[f.split(":", 1)[1]] == col)
        if not names:
            features[col] = pd.DataFrame(index=np.concatenate(list(split_rows.values())))
            continue
        blocks = []
        for split, rows in split_rows.items():
            blocks.append(pd.DataFrame(np.hstack([feats_by_split[split][f] for f in names]), index=rows))
        features[col] = pd.concat(blocks).sort_index()
        features[col].columns = [f"{f}#{j}" for f in names for j in range(feats_by_split["pool"][f].shape[1])]
    emb = base if base_kind == "tar" else get_frozen()

    def q(split, rows, yy):
        if not len(rows):
            return None
        vals = {}
        for f in feats_by_split["pool"]:
            a = feats_by_split[split][f]
            if is_cls:
                probs = np.hstack([1 - a.sum(axis=1, keepdims=True), a]) if a.shape[1] == len(classes) - 1 else a
                vals[f] = _metric(yy.astype(str).to_numpy(), probs, True, classes)
            else:
                vals[f] = _metric(pd.to_numeric(yy).to_numpy(dtype=float), a[:, 0], False, [])
        return vals
    quality = {"train_rows": q("pool", pool_rows, y_fit)}
    if "test" in extra:
        quality["test"] = q("test", np.asarray(extra["test"][0].index), extra["test"][1])
    diag = {"engine_version": "experts/1", "method": method, "base": base_kind, "experts": list(experts),
            "mode": mode, "expert_config": {**ecfg.__dict__}, "expert_cache_hits": hits,
            "pred_feature_quality": {"per_expert": quality,
                                     "gap_train_minus_test": {f: (quality["train_rows"][f] - quality["test"][f])
                                                              if quality.get("test") and quality["train_rows"][f] is not None
                                                              and quality["test"][f] is not None else None
                                                              for f in quality["train_rows"]}},
            "n_pred_features_per_col": {c: int(features[c].shape[1]) for c in text_cols},
            "seconds": {"base": round(t_base, 1), "experts": round(t_experts, 1)}}
    print(f"  [STACK:{method}] experts {list(experts)} ({mode}); cache hits {hits}; "
          f"base {t_base:.0f}s, experts {t_experts:.0f}s", flush=True)
    return StackResult(method=method, embedding_result=emb, base_result=base if base_kind == "tar" else base,
                       features=features, diagnostics=diag)

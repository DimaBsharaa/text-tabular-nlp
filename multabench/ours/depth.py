"""P11 (PROGRESS.md): Depth probes — target-aware read-outs of EVERY layer of the frozen encoder. Engine depth/1.

Motivation. Joint TAR adapts the top 3 of e5-small-v2's 12 layers with LoRA and then reads only the final layer.
Our cte_probe also reads only the final layer (a cross-fitted Ridge on the L2-normalised mean pool). Probing work on
transformers finds that task information often peaks in intermediate layers, and that the final layer of a
contrastively trained sentence encoder is shaped for retrieval similarity, not for an arbitrary target. Depth probes ask
whether the target signal that fine-tuning would surface is already *linearly available somewhere in the frozen
network*, and read it out without any gradient training:

  for every layer l = 1..L of the frozen encoder and every text column:
      h_l = masked mean pool of layer l's token states ("passage: col: val", the same string Joint Frozen encodes)
            (+ masked max pool with pooling="meanmax")
      standardised per dimension on the learner's training rows (label-free), / sqrt(width), so every column block has
      the scale of an L2-normalised embedding
  depth expert l : a cross-fitted Ridge on [h_l of all text columns] — cte_probe's machinery (p7.xfit_linear):
                   5-fold out-of-fold predictions for training rows, alpha from (1, 3, 10, 30, 100, 300) by the
                   training-row OOF metric, refit on all training rows for learner-val / test rows. Multiclass uses an
                   exact eigen-form solver for all one-hot targets at once (xfit_linear_eig; same folds / rule).
  aggregation    : "stack" — every layer's prediction is a learner feature (regression / binary: L columns);
                   "sl"    — one column: a stacked regression over the L layer predictions (Breiman 1996: non-negative
                             least squares with intercept, target = the experts' own target), itself cross-fitted
                             (5 folds, fold seed + 1) on the training rows' OOF layer predictions; val/test rows get
                             the NNLS fit on all training rows applied to their (full-fit) layer predictions;
                   "top3"  — the mean of the 3 layers with the best training-row OOF metric (ties -> deeper layer).
                   Multiclass (more than one output column per layer) always uses "top3".

Everything label-aware uses the learner's training rows only. The method is cte_probe (p7, unchanged, same caches)
plus the depth columns; `tests/test_depth.py` checks that dropping the depth columns gives cte_probe's features.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import os
import time
from os.path import exists, join
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.ours import cte
from multabench.ours import tar_engine as te
from multabench.ours.experts import ALL_COLS, cached_expert_features

DEPTH_VERSION = "depth/1"


@dataclasses.dataclass(frozen=True)
class DepthSpec:
    pooling: str = "mean"        # "mean" | "meanmax"
    aggregate: str = "stack"     # "stack" | "sl" | "top3" (multiclass -> top3)
    layers: Optional[tuple] = None  # None = all layers 1..L (hidden_states[1:]); 0 would be the embedding layer
    top_k: int = 3
    alphas: tuple = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)  # wider than cte's (0.3..10): standardised layer states

    def cte_config(self):
        return dataclasses.replace(cte.CTE_CONFIG, ridge_alphas=self.alphas)


DEPTH_METHODS = {
    "cte_depth": DepthSpec(),
    "cte_depth_mm": DepthSpec(pooling="meanmax"),
    "cte_depth_top3": DepthSpec(aggregate="top3"),
    "cte_depth_sl": DepthSpec(aggregate="sl"),
}


def _eig_ridge(X: np.ndarray, Y: np.ndarray):
    """Centred ridge in eigen form: one Gram + eigendecomposition per fit, then any alpha is cheap. Exactly sklearn's
    Ridge(fit_intercept=True) solution (up to float error), for all target columns at once."""
    X = np.asarray(X, dtype=np.float64)
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc = X - mx
    s, V = np.linalg.eigh(Xc.T @ Xc)
    return mx, my, V, np.clip(s, 0.0, None), V.T @ (Xc.T @ (Y - my))


def _eig_coef(f, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my, V, s, VtXtY = f
    W = V @ (VtXtY / (s + alpha)[:, None])
    return W, my - mx @ W


def xfit_linear_eig(X, X_extra: Dict[str, np.ndarray], y: np.ndarray, is_cls: bool, n_classes: int, c):
    """p7.xfit_linear (same folds, targets, alpha rule, output columns) with a direct multi-target solver: used for
    multiclass, where sparse_cg solves every one-hot column separately (Spotify: 114 targets x 432 fits per job).
    Also returns the all-training-rows (W, b) of the chosen alpha (packaging)."""
    from multabench.ours.experts import expert_folds
    folds = expert_folds(len(y), y, is_cls, c.expert_config())
    top = cte.top_classes(y, n_classes, c) if is_cls else None
    targets = lambda idx: cte._targets(y[idx], is_cls, n_classes, c)
    oofs = {a: None for a in c.ridge_alphas}
    for tr, te_ in folds:
        f = _eig_ridge(X[tr], targets(tr))
        for a in c.ridge_alphas:
            W, b = _eig_coef(f, a)
            p = np.asarray(X[te_], dtype=np.float64) @ W + b
            oofs[a] = np.zeros((len(y), p.shape[1])) if oofs[a] is None else oofs[a]
            oofs[a][te_] = p
    metrics = [cte.oof_metric(oofs[a], y, is_cls, n_classes) for a in c.ridge_alphas]
    alpha = c.ridge_alphas[int(np.argmax(metrics))]
    W, b = _eig_coef(_eig_ridge(X, targets(np.arange(len(y)))), alpha)
    feat = lambda P: cte.class_features(P, is_cls, n_classes, top)
    ext = {k: feat(np.asarray(v, dtype=np.float64) @ W + b) for k, v in X_extra.items()}
    return feat(oofs[alpha]), ext, {"alpha": alpha, "oof_metric": metrics}, (W, b)


def nnls_fit(P: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
    """Non-negative least squares with a free intercept (Breiman's stacked regression): weights w >= 0, b."""
    from scipy.optimize import nnls
    pm, tm = P.mean(axis=0), float(t.mean())
    w, _ = nnls(P - pm, t - tm)
    return w, tm - float(pm @ w)


def stacked_layers(P_pool: np.ndarray, P_ext: Dict[str, np.ndarray], targets, y: np.ndarray, is_cls: bool,
                   c) -> tuple[np.ndarray, Dict[str, np.ndarray], tuple[np.ndarray, float]]:
    """Cross-fitted NNLS combination of the per-layer predictions (columns of P_pool; one output per layer). Pool rows:
    out-of-fold over folds with seed c.seed + 1; extras: the fit on all pool rows. targets(idx) -> the expert target of
    those rows, built from those rows only (rank-Gaussian fit on them, or 1{class 1}), as the experts do. Returns
    (oof (n, 1), {split: (m, 1)}, (full-fit weights, intercept))."""
    from multabench.ours.experts import ExpertConfig, expert_folds
    n = len(P_pool)
    folds = expert_folds(n, y, is_cls, ExpertConfig(n_folds=c.n_folds, seed=c.seed + 1))
    oof = np.zeros(n)
    for tr, te_ in folds:
        w, b = nnls_fit(P_pool[tr], targets(tr))
        oof[te_] = P_pool[te_] @ w + b
    w, b = nnls_fit(P_pool, targets(np.arange(n)))
    return oof[:, None], {k: (v @ w + b)[:, None] for k, v in P_ext.items()}, (w, b)


# ---------------------------------------------------------------------------------------- layer states (label-free)

def _pool(h: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
    if pooling == "mean":
        return mean
    mx = h.masked_fill(m == 0, torch.finfo(h.dtype).min).max(dim=1).values
    return torch.cat([mean, mx], dim=-1)


@torch.no_grad()
def encode_layer_states(strings: list[str], model_name: str, max_length: int, pooling: str, device: torch.device,
                        batch_size: int = 64) -> np.ndarray:
    """(n, L, width) masked pools of hidden_states[1..L] of the frozen encoder, one row per string. Plain eval-mode
    forward pass of the vanilla checkpoint (= Joint Frozen's encoder); fp16 autocast on CUDA, fp32 on CPU."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa").to(device).eval()
    store = te.TokenStore(strings, tok, max_length)
    n_layers = model.config.num_hidden_layers
    width = model.config.hidden_size * (2 if pooling == "meanmax" else 1)
    out = np.zeros((len(strings), n_layers, width), dtype=np.float32)
    order = np.argsort(store.lengths, kind="stable")  # length-sorted batches: less padding, same results
    use_fp16 = device.type == "cuda"
    for i in range(0, len(order), batch_size):
        rows = order[i:i + batch_size]
        ids, mask = store.batch(rows, device)
        with te._autocast(device, use_fp16):
            hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        for li in range(n_layers):
            out[rows, li] = _pool(hs[li + 1].float(), mask, pooling).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def layer_states_cached(x_frames: list[DataFrame], text_cols: list[str], cfg: te.TarConfig, device: torch.device,
                        cache_root: Optional[str], dataset: str, pooling: str) -> Dict[str, tuple[np.ndarray, dict]]:
    """{col: (states (n_unique, L, width) float32, {row index: unique id})} for every row of x_frames, from a
    per-dataset cache of unique cell texts (target-free: shared by folds, learners and variants), like
    cte.frozen_embeddings_cached. Stored as float16."""
    out = {}
    for col in sorted(text_cols):
        rows = np.concatenate([np.asarray(f.index) for f in x_frames])
        raw = list(itertools.chain.from_iterable(te.encoding_texts(f, col) for f in x_frames))
        uniq, inverse = np.unique(np.array(raw, dtype=object), return_inverse=True)
        uniq = uniq.tolist()
        hashes = te._text_hash(uniq)
        path = join(cache_root, "depth_states", dataset, cfg.model_name.replace("/", "__"), f"len{cfg.max_length}", pooling,
                    hashlib.sha1(col.encode()).hexdigest()[:12] + ".npz") if cache_root else None
        with cte._store_lock(path):
            store_h, store_s = np.zeros(0, dtype=np.uint64), None
            if path and exists(path):
                z = np.load(path)
                store_h, store_s = z["hash"], z["states"]
            pos = {int(h): i for i, h in enumerate(store_h)}
            missing = [i for i, h in enumerate(hashes) if int(h) not in pos]
            if missing:
                from multabench.e5.constants import format_e5_passage
                new = encode_layer_states([format_e5_passage(col, uniq[i]) for i in missing], cfg.model_name,
                                          cfg.max_length, pooling, device)
                store_h = np.concatenate([store_h, hashes[missing]])
                store_s = new.astype(np.float16) if store_s is None else np.concatenate([store_s, new.astype(np.float16)])
                pos = {int(h): i for i, h in enumerate(store_h)}
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp = f"{path}.{os.getpid()}.tmp.npz"
                    np.savez(tmp, hash=store_h, states=store_s)
                    os.replace(tmp, path)
        states = store_s[[pos[int(h)] for h in hashes]].astype(np.float32)
        out[col] = (states, dict(zip(rows.tolist(), inverse.tolist())))
    return out


def layer_matrix(states: Dict[str, tuple[np.ndarray, dict]], text_cols: list[str], rows: np.ndarray, layer: int,
                 stats: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """[h_layer of every text column] for `rows`, standardised per dimension with statistics of the training rows
    (computed when stats is None, i.e. for the pool) and scaled by 1/sqrt(width) per column block."""
    blocks, new_stats = [], {}
    for col in sorted(text_cols):
        s, uid = states[col]
        a = s[[uid[int(r)] for r in rows], layer]
        mu, sd = (a.mean(axis=0), a.std(axis=0) + 1e-6) if stats is None else stats[col]
        new_stats[col] = (mu, sd)
        blocks.append((a - mu) / sd / np.sqrt(a.shape[1]))
    return np.hstack(blocks), new_stats


# ---------------------------------------------------------------------------------------- main entry

def run_depth(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], struct_cols: list[str],
              cat_cols: list[str], is_cls: bool, extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig,
              device: torch.device, cache_root: Optional[str], dataset: str, fold: int):
    from multabench.ours import p7
    from multabench.ours.stacking import StackResult
    spec = DEPTH_METHODS[method]
    text_cols = sorted(text_cols)
    extra = {k: v for k, v in extra.items() if v is not None}
    base = p7.run_p7("cte_probe", x_fit, y_fit, text_cols, struct_cols, cat_cols, is_cls, extra, cfg, device,
                     cache_root, dataset, fold)
    splits = {"pool": x_fit, **{k: x for k, (x, _) in extra.items()}}
    other = [k for k in splits if k != "pool"]
    rows = {k: np.asarray(x.index) for k, x in splits.items()}
    t0 = time.time()
    states = layer_states_cached([x[text_cols] for x in splits.values()], text_cols, cfg, device, cache_root, dataset,
                                 spec.pooling)
    t_states = time.time() - t0
    if is_cls:
        classes = sorted(y_fit.astype(str).unique())
        y = y_fit.astype(str).map({k: i for i, k in enumerate(classes)}).to_numpy()
    else:
        classes, y = [], pd.to_numeric(y_fit).to_numpy(dtype=float)
    n_classes = len(classes) if is_cls else 1
    n_layers = next(iter(states.values()))[0].shape[1]
    layers = list(spec.layers) if spec.layers is not None else list(range(1, n_layers + 1))
    t1 = time.time()
    per_layer, hits = {}, 0  # layer -> (oof features, {split: features}, best OOF metric, alpha)
    c = spec.cte_config()
    rows_key = te._rows_hash(x_fit.index, *[x.index for x, _ in extra.values()])
    texts_key = hashlib.sha1("\x00".join(t for col in text_cols for t in te.encoding_texts(x_fit, col)).encode()
                             ).hexdigest()[:10]
    y_key = cte.content_key(y, {k: {col: te.encoding_texts(x, col) for col in text_cols} for k, (x, _) in extra.items()},
                            text_cols)  # labels + prediction-row texts
    solver = "eig" if is_cls and n_classes > 2 else "sparse_cg"  # multiclass: one direct solve for all one-hot targets
    probe_key = hashlib.sha1(repr((DEPTH_VERSION, spec.pooling, spec.alphas, c.key(), solver)).encode()).hexdigest()[:10]
    for layer in layers:
        def compute(layer=layer):
            Xp, st = layer_matrix(states, text_cols, rows["pool"], layer - 1)
            Xe = {k: layer_matrix(states, text_cols, rows[k], layer - 1, st)[0] for k in other}
            if solver == "eig":
                oof, ext, meta, _ = xfit_linear_eig(Xp, Xe, y, is_cls, n_classes, c)
            else:
                oof, ext, meta = p7.xfit_linear(Xp, Xe, y, is_cls, n_classes, c)
            return {"pool": {"f": oof}, **{k: {"f": v} for k, v in ext.items()},
                    "meta": {"alpha": np.array([meta["alpha"]]), "oof_metric": np.array(meta["oof_metric"])}}
        key = f"{dataset}/fold{fold}/depth_L{layer:02d}_{probe_key}_{rows_key}_{texts_key}_{y_key}_{cte.encoder_key(cfg)}"
        out, hit = cached_expert_features(cache_root, key, compute)
        hits += int(hit)
        per_layer[layer] = (out["pool"]["f"], {k: out[k]["f"] for k in other}, float(out["meta"]["oof_metric"].max()),
                            float(out["meta"]["alpha"][0]))
    t_probe = time.time() - t1
    k_cols = next(iter(per_layer.values()))[0].shape[1]
    aggregate = spec.aggregate if k_cols == 1 else "top3"  # multiclass: bounded column count
    ranked = sorted(layers, key=lambda l: (per_layer[l][2], l), reverse=True)  # best OOF metric first; ties: deeper
    new, sl_weights = {}, None
    if aggregate == "stack":
        for layer in layers:
            oof, ext, _, _ = per_layer[layer]
            new[f"depth{layer:02d}:{ALL_COLS}"] = {"pool": oof, **ext}
    elif aggregate == "sl":
        targets = lambda idx: cte._targets(y[idx], is_cls, n_classes, c)[:, -1]  # rank-Gaussian y / 1{class 1}
        oof, ext, sl_weights = stacked_layers(np.hstack([per_layer[l][0] for l in layers]),
                                              {k: np.hstack([per_layer[l][1][k] for l in layers]) for k in other},
                                              targets, y, is_cls, c)
        new[f"depthsl:{ALL_COLS}"] = {"pool": oof, **ext}
    else:
        top = ranked[:spec.top_k]
        new[f"depthtop{spec.top_k}:{ALL_COLS}"] = {k: np.mean([per_layer[l][0] if k == "pool" else per_layer[l][1][k]
                                                               for l in top], axis=0) for k in splits}
    features = {c: base.features[c].copy() for c in text_cols}
    owner = text_cols[0]
    for name, arrs in new.items():
        block = pd.concat([DataFrame(arrs[k], index=rows[k]) for k in splits]).sort_index()
        block.columns = [f"{name}#{j}" for j in range(block.shape[1])]
        features[owner] = pd.concat([features[owner], block.loc[features[owner].index]], axis=1)
    diag = {**base.diagnostics, "engine_version": DEPTH_VERSION, "method": method, "spec": dataclasses.asdict(spec),
            "depth": {"n_layers": n_layers, "layers": layers, "aggregate": aggregate, "solver": solver,
                      "layer_oof_metric": {int(l): round(per_layer[l][2], 6) for l in layers},
                      "layer_alpha": {int(l): per_layer[l][3] for l in layers},
                      "ranked_layers": [int(l) for l in ranked],
                      "probe_oof_metric_final_layer_cte_probe": base.diagnostics.get("probe", {}).get("oof_metric"),
                      "probe_cache_hits": hits,
                      "sl_weights": None if sl_weights is None else [float(w) for w in sl_weights[0]],
                      "sl_intercept": None if sl_weights is None else float(sl_weights[1]),
                      "seconds": {"states": round(t_states, 1), "probes": round(t_probe, 1)}}}
    print(f"  [STACK:{method}] depth probes on {len(layers)} layers ({spec.pooling}, {aggregate}); best layers "
          f"{ranked[:3]}; states {t_states:.0f}s, probes {t_probe:.0f}s", flush=True)
    return StackResult(method=method, embedding_result=base.embedding_result, base_result=None, features=features,
                       diagnostics=diag)

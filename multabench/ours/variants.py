"""TAR variants (docs/method_spec.md). tar_repro itself lives in tar_engine.run_tar and is never changed here.

Every variant reuses TAR's split, texts, encoder, LoRA, head, optimizer, schedule, early stopping and PCA, and changes
only what its spec section says: `offset`, `prefix`, `prefix_offset`, `context`, `context_offset`.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from os.path import exists, join
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from multabench.baselines.training.metrics import calculate_metric
from multabench.e5.constants import format_e5_passage
from multabench.preprocessing.discretize import discretize_numerical
from multabench.ours import tar_engine as te
from multabench.ours.archive.structured_offset import StructuredOffset, fit_structured_offset

VARIANT_ENGINE_VERSION = "variants/1"
OFFSET_METHODS = {"offset", "prefix_offset", "context_offset"}
PREFIX_METHODS = {"prefix", "prefix_offset"}
CONTEXT_METHODS = {"context", "context_offset"}
METHODS = OFFSET_METHODS | PREFIX_METHODS | CONTEXT_METHODS
TOP_K = 8


STACK_METHOD_NAMES = {"insample_pred", "xfit_pred", "frozen_xfit_pred", "xfit_offset_pred"}  # multabench/ours/stacking.py
EXPERT_METHOD_NAMES = {"tar_experts", "frozen_experts", "tar_experts_lex", "tar_experts_nbr", "tar_experts_lex_insample"}


def engine_version(method: str) -> str:
    if method == "tar_repro":
        return te.ENGINE_VERSION
    if method in EXPERT_METHOD_NAMES:
        return "experts/1"
    if method in {"cte", "cte_lex", "cte_nbr", "cte_num", "cte_lex_insample"}:  # multabench/ours/cte.py
        return "cte/3"  # P5 v2 amendments 1+2 (cte/1 = original, cte/2 = amendment 1 only)
    from multabench.ours.depth import DEPTH_METHODS, DEPTH_VERSION
    if method in DEPTH_METHODS:  # multabench/ours/depth.py (P11)
        return DEPTH_VERSION
    from multabench.ours.archive.p9 import P9_METHODS, P9_VERSION
    if method in P9_METHODS:  # multabench/ours/p9.py
        return P9_VERSION
    from multabench.ours.p7 import P7_METHODS, P7_VERSION
    if method in P7_METHODS:  # multabench/ours/p7.py
        return P7_VERSION
    if method in {"catboost_native_text", "cte_plus_native", "cte_nb", "cte_residual"}:  # multabench/ours/p6.py
        from multabench.ours.archive.p6 import P6_ENGINE
        return P6_ENGINE[method]  # p6/3 for the native-text methods (amendment 2), p6/2 for cte_nb / cte_residual
    return "stacking/1" if method in STACK_METHOD_NAMES else VARIANT_ENGINE_VERSION


def _strings_hash(strings: list[str]) -> str:
    h = hashlib.sha1()
    for s in strings:
        h.update(s.encode())
        h.update(b"\x00")
    return h.hexdigest()[:10]


def _metric(y_true: np.ndarray, pred: np.ndarray, is_cls: bool, classes: list[str]) -> Optional[float]:
    """Upstream metric (AUROC / macro-OvR AUROC / R²). Classification rows with classes unknown to TAR are dropped."""
    if is_cls:
        y = np.asarray(y_true).astype(str)
        known = np.isin(y, classes)
        y_int = LabelEncoder().fit(classes).transform(y[known])
        if len(np.unique(y_int)) < 2:
            return None
        return float(calculate_metric(y_true=y_int, y_pred=pred[known], d_output=len(classes)).score)
    return float(calculate_metric(y_true=pd.to_numeric(pd.Series(y_true)).to_numpy(), y_pred=pred, d_output=1).score)


def _residual_bins(r_tr: Series, r_va: Series, n_bins: int) -> tuple[Series, Series]:
    """Spec 7.5: upstream discretize_numerical on the e5_train residuals; the same edges (outer ones open) for e5_val."""
    lab_tr = discretize_numerical(r_tr, n_bins=n_bins)
    _, edges = pd.qcut(pd.to_numeric(r_tr), q=n_bins, labels=False, duplicates="drop", retbins=True)
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    actual_n = len(edges) - 1
    step = 100 / actual_n
    label_map = {i: f"Quantile {round(i * step)}-{round((i + 1) * step)}%" for i in range(actual_n)}
    codes = pd.cut(pd.to_numeric(r_va), bins=edges, labels=False, include_lowest=True)
    return lab_tr, pd.Series(codes, index=r_va.index).map(label_map)


def run_variant(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], struct_cols: list[str],
                cat_cols: list[str], is_cls: bool, extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig,
                device: torch.device, cache_root: Optional[str], dataset: str, fold: int,
                structured: Optional[StructuredOffset] = None) -> te.TarResult:
    """Like tar_engine.run_tar, for a variant. x_fit / extras contain text AND structured columns (preprocessed).
    structured (P3 cross-fitting only): a pool-level OOF structured model whose pool covers x_fit; used instead of
    refitting one on x_fit."""
    assert method in METHODS, f"unknown variant {method!r} (implemented: {sorted(METHODS)})"
    text_cols = sorted(text_cols)
    struct_cols = [c for c in struct_cols if c not in text_cols]
    extra = {k: v for k, v in extra.items() if v is not None}
    uses_offset, uses_prefix, uses_context = method in OFFSET_METHODS, method in PREFIX_METHODS, method in CONTEXT_METHODS

    if not struct_cols:
        # Spec 7.6: no structured columns -> exactly the tar_repro computation (same cache entry).
        res = te.run_tar(x_fit[text_cols], y_fit, text_cols, is_cls, {k: (x[text_cols], y) for k, (x, y) in extra.items()},
                         cfg, device, cache_root, dataset, fold, variant="tar_repro")
        # In-memory relabel only (the cached tar_repro entry is untouched): the result belongs to this method.
        res.diagnostics = {**res.diagnostics, "offset_delegated": True}
        res.config = {**res.config, "engine_version": VARIANT_ENGINE_VERSION, "method": method,
                      "delegated_to": "tar_repro"}
        return res

    d = te.prepare_e5_data(x_fit[text_cols], y_fit, text_cols, is_cls, cfg.n_bins)  # TAR's split, texts, labels
    x_tr, x_va = d["x_tr"], d["x_va"]
    frames = [x_fit] + [x for x, _ in extra.values()]
    rows_by_col, raw_by_col = {}, {}
    for col in text_cols:
        rows_by_col[col] = np.concatenate([np.asarray(f.index) for f in frames])
        raw_by_col[col] = list(itertools.chain.from_iterable(te.encoding_texts(f, col) for f in frames))
    enc_strings = {col: [format_e5_passage(col, t) for t in raw_by_col[col]] for col in text_cols}
    # Same strings in the same order as run_tar -> the same lower-layer cache entry as tar_repro.
    all_strings = list(dict.fromkeys(itertools.chain(d["train_texts"], d["val_texts"], *enc_strings.values())))

    struct_cfg = {"struct_cols": struct_cols, "cat_cols": [c for c in cat_cols if c in struct_cols]}
    rows_hash = te._rows_hash(x_fit.index, *[x.index for x, _ in extra.values()])
    key = (f"{dataset}/fold{fold}/{method}/seed{cfg.seed}/{te.config_hash(cfg, method, device)}_{rows_hash}_"
           f"{_strings_hash(all_strings)}_{_strings_hash([json.dumps(struct_cfg, sort_keys=True)])}"
           + ("" if structured is None else f"_pool{_strings_hash([str(len(structured.pool_index))])}"
              f"{te._rows_hash(pd.Index(structured.pool_index))}"))
    path = join(cache_root, key) if cache_root else None
    if path and exists(join(path, "meta.json")):
        print(f"  [TAR] cache hit: {key}", flush=True)
        return te._load(path)

    t0 = time.time()
    pool = pd.Index(np.concatenate([np.asarray(x_tr.index), np.asarray(x_va.index)]))
    x_pool = x_fit.loc[pool]
    y_pool = y_fit.loc[pool]
    strata = pd.concat([d["y_tr"], d["y_va"]]).loc[pool].astype(str).to_numpy()  # TAR's labels / y-bins
    so = structured if structured is not None else fit_structured_offset(
        x_pool, y_pool, struct_cols, struct_cfg["cat_cols"], is_cls, d["classes"] if is_cls else [], strata=strata)
    t_struct = time.time() - t0
    rep = len(text_cols)

    # ---- target and loss
    train_offsets = val_offsets = None
    classes, train_labels, val_labels = d["classes"], d["train_labels"], d["val_labels"]
    if uses_offset and is_cls:
        train_offsets = np.repeat(so.log_offset(so.oof_for(x_tr.index)), rep, axis=0)
        val_offsets = np.repeat(so.log_offset(so.oof_for(x_va.index)), rep, axis=0)
    elif uses_offset:
        y_num = pd.to_numeric(y_fit)
        r_tr = y_num.loc[x_tr.index] - so.oof_for(x_tr.index)
        r_va = y_num.loc[x_va.index] - so.oof_for(x_va.index)
        lab_tr, lab_va = _residual_bins(r_tr, r_va, cfg.n_bins)
        enc = LabelEncoder().fit(lab_tr.astype(str).to_numpy())
        classes = enc.classes_.tolist()
        train_labels = np.repeat(enc.transform(lab_tr.astype(str).to_numpy()), rep)
        val_labels = np.repeat(enc.transform(lab_va.astype(str).to_numpy()), rep)
        resid_means = r_tr.groupby(lab_tr.astype(str)).mean()
        _, resid_edges = pd.qcut(pd.to_numeric(r_tr), q=cfg.n_bins, labels=False, duplicates="drop", retbins=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if uses_context:  # spec 4: "passage: {context} | {col}: {val}" with capped context (new lower-layer cache)
        from multabench.ours.archive.context_text import ContextTokenStore, Verbalizer
        verb = Verbalizer(so.top_features(TOP_K), struct_cfg["cat_cols"], x_pool)
        part = lambda t: t[len("passage: "):]  # TAR's training cell text without the prefix
        ctx_tr = np.repeat(verb.contexts(x_fit.loc[x_tr.index]), rep).tolist()  # x_tr/x_va hold text cols only
        ctx_va = np.repeat(verb.contexts(x_fit.loc[x_va.index]), rep).tolist()
        parts_tr, parts_va = [part(t) for t in d["train_texts"]], [part(t) for t in d["val_texts"]]
        ctx_frames = [verb.contexts(f) for f in frames]
        ctx_enc = {col: list(itertools.chain.from_iterable(ctx_frames)) for col in text_cols}
        parts_enc = {col: [f"{col}: {t}" for t in raw_by_col[col]] for col in text_cols}  # = format_e5_passage
        tokens = ContextTokenStore(ctx_tr + ctx_va + list(itertools.chain.from_iterable(ctx_enc[c] for c in text_cols)),
                                   parts_tr + parts_va + list(itertools.chain.from_iterable(parts_enc[c] for c in text_cols)),
                                   tokenizer, cfg.max_length)
    else:
        tokens = te.TokenStore(all_strings, tokenizer, cfg.max_length)
    te.set_seed(cfg.seed)
    model = te.build_model(cfg, d_output=len(classes), device=device)
    t1 = time.time()
    feature_info = {"top_features": so.top_features(TOP_K)}

    # ---- inputs: which cells, through which source
    if uses_prefix:
        from multabench.ours.archive.feature_tokens import (FeatureSpec, FeatureTokenizer, PrefixSource, RowFeatures,
                                                    frozen_name_vectors)
        assert cfg.fast, "prefix variants need the fast (cached frozen layers) path"
        cache = te.LowerCache.get_or_build(model, tokens, cfg, device)
        assert cache is not None, "prefix variants need the lower-layer cache (not enough local disk)"
        top = so.top_features(TOP_K)
        specs = [FeatureSpec(c, "cat" if c in struct_cfg["cat_cols"] else "num", x_pool[c]) for c in top]
        all_rows = pd.concat([f[top] for f in frames])
        feats = RowFeatures(specs, all_rows)
        model.feature_tokenizer = FeatureTokenizer(specs, frozen_name_vectors(model, tokenizer, top, cfg, device),
                                                   model.head.in_features).to(device)
        # cell table: train cells, val cells, then one cell per (column, row) for encoding
        tr_rows = np.repeat(feats.positions(x_tr.index), rep)
        va_rows = np.repeat(feats.positions(x_va.index), rep)
        enc_uid = {col: tokens.uids(enc_strings[col]) for col in text_cols}
        enc_pos = {col: feats.positions(rows_by_col[col]) for col in text_cols}
        cell_uid = np.concatenate([tokens.uids(d["train_texts"]), tokens.uids(d["val_texts"])]
                                  + [enc_uid[c] for c in text_cols])
        cell_pos = np.concatenate([tr_rows, va_rows] + [enc_pos[c] for c in text_cols])
        source = PrefixSource(model, cache, cfg, cell_uid, cell_pos, feats)
        n_tr, n_va = len(d["train_texts"]), len(d["val_texts"])
        train_ids, val_ids = np.arange(n_tr), np.arange(n_tr, n_tr + n_va)
        start, enc_ids = n_tr + n_va, {}
        for c in text_cols:
            enc_ids[c] = np.arange(start, start + len(enc_uid[c]))
            start += len(enc_uid[c])
        feature_info["feature_kinds"] = {s.name: s.kind for s in specs}
    elif uses_context:
        source = te.make_source(model, tokens, cfg, device)
        train_ids, val_ids = tokens.uids_for(ctx_tr, parts_tr), tokens.uids_for(ctx_va, parts_va)
        enc_ids = {c: tokens.uids_for(ctx_enc[c], parts_enc[c]) for c in text_cols}
        lens = tokens.lengths[train_ids]
        feature_info.update({"context_example": tokens.strings[train_ids[0]][:300],
                             "mean_train_tokens": float(lens.mean()), "max_train_tokens": int(lens.max())})
    else:
        source = te.make_source(model, tokens, cfg, device)
        train_ids, val_ids = tokens.uids(d["train_texts"]), tokens.uids(d["val_texts"])
    t_lower = time.time() - t1
    print(f"  [TAR:{method}] fine-tuning on {len(x_tr)} rows × {rep} text cols = {len(d['train_texts'])} cells "
          f"(val {len(x_va)} rows), d_output={len(classes)}, struct cols={len(struct_cols)}, "
          f"path={'fast' if source.fast else 'full model'}", flush=True)
    train_log = te.train_tar(cfg, model, source, train_ids, train_labels, val_ids, val_labels, device,
                             train_offsets, val_offsets)
    t_train = time.time() - t1 - t_lower

    # ---- embeddings for every row we will be asked about
    t2 = time.time()
    columns = {}
    for col in text_cols:
        if uses_prefix or uses_context:  # per (row, column): depends on the row's structured features (spec 7.9)
            emb, logits = te.encode(model, source, enc_ids[col], cfg, device)
            columns[col] = te.ColumnOutputs(emb=emb, logits=logits, text_hash=te._text_hash(raw_by_col[col]),
                                            row_uid={int(r): i for i, r in enumerate(rows_by_col[col])})
            continue
        uniq, inverse = np.unique(np.array(raw_by_col[col], dtype=object), return_inverse=True)
        uniq = uniq.tolist()
        emb, logits = te.encode(model, source, tokens.uids([format_e5_passage(col, t) for t in uniq]), cfg, device)
        columns[col] = te.ColumnOutputs(emb=emb, logits=logits, text_hash=te._text_hash(uniq),
                                        row_uid=dict(zip(rows_by_col[col].tolist(), inverse.tolist())))
    t_encode = time.time() - t2

    head = {"classes": classes, "d_output": len(classes)}
    if uses_offset and not is_cls:
        head["bin_means"] = [float(resid_means[c]) for c in classes]  # head predicts the residual
        head["bin_edges"] = [float(e) for e in resid_edges]  # residual quantile edges (e5_train), for midpoints
    elif not is_cls:
        yb = pd.Series(d["y_tr"].astype(str).to_numpy(), index=d["y_tr"].index)
        means = pd.to_numeric(y_fit.loc[d["y_tr"].index]).groupby(yb).mean()
        head["bin_means"] = [float(means[c]) for c in classes]
    splits = {"e5_train": np.asarray(x_tr.index), "e5_val": np.asarray(x_va.index)}
    splits.update({k: np.asarray(x.index) for k, (x, _) in extra.items()})
    config = {**cfg.result_dict(device), "engine_version": VARIANT_ENGINE_VERSION, "method": method,
              "structured": {**struct_cfg, "catboost": {"iterations": 500, "depth": 6, "seed": 42}, "oof_folds": 5},
              **({"prefix": {"top_k": TOP_K, "ple_bins": 16, "min_category_count": 3}} if uses_prefix else {}),
              **({"context": {"top_k": TOP_K, "max_context_tokens": 80, "cat_max_chars": 30}} if uses_context else {})}
    result = te.TarResult(variant=method, cache_key=key, cache_hit=False, config=config, text_cols=text_cols,
                          splits=splits, columns=columns, head=head, diagnostics={})
    if uses_offset:
        diag = _offset_diagnostics(result, so, y_fit, extra, x_tr, x_va, is_cls)
    else:
        y_by_split = {"e5_train": y_fit.loc[x_tr.index], "e5_val": y_fit.loc[x_va.index]}
        y_by_split.update({k: y for k, (_, y) in extra.items()})
        hq = te._head_quality(result, y_by_split, is_cls)
        diag = {"head_quality": hq, "head_quality_scores": {k: v["score"] for k, v in hq.items()},
                "offset_info": {"has_features": so.has_features, "n_struct_cols": len(so.feature_cols),
                                "top_features": so.top_features(8),
                                "importances": {k: round(v, 3) for k, v in so.importances.items()}}}
    result.diagnostics = {"train": train_log, **diag, "feature_info": feature_info,
                          "seconds_struct": round(t_struct, 1), "seconds_lower_cache": round(t_lower, 1),
                          "seconds_train": round(t_train, 1), "seconds_encode": round(t_encode, 1),
                          "seconds_total_tar": round(time.time() - t0, 1),
                          "n_unique_texts": {c: int(len(columns[c].emb)) for c in text_cols}, "device": str(device)}
    dg = result.diagnostics
    parts = [k for k in ("head_quality_scores", "struct_only_quality", "combined_quality") if k in dg]
    print(f"  [TAR:{method}] " + " | ".join(
        f"{k.replace('_quality', '').replace('_scores', '')} " +
        ", ".join(f"{s}={v:.4f}" for s, v in dg[k].items() if v is not None) for k in parts), flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        te._save(result, model, path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _offset_diagnostics(result: te.TarResult, so: StructuredOffset, y_fit: Series, extra: dict, x_tr: DataFrame,
                        x_va: DataFrame, is_cls: bool) -> dict:
    """head_quality (vs the variant's own target), struct_only_quality, combined_quality and lift, per split.
    Pool rows use OOF structured predictions; learner_val / test use the full fit (diagnostics only)."""
    struct_pred = {"e5_train": so.oof_for(x_tr.index), "e5_val": so.oof_for(x_va.index)}
    y_split = {"e5_train": y_fit.loc[x_tr.index], "e5_val": y_fit.loc[x_va.index]}
    for k, (x, y) in extra.items():
        struct_pred[k] = so.predict_full(x)
        y_split[k] = y
    classes = result.head["classes"]
    head_q, struct_q, comb_q, lift = {}, {}, {}, {}
    for split, y in y_split.items():
        rows = np.asarray(y.index)
        sp = struct_pred[split]
        if is_cls:
            # combined = softmax(head logits + log offset) per cell, averaged over the row's text cells
            off = torch.tensor(so.log_offset(sp), dtype=torch.float32)
            comb = []
            for col in result.text_cols:
                c = result.columns[col]
                lg = torch.tensor(c.logits[[c.row_uid[int(r)] for r in rows]])
                comb.append(torch.softmax(lg + off, dim=-1).numpy())
            comb = np.mean(comb, axis=0)
            head_q[split] = _metric(y, result.head_probs(rows), True, classes)
            struct_q[split] = _metric(y, sp, True, classes)
            comb_q[split] = _metric(y, comb, True, classes)
        else:
            resid_pred = result.head_probs(rows) @ np.asarray(result.head["bin_means"])
            y_num = pd.to_numeric(y).to_numpy(dtype=float)
            head_q[split] = _metric(y_num - sp, resid_pred, False, [])
            struct_q[split] = _metric(y_num, sp, False, [])
            comb_q[split] = _metric(y_num, sp + resid_pred, False, [])
        lift[split] = None if comb_q[split] is None or struct_q[split] is None else comb_q[split] - struct_q[split]
    return {"head_quality": {k: {"score": v} for k, v in head_q.items()}, "head_quality_scores": head_q,
            "struct_only_quality": struct_q, "combined_quality": comb_q, "lift": lift,
            "offset_info": {"has_features": so.has_features, "n_struct_cols": len(so.feature_cols),
                            "top_features": so.top_features(8),
                            "importances": {k: round(v, 3) for k, v in so.importances.items()}}}

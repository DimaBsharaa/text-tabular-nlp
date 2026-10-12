"""P5 v2 (PROGRESS.md): Cross-fitted Text Experts (CTE) — no fine-tuning.

Learner features = structured columns + frozen e5-small-v2 PCA-30 per text column (PCA fit on training rows, in the
learner) + per text column and on all text columns concatenated, cross-fitted lexical and neighbour expert
predictions (multabench/ours/experts.py with the CTE config) + target-free numeracy / form features per text column.
Frozen embeddings are target-free and cached once per dataset (per unique cell text).
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from os.path import exists, join
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.ours import tar_engine as te
from multabench.ours.experts import ALL_COLS, ExpertConfig, cached_expert_features, compute_expert_features

CTE_VERSION = "cte/1"
CTE_METHODS = {  # method -> (experts, mode, numeracy)
    "cte": (("lex", "nbr"), "oof", True),
    "cte_lex": (("lex",), "oof", False),
    "cte_nbr": (("nbr",), "oof", False),
    "cte_num": ((), "oof", True),
    "cte_lex_insample": (("lex",), "insample", False),
}
CTE_EXPERT_CONFIG = ExpertConfig(max_word_features=200_000, max_char_features=200_000,
                                 ridge_alphas=(0.1, 1.0, 10.0, 100.0), logreg_Cs=(0.1, 1.0, 10.0),
                                 neighbour_weights="inverse_distance", reg_target="rank_gauss")

# ---------------------------------------------------------------------------------------- numeracy / form features

_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?")
_YEAR = re.compile(r"(?<!\d)(19\d{2}|20[0-2]\d|2030)(?!\d)")
_MONEY = re.compile(r"[$€£¥₹]\s?(\d+(?:[.,]\d+)*)|(\d+(?:[.,]\d+)*)\s?(?:usd|eur|gbp|dollars?|euros?)\b", re.I)
_PCT = re.compile(r"\d\s?%|\bpercent\b", re.I)
_SENT = re.compile(r"[.!?]+(?:\s|$)")
_PUNCT = re.compile(r"[^\w\s]")
NUMERACY_NAMES = ["n_numbers", "first_number", "max_number", "min_number", "n_years", "max_year", "has_currency",
                  "max_money", "n_percent", "n_chars", "n_words", "n_sentences", "mean_word_len", "upper_ratio",
                  "digit_ratio", "punct_ratio", "n_exclaim", "n_question", "unique_word_ratio", "is_empty"]


def _slog1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def _to_float(s: str) -> Optional[float]:
    t = s.replace(",", "") if s.count(",") and s.count(".") <= 1 else s.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def numeracy_features(text: str) -> list[float]:
    """Target-free numeracy and form features of one text cell (NaN where undefined)."""
    from tabstar.preprocessing.feat_types import MISSING_VALUE
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    empty = float(t.strip() == "" or t == MISSING_VALUE or t.lower() == "nan")
    if empty:
        t = ""
    nums = [v for v in (_to_float(m) for m in _NUM.findall(t)) if v is not None]
    years = [int(y) for y in _YEAR.findall(t)]
    money = [v for v in (_to_float(a or b) for a, b in _MONEY.findall(t)) if v is not None]
    words = t.split()
    n_chars = len(t)
    nan = float("nan")
    return [
        float(len(nums)), _slog1p(nums[0]) if nums else nan, _slog1p(max(nums)) if nums else nan,
        _slog1p(min(nums)) if nums else nan, float(len(years)), float(max(years)) if years else nan,
        float(bool(re.search(r"[$€£¥₹]", t))), _slog1p(max(money)) if money else nan, float(len(_PCT.findall(t))),
        float(n_chars), float(len(words)), float(len(_SENT.findall(t))) if t else 0.0,
        float(np.mean([len(w) for w in words])) if words else 0.0,
        sum(c.isupper() for c in t) / n_chars if n_chars else 0.0,
        sum(c.isdigit() for c in t) / n_chars if n_chars else 0.0,
        len(_PUNCT.findall(t)) / n_chars if n_chars else 0.0,
        float(t.count("!")), float(t.count("?")),
        len({w.lower() for w in words}) / len(words) if words else 0.0, empty,
    ]


def numeracy_matrix(texts: list[str]) -> np.ndarray:
    return np.array([numeracy_features(t) for t in texts], dtype=float)


# ---------------------------------------------------------------------------------------- frozen embedding cache

def _h64(texts: list[str]) -> np.ndarray:
    return te._text_hash(texts)


def frozen_embeddings_cached(x_frames: list[DataFrame], text_cols: list[str], cfg: te.TarConfig, device: torch.device,
                             cache_root: Optional[str], dataset: str) -> tuple[te.TarResult, dict]:
    """Frozen e5 embeddings ("passage: col: val", L2-normalised mean pool) for every row of x_frames, from a per-dataset
    cache of unique cell texts (target-free, so shared by all folds / learners / variants); missing texts are encoded
    and added. Numerically the encoder of stacking.frozen_embeddings (= upstream's frozen encoder, tests)."""
    from transformers import AutoTokenizer
    from multabench.e5.constants import format_e5_passage
    import itertools
    tok = None
    model = None
    columns, stats = {}, {"encoded": 0, "cached": 0}
    for col in sorted(text_cols):
        rows = np.concatenate([np.asarray(f.index) for f in x_frames])
        raw = list(itertools.chain.from_iterable(te.encoding_texts(f, col) for f in x_frames))
        uniq, inverse = np.unique(np.array(raw, dtype=object), return_inverse=True)
        uniq = uniq.tolist()
        hashes = _h64(uniq)
        path = join(cache_root, "frozen_emb", dataset, cfg.model_name.replace("/", "__"),
                    hashlib.sha1(col.encode()).hexdigest()[:12] + ".npz") if cache_root else None
        store_h, store_e = np.zeros(0, dtype=np.uint64), np.zeros((0, 384), dtype=np.float32)
        if path and exists(path):
            z = np.load(path)
            store_h, store_e = z["hash"], z["emb"]
        pos = {int(h): i for i, h in enumerate(store_h)}
        missing = [i for i, h in enumerate(hashes) if int(h) not in pos]
        if missing:
            if model is None:
                tok = AutoTokenizer.from_pretrained(cfg.model_name)
                te.set_seed(cfg.seed)
                model = te.build_model(cfg, d_output=2, device=device)  # fresh LoRA (B = 0) = vanilla e5
            strings = [format_e5_passage(col, uniq[i]) for i in missing]
            tokens = te.TokenStore(list(dict.fromkeys(strings)), tok, cfg.max_length)
            emb_u, _ = te.encode(model, te.make_source(model, tokens, cfg, device), tokens.uids(strings), cfg, device)
            store_h = np.concatenate([store_h, hashes[missing]])
            store_e = np.concatenate([store_e, emb_u.astype(np.float32)])
            pos = {int(h): i for i, h in enumerate(store_h)}
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp.npz"
                np.savez(tmp, hash=store_h, emb=store_e)
                os.replace(tmp, path)
        stats["encoded"] += len(missing)
        stats["cached"] += len(uniq) - len(missing)
        emb = store_e[[pos[int(h)] for h in hashes]]
        columns[col] = te.ColumnOutputs(emb=emb, logits=np.zeros((len(uniq), 2), dtype=np.float32), text_hash=hashes,
                                        row_uid=dict(zip(rows.tolist(), inverse.tolist())))
    if model is not None:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res = te.TarResult(variant="frozen", cache_key="", cache_hit=False, config={"model_name": cfg.model_name},
                       text_cols=sorted(text_cols), splits={}, columns=columns, head={"classes": [], "d_output": 0},
                       diagnostics={})
    return res, stats


# ---------------------------------------------------------------------------------------- main entry

def expert_jobs() -> int:
    """Parallel lexical views (OURS_EXPERT_JOBS overrides; default = CPU count). Does not change any result."""
    return max(1, int(os.environ.get("OURS_EXPERT_JOBS", os.cpu_count() or 1)))


def run_cte(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], is_cls: bool,
            extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig, device: torch.device,
            cache_root: Optional[str], dataset: str, fold: int):
    from multabench.ours.stacking import StackResult
    from multabench.ours.variants import _metric
    experts, mode, use_num = CTE_METHODS[method]
    ecfg = CTE_EXPERT_CONFIG
    text_cols = sorted(text_cols)
    extra = {k: v for k, v in extra.items() if v is not None}
    x_t = x_fit[text_cols]
    extra_t = {k: (x[text_cols], y) for k, (x, y) in extra.items()}
    t0 = time.time()
    frozen, emb_stats = frozen_embeddings_cached([x_t] + [x for x, _ in extra_t.values()], text_cols, cfg, device,
                                                 cache_root, dataset)
    t_emb = time.time() - t0

    pool_rows = np.asarray(x_fit.index)
    if is_cls:
        classes = sorted(y_fit.astype(str).unique())
        pos = {c: i for i, c in enumerate(classes)}
        y = y_fit.astype(str).map(pos).to_numpy()
    else:
        classes, y = [], pd.to_numeric(y_fit).to_numpy(dtype=float)
    texts = {c: te.encoding_texts(x_t, c) for c in text_cols}
    extra_texts = {k: {c: te.encoding_texts(x, c) for c in text_cols} for k, (x, _) in extra_t.items()}
    rows_key = te._rows_hash(x_fit.index, *[x.index for x, _ in extra.values()])
    texts_key = hashlib.sha1("\x00".join(t for c in text_cols for t in texts[c]).encode()).hexdigest()[:10]
    t1 = time.time()
    feats_by_split, hits = {"pool": {}, **{k: {} for k in extra}}, {}
    for kind in experts:
        key = f"{dataset}/fold{fold}/cte_{kind}_{mode}_{ecfg.key()}_{rows_key}_{texts_key}"

        def compute(kind=kind):
            embs = extra_embs = None
            if kind == "nbr":
                embs = {c: frozen.row_embeddings(c, pool_rows) for c in text_cols}
                extra_embs = {k: {c: frozen.row_embeddings(c, np.asarray(x.index)) for c in text_cols}
                              for k, (x, _) in extra_t.items()}
            return compute_expert_features(texts, embs, y, is_cls, len(classes), extra_texts, extra_embs, (kind,),
                                           mode, ecfg, n_jobs=expert_jobs())
        out, hits[kind] = cached_expert_features(cache_root, key, compute)
        for split, d in out.items():
            feats_by_split.setdefault(split, {}).update(d)
    if use_num:
        for c in text_cols:
            feats_by_split["pool"][f"num:{c}"] = numeracy_matrix(texts[c])
            for k in extra_texts:
                feats_by_split[k][f"num:{c}"] = numeracy_matrix(extra_texts[k][c])
    t_experts = time.time() - t1

    owner = {c: c for c in text_cols} | {ALL_COLS: text_cols[0]}
    split_rows = {"pool": pool_rows, **{k: np.asarray(x.index) for k, (x, _) in extra.items()}}
    features = {}
    for col in text_cols:
        names = sorted(f for f in feats_by_split["pool"] if owner[f.split(":", 1)[1]] == col)
        if not names:
            features[col] = pd.DataFrame(index=np.sort(np.concatenate(list(split_rows.values()))))
            continue
        blocks = [pd.DataFrame(np.hstack([feats_by_split[s][f] for f in names]), index=r) for s, r in split_rows.items()]
        features[col] = pd.concat(blocks).sort_index()
        features[col].columns = [f"{f}#{NUMERACY_NAMES[j] if f.startswith('num:') else j}" for f in names
                                 for j in range(feats_by_split["pool"][f].shape[1])]

    def quality(split, rows, yy):
        vals = {}
        for f in feats_by_split["pool"]:
            if f.startswith("num:"):
                continue  # target-free
            a = feats_by_split[split][f]
            if is_cls:
                probs = np.hstack([1 - a.sum(axis=1, keepdims=True), a]) if a.shape[1] == len(classes) - 1 else a
                vals[f] = _metric(yy.astype(str).to_numpy(), probs, True, classes)
            else:  # expert predicts the rank-Gaussian target; Spearman-like check via R² against the raw target's ranks
                from scipy.stats import spearmanr
                vals[f] = float(spearmanr(a[:, 0], pd.to_numeric(yy).to_numpy(dtype=float)).statistic)
        return vals
    q = {"train_rows": quality("pool", pool_rows, y_fit)}
    if "test" in extra:
        q["test"] = quality("test", np.asarray(extra["test"][0].index), extra["test"][1])
    gap = {f: (q["train_rows"][f] - q["test"][f]) if q.get("test") and q["train_rows"][f] is not None
           and q["test"][f] is not None else None for f in q["train_rows"]}
    diag = {"engine_version": CTE_VERSION, "method": method, "experts": list(experts), "mode": mode,
            "numeracy": use_num, "expert_config": {**ecfg.__dict__}, "expert_cache_hits": hits,
            "frozen_embedding_cache": emb_stats,
            "pred_feature_quality": {"per_expert": q, "gap_train_minus_test": gap,
                                     "metric": "AUROC (cls) / Spearman with y (reg)"},
            "n_pred_features_per_col": {c: int(features[c].shape[1]) for c in text_cols},
            "seconds": {"frozen_embeddings": round(t_emb, 1), "experts": round(t_experts, 1)}}
    print(f"  [STACK:{method}] experts {list(experts)} ({mode}) numeracy={use_num}; emb {emb_stats}; "
          f"expert cache hits {hits}; emb {t_emb:.0f}s, experts {t_experts:.0f}s", flush=True)
    return StackResult(method=method, embedding_result=frozen, base_result=None, features=features, diagnostics=diag)

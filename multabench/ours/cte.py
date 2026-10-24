"""P5 v2 (PROGRESS.md, amendments 1+2): Cross-fitted Text Experts (CTE) — no fine-tuning. Engine cte/3.

Learner features = structured columns + frozen e5-small-v2 PCA-30 per text column (PCA fit on training rows, in the
learner) + cross-fitted expert predictions + target-free numeracy / form features per text column:
  lexical expert   : on all text columns concatenated ("col: val | ..."). TF-IDF (word 1-2 + char_wb 3-5, 50k
                     vocabulary each, sublinear, min_df 2; char on the first 2,000 characters) fit ONCE per job on the
                     learner's training rows (label-free), then Ridge(sparse_cg) on the rank-Gaussian y (regression)
                     or on one-hot class indicators (classification). Alpha is chosen from (0.3, 1, 3, 10) by the
                     5-fold OOF metric on the training rows (AUROC / macro OvR AUROC / Spearman); the chosen alpha's
                     OOF predictions are the feature.
  neighbour expert : per text column + concatenation (multiclass: concatenation only); k=20 cosine neighbours by
                     matrix product of L2-normalised frozen embeddings, 1/d weights.
  column budget    : regression 1 column; binary 1; 3..10 classes C-1; C > 10 the top-10 training classes' scores +
                     max probability + entropy (12 columns).
Pool (training) rows get 5-fold out-of-fold predictions; learner-val / test rows get predictions of experts fitted on
all training rows. Frozen embeddings are target-free and cached once per dataset (per unique cell text).
From experts.py, cte/3 uses only ExpertConfig (folds / rank-Gauss settings), expert_folds, rank_gauss,
cached_expert_features (npz cache), ALL_COLS and SEED; P6 hooks: compute_cte_features(lex_fit=..., residual=...).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from os.path import exists, join
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.ours import tar_engine as te
from multabench.ours.experts import ALL_COLS, SEED, ExpertConfig, cached_expert_features, expert_folds
from multabench.ours.experts import rank_gauss

CTE_VERSION = "cte/3"  # cte/2 = amendment 1 only; cte/3 = + amendment 2 (column budget, OOF alpha)
CTE_METHODS = {  # method -> (experts, mode, numeracy)
    "cte": (("lex", "nbr"), "oof", True),
    "cte_lex": (("lex",), "oof", False),
    "cte_nbr": (("nbr",), "oof", False),
    "cte_num": ((), "oof", True),
    "cte_lex_insample": (("lex",), "insample", False),
}


@dataclass(frozen=True)
class CTEConfig:
    n_folds: int = 5
    word_ngrams: tuple = (1, 2)
    char_ngrams: tuple = (3, 5)
    min_df: int = 2
    max_word_features: int = 50_000
    max_char_features: int = 50_000
    char_max_chars: int = 2000  # the char vectorizer sees the first 2,000 characters of each text
    ridge_alphas: tuple = (0.3, 1.0, 3.0, 10.0)  # chosen per job by the OOF metric on training rows (amendment 2B)
    max_class_columns: int = 10  # C > 10: top-10 class scores + max prob + entropy (amendment 2A)
    k_neighbors: int = 20
    seed: int = SEED

    def key(self) -> str:
        return hashlib.sha1(json.dumps([CTE_VERSION, asdict(self)], sort_keys=True).encode()).hexdigest()[:10]

    def expert_config(self) -> ExpertConfig:
        """Folds and the rank-Gaussian target come from experts.py (same definitions as P5 v2 before amendment 1)."""
        return ExpertConfig(n_folds=self.n_folds, seed=self.seed, k_neighbors=self.k_neighbors,
                            neighbour_weights="inverse_distance", reg_target="rank_gauss")


# P17: the cross-fitting fold seed can be changed per process (method names *_xs<seed>, scripts/run_experiment.py set
# OURS_XFIT_SEED before this module is imported); unset = 42, the configuration of every other result.
CTE_CONFIG = CTEConfig(seed=int(os.environ.get("OURS_XFIT_SEED", SEED)))

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


@contextlib.contextmanager
def _store_lock(path: Optional[str]):
    """Exclusive lock (local disk) around read-encode-write of one frozen-embedding store. Parallel jobs on one machine
    otherwise encode the same texts at once and race on tar_engine's lower-layer cache dir (E24's crash)."""
    if not path:
        yield
        return
    import fcntl
    root = join(os.path.expanduser("~"), ".cache", "ours_locks")
    os.makedirs(root, exist_ok=True)
    with open(join(root, hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16] + ".lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def frozen_embeddings_cached(x_frames: list[DataFrame], text_cols: list[str], cfg: te.TarConfig, device: torch.device,
                             cache_root: Optional[str], dataset: str,
                             prefix: str = "passage: ") -> tuple[te.TarResult, dict]:
    """Frozen e5 embeddings ("<prefix>col: val", default "passage: col: val" as upstream; L2-normalised mean pool) for every row of x_frames, from a per-dataset
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
        sub = "" if prefix == "passage: " else hashlib.sha1(prefix.encode()).hexdigest()[:8]  # other prefixes: own store
        path = join(cache_root, "frozen_emb", dataset, cfg.model_name.replace("/", "__"), f"len{cfg.max_length}",
                    *([sub] if sub else []),
                    hashlib.sha1(col.encode()).hexdigest()[:12] + ".npz") if cache_root else None
        with _store_lock(path):  # parallel jobs: one encodes, the others wait and then read the cache
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
                strings = [format_e5_passage(col, uniq[i]) if prefix == "passage: " else f"{prefix}{col}: {uniq[i]}"
                           for i in missing]
                tokens = te.TokenStore(list(dict.fromkeys(strings)), tok, cfg.max_length)
                emb_u, _ = te.encode(model, te.make_source(model, tokens, cfg, device), tokens.uids(strings), cfg, device)
                store_h = np.concatenate([store_h, hashes[missing]])
                store_e = np.concatenate([store_e, emb_u.astype(np.float32)])
                pos = {int(h): i for i, h in enumerate(store_h)}
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp = f"{path}.{os.getpid()}.tmp.npz"  # unique per process: parallel jobs may share a cache
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


# ---------------------------------------------------------------------------------------- fast experts

def tfidf_matrices(pool: list[str], extras: Dict[str, list[str]], c: CTEConfig):
    """TF-IDF fit once on the pool (= the learner's training rows; target-free); returns (X_pool, {split: X})."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    cut = lambda xs: [t[:c.char_max_chars] for t in xs]
    mats_pool, mats_extra = [], {k: [] for k in extras}
    for kw, prep in ((dict(analyzer="word", ngram_range=c.word_ngrams, max_features=c.max_word_features), list),
                     (dict(analyzer="char_wb", ngram_range=c.char_ngrams, max_features=c.max_char_features), cut)):
        v = TfidfVectorizer(sublinear_tf=True, min_df=c.min_df, dtype=np.float32, **kw)
        try:
            mats_pool.append(v.fit_transform(prep(pool)))
        except ValueError:  # empty vocabulary (e.g. all texts identical / too few) -> skip this analyzer
            continue
        for k, xs in extras.items():
            mats_extra[k].append(v.transform(prep(xs)))
    if not mats_pool:
        return csr_matrix((len(pool), 0), dtype=np.float32), {k: csr_matrix((len(x), 0), dtype=np.float32)
                                                              for k, x in extras.items()}
    return hstack(mats_pool).tocsr(), {k: hstack(m).tocsr() for k, m in mats_extra.items()}


def _targets(y_tr: np.ndarray, is_cls: bool, n_classes: int, c: CTEConfig) -> np.ndarray:
    if is_cls:
        return np.eye(n_classes)[y_tr]  # one-hot class indicators: one multi-output least-squares fit
    return rank_gauss(y_tr, c.expert_config())[:, None]  # QuantileTransformer fit on exactly these rows


def ridge_fit(X_tr, Y: np.ndarray, X_tes: list, alpha: float) -> list:
    """One multi-output least-squares fit on the target matrix Y; predictions (n, Y.shape[1]) for each of X_tes."""
    from sklearn.linear_model import Ridge
    if X_tr.shape[1] == 0:  # no vocabulary: constant prediction
        return [np.tile(Y.mean(axis=0), (X.shape[0], 1)) for X in X_tes]
    m = Ridge(alpha=alpha, solver="sparse_cg").fit(X_tr, Y)
    return [np.asarray(m.predict(X)).reshape(X.shape[0], -1) for X in X_tes]


def ridge_scores(X_tr, y_tr: np.ndarray, X_tes: list, is_cls: bool, n_classes: int, alpha: float,
                 c: CTEConfig) -> list:
    """Full scores (n, C) for classification, (n, 1) for regression, for each matrix in X_tes."""
    return ridge_fit(X_tr, _targets(y_tr, is_cls, n_classes, c), X_tes, alpha)


def top_classes(y_pool: np.ndarray, n_classes: int, c: CTEConfig) -> Optional[np.ndarray]:
    """Amendment 2A: the max_class_columns most frequent training classes (ties by class index) when C > 10."""
    if n_classes <= c.max_class_columns:
        return None
    counts = np.bincount(y_pool, minlength=n_classes)
    return np.argsort(-counts, kind="stable")[:c.max_class_columns]


def class_features(P: np.ndarray, is_cls: bool, n_classes: int, top: Optional[np.ndarray]) -> np.ndarray:
    """Expert output columns: regression 1; binary the class-1 score; 3..10 classes C-1 scores; C > 10 the top-10
    class scores + max probability + entropy over all C (scores clipped at 0 and renormalised; uniform if all 0)."""
    if not is_cls:
        return P
    if n_classes == 2:
        return P[:, 1:2]
    if top is None:
        return P[:, 1:]
    q = np.clip(P, 0.0, None)
    tot = q.sum(axis=1, keepdims=True)
    q = np.where(tot > 0, q / np.where(tot > 0, tot, 1.0), 1.0 / n_classes)
    ent = -(q * np.log(np.clip(q, 1e-12, None))).sum(axis=1, keepdims=True)
    return np.hstack([P[:, top], q.max(axis=1, keepdims=True), ent])


def oof_metric(P: np.ndarray, y: np.ndarray, is_cls: bool, n_classes: int) -> float:
    """Amendment 2B selection metric on training-row OOF scores: AUROC (binary), macro one-vs-rest AUROC over the
    classes present (multiclass), Spearman with y (regression)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    if not is_cls:
        r = spearmanr(P[:, 0], y).statistic
        return float(r) if np.isfinite(r) else -np.inf
    if n_classes == 2:
        return float(roc_auc_score(y == 1, P[:, 1])) if 0 < (y == 1).sum() < len(y) else -np.inf
    aucs = [roc_auc_score(y == k, P[:, k]) for k in range(n_classes) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else -np.inf


def knn_topk(E_tr: np.ndarray, E_te: np.ndarray, k: int, device: torch.device, chunk: int = 4096):
    """Top-k cosine neighbours of L2-normalised rows by one matrix product; returns (cosine distance, index)."""
    dev = device if device.type == "cuda" else torch.device("cpu")
    A = torch.as_tensor(np.ascontiguousarray(E_tr, dtype=np.float32), device=dev)
    k = min(k, len(E_tr))
    dists, idxs = [], []
    for s in range(0, len(E_te), chunk):
        B = torch.as_tensor(np.ascontiguousarray(E_te[s:s + chunk], dtype=np.float32), device=dev)
        sim, idx = torch.topk(B @ A.T, k, dim=1)
        dists.append(np.clip(1.0 - sim.cpu().numpy().astype(np.float64), 0.0, None))
        idxs.append(idx.cpu().numpy())
    return np.concatenate(dists), np.concatenate(idxs)


def neighbour_average(E_tr, Y: np.ndarray, E_te, c: CTEConfig, device: torch.device) -> np.ndarray:
    """1/d-weighted average of the k nearest training rows' target vectors (rows of Y)."""
    dist, idx = knn_topk(E_tr, E_te, c.k_neighbors, device)
    w = 1.0 / (dist + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkc->nc", w, Y[idx])


def neighbour_predict(E_tr, y_tr: np.ndarray, E_te, is_cls: bool, n_classes: int, c: CTEConfig,
                      device: torch.device) -> np.ndarray:
    """Full scores: class distribution (n, C) for classification, (n, 1) for regression."""
    return neighbour_average(E_tr, _targets(y_tr, is_cls, n_classes, c), E_te, c, device)


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def compute_cte_features(kind: str, texts: Dict[str, list[str]], embs: Optional[Dict[str, np.ndarray]],
                         y: np.ndarray, is_cls: bool, n_classes: int, extra_texts: Dict[str, Dict[str, list[str]]],
                         extra_embs: Optional[Dict[str, Dict[str, np.ndarray]]], mode: str, c: CTEConfig,
                         device: torch.device, timings: dict, *, lex_fit: Optional[Callable] = None,
                         residual=None) -> Dict[str, Dict[str, np.ndarray]]:
    """{split: {feature: (n_rows, k)}} for split in {"pool", *extras} (+ "meta" for the lexical alpha). Pool rows
    get out-of-fold predictions (mode "oof") or in-sample ones ("insample", ablation); extras come from the fit on
    all pool rows. Only pool (= the learner's training) rows and their labels are ever fitted on or scored.
    P6 hooks (P5 / cte/3 leaves both None): lex_fit(X_tr, y_tr, X_tes, alpha) -> full scores replaces the one-hot /
    rank-Gaussian Ridge (cte_nb); residual (p6.StructuredResidual) replaces the targets by residuals of a
    structured-only model (cte_residual)."""
    assert mode in ("oof", "insample") and kind in ("lex", "nbr")
    assert residual is None or (mode == "oof" and lex_fit is None)
    cols = sorted(texts)
    folds = expert_folds(len(y), y, is_cls, c.expert_config())
    top = top_classes(y, n_classes, c) if is_cls else None
    out = {"pool": {}, **{k: {} for k in extra_texts}}
    names = list(extra_texts)
    all_rows = np.arange(len(y))
    if residual is None:
        feat = lambda P: class_features(P, is_cls, n_classes, top)
        targets = lambda tr, full: _targets(y[tr], is_cls, n_classes, c)
        select = lambda P: oof_metric(P, y, is_cls, n_classes)
    else:
        feat = lambda P: residual.features(P, top)
        targets = lambda tr, full: residual.full_targets() if full else residual.train_targets(tr)
        select = lambda P: oof_metric(residual.offset_full() + P, y, is_cls, n_classes)

    def oof(fit_predict, take, pool_in):
        P = None
        for tr, te_ in folds:
            p = fit_predict(take(pool_in, tr), tr, [take(pool_in, te_)], False)[0]
            P = np.zeros((len(y), p.shape[1])) if P is None else P
            P[te_] = p
        return P

    if kind == "lex":
        concat = lambda d: [" | ".join(f"{col}: {d[col][i]}" for col in cols) for i in range(len(d[cols[0]]))]
        t0 = time.perf_counter()
        X, X_extra = tfidf_matrices(concat(texts), {k: concat(v) for k, v in extra_texts.items()}, c)
        timings["tfidf"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        take = lambda a, idx: a[idx]
        if lex_fit is None:
            fp = {a: (lambda A, tr, Bs, full, a=a: ridge_fit(A, targets(tr, full), Bs, a)) for a in c.ridge_alphas}
        else:
            fp = {a: (lambda A, tr, Bs, full, a=a: lex_fit(A, y[tr], Bs, a)) for a in c.ridge_alphas}
        oofs = {a: oof(fp[a], take, X) for a in c.ridge_alphas}
        metrics = [select(oofs[a]) for a in c.ridge_alphas]
        alpha = c.ridge_alphas[int(np.argmax(metrics))]  # first maximum = the smaller alpha on ties
        full = fp[alpha](X, all_rows, [X_extra[k] for k in names] + ([X] if mode == "insample" else []), True)
        name = f"lex:{ALL_COLS}"
        for k, P in zip(names, full):
            out[k][name] = feat(P)
        out["pool"][name] = feat(full[-1] if mode == "insample" else oofs[alpha])
        out["meta"] = {"lex_alpha": np.array([alpha]), "lex_alpha_oof_metric": np.array(metrics)}
        timings["ridge"] = round(time.perf_counter() - t0, 2)
        timings["n_tfidf_features"] = int(X.shape[1])
    else:
        assert mode == "oof", "the neighbour expert has no in-sample ablation"
        t0 = time.perf_counter()
        views = {ALL_COLS: _normalise(np.mean([embs[col] for col in cols], axis=0))}
        extra_views = {k: {ALL_COLS: _normalise(np.mean([v[col] for col in cols], axis=0))} for k, v in extra_embs.items()}
        if not (is_cls and n_classes > 2):  # amendment 2A: multiclass -> concatenation only
            views = {**{col: embs[col] for col in cols}, **views}
            extra_views = {k: {**{col: extra_embs[k][col] for col in cols}, **v} for k, v in extra_views.items()}
        fp = lambda A, tr, Bs, full: [neighbour_average(A, targets(tr, full), B, c, device) for B in Bs]
        for view, E in views.items():
            full = fp(E, all_rows, [extra_views[k][view] for k in names], True)
            for k, P in zip(names, full):
                out[k][f"nbr:{view}"] = feat(P)
            out["pool"][f"nbr:{view}"] = feat(oof(fp, lambda a, idx: a[idx], E))
        timings["knn"] = round(time.perf_counter() - t0, 2)
    return out


# ---------------------------------------------------------------------------------------- main entry

def content_key(y: np.ndarray, extra_texts: Dict[str, Dict[str, list]], text_cols: list[str]) -> str:
    """Fingerprint of what a supervised cache entry also depends on beyond the row ids and training texts: the training
    labels and the texts of the rows it predicts (learner-val / test). Without it, a rerun with other labels or edited
    prediction texts under the same dataset / fold / row ids would reuse stale features (audit, 2026-10-22)."""
    h = hashlib.sha1(np.ascontiguousarray(y).tobytes())
    for k in sorted(extra_texts):
        h.update(("\x01" + k + "\x00" + "\x00".join(t for c in text_cols for t in extra_texts[k][c])).encode())
    return h.hexdigest()[:10]


def encoder_key(cfg: te.TarConfig) -> str:
    """Fingerprint of the frozen-encoder settings that cached predictions built on embeddings depend on."""
    return hashlib.sha1(repr((cfg.model_name, cfg.max_length)).encode()).hexdigest()[:8]


def run_cte(method: str, x_fit: DataFrame, y_fit: Series, text_cols: list[str], is_cls: bool,
            extra: Dict[str, tuple[DataFrame, Series]], cfg: te.TarConfig, device: torch.device,
            cache_root: Optional[str], dataset: str, fold: int, struct_cols: Optional[list[str]] = None,
            cat_cols: Optional[list[str]] = None):
    from multabench.ours.stacking import StackResult
    from multabench.ours.variants import _metric
    p6 = None  # earlier version (claim C3); imported only on its own branch
    if method in CTE_METHODS:  # P5 (claim C2)
        experts, mode, use_num = CTE_METHODS[method]
        lex_kind, target_kind, version = "tfidf", "y", CTE_VERSION
    else:  # P6 (exploratory): same CTE experts with a different lexical model or residual targets
        from multabench.ours.archive import p6
        base, lex_kind, target_kind, _ = p6.P6_METHODS[method]
        experts, mode, use_num = CTE_METHODS[base]
        version = p6.P6_ENGINE[method]
    ccfg = CTE_CONFIG
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
    content = content_key(y, extra_texts, text_cols)
    t1 = time.time()
    feats_by_split, hits, timings = {"pool": {}, **{k: {} for k in extra}}, {}, {}
    residual, hooks, p6_info = None, {}, {}
    if target_kind == "residual":
        sc = sorted(struct_cols or [])
        residual = p6.StructuredResidual(p6.struct_frame(x_fit, sc, list(cat_cols or [])), list(cat_cols or []), y,
                                         is_cls, len(classes), expert_folds(len(y), y, is_cls, ccfg.expert_config()),
                                         ccfg.expert_config())
        hooks["residual"] = residual
        p6_info = {"struct_cols": sc, "p6_config": asdict(p6.P6_CONFIG)}
    nb = lex_kind == "nb" and is_cls  # regression: cte_nb = cte (same cache entry)
    for kind in experts:
        tag = kind + ("nb" if nb and kind == "lex" else "")
        if residual is not None:
            tag += "res_" + p6.P6_CONFIG.key() + "_" + hashlib.sha1("|".join(residual.xs.columns).encode()).hexdigest()[:8]
        enc = f"_{encoder_key(cfg)}" if kind == "nbr" else ""  # the lexical expert never reads the encoder
        key = f"{dataset}/fold{fold}/cte_{tag}_{mode}_{ccfg.key()}_{rows_key}_{texts_key}_{content}{enc}"

        def compute(kind=kind):
            embs = extra_embs = None
            if kind == "nbr":
                embs = {c: frozen.row_embeddings(c, pool_rows) for c in text_cols}
                extra_embs = {k: {c: frozen.row_embeddings(c, np.asarray(x.index)) for c in text_cols}
                              for k, (x, _) in extra_t.items()}
            kw = dict(hooks)
            if nb and kind == "lex":
                kw["lex_fit"] = lambda A, yy, Bs, a: p6.nb_ridge_scores(A, yy, Bs, len(classes), a)
            return compute_cte_features(kind, texts, embs, y, is_cls, len(classes), extra_texts, extra_embs, mode,
                                        ccfg, device, timings, **kw)
        out, hits[kind] = cached_expert_features(cache_root, key, compute)
        for split, d in out.items():
            feats_by_split.setdefault(split, {}).update(d)
    meta = feats_by_split.pop("meta", {})
    lex_alpha = ({"chosen": float(meta["lex_alpha"][0]), "alphas": list(ccfg.ridge_alphas),
                  "oof_metric": [round(float(v), 6) for v in meta["lex_alpha_oof_metric"]]} if "lex_alpha" in meta else None)
    top = top_classes(y, len(classes), ccfg) if is_cls else None
    t_num = time.time()
    if use_num:
        for c in text_cols:
            feats_by_split["pool"][f"num:{c}"] = numeracy_matrix(texts[c])
            for k in extra_texts:
                feats_by_split[k][f"num:{c}"] = numeracy_matrix(extra_texts[k][c])
    timings["numeracy"] = round(time.time() - t_num, 2)
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
            if is_cls and top is not None:  # C > 10: mean one-vs-rest AUROC of the top-10 class score columns
                from sklearn.metrics import roc_auc_score
                yi = yy.astype(str).map(pos).to_numpy()
                aucs = [roc_auc_score(yi == k, a[:, j]) for j, k in enumerate(top) if 0 < (yi == k).sum() < len(yi)]
                vals[f] = float(np.mean(aucs)) if aucs else None
            elif is_cls:
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
    if residual is not None:
        p6_info["n_struct_fits"] = residual.n_struct_fits  # 0 when both experts came from the cache
        if "offset" in residual._memo:
            p6_info["struct_oof_metric"] = round(oof_metric(residual.offset_full(), y, is_cls, len(classes)), 6)
    if method not in CTE_METHODS:
        p6_info.update({"lexical": "nb" if nb else "tfidf", "targets": target_kind})
    diag = {"engine_version": version, "method": method, "p6": p6_info or None, "experts": list(experts), "mode": mode,
            "numeracy": use_num, "expert_config": asdict(ccfg), "expert_timings": timings, "expert_cache_hits": hits,
            "lex_alpha": lex_alpha, "n_classes": len(classes) if is_cls else None,
            "class_columns": "top10+maxp+entropy" if top is not None else "C-1" if is_cls else "1",
            "frozen_embedding_cache": emb_stats,
            "pred_feature_quality": {"per_expert": q, "gap_train_minus_test": gap,
                                     "metric": "AUROC (cls; C > 10: mean OvR over the top-10 class columns) / Spearman with y (reg)"},
            "n_pred_features_per_col": {c: int(features[c].shape[1]) for c in text_cols},
            "seconds": {"frozen_embeddings": round(t_emb, 1), "experts": round(t_experts, 1)}}
    print(f"  [STACK:{method}] experts {list(experts)} ({mode}) numeracy={use_num}; emb {emb_stats}; "
          f"expert cache hits {hits}; lex alpha {lex_alpha}; emb {t_emb:.0f}s, experts {t_experts:.0f}s", flush=True)
    return StackResult(method=method, embedding_result=frozen, base_result=None, features=features, diagnostics=diag)

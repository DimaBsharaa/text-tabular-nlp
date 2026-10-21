"""Packaged CTE model (scripts/train.py saves it, scripts/predict.py loads it): the fitted parts of our method, so new
rows are featurised without re-fitting anything.

After the learner is trained, the CTE experts are fitted once on ALL training rows (the pool): exactly the fit that
run_cte uses for learner-val / test rows. That is, the TF-IDF vectorizers, the Ridge with the alpha chosen by the
training-row OOF metric, and the kNN index (pool embeddings + targets per view). New rows get these predictions, plus
numeracy features, laid out in the same feature columns as in training. Then the fitted PCA and learner are applied.

Model folder:
  meta.json         task, classes, text columns, raw feature columns, chosen alpha, encoder config, versions
  vectorizers.joblib TF-IDF vectorizers (word 1-2, char_wb 3-5) of the lexical expert
  ridge.joblib      the lexical Ridge (or the constant prediction if there was no vocabulary)
  probe.joblib      (cte_probe) the linear probe: Ridge on the full frozen embeddings of all text columns
  knn_index.npz     per neighbour view: L2-normalised training-row embeddings + training targets
  depth.joblib      (cte_depth*) per used layer: the training-row standardisation and the all-rows Ridge
  pca.joblib        the fitted PCA-30 per text column
  the fitted learner, in its own format (meta.json "learner_format"; folders without it are CatBoost):
    catboost.cbm                  CatBoost (native format)
    learner_model.tabpfn_fit      TabPFN-v2 / TabPFN-2.5 (tabpfn's save_fitted_tabpfn_model: fitted state without the
                                  foundation-model weights, which are downloaded again on load)
    learner_model.joblib          LightGBM, TabM
  learner.joblib    upstream preprocessing state (dates, feature types, label encoder), heavy parts removed
  For learners with upstream's median filling (TabM), meta.json "fill_medians" holds the training-row medians of our
  appended columns, applied to new rows exactly as in training (learners.py, E41).
"""
from __future__ import annotations

import json
import os
from os.path import join
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame

from multabench.ours import cte
from multabench.ours import tar_engine as te
from multabench.ours.experts import ALL_COLS

FORMAT_VERSION = "cte_model/1"


def _learner_format(learner_model) -> str:
    lib = type(learner_model).__module__.split(".")[0]
    return {"catboost": "catboost", "tabpfn": "tabpfn"}.get(lib, "joblib")


def _fill_medians(model) -> Optional[dict]:
    """The training-row medians learners.py fills our appended columns with (USE_MEDIAN_FILLING learners only)."""
    if not getattr(model, "USE_MEDIAN_FILLING", False):
        return None
    rows = np.asarray(model.ours_stack_inputs["x_fit"].index)  # the learner's training rows, as in learners.py
    out = {}
    for col, feats in model.stack_result.features.items():
        if feats.shape[1]:
            med = feats.loc[rows].median()
            out[col] = {"columns": [str(c) for c in med.index], "values": [None if pd.isna(v) else float(v) for v in med]}
    return out


class CTEFeaturizer:
    """The full-pool CTE experts (method `cte`, engine cte/3) of one fitted learner."""

    def __init__(self, text_cols, is_cls, classes, top, alpha, vectorizers, ridge, ridge_const, knn, c=cte.CTE_CONFIG,
                 probe=None, probe_alpha=None, depth=None):
        self.text_cols, self.is_cls, self.classes, self.top, self.alpha = text_cols, is_cls, classes, top, alpha
        self.vectorizers, self.ridge, self.ridge_const, self.knn, self.c = vectorizers, ridge, ridge_const, knn, c
        self.probe, self.probe_alpha = probe, probe_alpha  # cte_probe (and cte_depth*, which contain it)
        self.depth = depth  # cte_depth*: {pooling, aggregate, top_k, layers, stats {layer: {col: (mu, sd)}}, ridges}

    @property
    def n_classes(self) -> int:
        return len(self.classes) if self.is_cls else 1

    # ------------------------------------------------------------------ fit (mirrors cte.run_cte / compute_cte_features)
    @classmethod
    def from_learner(cls, model) -> "CTEFeaturizer":
        from multabench.ours.depth import DEPTH_METHODS
        assert model.ours_variant in ("cte", "cte_probe", *DEPTH_METHODS), \
            "packaging is implemented for `cte`, `cte_probe` and the depth methods"
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        c = cte.CTE_CONFIG
        inp = model.ours_stack_inputs
        text_cols = sorted(inp["text_cols"])
        y_fit = inp["y_fit"]
        if model.is_cls:
            classes = sorted(y_fit.astype(str).unique())
            y = y_fit.astype(str).map({k: i for i, k in enumerate(classes)}).to_numpy()
        else:
            classes, y = [], pd.to_numeric(y_fit).to_numpy(dtype=float)
        n_classes = len(classes) if model.is_cls else 1
        top = cte.top_classes(y, n_classes, c) if model.is_cls else None
        diag = model.stack_result.diagnostics
        alpha = float((diag["cte"] if "cte" in diag else diag)["lex_alpha"]["chosen"])  # cte_probe / depth: nested
        x_t = inp["x_fit"][text_cols]
        texts = {col: te.encoding_texts(x_t, col) for col in text_cols}
        pool = _concat(texts, text_cols)
        vectorizers, mats = [], []
        for kw, cut in _analyzers(c):
            v = TfidfVectorizer(sublinear_tf=True, min_df=c.min_df, dtype=np.float32, **kw)
            try:
                mats.append(v.fit_transform(_prep(pool, cut, c)))
            except ValueError:  # empty vocabulary -> analyzer skipped (as in cte.tfidf_matrices)
                continue
            vectorizers.append((v, cut))
        X = hstack(mats).tocsr() if mats else csr_matrix((len(pool), 0), dtype=np.float32)
        Y = cte._targets(y, model.is_cls, n_classes, c)  # one-hot / rank-Gaussian of ALL training rows
        ridge, ridge_const = (None, Y.mean(axis=0)) if X.shape[1] == 0 else (
            Ridge(alpha=alpha, solver="sparse_cg").fit(X, Y), None)
        rows = np.asarray(inp["x_fit"].index)
        embs = {col: model.tar_result.row_embeddings(col, rows) for col in text_cols}
        views = {ALL_COLS: cte._normalise(np.mean([embs[col] for col in text_cols], axis=0))}
        if not (model.is_cls and n_classes > 2):  # amendment 2A: multiclass -> concatenation only
            views = {**{col: embs[col] for col in text_cols}, **views}
        knn = {view: (np.asarray(E, dtype=np.float32), np.asarray(Y, dtype=np.float64)) for view, E in views.items()}
        probe = probe_alpha = None
        if model.ours_variant != "cte":  # the all-training-rows fit p7.xfit_linear uses for val/test rows
            probe_alpha = float(diag["probe"]["alpha"])
            probe = Ridge(alpha=probe_alpha, solver="sparse_cg").fit(np.hstack([embs[col] for col in text_cols]), Y)
        depth_fit = None
        if model.ours_variant in DEPTH_METHODS:
            depth_fit = _fit_depth(model, inp, text_cols, rows, y, n_classes)
        return cls(text_cols, model.is_cls, classes, top, alpha, vectorizers, ridge, ridge_const, knn, c, probe,
                   probe_alpha, depth_fit)

    # ------------------------------------------------------------------ transform
    def transform(self, texts: Dict[str, list[str]], embs: Dict[str, np.ndarray], index,
                  device: torch.device = torch.device("cpu")) -> Dict[str, DataFrame]:
        """Feature frames per text column for new rows (same names / order as run_cte's features)."""
        from scipy.sparse import csr_matrix, hstack
        n, c = len(index), self.c
        feat = lambda P: cte.class_features(P, self.is_cls, self.n_classes, self.top)
        out: Dict[str, np.ndarray] = {}
        new = _concat(texts, self.text_cols)
        mats = [v.transform(_prep(new, cut, c)) for v, cut in self.vectorizers]
        X = hstack(mats).tocsr() if mats else csr_matrix((n, 0), dtype=np.float32)
        P = (np.tile(self.ridge_const, (n, 1)) if self.ridge is None
             else np.asarray(self.ridge.predict(X)).reshape(n, -1))
        out[f"lex:{ALL_COLS}"] = feat(P)
        new_views = {ALL_COLS: cte._normalise(np.mean([embs[col] for col in self.text_cols], axis=0))}
        new_views.update({col: embs[col] for col in self.text_cols})
        for view, (E, Y) in self.knn.items():
            out[f"nbr:{view}"] = feat(cte.neighbour_average(E, Y, new_views[view], c, device))
        for col in self.text_cols:
            out[f"num:{col}"] = cte.numeracy_matrix(texts[col])
        if self.probe is not None:
            Pp = np.asarray(self.probe.predict(np.hstack([embs[col] for col in self.text_cols]))).reshape(n, -1)
            out[f"probe:{ALL_COLS}"] = feat(Pp)
        depth_names = []
        if self.depth is not None:  # same names and order as depth.run_depth (appended after the base features)
            from multabench.ours import depth as dp
            d = self.depth
            states = dp.layer_states_cached([DataFrame({col: texts[col] for col in self.text_cols}, index=index)],
                                            self.text_cols, d["cfg"], device, None, "predict", d["pooling"])
            P = {l: np.asarray(d["ridges"][l].predict(dp.layer_matrix(states, self.text_cols, np.asarray(index), l - 1,
                                                                      d["stats"][l])[0])).reshape(n, -1)
                 for l in d["layers"]}
            if d["aggregate"] == "stack":
                depth_names = [f"depth{l:02d}:{ALL_COLS}" for l in d["layers"]]
                for l, name in zip(d["layers"], depth_names):
                    out[name] = feat(P[l])
            elif d["aggregate"] == "sl":  # NNLS over the layers' (binary: class-1 / regression) predictions
                depth_names = [f"depthsl:{ALL_COLS}"]
                Pl = np.hstack([feat(P[l]) for l in d["layers"]])
                out[depth_names[0]] = (Pl @ np.asarray(d["sl_weights"]) + d["sl_intercept"])[:, None]
            else:
                depth_names = [f"depthtop{d['top_k']}:{ALL_COLS}"]
                out[depth_names[0]] = np.mean([feat(P[l]) for l in d["layers"]], axis=0)
        owner = {col: col for col in self.text_cols} | {ALL_COLS: self.text_cols[0]}
        frames = {}
        for col in self.text_cols:
            names = sorted(f for f in out if owner[f.split(":", 1)[1]] == col and f not in depth_names)
            names += [f for f in depth_names if owner[f.split(":", 1)[1]] == col]
            if not names:
                frames[col] = DataFrame(index=index)
                continue
            frames[col] = DataFrame(np.hstack([out[f] for f in names]), index=index)
            frames[col].columns = [f"{f}#{cte.NUMERACY_NAMES[j] if f.startswith('num:') else j}" for f in names
                                   for j in range(out[f].shape[1])]
        return frames

    # ------------------------------------------------------------------ save / load
    def save(self, folder: str) -> dict:
        import joblib
        joblib.dump(self.vectorizers, join(folder, "vectorizers.joblib"), compress=3)
        joblib.dump({"ridge": self.ridge, "const": self.ridge_const}, join(folder, "ridge.joblib"), compress=3)
        np.savez_compressed(join(folder, "knn_index.npz"), **{f"{v}|E": E for v, (E, _) in self.knn.items()},
                            **{f"{v}|Y": Y for v, (_, Y) in self.knn.items()})
        if self.probe is not None:
            joblib.dump(self.probe, join(folder, "probe.joblib"), compress=3)
        if self.depth is not None:
            joblib.dump({k: v for k, v in self.depth.items() if k != "cfg"}, join(folder, "depth.joblib"), compress=3)
        return {"text_cols": self.text_cols, "is_cls": self.is_cls, "classes": self.classes, "probe_alpha": self.probe_alpha,
                "top": None if self.top is None else [int(t) for t in self.top], "alpha": self.alpha,
                "knn_views": list(self.knn),
                "depth": None if self.depth is None else {k: self.depth[k] for k in ("pooling", "aggregate", "top_k")}
                | {"layers": [int(l) for l in self.depth["layers"]]}}

    @classmethod
    def load(cls, folder: str, meta: dict) -> "CTEFeaturizer":
        import joblib
        r = joblib.load(join(folder, "ridge.joblib"))
        z = np.load(join(folder, "knn_index.npz"))
        knn = {v: (z[f"{v}|E"], z[f"{v}|Y"]) for v in meta["knn_views"]}
        top = None if meta["top"] is None else np.asarray(meta["top"])
        probe = joblib.load(join(folder, "probe.joblib")) if meta.get("probe_alpha") is not None else None
        depth_fit = None
        if meta.get("depth"):
            depth_fit = joblib.load(join(folder, "depth.joblib"))
            depth_fit["cfg"] = te.TarConfig(**meta["encoder"])
        return cls(meta["text_cols"], meta["is_cls"], meta["classes"], top, meta["alpha"],
                   joblib.load(join(folder, "vectorizers.joblib")), r["ridge"], r["const"], knn, probe=probe,
                   probe_alpha=meta.get("probe_alpha"), depth=depth_fit)


def _fit_depth(model, inp, text_cols, rows, y, n_classes) -> dict:
    """The depth experts' all-training-rows fits that depth.run_depth used for val/test rows: per used layer, the
    training-row standardisation and Ridge(alpha chosen by that layer's training-row OOF metric)."""
    from sklearn.linear_model import Ridge
    from multabench.ours import depth as dp
    spec = dp.DEPTH_METHODS[model.ours_variant]
    dd = model.stack_result.diagnostics["depth"]
    layers = dd["ranked_layers"][:spec.top_k] if dd["aggregate"] == "top3" else dd["layers"]
    states = dp.layer_states_cached([inp["x_fit"][text_cols]], text_cols, model.ours_cfg, model.device,
                                    model.ours_cache_dir, model.ours_dataset, spec.pooling)
    Y = cte._targets(y, model.is_cls, n_classes, spec.cte_config())
    stats, ridges = {}, {}
    solver = "cholesky" if dd.get("solver") == "eig" else "sparse_cg"  # eig = exact closed form, like cholesky
    for l in layers:
        X, stats[l] = dp.layer_matrix(states, text_cols, rows, l - 1)
        ridges[l] = Ridge(alpha=float(dd["layer_alpha"][l]), solver=solver).fit(X, Y)
    return {"pooling": spec.pooling, "aggregate": dd["aggregate"], "top_k": spec.top_k, "layers": [int(l) for l in layers],
            "stats": stats, "ridges": ridges, "cfg": model.ours_cfg, "sl_weights": dd.get("sl_weights"),
            "sl_intercept": dd.get("sl_intercept")}


def _analyzers(c):
    return ((dict(analyzer="word", ngram_range=c.word_ngrams, max_features=c.max_word_features), False),
            (dict(analyzer="char_wb", ngram_range=c.char_ngrams, max_features=c.max_char_features), True))


def _prep(xs: list[str], cut: bool, c) -> list[str]:
    return [t[:c.char_max_chars] for t in xs] if cut else list(xs)


def _concat(texts: Dict[str, list[str]], cols: list[str]) -> list[str]:
    return [" | ".join(f"{col}: {texts[col][i]}" for col in cols) for i in range(len(texts[cols[0]]))]


# ---------------------------------------------------------------------------------------- whole model folder

def save_model(model, folder: str, extra_meta: Optional[dict] = None) -> None:
    """Writes the model folder for a fitted `cte` / `cte_probe` / `cte_depth*` learner (the learner object is stripped of
    heavy, row-bound parts)."""
    import dataclasses
    import joblib
    os.makedirs(folder, exist_ok=True)
    fz = CTEFeaturizer.from_learner(model)
    from multabench.ours.variants import engine_version
    meta = {"format": FORMAT_VERSION, "method": model.ours_variant, "engine_version": engine_version(model.ours_variant),
            **fz.save(folder),
            "encoder": {k: v for k, v in dataclasses.asdict(model.ours_cfg).items() if k != "lower_cache_dir"},
            **(extra_meta or {})}
    joblib.dump({col: enc.encoder.pca for col, enc in model.text_transformers.items()}, join(folder, "pca.joblib"))
    meta["learner_format"] = _learner_format(model.model_)
    meta["fill_medians"] = _fill_medians(model)
    if meta["learner_format"] == "catboost":
        model.model_.save_model(join(folder, "catboost.cbm"))
    elif meta["learner_format"] == "tabpfn":
        from tabpfn.model_loading import save_fitted_tabpfn_model
        save_fitted_tabpfn_model(model.model_, join(folder, "learner_model.tabpfn_fit"))
    else:
        joblib.dump(model.model_, join(folder, "learner_model.joblib"), compress=3)
    keep = {}
    for attr in ("model_", "text_transformers", "tar_result", "stack_result", "ours_stack_inputs", "ours_test",
                 "ours_learner_val", "ours_probe_sets"):
        keep[attr] = getattr(model, attr, None)
        setattr(model, attr, None)
    try:
        joblib.dump(model, join(folder, "learner.joblib"), compress=3)
    finally:
        for attr, v in keep.items():
            setattr(model, attr, v)
    json.dump(meta, open(join(folder, "meta.json"), "w"), indent=1)


def predict_folder(folder: str, x_new: DataFrame, device: Optional[torch.device] = None) -> DataFrame:
    """Predictions for raw new rows (the training feature columns) from a saved model folder."""
    import joblib
    from catboost import CatBoostClassifier, CatBoostRegressor
    from multabench.ours.learners import PredColumnEncoder
    device = device or torch.device("cpu")
    meta = json.load(open(join(folder, "meta.json")))
    fz = CTEFeaturizer.load(folder, meta)
    model = joblib.load(join(folder, "learner.joblib"))
    fmt = meta.get("learner_format", "catboost")
    if fmt == "catboost":
        model.model_ = (CatBoostClassifier if meta["is_cls"] else CatBoostRegressor)().load_model(
            join(folder, "catboost.cbm"))
    elif fmt == "tabpfn":
        from tabpfn.model_loading import load_fitted_tabpfn_model
        model.model_ = load_fitted_tabpfn_model(join(folder, "learner_model.tabpfn_fit"), device=device)
    else:
        model.model_ = joblib.load(join(folder, "learner_model.joblib"))
    pca = joblib.load(join(folder, "pca.joblib"))
    x = x_new[meta["feature_columns"]].copy()
    x.index = pd.RangeIndex(len(x))
    pre = model._ours_preprocess_like_transform(x)[fz.text_cols]
    cfg = te.TarConfig(**meta["encoder"])
    emb, _ = cte.frozen_embeddings_cached([pre], fz.text_cols, cfg, device, None, "predict")
    texts = {col: te.encoding_texts(pre, col) for col in fz.text_cols}
    feats = fz.transform(texts, {col: emb.row_embeddings(col, x.index) for col in fz.text_cols}, x.index, device)
    for col, med in (meta.get("fill_medians") or {}).items():  # TabM: the training-row medians, as in learners.py
        assert [str(c) for c in feats[col].columns] == med["columns"], col
        values = pd.Series([np.nan if v is None else v for v in med["values"]], index=feats[col].columns, dtype=float)
        feats[col] = feats[col].fillna(values).fillna(0.0)
    model.text_transformers = {col: PredColumnEncoder(result=emb, col_name=col, encoder=pca[col], features=feats[col])
                               for col in fz.text_cols}  # sorted, as in training (column order)
    model.tar_result = emb
    pred = np.asarray(model.predict(x))
    if not meta["is_cls"]:
        return DataFrame({"prediction": pred}, index=x_new.index)
    labels = [str(k) for k in model.target_transformer.classes_]
    probs = np.column_stack([1 - pred, pred]) if pred.ndim == 1 else pred
    return DataFrame(probs, columns=[f"proba_{k}" for k in labels], index=x_new.index)

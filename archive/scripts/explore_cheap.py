"""EXPLORATION ONLY (not a pre-registered method): cheap label-aware text features on top of Joint Frozen.

Per (dataset, fold), with upstream loading / splits / preprocessing and upstream CatBoost hyperparameters:
  base   = structured columns + frozen E5 PCA-30 per text column (≈ upstream Joint Frozen)
  +lex   = base + 5-fold out-of-fold TF-IDF -> ridge / logistic predictions per text column
  +knn   = base + 5-fold out-of-fold kNN target mean (cosine, k=20) in frozen E5 space per text column
  +both
Test rows get the full-fit predictions. Compared with the leaderboard `all` (Joint Frozen) and `ft` (Joint TAR).

    python scripts/explore_cheap.py --dataset REG_TEXT_VIDEO_GAMES_SALES --folds 0 1 2
"""
from __future__ import annotations
import os as _os
import sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # archive/scripts -> repo
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "scripts"), _os.path.dirname(_os.path.abspath(__file__))]

import argparse
import dataclasses
import json
import os
import sys
import time
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, join(REPO_ROOT, "scripts"))
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import pandas as pd


def oof_lexical(texts_tr, y_tr, texts_te, is_cls, n_classes, seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold

    def fit_pred(tr_t, tr_y, te_t):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
        Xtr = vec.fit_transform(tr_t)
        Xte = vec.transform(te_t)
        if is_cls:
            m = LogisticRegression(C=4.0, max_iter=2000)
            m.fit(Xtr, tr_y)
            p = np.zeros((Xte.shape[0], n_classes))
            p[:, m.classes_] = m.predict_proba(Xte)
            return p[:, 1:2] if n_classes == 2 else p[:, 1:]
        m = Ridge(alpha=3.0)
        m.fit(Xtr, tr_y)
        return m.predict(Xte)[:, None]

    splitter = StratifiedKFold(5, shuffle=True, random_state=seed) if is_cls else KFold(5, shuffle=True, random_state=seed)
    oof = None
    for tr, va in splitter.split(np.zeros(len(texts_tr)), y_tr if is_cls else None):
        p = fit_pred([texts_tr[i] for i in tr], y_tr[tr], [texts_tr[i] for i in va])
        oof = np.zeros((len(texts_tr), p.shape[1])) if oof is None else oof
        oof[va] = p
    return oof, fit_pred(texts_tr, y_tr, texts_te)


def oof_knn(emb_tr, y_tr, emb_te, is_cls, n_classes, k=20, seed=42):
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.neighbors import NearestNeighbors

    def target(y):
        if is_cls:
            Y = np.eye(n_classes)[y]
            return Y[:, 1:2] if n_classes == 2 else Y[:, 1:]
        return y[:, None].astype(float)

    def fit_pred(etr, ytr, ete):
        nn = NearestNeighbors(n_neighbors=min(k, len(etr)), metric="cosine").fit(etr)
        _, idx = nn.kneighbors(ete)
        return target(ytr)[idx].mean(axis=1)

    splitter = StratifiedKFold(5, shuffle=True, random_state=seed) if is_cls else KFold(5, shuffle=True, random_state=seed)
    oof = np.zeros((len(emb_tr), target(y_tr[:1]).shape[1]))
    for tr, va in splitter.split(np.zeros(len(emb_tr)), y_tr if is_cls else None):
        oof[va] = fit_pred(emb_tr[tr], y_tr[tr], emb_tr[va])
    return oof, fit_pred(emb_tr, y_tr, emb_te)


def main():
    import torch
    from sklearn.decomposition import PCA
    from tabstar.preprocessing.splits import split_to_test, split_to_val
    from catboost import CatBoostClassifier, CatBoostRegressor

    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.baselines.training.metrics import calculate_metric
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.datasets.utils import dataset_from_name
    from multabench.ours import tar_engine as te
    from multabench.ours.learners import make_learner
    from multabench.ours.stacking import frozen_embeddings
    from run_experiment import reference_scores

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default=join(REPO_ROOT, ".explore"))
    p.add_argument("--base", choices=["frozen", "tar"], default="frozen",
                   help="embeddings: frozen E5 (Joint Frozen) or a tar_repro fine-tune on the training rows (CPU: slow)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cpu = torch.device("cpu")
    cfg = te.TarConfig(eval_batch_size=64, lower_cache_dir=join(os.path.expanduser("~"), ".cache", "tar_lower_cache"))
    ds_id = dataset_from_name(name=a.dataset)
    ds = download_dataset(dataset_id=ds_id, multimodal_state=MultimodalState.ALL, target_override=None)
    rows = []
    for fold in a.folds:
        t0 = time.time()
        x, y = subsample_dataset(x=ds.x, y=ds.y, is_cls=ds.is_cls, train_examples=10_000, fold=fold)
        x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=ds.is_cls, fold=fold, train_examples=10_000)
        lr = make_learner("catboost")(problem_type=ds.task_type, device=cpu, dataset=ds_id, image_folder=None, tune_e5=False)
        lr.configure_ours("tar_repro", cfg, None, a.dataset, fold, (x_test, y_test))
        xp_tr, yp_tr = lr.do_model_agnostic_preprocessing(x=x_train.copy(), y=y_train.copy())
        xp_te = lr._ours_preprocess_like_transform(x_test)
        text_cols = sorted(lr.text_features)
        cat_cols = sorted(lr.categorical_features)
        struct = [c for c in xp_tr.columns if c not in text_cols]
        if ds.is_cls:
            enc = {c: i for i, c in enumerate(sorted(yp_tr.astype(str).unique()))}
            ytr = yp_tr.astype(str).map(enc).to_numpy()
            yte = y_test.astype(str).map(enc).fillna(-1).astype(int).to_numpy()
            n_cls = len(enc)
        else:
            ytr, yte, n_cls = pd.to_numeric(yp_tr).to_numpy(float), pd.to_numeric(y_test).to_numpy(float), 1
        if a.base == "tar":  # tar_repro fine-tune on the training rows (its own 90/10 split); test = predict-only
            emb = te.run_tar(xp_tr[text_cols], yp_tr, text_cols, ds.is_cls, {"test": (xp_te[text_cols], y_test)},
                             dataclasses.replace(cfg, micro_batch_size=32), cpu, join(a.out, "_tar_cache"),
                             a.dataset, fold, variant="tar_repro")
        else:
            emb = frozen_embeddings(xp_tr[text_cols], {"test": (xp_te[text_cols], y_test)}, text_cols, cfg, cpu)
        feats = {"base": ([], []), "lex": ([], []), "knn": ([], [])}
        for c in text_cols:
            etr = emb.row_embeddings(c, xp_tr.index)
            ete = emb.row_embeddings(c, xp_te.index)
            pca = PCA(n_components=min(30, etr.shape[0] - 1), random_state=42).fit(etr)
            feats["base"][0].append(pca.transform(etr)); feats["base"][1].append(pca.transform(ete))
            ttr, tte = te.encoding_texts(xp_tr, c), te.encoding_texts(xp_te, c)
            o, t = oof_lexical(ttr, ytr, tte, ds.is_cls, n_cls)
            feats["lex"][0].append(o); feats["lex"][1].append(t)
            o, t = oof_knn(etr, ytr, ete, ds.is_cls, n_cls)
            feats["knn"][0].append(o); feats["knn"][1].append(t)
        S_tr = xp_tr[struct].reset_index(drop=True)
        S_te = xp_te[struct].reset_index(drop=True)
        cats = [c for c in struct if c in cat_cols]
        for c in cats:
            S_tr[c], S_te[c] = S_tr[c].astype(str), S_te[c].astype(str)

        def frame(parts, prefix):
            M = np.hstack(parts) if parts else np.zeros((0, 0))
            return pd.DataFrame(M, columns=[f"{prefix}{i}" for i in range(M.shape[1])])

        ref = reference_scores(a.dataset, "CatBoost 😸", fold)
        res = {"dataset": a.dataset, "fold": fold, "lb_all": ref.get("joint_frozen"), "lb_ft": ref.get("joint_tar")}
        for name, keys in (("base", ["base"]), ("+lex", ["base", "lex"]), ("+knn", ["base", "knn"]),
                           ("+both", ["base", "lex", "knn"])):
            Xtr = pd.concat([S_tr] + [frame(feats[k][0], f"{k}_") for k in keys], axis=1)
            Xte = pd.concat([S_te] + [frame(feats[k][1], f"{k}_") for k in keys], axis=1)
            idx = np.arange(len(Xtr))
            tr_i, va_i, _, _ = split_to_val(x=pd.DataFrame({"i": idx}), y=pd.Series(ytr), is_cls=ds.is_cls)
            tr_i, va_i = tr_i["i"].to_numpy(), va_i["i"].to_numpy()
            M = (CatBoostClassifier if ds.is_cls else CatBoostRegressor)(
                iterations=2000, early_stopping_rounds=50, od_pval=0.001, random_state=0, verbose=0,
                cat_features=cats, thread_count=-1)
            M.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=(Xtr.iloc[va_i], ytr[va_i]), use_best_model=True)
            pred = M.predict_proba(Xte) if ds.is_cls else M.predict(Xte)
            if ds.is_cls and n_cls == 2:
                pred = pred[:, 1]
            keep = yte >= 0 if ds.is_cls else np.ones(len(yte), bool)
            res[name] = float(calculate_metric(y_true=yte[keep], y_pred=pred[keep], d_output=n_cls if ds.is_cls else 1).score)
        res["seconds"] = round(time.time() - t0)
        rows.append(res)
        print(json.dumps(res), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(join(a.out, f"{a.dataset}_{a.base}.csv"), index=False)
    for k in ("base", "+lex", "+knn", "+both"):
        print(f"{k:<6} mean {df[k].mean():.4f} | vs lb_ft {np.mean(df[k] - df.lb_ft):+.4f} | vs base {np.mean(df[k] - df.base):+.4f}")
    print(f"lb_all {df.lb_all.mean():.4f} | lb_ft {df.lb_ft.mean():.4f}")


if __name__ == "__main__":
    main()

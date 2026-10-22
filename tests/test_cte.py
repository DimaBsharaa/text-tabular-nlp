"""CPU tests for P5 v2 Cross-fitted Text Experts (multabench/ours/cte.py)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import cte
from multabench.ours import experts as ex
from multabench.ours import stacking as st
from multabench.ours import tar_engine as te

CPU = torch.device("cpu")
BASE = te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16)


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(BASE, lower_cache_dir=str(tmp_path / "lower"))


def test_numeracy_features():
    f = dict(zip(cte.NUMERACY_NAMES, cte.numeracy_features("Released in 1998, sold 2,500 copies at $19.99 (up 15%)! Wow?")))
    assert f["n_numbers"] >= 4 and f["n_years"] == 1 and f["max_year"] == 1998
    assert f["has_currency"] == 1 and np.isclose(f["max_money"], np.log1p(19.99))
    assert f["n_percent"] == 1 and f["n_exclaim"] == 1 and f["n_question"] == 1 and f["is_empty"] == 0
    assert np.isclose(f["max_number"], np.log1p(2500))
    e = dict(zip(cte.NUMERACY_NAMES, cte.numeracy_features("Unknown Value")))  # upstream's missing placeholder
    assert e["is_empty"] == 1 and e["n_chars"] == 0 and np.isnan(e["first_number"])
    neg = dict(zip(cte.NUMERACY_NAMES, cte.numeracy_features("temperature -40 degrees")))
    assert np.isclose(neg["min_number"], -np.log1p(40))
    assert cte.numeracy_matrix(["a 1", "b"]).shape == (2, len(cte.NUMERACY_NAMES))
    assert np.array_equal(cte.numeracy_matrix(["x 3 4"]), cte.numeracy_matrix(["x 3 4"]), equal_nan=True)


def _texts_y(n=150, is_cls=True, seed=3, cols=("a", "b")):
    rng = np.random.default_rng(seed)
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "noisy"]
    texts = {c: [" ".join(rng.choice(words, 5)) + f" {i % 9}" for i in range(n)] for c in cols}
    y = rng.integers(0, 3, n) if is_cls else rng.lognormal(size=n)
    embs = {c: cte._normalise(rng.normal(size=(n, 8))) for c in cols}
    return texts, y, embs


def _features(kind, texts, y, embs, is_cls, mode="oof", n_extra=20):
    extra_t = {"test": {c: v[:n_extra] for c, v in texts.items()}}
    extra_e = {"test": {c: v[:n_extra] for c, v in embs.items()}}
    return cte.compute_cte_features(kind, texts, embs if kind == "nbr" else None, y, is_cls, 3 if is_cls else 1,
                                    extra_t, extra_e if kind == "nbr" else None, mode, cte.CTE_CONFIG, CPU, {})


@pytest.mark.parametrize("kind", ["lex", "nbr"])
@pytest.mark.parametrize("is_cls", [True, False])
def test_oof_prediction_never_sees_its_own_label(monkeypatch, kind, is_cls):
    """Change the labels of the rows held out in fold 0: their own OOF predictions must not move (bit-identical)."""
    texts, y, embs = _texts_y(is_cls=is_cls)
    folds = ex.expert_folds(len(y), y, is_cls, cte.CTE_CONFIG.expert_config())
    monkeypatch.setattr(cte, "expert_folds", lambda *a, **k: folds)  # same folds for both label vectors
    held = folds[0][1]
    y2 = y.copy()
    y2[held] = (y[held] + 1) % 3 if is_cls else y[held] * 7 + 3
    f1, f2 = _features(kind, texts, y, embs, is_cls), _features(kind, texts, y2, embs, is_cls)
    for name in f1["pool"]:
        assert np.array_equal(f1["pool"][name][held], f2["pool"][name][held]), name
        assert not np.array_equal(f1["pool"][name], f2["pool"][name]), name  # the other rows do see them
    for name in f1["test"]:
        assert not np.array_equal(f1["test"][name], f2["test"][name])  # val/test: fit on all training rows


def test_rank_gauss_fit_only_on_each_fits_training_rows(monkeypatch):
    texts, y, embs = _texts_y(n=100, is_cls=False)
    seen, real = [], cte.rank_gauss

    def spy(y_tr, c):
        seen.append(len(y_tr))
        assert c.reg_target == "rank_gauss"
        return real(y_tr, c)
    monkeypatch.setattr(cte, "rank_gauss", spy)
    for kind in ("lex", "nbr"):
        seen.clear()
        _features(kind, texts, y, embs, False)
        folds = ex.expert_folds(100, y, False, cte.CTE_CONFIG.expert_config())
        fold_sizes = [len(tr) for tr, _ in folds]
        if kind == "lex":  # concatenation only; 4 alphas x 5 OOF fits + 1 full fit with the chosen alpha
            expected = [100] + fold_sizes * len(cte.CTE_CONFIG.ridge_alphas)
        else:  # regression: 2 columns + concatenation
            expected = ([100] + fold_sizes) * 3
        assert sorted(seen) == sorted(expected)


def test_tfidf_fit_once_on_training_rows_only(monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts, y, embs = _texts_y(n=120)
    texts["a"][0] = "x" * 5000
    fitted, real = [], TfidfVectorizer.fit_transform

    def spy(self, raw, *a, **k):
        fitted.append((self.analyzer, list(raw)))
        return real(self, raw, *a, **k)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", spy)
    _features("lex", texts, y, embs, True)
    assert [an for an, _ in fitted] == ["word", "char_wb"]  # once per job, not per fold / view
    pool = [f"a: {texts['a'][i]} | b: {texts['b'][i]}" for i in range(120)]
    assert fitted[0][1] == pool  # exactly the training rows (no test rows), all columns concatenated
    assert fitted[1][1] == [t[:2000] for t in pool] and len(fitted[1][1][0]) == 2000  # char: first 2,000 chars
    assert cte.CTE_CONFIG.max_word_features == cte.CTE_CONFIG.max_char_features == 50_000


def test_ridge_expert_outputs():
    texts, y, embs = _texts_y(n=120, is_cls=True)
    f = _features("lex", texts, y, embs, True)
    assert f["pool"]["lex:__all__"].shape == (120, 2) and f["test"]["lex:__all__"].shape == (20, 2)  # C-1 = 2
    yb = (y > 0).astype(int)
    fb = cte.compute_cte_features("lex", texts, None, yb, True, 2, {"test": {c: v[:20] for c, v in texts.items()}},
                                  None, "oof", cte.CTE_CONFIG, CPU, {})
    assert fb["pool"]["lex:__all__"].shape == (120, 1)  # binary: the class-1 score
    from scipy.sparse import random as sprandom
    X = sprandom(50, 30, density=0.3, random_state=0, format="csr")
    full = np.asarray(cte._targets(yb[:50], True, 2, cte.CTE_CONFIG))
    from sklearn.linear_model import Ridge
    p = Ridge(alpha=1.0, solver="sparse_cg").fit(X, full).predict(X)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)  # one-hot least squares: scores sum to 1, so C-1 is lossless
    assert np.allclose(cte.ridge_scores(X, yb[:50], [X], True, 2, 1.0, cte.CTE_CONFIG)[0], p)


def test_knn_matrix_product_matches_sklearn_and_inverse_distance():
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0)
    E_tr, E_te = cte._normalise(rng.normal(size=(300, 16))), cte._normalise(rng.normal(size=(40, 16)))
    d, i = cte.knn_topk(E_tr, E_te, 20, CPU, chunk=7)
    d_ref, i_ref = NearestNeighbors(n_neighbors=20, metric="cosine").fit(E_tr).kneighbors(E_te)
    assert np.array_equal(i, i_ref) and np.allclose(d, d_ref, atol=1e-5)
    E = np.eye(3)
    q = cte._normalise(np.array([[1.0, 0.1, 0.0]]))
    p = cte.neighbour_predict(E, np.array([1, 0, 0]), q, True, 2, dataclasses.replace(cte.CTE_CONFIG, k_neighbors=2),
                              CPU)
    assert p.shape == (1, 2) and p[0, 1] > 0.9  # the closest neighbour (class 1) dominates under 1/d weights


def test_cte_features_deterministic():
    texts, y, embs = _texts_y()
    for kind in ("lex", "nbr"):
        a, b = _features(kind, texts, y, embs, True), _features(kind, texts, y, embs, True)
        assert all(np.array_equal(a[s][f], b[s][f]) for s in a for f in a[s])


def test_insample_lexical_ablation_fits_on_all_rows():
    texts, y, embs = _texts_y(n=120, is_cls=False)
    ins, oof = _features("lex", texts, y, embs, False, "insample"), _features("lex", texts, y, embs, False)
    from scipy.stats import spearmanr
    yr = ex.rank_gauss(y, cte.CTE_CONFIG.expert_config())
    assert spearmanr(ins["pool"]["lex:__all__"][:, 0], yr).statistic > spearmanr(oof["pool"]["lex:__all__"][:, 0],
                                                                                  yr).statistic
    assert np.allclose(ins["test"]["lex:__all__"], oof["test"]["lex:__all__"])  # val/test identical in both modes


def test_parallel_plan_and_learner_threads(monkeypatch):
    import argparse
    import sys
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "scripts"))
    import run_experiment as rx
    monkeypatch.setattr(rx.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(rx.shutil, "which", lambda _: None)  # no GPU here
    a = argparse.Namespace(parallel_jobs="auto")
    n, env = rx.parallel_plan(a, [("cte", "catboost", "D", 0, 42), ("cte_lex", "lgbm", "D", 1, 42)])
    assert n == 6 and env["OURS_LEARNER_THREADS"] == "2" and env["OMP_NUM_THREADS"] == "2"
    assert rx.parallel_plan(a, [("cte", "catboost", "D", 0, 42), ("tar_repro", "catboost", "D", 0, 42)]) == (1, {})
    assert rx.parallel_plan(argparse.Namespace(parallel_jobs="3"), [("tar_repro", "catboost", "D", 0, 42)])[0] == 3
    from multabench.datasets.objects import SupervisedTask
    from multabench.ours.learners import make_learner
    monkeypatch.setenv("OURS_LEARNER_THREADS", "2")
    for name, key in (("catboost", "thread_count"), ("lgbm", "n_jobs")):
        m = make_learner(name)(problem_type=SupervisedTask.BINARY, device=CPU, tune_e5=False)
        assert m.initialize_model().get_params()[key] == 2


def _table(n=330, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(9000, 9000 + n)
    words = ["great", "awful", "fine", "broken", "lovely", "cheap"]
    x = pd.DataFrame({"review": [" ".join(rng.choice(words, 6)) + f" {rng.integers(1990, 2020)} ${rng.integers(5, 99)}"
                                 for _ in range(n)],
                      "title": rng.choice(["Phone", "Lamp", "Chair"], n), "price": rng.lognormal(3, 1, n)}, index=idx)
    y_cls = pd.Series(rng.choice(np.array(["a", "b", "c"])[:n_classes], n), index=idx)
    y_reg = pd.Series(np.log(x["price"]) + rng.normal(0, 0.3, n), index=idx)
    return x, y_cls, y_reg


def test_frozen_embedding_cache_reused_and_equal(tmp_path, cfg):
    x, _, _ = _table(60)
    root = str(tmp_path / "c")
    a, s1 = cte.frozen_embeddings_cached([x.iloc[:40], x.iloc[40:]], ["review", "title"], cfg, CPU, root, "SYNTH")
    b, s2 = cte.frozen_embeddings_cached([x.iloc[:40], x.iloc[40:]], ["review", "title"], cfg, CPU, root, "SYNTH")
    assert s1["encoded"] > 0 and s2["encoded"] == 0 and s2["cached"] == s1["encoded"]
    ref = st.frozen_embeddings(x[["review", "title"]], {}, ["review", "title"], cfg, CPU)
    for c in ("review", "title"):
        assert np.allclose(a.row_embeddings(c, x.index), b.row_embeddings(c, x.index))
        assert np.abs(a.row_embeddings(c, x.index) - ref.row_embeddings(c, x.index)).max() < 1e-5


@pytest.mark.parametrize("method", sorted(cte.CTE_METHODS))
@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_every_cte_variant_end_to_end(tmp_path, cfg, method, task):
    x, y_cls, y_reg = _table(n_classes=3 if task == "mul" else 2)
    y = y_reg if task == "reg" else y_cls
    x_fit, y_fit = x.iloc[:250], y.iloc[:250]
    extra = {"learner_val": (x.iloc[250:280], y.iloc[250:280]), "test": (x.iloc[280:], y.iloc[280:])}
    res = st.run_stacking(method, x_fit, y_fit, ["review", "title"], ["price"], [], task != "reg", extra, cfg, CPU,
                          str(tmp_path / "c"), "SYNTH", 0)
    experts, mode, num = cte.CTE_METHODS[method]
    per = 2 if task == "mul" else 1
    per_col_nbr = "nbr" in experts and task != "mul"  # amendment 2A: multiclass neighbours on the concatenation only
    exp_title = per_col_nbr * per + (len(cte.NUMERACY_NAMES) if num else 0)  # lexical: concatenation only
    assert res.features["title"].shape == (330, exp_title)
    assert res.features["review"].shape == (330, exp_title + len(experts) * per)  # + the "__all__" views
    assert res.base_result is None and res.embedding_result.variant == "frozen"
    assert res.diagnostics["engine_version"] == "cte/3"
    if "lex" in experts:
        la = res.diagnostics["lex_alpha"]
        assert la["chosen"] in cte.CTE_CONFIG.ridge_alphas and len(la["oof_metric"]) == 4
    f = res.features["review"].to_numpy()
    assert np.isfinite(f[:, [i for i, c in enumerate(res.features["review"].columns) if not c.startswith("num:")]]).all()


def test_learner_cte_pca_on_train_and_group_importance(tmp_path, cfg):
    from sklearn.decomposition import PCA
    from multabench.datasets.objects import SupervisedTask
    from multabench.ours.learners import make_learner
    rng = np.random.default_rng(1)
    n = 420
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "flimsy", "bright", "noisy"]
    x = pd.DataFrame({"review": [" ".join(rng.choice(words, 12)) + f" #{i}" for i in range(n)],
                      "price": rng.lognormal(3, 1, n)}, index=pd.RangeIndex(n))
    y = pd.Series(np.where(x["price"] > x["price"].median(), "high", "low"), index=x.index)
    model = make_learner("catboost")(problem_type=SupervisedTask.BINARY, device=CPU, tune_e5=False)
    model.configure_ours(variant="cte", cfg=cfg, cache_dir=str(tmp_path / "c"), dataset="SYNTH", fold=0,
                         test=(x.iloc[380:], y.iloc[380:]))
    model.fit(x.iloc[:380], y.iloc[:380])
    train_rows = x.index[:380].difference(model.ours_learner_val[0].index)
    ref = PCA(n_components=30, random_state=42).fit(model.tar_result.row_embeddings("review", np.asarray(train_rows)))
    assert np.allclose(np.abs(model.text_transformers["review"].encoder.pca.components_), np.abs(ref.components_),
                       atol=1e-5)
    imp = model.stack_feature_importance()
    assert {"structured", "pca", "lex", "nbr", "num"} <= set(imp["group_share"])
    assert np.isclose(sum(imp["group_share"].values()), 1.0, atol=1e-3)
    assert 0 <= model.score_all_metrics(X=x.iloc[380:], y=y.iloc[380:]).score <= 1


@pytest.mark.parametrize("is_cls", [True, False])
def test_parallel_lexical_views_match_serial(is_cls):
    rng = np.random.default_rng(3)
    n = 150
    words = ["great", "awful", "fine", "broken", "lovely", "cheap"]
    texts = {c: [" ".join(rng.choice(words, 5)) + f" {i % 9}" for i in range(n)] for c in ("a", "b")}
    y = rng.integers(0, 2, n) if is_cls else rng.normal(size=n)
    extra = {"test": {c: v[:20] for c, v in texts.items()}}
    args = (texts, None, y, is_cls, 2, extra, None, ("lex",), "oof", ex.ExpertConfig(reg_target="rank_gauss"))
    s, p = ex.compute_expert_features(*args, n_jobs=1), ex.compute_expert_features(*args, n_jobs=3)
    assert s.keys() == p.keys() and all(s[k].keys() == p[k].keys() for k in s)
    # BLAS thread count differs inside workers -> float noise only (measured ~1e-17); each mode is bit-reproducible
    assert all(np.allclose(s[k][f], p[k][f], rtol=0, atol=1e-10) for k in s for f in s[k])
    p2 = ex.compute_expert_features(*args, n_jobs=3)
    assert all(np.array_equal(p[k][f], p2[k][f]) for k in p for f in p[k])


def test_column_budget_for_many_classes():
    """Amendment 2A: C > 10 -> top-10 training classes by frequency + max prob + entropy; 3..10 -> C-1."""
    rng = np.random.default_rng(5)
    y = np.repeat(np.arange(13), np.arange(13) + 5)  # class k appears k + 5 times: the top 10 are 12, 11, ..., 3
    top = cte.top_classes(y, 13, cte.CTE_CONFIG)
    assert list(top) == list(range(12, 2, -1))
    assert cte.top_classes(np.arange(10).repeat(3), 10, cte.CTE_CONFIG) is None
    P = rng.dirichlet(np.ones(13), size=7)
    F = cte.class_features(P, True, 13, top)
    assert F.shape == (7, 12) and np.array_equal(F[:, :10], P[:, top])
    assert np.allclose(F[:, 10], P.max(axis=1)) and np.allclose(F[:, 11], -(P * np.log(P)).sum(axis=1))
    R = np.array([[-0.2, 0.5, 0.0] + [0.0] * 10, [0.0] * 13])  # Ridge scores: clipped + renormalised; all-0 -> uniform
    G = cte.class_features(R, True, 13, top)
    assert np.isclose(G[0, 10], 1.0) and np.isclose(G[0, 11], 0.0) and np.isclose(G[1, 11], np.log(13))
    assert cte.class_features(P[:, :5], True, 5, None).shape == (7, 4)
    assert cte.class_features(P[:, :2], True, 2, None).shape == (7, 1)
    texts, _, embs = _texts_y(n=200, cols=("a", "b"))
    y13 = np.random.default_rng(1).integers(0, 13, 200)
    ex_t = {"test": {c: v[:20] for c, v in texts.items()}}
    lex = cte.compute_cte_features("lex", texts, None, y13, True, 13, ex_t, None, "oof", cte.CTE_CONFIG, CPU, {})
    nbr = cte.compute_cte_features("nbr", texts, embs, y13, True, 13, ex_t,
                                   {"test": {c: v[:20] for c, v in embs.items()}}, "oof", cte.CTE_CONFIG, CPU, {})
    assert {k: v.shape for k, v in lex["pool"].items()} == {"lex:__all__": (200, 12)}
    assert {k: v.shape for k, v in nbr["pool"].items()} == {"nbr:__all__": (200, 12)}  # concatenation only
    assert nbr["test"]["nbr:__all__"].shape == (20, 12)


def test_alpha_chosen_by_training_oof_only(monkeypatch):
    """Amendment 2B: the chosen alpha maximises the OOF metric on training rows; val/test rows never influence it."""
    texts, y, embs = _texts_y(n=150, is_cls=False)
    f = _features("lex", texts, y, embs, False)
    alphas, metrics = cte.CTE_CONFIG.ridge_alphas, f["meta"]["lex_alpha_oof_metric"]
    assert f["meta"]["lex_alpha"][0] == alphas[int(np.argmax(metrics))]
    # recompute the OOF metric of every alpha by hand
    pool = [f"a: {texts['a'][i]} | b: {texts['b'][i]}" for i in range(150)]
    X, _ = cte.tfidf_matrices(pool, {}, cte.CTE_CONFIG)
    folds = ex.expert_folds(150, y, False, cte.CTE_CONFIG.expert_config())
    for a, m in zip(alphas, metrics):
        P = np.zeros((150, 1))
        for tr, te_ in folds:
            P[te_] = cte.ridge_scores(X[tr], y[tr], [X[te_]], False, 1, a, cte.CTE_CONFIG)[0]
        assert np.isclose(cte.oof_metric(P, y, False, 1), m)
    chosen = f["meta"]["lex_alpha"][0]
    P_best = np.zeros((150, 1))
    for tr, te_ in folds:
        P_best[te_] = cte.ridge_scores(X[tr], y[tr], [X[te_]], False, 1, chosen, cte.CTE_CONFIG)[0]
    assert np.allclose(f["pool"]["lex:__all__"], P_best)  # the chosen alpha's OOF predictions are the feature
    # garbage test texts: pool features and the chosen alpha do not move
    texts_bad = {"test": {c: ["zzz qqq"] * 20 for c in texts}}
    g = cte.compute_cte_features("lex", texts, None, y, False, 1, texts_bad, None, "oof", cte.CTE_CONFIG, CPU, {})
    assert np.array_equal(g["pool"]["lex:__all__"], f["pool"]["lex:__all__"])
    assert np.array_equal(g["meta"]["lex_alpha_oof_metric"], f["meta"]["lex_alpha_oof_metric"])


def test_end_to_end_many_classes(tmp_path, cfg):
    x, _, _ = _table(n=400)
    rng = np.random.default_rng(2)
    y = pd.Series(rng.choice([f"c{k}" for k in range(12)], 400), index=x.index)
    extra = {"learner_val": (x.iloc[300:340], y.iloc[300:340]), "test": (x.iloc[340:], y.iloc[340:])}
    res = st.run_stacking("cte", x.iloc[:300], y.iloc[:300], ["review", "title"], ["price"], [], True, extra, cfg, CPU,
                          str(tmp_path / "c"), "SYNTH", 0)
    n_num = len(cte.NUMERACY_NAMES)
    assert res.features["title"].shape == (400, n_num)  # no per-column neighbours for multiclass
    assert res.features["review"].shape == (400, 12 + 12 + n_num)  # lex + nbr on the concatenation, 12 columns each
    assert res.diagnostics["class_columns"] == "top10+maxp+entropy" and res.diagnostics["n_classes"] == 12
    q = res.diagnostics["pred_feature_quality"]["per_expert"]["test"]
    assert set(q) == {"lex:__all__", "nbr:__all__"} and all(0 <= v <= 1 for v in q.values())


def _encode_in_process(args):
    root, lower = args
    import torch as _torch
    from multabench.ours import cte as _cte
    from multabench.ours import tar_engine as _te
    x, _, _ = _table(40)
    c = _te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16, lower_cache_dir=lower)
    res, stats = _cte.frozen_embeddings_cached([x], ["review"], c, _torch.device("cpu"), root, "SYNTH_PAR")
    return stats, res.row_embeddings("review", x.index)


def test_parallel_jobs_encode_each_text_once(tmp_path):
    """E24's crash: parallel jobs of one dataset raced on the same caches. With the store lock exactly one process
    encodes; the others wait and read its cache (identical embeddings)."""
    import multiprocessing as mp
    args = (str(tmp_path / "c"), str(tmp_path / "lower"))
    with mp.get_context("spawn").Pool(3) as pool:
        out = pool.map(_encode_in_process, [args] * 3)
    encoded = sorted(s["encoded"] for s, _ in out)
    assert encoded[0] == encoded[1] == 0 and encoded[2] > 0
    assert all(np.array_equal(out[0][1], e) for _, e in out)


def test_cte3_features_unchanged_golden():
    """P5's code path (cte/3) must reproduce the features generated at f1514bc. Bit for bit in the pinned environment
    (Python 3.11, uv.lock); other BLAS / Python builds differ by ~2e-4 (external audit), so the check allows 1e-3 and
    requires identical missing-value positions: a code change moves features far more than that."""
    import sys
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(abspath(__file__)), "golden"))
    import make_cte3_golden as g
    ref = np.load(g.OUT)
    new = g.all_features()
    assert set(new) == set(ref.files)
    for k in ref.files:
        assert new[k].shape == ref[k].shape and np.array_equal(np.isnan(new[k]), np.isnan(ref[k])), k
        assert np.allclose(new[k], ref[k], rtol=0, atol=1e-3, equal_nan=True), k


def test_cache_key_depends_on_labels_and_prediction_texts():
    """A supervised cache entry must not be reused after the training labels or the predicted rows' texts change under
    the same dataset / fold / row ids (external audit, 2026-10-22)."""
    y = np.array([0, 1, 1, 0, 2])
    extra = {"test": {"a": ["x", "y"]}, "learner_val": {"a": ["z"]}}
    k = cte.content_key(y, extra, ["a"])
    assert k == cte.content_key(y.copy(), {"learner_val": {"a": ["z"]}, "test": {"a": ["x", "y"]}}, ["a"])  # order-free
    assert k != cte.content_key(np.array([0, 1, 1, 0, 1]), extra, ["a"])  # one label changed
    assert k != cte.content_key(y, {"test": {"a": ["x", "y!"]}, "learner_val": {"a": ["z"]}}, ["a"])  # a test text edited
    assert k != cte.content_key(y, {"test": {"a": ["x", "y"]}}, ["a"])  # a split missing

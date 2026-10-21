"""CPU tests for P6 (exploratory CatBoost variants, multabench/ours/p6.py) + the P5 (cte/3) golden guard."""
import dataclasses
import sys
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import cte
from multabench.ours import experts as ex
from multabench.ours.archive import p6
from multabench.ours import tar_engine as te

CPU = torch.device("cpu")
ONE_ALPHA = dataclasses.replace(cte.CTE_CONFIG, ridge_alphas=(1.0,))  # leakage tests: no alpha selection in the way


def _data(n=150, is_cls=True, n_classes=3, seed=4):
    rng = np.random.default_rng(seed)
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "noisy"]
    texts = {c: [" ".join(rng.choice(words, 5)) + f" {i % 9}" for i in range(n)] for c in ("a", "b")}
    embs = {c: cte._normalise(rng.normal(size=(n, 8))) for c in ("a", "b")}
    xs = pd.DataFrame({"num": rng.normal(size=n), "cat": rng.choice(["u", "v", "w"], n)})
    y = rng.integers(0, n_classes, n) if is_cls else rng.lognormal(size=n) + 2 * (xs["num"].to_numpy() > 0)
    return texts, embs, xs, y


def _residual(xs, y, is_cls, n_classes, c=ONE_ALPHA, cfg=p6.P6Config(struct_iterations=30)):
    folds = ex.expert_folds(len(y), y, is_cls, c.expert_config())
    return p6.StructuredResidual(p6.struct_frame(xs, ["cat", "num"], ["cat"]), ["cat"], y, is_cls, n_classes, folds,
                                 c.expert_config(), cfg)


def _run(kind, texts, embs, y, is_cls, n_classes, c=ONE_ALPHA, **hooks):
    ext_t = {"test": {k: v[:20] for k, v in texts.items()}}
    ext_e = {"test": {k: v[:20] for k, v in embs.items()}}
    return cte.compute_cte_features(kind, texts, embs if kind == "nbr" else None, y, is_cls, n_classes, ext_t,
                                    ext_e if kind == "nbr" else None, "oof", c, CPU, {}, **hooks)


# ---------------------------------------------------------------------------------------- cte_nb

def test_nb_ratio_and_binary_scores():
    from scipy.sparse import csr_matrix
    X = csr_matrix(np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=float))
    y = np.array([1, 1, 0, 0])
    P = p6.nb_ridge_scores(X, y, [X], 2, alpha=0.1)[0]
    assert P.shape == (4, 2) and np.allclose(P[:, 0], 1 - P[:, 1])
    assert P[:2, 1].min() > P[2:, 1].max()  # word 0 only in class 1 -> NB weight up-weights it
    P3 = p6.nb_ridge_scores(X, np.array([0, 1, 2, 2]), [X], 3, alpha=0.1)[0]
    assert P3.shape == (4, 3)


@pytest.mark.parametrize("n_classes", [2, 4])
def test_nb_oof_never_sees_its_own_label(monkeypatch, n_classes):
    texts, embs, _, y = _data(is_cls=True, n_classes=n_classes)
    folds = ex.expert_folds(len(y), y, True, ONE_ALPHA.expert_config())
    monkeypatch.setattr(cte, "expert_folds", lambda *a, **k: folds)
    held = folds[0][1]
    y2 = y.copy()
    y2[held] = (y[held] + 1) % n_classes
    fit = lambda A, yy, Bs, a: p6.nb_ridge_scores(A, yy, Bs, n_classes, a)
    f1 = _run("lex", texts, embs, y, True, n_classes, lex_fit=fit)
    f2 = _run("lex", texts, embs, y2, True, n_classes, lex_fit=fit)
    assert np.array_equal(f1["pool"]["lex:__all__"][held], f2["pool"]["lex:__all__"][held])
    assert not np.array_equal(f1["pool"]["lex:__all__"], f2["pool"]["lex:__all__"])
    plain = _run("lex", texts, embs, y, True, n_classes)
    assert not np.allclose(plain["pool"]["lex:__all__"], f1["pool"]["lex:__all__"])  # NB weighting changes the expert


# ---------------------------------------------------------------------------------------- cte_residual

@pytest.mark.parametrize("kind", ["lex", "nbr"])
@pytest.mark.parametrize("is_cls,n_classes", [(True, 2), (True, 3), (False, 1)])
def test_residual_oof_never_sees_its_own_label_nested(monkeypatch, kind, is_cls, n_classes):
    """Fold-0 labels change -> fold-0 rows' residual-expert OOF predictions are bit-identical (the structured model
    that builds the training residuals of fold 0's expert is cross-fitted inside fold 0's training rows only)."""
    texts, embs, xs, y = _data(is_cls=is_cls, n_classes=n_classes)
    folds = ex.expert_folds(len(y), y, is_cls, ONE_ALPHA.expert_config())
    monkeypatch.setattr(cte, "expert_folds", lambda *a, **k: folds)
    held = folds[0][1]
    y2 = y.copy()
    y2[held] = (y[held] + 1) % n_classes if is_cls else y[held] * 5 + 7
    r1, r2 = _residual(xs, y, is_cls, n_classes), _residual(xs, y2, is_cls, n_classes)
    r1.folds = r2.folds = folds
    f1 = _run(kind, texts, embs, y, is_cls, n_classes, residual=r1)
    f2 = _run(kind, texts, embs, y2, is_cls, n_classes, residual=r2)
    for name in f1["pool"]:
        assert np.array_equal(f1["pool"][name][held], f2["pool"][name][held]), name
        assert not np.array_equal(f1["pool"][name], f2["pool"][name]), name
    assert np.array_equal(r1.offset_full()[held], r2.offset_full()[held])  # S's offsets are OOF too
    assert r1.n_struct_fits > 0


def test_residual_targets_definition():
    _, _, xs, y = _data(is_cls=True, n_classes=3)
    r = _residual(xs, y, True, 3)
    R = r.full_targets()
    assert R.shape == (150, 3) and np.allclose(R.sum(axis=1), 0, atol=1e-9)  # one-hot - probabilities
    assert np.allclose(R + r.offset_full(), np.eye(3)[y])
    _, _, xs, yr = _data(is_cls=False)
    rr = _residual(xs, yr, False, 1)
    assert np.allclose(rr.full_targets() + rr.offset_full(), ex.rank_gauss(yr, ONE_ALPHA.expert_config())[:, None])
    tr = np.arange(30, 150)
    assert rr.train_targets(tr).shape == (120, 1) and rr.train_targets(tr) is rr.train_targets(tr)  # memoised
    empty = p6.StructuredResidual(xs.iloc[:, :0], [], yr, False, 1, r.folds, ONE_ALPHA.expert_config())
    assert np.allclose(empty.offset_full().std(), empty.offset_full().std()) and empty.n_struct_fits == 0


def test_residual_column_layout_many_classes():
    r = p6.StructuredResidual(pd.DataFrame(index=range(5)), [], np.zeros(5, int), True, 12, [], ex.ExpertConfig())
    R = np.random.default_rng(0).normal(size=(5, 12))
    top = np.arange(10)[::-1]
    F = r.features(R, top)
    assert F.shape == (5, 12) and np.array_equal(F[:, :10], R[:, top])
    assert np.allclose(F[:, 10], R.max(axis=1)) and np.allclose(F[:, 11], R.min(axis=1))


def test_residual_deterministic():
    texts, embs, xs, y = _data(is_cls=False)
    a = _run("lex", texts, embs, y, False, 1, residual=_residual(xs, y, False, 1))
    b = _run("lex", texts, embs, y, False, 1, residual=_residual(xs, y, False, 1))
    assert all(np.array_equal(a[s][f], b[s][f]) for s in a for f in a[s])


# ---------------------------------------------------------------------------------------- learner end to end

def _table(n, task, seed=0):
    rng = np.random.default_rng(seed)
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "noisy", "bright", "dull"]
    x = pd.DataFrame({"review": [" ".join(rng.choice(words, 8)) + f" {rng.integers(1990, 2020)}" for _ in range(n)],
                      "title": [f"{rng.choice(['Phone', 'Lamp', 'Chair', 'Desk'])} {rng.choice(['A', 'B', 'C'])} model"
                                for _ in range(n)],
                      "price": rng.lognormal(3, 1, n), "brand": rng.choice(["x", "y", "z"], n)},
                     index=pd.RangeIndex(5000, 5000 + n))
    good = x["review"].str.count("great|lovely|sturdy").to_numpy()
    if task == "reg":
        y = pd.Series(np.log(x["price"]) + 0.5 * good + rng.normal(0, 0.3, n), index=x.index)
    elif task == "bin":
        y = pd.Series(np.where(good + rng.normal(0, 0.7, n) > 2.4, "pos", "neg"), index=x.index)
    else:
        y = pd.Series(np.array(["lo", "mid", "hi"])[np.clip(good, 0, 2)], index=x.index)
    return x, y


@pytest.mark.parametrize("method", sorted(p6.P6_METHODS))
@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_every_p6_variant_end_to_end(tmp_path, method, task):
    from multabench.datasets.objects import SupervisedTask
    from multabench.ours.learners import make_learner
    from multabench.ours.variants import engine_version
    x, y = _table(320, task)
    problem = {"bin": SupervisedTask.BINARY, "mul": SupervisedTask.MULTICLASS, "reg": SupervisedTask.REGRESSION}[task]
    cfg = te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16, lower_cache_dir=str(tmp_path / "lower"))
    model = make_learner("catboost")(problem_type=problem, device=CPU, tune_e5=False)
    model.configure_ours(variant=method, cfg=cfg, cache_dir=str(tmp_path / "c"), dataset="SYNTH", fold=0,
                         test=(x.iloc[270:], y.iloc[270:]))
    model.fit(x.iloc[:270], y.iloc[:270])
    names = list(model.model_.feature_names_)
    native = [c for c in names if c.endswith(p6.NATIVE_SUFFIX)]
    pca = [c for c in names if "_txt_pca_" in c]
    assert engine_version(method) == p6.P6_ENGINE[method]
    if p6.P6_METHODS[method][3]:  # native text variants
        assert sorted(native) == [c + p6.NATIVE_SUFFIX for c in sorted(model.text_features)] and native  # upstream's text cols
        info = model.native_text_info()
        assert info["feature_calcers"] == p6.native_feature_calcers(task != "reg", 3) and 0 <= info["importance_share"] <= 1
    else:
        assert not native and model.native_text_info() is None
    if method == "catboost_native_text":
        assert not pca and model.tar_result is None  # no embeddings at all
    else:
        assert len(pca) > 30 * len(model.text_features)  # PCA-30 per text column + expert / numeracy columns
        d = model.stack_result.diagnostics
        assert d["engine_version"] == p6.P6_ENGINE[method] and d["p6"]["targets"] == ("residual" if method == "cte_residual" else "y")
        if method == "cte_residual":
            assert "price" in d["p6"]["struct_cols"] and d["p6"]["n_struct_fits"] > 0
            assert not set(d["p6"]["struct_cols"]) & set(model.text_features)
    s = model.score_all_metrics(X=x.iloc[270:], y=y.iloc[270:]).score
    assert np.isfinite(s) and (task == "reg" or 0 <= s <= 1)


def test_cte_plus_native_reuses_cte_expert_cache(tmp_path):
    from multabench.datasets.objects import SupervisedTask
    from multabench.ours.learners import make_learner
    x, y = _table(320, "bin")
    cfg = te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16, lower_cache_dir=str(tmp_path / "lower"))
    feats = {}
    for method in ("cte", "cte_plus_native"):
        model = make_learner("catboost")(problem_type=SupervisedTask.BINARY, device=CPU, tune_e5=False)
        model.configure_ours(variant=method, cfg=cfg, cache_dir=str(tmp_path / "c"), dataset="SYNTH", fold=0,
                             test=(x.iloc[270:], y.iloc[270:]))
        model.fit(x.iloc[:270], y.iloc[:270])
        feats[method] = model.stack_result
    assert feats["cte_plus_native"].diagnostics["expert_cache_hits"] == {"lex": True, "nbr": True}
    assert feats["cte"].features.keys() == feats["cte_plus_native"].features.keys()
    for col in feats["cte"].features:
        assert feats["cte"].features[col].equals(feats["cte_plus_native"].features[col])


def test_native_calcers_amendment():
    """P6 amendment 2: BoW only for every task (binary, multiclass of any size, regression)."""
    for is_cls, c in ((True, 2), (True, 3), (True, 10), (True, 11), (True, 30), (False, 1)):
        assert p6.native_feature_calcers(is_cls, c) == ["BoW"]
    assert p6.P6_ENGINE == {"catboost_native_text": "p6/3", "cte_plus_native": "p6/3", "cte_nb": "p6/2",
                            "cte_residual": "p6/2"}
    assert p6.P6_CONFIG.key() == "9280a862b6"  # cte_residual's cache key did not change with amendment 2

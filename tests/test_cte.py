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


def test_rank_gauss_fit_only_on_each_experts_training_rows(monkeypatch):
    n = 100
    y = np.random.default_rng(0).lognormal(size=n)
    seen, real = [], ex.rank_gauss

    def spy(y_tr, c):
        seen.append(set(np.round(np.asarray(y_tr), 12)))
        return real(y_tr, c)
    monkeypatch.setattr(ex, "rank_gauss", spy)
    texts = {"t": [f"w{i % 7} x{i}" for i in range(n)]}
    ex.compute_expert_features(texts, None, y, False, 1, {"test": {"t": ["w1 q"]}}, None, ("lex",), "oof",
                               cte.CTE_EXPERT_CONFIG)
    folds = ex.expert_folds(n, y, False, cte.CTE_EXPERT_CONFIG)
    allowed = [set(np.round(y[tr], 12)) for tr, _ in folds] + [set(np.round(y, 12))]
    assert seen and all(any(s == a for a in allowed) for s in seen)
    # the held-out rows of a fold never enter that fold's transform
    for (tr, te_), s in zip(folds * 2, [x for x in seen if len(x) < n]):
        assert not (set(np.round(y[te_], 12)) & s) or s == set(np.round(y[tr], 12))


def test_oof_cte_experts_never_fit_on_predicted_rows(monkeypatch):
    n = 120
    texts = {"review": [f"good bad {i} uid{i}" for i in range(n)]}
    y = np.random.default_rng(1).integers(0, 2, n)
    calls, real = [], ex.LexicalExpert.fit_predict

    def spy(self, tr, y_tr, te_):
        calls.append((set(tr), set(te_)))
        return real(self, tr, y_tr, te_)
    monkeypatch.setattr(ex.LexicalExpert, "fit_predict", spy)
    ex.compute_expert_features(texts, None, y, True, 2, {}, None, ("lex",), "oof", cte.CTE_EXPERT_CONFIG)
    oof = [(a, b) for a, b in calls if len(b) < n // 2]
    assert len(oof) == 10 and all(not (a & b) for a, b in oof)


def test_inverse_distance_neighbours():
    c = cte.CTE_EXPERT_CONFIG
    E = np.eye(3)
    E_te = np.array([[1.0, 0.1, 0.0]])
    nb = ex.NeighbourExpert(True, 2, dataclasses.replace(c, k_neighbors=2))
    p = nb.fit_predict(E, np.array([1, 0, 0]), E_te / np.linalg.norm(E_te))
    assert p.shape == (1, 1) and p[0, 0] > 0.9  # the closest neighbour (class 1) dominates under 1/d weights


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
    exp_title = len(experts) * per + (len(cte.NUMERACY_NAMES) if num else 0)
    assert res.features["title"].shape == (330, exp_title)
    assert res.features["review"].shape == (330, exp_title + len(experts) * per)  # + "__all__" view
    assert res.base_result is None and res.embedding_result.variant == "frozen"
    assert res.diagnostics["engine_version"] == "cte/1"
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
    args = (texts, None, y, is_cls, 2, extra, None, ("lex",), "oof", cte.CTE_EXPERT_CONFIG)
    s, p = ex.compute_expert_features(*args, n_jobs=1), ex.compute_expert_features(*args, n_jobs=3)
    assert s.keys() == p.keys() and all(s[k].keys() == p[k].keys() for k in s)
    # BLAS thread count differs inside workers -> float noise only (measured ~1e-17); each mode is bit-reproducible
    assert all(np.allclose(s[k][f], p[k][f], rtol=0, atol=1e-10) for k in s for f in s[k])
    p2 = ex.compute_expert_features(*args, n_jobs=3)
    assert all(np.array_equal(p[k][f], p2[k][f]) for k in p for f in p[k])

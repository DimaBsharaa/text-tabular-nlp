"""CPU tests for P5 experts (PROGRESS.md P5): OOF never sees own label, determinism, in-sample leaks, PCA on train
only, tiny end-to-end for every P5 method."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import experts as ex
from multabench.ours import stacking as st
from multabench.ours import tar_engine as te

CPU = torch.device("cpu")
BASE = te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16)


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(BASE, lower_cache_dir=str(tmp_path / "lower"))


def _texts(n, seed=0):
    rng = np.random.default_rng(seed)
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "flimsy"]
    return [" ".join(rng.choice(words, 6)) + f" uid{i}x" for i in range(n)]


def test_oof_lexical_never_fits_on_the_rows_it_predicts(monkeypatch):
    n = 120
    texts = {"review": _texts(n)}
    y = np.random.default_rng(1).normal(size=n)
    calls, real = [], ex.LexicalExpert.fit_predict

    def spy(self, tr, y_tr, te_):
        calls.append((set(tr), set(te_)))
        return real(self, tr, y_tr, te_)
    monkeypatch.setattr(ex.LexicalExpert, "fit_predict", spy)
    ex.compute_expert_features(texts, None, y, False, 1, {"test": {"review": _texts(10, seed=9)}}, None, ("lex",),
                               "oof", ex.ExpertConfig())
    pools = [set(texts["review"]), {f"review: {t}" for t in texts["review"]}]  # column view and __all__ view
    oof_calls = [(tr, te_) for tr, te_ in calls if any(te_ <= p for p in pools)]
    assert len(oof_calls) == 2 * 5  # 5 folds x 2 views (column + __all__ uses its own strings)
    for tr, te_ in oof_calls:
        assert not (tr & te_), "an out-of-fold expert was fit on rows it predicts"
    assert sum(len(te_) for tr, te_ in oof_calls[:5]) == n  # each row predicted exactly once per view


def test_insample_lexical_leaks_and_oof_does_not():
    """Unique tokens per row + random labels: in-sample ridge memorizes, out-of-fold cannot (ablation premise)."""
    n = 300
    texts = {"t": [f"tok{i} common words here" for i in range(n)]}
    y = np.random.default_rng(0).normal(size=n)
    cfg = ex.ExpertConfig(ridge_alphas=(0.1,))
    r2 = lambda p: 1 - np.mean((y - p[:, 0]) ** 2) / np.var(y)
    ins = ex.compute_expert_features(texts, None, y, False, 1, {}, None, ("lex",), "insample", cfg)["pool"]["lex:t"]
    oof = ex.compute_expert_features(texts, None, y, False, 1, {}, None, ("lex",), "oof", cfg)["pool"]["lex:t"]
    assert r2(ins) > 0.3 and r2(oof) < 0.05, (r2(ins), r2(oof))  # random labels: memorized in-sample only


def test_neighbours_and_determinism():
    n = 100
    rng = np.random.default_rng(2)
    E = rng.normal(size=(n, 16))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    y = rng.integers(0, 3, n)
    texts = {"a": _texts(n)}
    kw = dict(texts=texts, embs={"a": E}, y=y, is_cls=True, n_classes=3, extra_texts={"test": {"a": _texts(5, 3)}},
              extra_embs={"test": {"a": E[:5]}}, experts=("lex", "nbr"), mode="oof", cfg=ex.ExpertConfig(k_neighbors=5))
    a = ex.compute_expert_features(**kw)
    b = ex.compute_expert_features(**kw)
    assert set(a["pool"]) == {"lex:a", "lex:__all__", "nbr:a", "nbr:__all__"}
    for split in a:
        for f in a[split]:
            assert np.array_equal(a[split][f], b[split][f])
    assert a["pool"]["nbr:a"].shape == (n, 2) and a["test"]["lex:a"].shape == (5, 2)  # 3 classes -> 2 columns
    nb = a["pool"]["nbr:a"]
    assert (nb >= 0).all() and (nb.sum(axis=1) <= 1 + 1e-9).all()


def test_folds_deterministic_and_stratified():
    y = np.array([0] * 60 + [1] * 40)
    f1 = ex.expert_folds(100, y, True, ex.ExpertConfig())
    f2 = ex.expert_folds(100, y, True, ex.ExpertConfig())
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(f1, f2))
    assert all(abs(y[te_].mean() - 0.4) < 0.06 for _, te_ in f1)


def _table(n=330, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(7000, 7000 + n)
    price = rng.lognormal(3, 1, n)
    y_cls = pd.Series(rng.choice(np.array(["a", "b", "c"])[:n_classes], n), index=idx)
    y_reg = pd.Series(np.log(price) + rng.normal(0, 0.3, n), index=idx)
    x = pd.DataFrame({"review": _texts(n, seed), "title": rng.choice(["Phone", "Lamp", "Chair"], n),
                      "price": price}, index=idx)
    return x, y_cls, y_reg


@pytest.mark.parametrize("method", sorted(st.EXPERT_METHODS))
@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_every_p5_method_end_to_end(tmp_path, cfg, method, task):
    x, y_cls, y_reg = _table(n_classes=3 if task == "mul" else 2)
    y = y_reg if task == "reg" else y_cls
    x_fit, y_fit = x.iloc[:250], y.iloc[:250]
    extra = {"learner_val": (x.iloc[250:280], y.iloc[250:280]), "test": (x.iloc[280:], y.iloc[280:])}
    res = st.run_stacking(method, x_fit, y_fit, ["review", "title"], ["price"], [], task != "reg", extra, cfg, CPU,
                          str(tmp_path / "c"), "SYNTH", 0)
    per_view = 1 if task != "mul" else 2
    n_exp = len(st.EXPERT_METHODS[method][1])
    assert res.features["review"].shape == (330, 2 * n_exp * per_view)  # review + __all__ views
    assert res.features["title"].shape == (330, n_exp * per_view)
    assert np.isfinite(res.features["review"].to_numpy()).all()
    again = st.run_stacking(method, x_fit, y_fit, ["review", "title"], ["price"], [], task != "reg", extra, cfg, CPU,
                            str(tmp_path / "c"), "SYNTH", 0)
    assert all(again.diagnostics["expert_cache_hits"].values())
    assert np.array_equal(again.features["review"].to_numpy(), res.features["review"].to_numpy())


def test_learner_pca_fit_on_training_rows_only(tmp_path, cfg):
    """Through the real learner: PCA components equal a PCA fit on the training rows' embeddings alone."""
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
    model.configure_ours(variant="tar_experts", cfg=cfg, cache_dir=str(tmp_path / "c"), dataset="SYNTH", fold=0,
                         test=(x.iloc[380:], y.iloc[380:]))
    model.fit(x.iloc[:380], y.iloc[:380])
    enc = model.text_transformers["review"]
    train_rows = model.ours_learner_val[0].index.symmetric_difference(x.index[:380])  # learner's fit rows
    emb = model.tar_result.row_embeddings("review", np.asarray(train_rows))
    ref = PCA(n_components=30, random_state=42).fit(emb)
    assert np.allclose(np.abs(enc.encoder.pca.components_), np.abs(ref.components_), atol=1e-5)
    assert any(c.startswith("review_txt_pca_3") and c != "review_txt_pca_3" for c in model.model_.feature_names_)
    assert 0 <= model.score_all_metrics(X=x.iloc[380:], y=y.iloc[380:]).score <= 1

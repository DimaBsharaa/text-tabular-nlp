"""CPU tests for P3 stacking: cross-fitted head predictions as features."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import stacking as st
from multabench.ours import tar_engine as te

CPU = torch.device("cpu")
BASE = te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16)


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(BASE, lower_cache_dir=str(tmp_path / "lower"))


def _table(n=400, seed=0, n_classes=2):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(5000, 5000 + n)
    price = rng.lognormal(3, 1, n)
    labels = np.array(["a", "b", "c"])[:n_classes]
    y_cls = pd.Series(rng.choice(labels, n), index=idx)
    y_reg = pd.Series(np.log(price) + rng.normal(0, 0.3, n), index=idx)
    x = pd.DataFrame({"review": [f"{rng.choice(['great', 'bad'])} item {i}" for i in range(n)],
                      "title": rng.choice(["Phone", "Lamp", "Chair"], n), "price": price,
                      "brand": rng.choice(["acme", "globex", "initech"], n)}, index=idx)
    return x, y_cls, y_reg


def _split(x, y):  # pool 330: every fold trains on 220 rows (upstream regression needs >= ~200), learner_val 20, test 50
    return x.iloc[:330], y.iloc[:330], {"learner_val": (x.iloc[330:350], y.iloc[330:350]),
                                        "test": (x.iloc[350:], y.iloc[350:])}


class _Spy:
    """Wraps tar_engine.run_tar to record which rows each call trains on and which it only predicts."""
    def __init__(self, monkeypatch):
        self.calls, real = [], te.run_tar

        def spy(x_fit, y_fit, text_cols, is_cls, extra, *a, **kw):
            self.calls.append({"train": set(x_fit.index), "extra": {k: set(v[0].index) for k, v in extra.items()},
                               "variant": kw.get("variant")})
            return real(x_fit, y_fit, text_cols, is_cls, extra, *a, **kw)
        monkeypatch.setattr(te, "run_tar", spy)


# (2) the fold split is deterministic; folds are disjoint and cover the pool
def test_folds_deterministic_disjoint_covering():
    x, y, _ = _table()
    pool = x.index[:330]
    a = st.xfit_folds(pool, y, True)
    b = st.xfit_folds(pool, y, True)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert len(a) == 3 and sum(len(f) for f in a) == len(pool)
    assert np.array_equal(np.sort(np.concatenate(a)), np.sort(np.asarray(pool)))
    assert all(len(set(a[i]) & set(a[j])) == 0 for i in range(3) for j in range(i + 1, 3))
    # stratified for classification: class shares per fold close to the pool's
    share = y.loc[pool].eq("a").mean()
    assert all(abs(y.loc[f].eq("a").mean() - share) < 0.05 for f in a)


# (1) no training row's prediction comes from a model that saw its label; (3) folds run tar_repro's code path
@pytest.mark.parametrize("task", ["cls", "reg"])
def test_xfit_never_predicts_seen_rows_and_uses_tar_repro(tmp_path, cfg, monkeypatch, task):
    x, y_cls, y_reg = _table()
    y = y_cls if task == "cls" else y_reg
    x_fit, y_fit, extra = _split(x, y)
    spy = _Spy(monkeypatch)
    res = st.run_stacking("xfit_pred", x_fit, y_fit, ["review", "title"], ["price", "brand"], ["brand"],
                          task == "cls", extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    pool = set(x_fit.index)
    fold_calls = [c for c in spy.calls if "heldout" in c["extra"]]
    assert len(fold_calls) == 3 and all(c["variant"] == "tar_repro" for c in fold_calls)
    held_union = set()
    for c in fold_calls:
        assert not (c["train"] & c["extra"]["heldout"]), "a fold model trained on rows it predicts"
        assert c["train"] | c["extra"]["heldout"] == pool
        assert not (c["train"] & (c["extra"]["test"] | c["extra"]["learner_val"]))
        held_union |= c["extra"]["heldout"]
    assert held_union == pool
    # (3) the fold model is exactly tar_repro on that subset: a direct run_tar on the same rows hits the same cache
    sub = sorted(fold_calls[0]["train"])
    sub_extra = {"heldout": (x_fit.loc[sorted(fold_calls[0]["extra"]["heldout"])][["review", "title"]],
                             y_fit.loc[sorted(fold_calls[0]["extra"]["heldout"])]),
                 **{k: (v[0][["review", "title"]], v[1]) for k, v in extra.items()}}
    direct = te.run_tar(x_fit.loc[sub][["review", "title"]], y_fit.loc[sub], ["review", "title"], task == "cls",
                        sub_extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0, variant="tar_repro")
    assert direct.cache_hit and direct.cache_key == res.diagnostics["fold_models"][0]["cache_key"]
    assert res.diagnostics["folds_cover_pool"]


# (4) all four variants end to end; feature shapes
@pytest.mark.parametrize("method", ["insample_pred", "xfit_pred", "frozen_xfit_pred", "xfit_offset_pred"])
@pytest.mark.parametrize("task,n_feat", [("bin", 1), ("mul", 2), ("reg", 2)])
def test_stacking_methods_end_to_end(tmp_path, cfg, method, task, n_feat):
    x, y_cls, y_reg = _table(n_classes=3 if task == "mul" else 2)
    y = y_reg if task == "reg" else y_cls
    x_fit, y_fit, extra = _split(x, y)
    res = st.run_stacking(method, x_fit, y_fit, ["review", "title"], ["price", "brand"], ["brand"], task != "reg",
                          extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    for col in ("review", "title"):
        f = res.features[col]
        assert f.shape == (400, n_feat) and np.isfinite(f.to_numpy()).all()
        assert set(f.index) == set(x.index)
        if task != "reg":
            assert ((f.to_numpy() >= 0) & (f.to_numpy() <= 1)).all()
    q = res.diagnostics["pred_feature_quality"]
    assert {"train_rows", "test", "gap_train_minus_test"} <= set(q)
    if method == "frozen_xfit_pred":
        assert res.embedding_result.variant == "frozen"
    if method == "insample_pred":
        assert res.diagnostics["fold_models"] == []
    else:
        assert len(res.diagnostics["fold_models"]) == 3


def test_frozen_embeddings_equal_untrained_tar_encoder(tmp_path, cfg):
    """frozen_xfit_pred's embeddings are vanilla e5 (fresh LoRA = no-op), matching upstream's frozen encoder."""
    from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
    x, y, _ = _table(60)
    res = st.frozen_embeddings(x[["review", "title"]], {}, ["review", "title"], cfg, CPU)
    vanilla, tok = get_vanilla_e5(CPU)
    theirs = encode_texts_with_e5(texts=te.encoding_texts(x, "review"), col_name="review", model=vanilla,
                                  tokenizer=tok, device=CPU)
    ours = res.row_embeddings("review", x.index, texts=te.encoding_texts(x, "review"))
    assert np.abs(ours - theirs).max() < 1e-3


def test_learner_receives_prediction_columns(tmp_path, cfg):
    """Through the real learner path (upstream TabularModel + our mixin): the appended columns reach CatBoost."""
    from multabench.datasets.objects import SupervisedTask
    from multabench.ours.learners import make_learner
    rng = np.random.default_rng(1)
    n = 420
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "flimsy", "bright", "noisy"]
    x = pd.DataFrame({"review": [" ".join(rng.choice(words, 12)) + f" #{i}" for i in range(n)],
                      "price": rng.lognormal(3, 1, n), "brand": rng.choice(["acme", "globex", "initech"], n)},
                     index=pd.RangeIndex(n))
    y = pd.Series(np.where(x["price"] > x["price"].median(), "high", "low"), index=x.index)
    x_tr, y_tr, x_te, y_te = x.iloc[:380], y.iloc[:380], x.iloc[380:], y.iloc[380:]
    model = make_learner("catboost")(problem_type=SupervisedTask.BINARY, device=CPU, tune_e5=False)
    model.configure_ours(variant="xfit_pred", cfg=cfg, cache_dir=str(tmp_path / "tar"), dataset="SYNTH", fold=0,
                         test=(x_te, y_te))
    model.fit(x_tr, y_tr)
    assert "review" in model.text_features
    names = model.model_.feature_names_
    assert "review_txt_pca_30" in names and "review_txt_pca_31" not in names  # binary: 1 prediction column
    imp = model.stack_feature_importance()
    assert set(imp["pred_columns"]) == {"review_txt_pca_30"} and 0 <= imp["pred_share"] <= 1
    assert 0.0 <= model.score_all_metrics(X=x_te, y=y_te).score <= 1.0


def test_companion_reuses_the_first_methods_device(tmp_path, monkeypatch):
    """tabstar's get_device() refuses a GPU already holding memory (ours, after xfit_pred): the companion
    (frozen_xfit_pred) must reuse the device chosen for the first method instead of asking again."""
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "scripts"))
    import run_experiment as rx
    seen, sentinel = [], object()

    def fake_run_one(margs):
        seen.append((margs.method, margs._device))
        margs._device = margs._device or sentinel
        return {"score": 0.5, "reference": {}, "method": margs.method}
    monkeypatch.setattr(rx, "run_one", fake_run_one)
    monkeypatch.setattr(rx, "existing_result_ok", lambda p: False)
    monkeypatch.setattr(_sys, "argv", ["run_experiment.py", "--method", "xfit_pred", "--learner", "catboost",
                                       "--dataset", "X", "--fold", "0", "--out", str(tmp_path)])
    assert rx.main() == 0
    assert [m for m, _ in seen] == ["xfit_pred", "frozen_xfit_pred"]
    assert seen[0][1] is None and seen[1][1] is sentinel


def test_frozen_base_loaded_if_cached_else_skipped(tmp_path, cfg):
    """P4: frozen_xfit_pred loads the full-pool tar_repro result for diagnostics only if it is cached; else skips it."""
    x, y, _ = _table()
    x_fit, y_fit, extra = _split(x, y)
    cols = ["review", "title"]
    root = str(tmp_path / "tar")
    x_t, extra_t = st._tar_inputs(x_fit, cols, extra)
    assert st.cached_tar_repro(x_t, extra_t, cfg, CPU, root, "SYNTH", 0) is None
    r0 = st.run_stacking("frozen_xfit_pred", x_fit, y_fit, cols, ["price", "brand"], ["brand"], True, extra, cfg, CPU,
                         root, "SYNTH", 0)
    assert r0.base_result is None and r0.embedding_result.variant == "frozen"
    tar = te.run_tar(x_t, y_fit, cols, True, extra_t, cfg, CPU, root, "SYNTH", 0, variant="tar_repro")
    assert st.tar_repro_cache_key(x_t, extra_t, cfg, CPU, "SYNTH", 0) == tar.cache_key  # same formula as run_tar
    r1 = st.run_stacking("frozen_xfit_pred", x_fit, y_fit, cols, ["price", "brand"], ["brand"], True, extra, cfg, CPU,
                         root, "SYNTH", 0)
    assert r1.base_result is not None and r1.base_result.cache_key == tar.cache_key
    for c in cols:  # the method's own output does not depend on whether the base was available
        assert np.allclose(r0.features[c].to_numpy(), r1.features[c].to_numpy())

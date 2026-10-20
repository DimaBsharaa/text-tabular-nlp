"""CPU tests for P7 (multabench/ours/p7.py): every variant end to end, OOF leakage of the new label-aware experts,
label-free groups fitted on training rows only, cte/3 unchanged."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import cte, p7
from multabench.ours import experts as ex
from multabench.ours import stacking as st
from multabench.ours import tar_engine as te

CPU = torch.device("cpu")


@pytest.fixture
def cfg(tmp_path):
    return te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16, lower_cache_dir=str(tmp_path / "lower"))


def _table(n=300, task="bin", seed=0):
    rng = np.random.default_rng(seed)
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "noisy"]
    x = pd.DataFrame({"review": [" ".join(rng.choice(words, 6)) + f" {rng.integers(1990, 2020)}" for _ in range(n)],
                      "title": rng.choice(["Phone A", "Lamp B", "Chair C"], n), "price": rng.lognormal(3, 1, n),
                      "brand": rng.choice(["x", "y", "z"], n)}, index=pd.RangeIndex(7000, 7000 + n))
    good = x["review"].str.count("great|lovely").to_numpy()
    if task == "reg":
        y = pd.Series(np.log(x["price"]) + good + rng.normal(0, 0.3, n), index=x.index)
    else:
        k = 2 if task == "bin" else 3
        y = pd.Series(np.array(["a", "b", "c"])[np.clip(good, 0, k - 1)], index=x.index)
    return x, y


@pytest.mark.parametrize("method", sorted(p7.P7_METHODS))
@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_every_p7_variant_end_to_end(tmp_path, cfg, method, task):
    x, y = _table(task=task)
    extra = {"learner_val": (x.iloc[240:270], y.iloc[240:270]), "test": (x.iloc[270:], y.iloc[270:])}
    res = st.run_stacking(method, x.iloc[:240], y.iloc[:240], ["review", "title"], ["brand", "price"], ["brand"],
                          task != "reg", extra, cfg, CPU, str(tmp_path / "c"), "SYNTH", 0)
    use_cte, groups = p7.P7_METHODS[method]
    assert res.diagnostics["engine_version"] == "p7/1"
    f = pd.concat([res.features[c] for c in ("review", "title")], axis=1)
    assert len(f) == 300 and f.index.is_unique and set(f.index) == set(x.index)
    prefixes = {c.split(":")[0] for c in f.columns}
    assert set(groups) <= prefixes and (("lex" in prefixes) == use_cte)
    if method == "frozen0":
        assert f.shape[1] == 0  # = our Joint Frozen: PCA only
    if use_cte:  # the cte part is exactly cte's features
        base = st.run_stacking("cte", x.iloc[:240], y.iloc[:240], ["review", "title"], ["brand", "price"], ["brand"],
                               task != "reg", extra, cfg, CPU, str(tmp_path / "c"), "SYNTH", 0)
        for c in ("review", "title"):
            assert res.features[c][base.features[c].columns].equals(base.features[c])


@pytest.mark.parametrize("is_cls", [True, False])
def test_xfit_linear_oof_never_sees_its_own_label(monkeypatch, is_cls):
    rng = np.random.default_rng(1)
    n = 150
    X = rng.normal(size=(n, 20))
    y = rng.integers(0, 3, n) if is_cls else rng.normal(size=n)
    one = dataclasses.replace(cte.CTE_CONFIG, ridge_alphas=(1.0,))
    folds = ex.expert_folds(n, y, is_cls, one.expert_config())
    monkeypatch.setattr(p7, "expert_folds", lambda *a, **k: folds)
    held = folds[0][1]
    y2 = y.copy()
    y2[held] = (y[held] + 1) % 3 if is_cls else y[held] * 5 + 2
    a, ea, _ = p7.xfit_linear(X, {"test": X[:10]}, y, is_cls, 3 if is_cls else 1, one)
    b, eb, _ = p7.xfit_linear(X, {"test": X[:10]}, y2, is_cls, 3 if is_cls else 1, one)
    assert np.array_equal(a[held], b[held]) and not np.array_equal(a, b) and not np.array_equal(ea["test"], eb["test"])


def test_row_tokenizer_label_free_and_fitted_on_train():
    x, y = _table(task="reg")
    tok = p7.RowTokenizer(x.iloc[:200], ["review", "title"], ["brand", "price"], ["brand"])
    s = tok.transform(x.iloc[200:203])
    assert all(t.startswith("price__q") and "brand__" in t and "review: " in t for t in s)
    edges = np.quantile(x["price"].iloc[:200], np.linspace(0.1, 0.9, 9))
    assert np.allclose(tok.edges["price"], np.unique(edges))  # deciles of the training rows only
    v = p7.verbalize_rows(x.iloc[:1], ["review"], ["brand", "price"])[0]
    assert v.startswith("brand: ") and v.index("price: ") < v.index("review: ")  # structure first


def test_cte3_golden_still_holds():
    import sys
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(abspath(__file__)), "golden"))
    import make_cte3_golden as g
    ref = np.load(g.OUT)
    new = g.all_features()
    assert all(np.array_equal(new[k], ref[k]) for k in ref.files)


def test_median_filling_for_tabm_only(tmp_path, cfg):
    """E41: TabM (USE_MEDIAN_FILLING) gets the training-row median for NaN in our appended columns; CatBoost keeps NaN."""
    from multabench.datasets.objects import SupervisedTask
    from multabench.ours.learners import make_learner
    x, y = _table(n=300, task="reg")
    x["review"] = [t if i % 3 else "no digits here at all" for i, t in enumerate(x["review"])]  # numeracy NaN rows
    for name, expect_nan in (("catboost", True), ("tabm", False)):
        m = make_learner(name)(problem_type=SupervisedTask.REGRESSION, device=CPU, tune_e5=False)
        m.configure_ours(variant="cte_probe", cfg=cfg, cache_dir=str(tmp_path / "c"), dataset="SYNTH", fold=0,
                         test=(x.iloc[260:], y.iloc[260:]))
        m.fit(x.iloc[:260], y.iloc[:260])
        f = m.text_transformers["review"].features
        assert bool(f.isna().to_numpy().any()) == expect_nan, name
        if not expect_nan:  # the filled value is the median over the learner's training rows (no val / test rows)
            raw = m.stack_result.features["review"]
            col = raw.columns[raw.isna().any()][0]
            train_rows = m.ours_stack_inputs["x_fit"].index
            filled = f.loc[raw[col].isna(), col].unique()
            assert len(filled) == 1 and np.isclose(filled[0], raw.loc[train_rows, col].median())
        assert np.isfinite(m.score_all_metrics(X=x.iloc[260:], y=y.iloc[260:]).score)

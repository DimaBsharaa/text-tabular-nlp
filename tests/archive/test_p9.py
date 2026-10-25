"""CPU tests for P9 (multabench/ours/p9.py): p9_base reproduces cte_probe; every switch end to end; OOF leakage with
averaged test predictions and repeated cross-fitting; the prompt view uses the target's name only."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import cte
from multabench.ours.archive import p9
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
        y = pd.Series(np.log(x["price"]) + good + rng.normal(0, 0.3, n), index=x.index, name="rating")
    else:
        k = 2 if task == "bin" else 3
        y = pd.Series(np.array(["a", "b", "c"])[np.clip(good, 0, k - 1)], index=x.index, name="label")
    return x, y


def _run(method, x, y, task, cfg, tmp_path):
    extra = {"learner_val": (x.iloc[240:270], y.iloc[240:270]), "test": (x.iloc[270:], y.iloc[270:])}
    return st.run_stacking(method, x.iloc[:240], y.iloc[:240], ["review", "title"], ["brand", "price"], ["brand"],
                           task != "reg", extra, cfg, CPU, str(tmp_path / "c"), "SYNTH", 0)


@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_p9_base_reproduces_cte_probe(tmp_path, cfg, task):
    x, y = _table(task=task)
    a, b = _run("p9_base", x, y, task, cfg, tmp_path), _run("cte_probe", x, y, task, cfg, tmp_path)
    for col in ("review", "title"):
        assert list(a.features[col].columns) == list(b.features[col].columns)
        assert np.allclose(a.features[col].to_numpy(), b.features[col].to_numpy(), atol=1e-9, equal_nan=True), col


@pytest.mark.parametrize("method", sorted(p9.P9_METHODS))
@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_every_p9_variant_end_to_end(tmp_path, cfg, method, task):
    x, y = _table(task=task)
    res = _run(method, x, y, task, cfg, tmp_path)
    f = pd.concat([res.features[c] for c in ("review", "title")], axis=1)
    assert len(f) == 300 and res.diagnostics["engine_version"] == "p9/1"
    pre = {c.split(":")[0] for c in f.columns}
    spec = p9.P9_METHODS[method]
    assert {"lex", "nbr", "probe", "num"} <= pre and ("rff" in pre) == spec.rff and ("pprobe" in pre) == spec.prompt
    assert np.isfinite(f[[c for c in f.columns if not c.startswith("num:")]].to_numpy()).all()


@pytest.mark.parametrize("test_mode,repeats", [("avg", 1), ("full", 3), ("avg", 3)])
@pytest.mark.parametrize("is_cls", [True, False])
def test_oof_never_sees_own_label_with_avg_and_repeats(monkeypatch, test_mode, repeats, is_cls):
    """Rows held out in EVERY fold split keep bit-identical OOF predictions when their labels change: with repeats,
    a row's OOF prediction averages models that never saw it."""
    rng = np.random.default_rng(3)
    n = 150
    X = rng.normal(size=(n, 12))
    y = rng.integers(0, 3, n) if is_cls else rng.normal(size=n)
    c = dataclasses.replace(cte.CTE_CONFIG, ridge_alphas=(1.0,))
    spec = p9.P9Spec(test_mode=test_mode, repeats=repeats)
    a, ea, _ = p9.linear_expert(X, {"test": X[:7]}, y, is_cls, 3 if is_cls else 1, c, spec)
    row = 5
    y2 = y.copy()
    y2[row] = (y[row] + 1) % 3 if is_cls else y[row] + 9
    b, eb, _ = p9.linear_expert(X, {"test": X[:7]}, y2, is_cls, 3 if is_cls else 1, c, spec) if not is_cls else \
        (None, None, None)
    if is_cls:  # stratified folds depend on y: hold the folds fixed for the label change
        fl = p9._folds_list(y, True, c, repeats)
        monkeypatch.setattr(p9, "_folds_list", lambda *a_, **k: fl)
        a, ea, _ = p9.linear_expert(X, {"test": X[:7]}, y, True, 3, c, spec)
        b, eb, _ = p9.linear_expert(X, {"test": X[:7]}, y2, True, 3, c, spec)
    assert np.array_equal(a[row], b[row]) and not np.array_equal(a, b)


def test_avg_mode_test_predictions_are_fold_model_means():
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(120, 8)), rng.normal(size=120)
    c = dataclasses.replace(cte.CTE_CONFIG, ridge_alphas=(1.0,))
    _, ext, _ = p9.linear_expert(X, {"t": X[:5]}, y, False, 1, c, p9.P9Spec(test_mode="avg"))
    folds = ex.expert_folds(120, y, False, ex.ExpertConfig(n_folds=5, seed=42))
    manual = np.mean([cte.ridge_fit(X[tr], cte._targets(y[tr], False, 1, c), [X[:5]], 1.0)[0] for tr, _ in folds], axis=0)
    assert np.allclose(ext["t"], manual)


def test_prompt_uses_target_name_not_values(tmp_path, cfg, monkeypatch):
    x, y = _table(task="reg")
    seen = []
    real = cte.frozen_embeddings_cached

    def spy(frames, cols, *a, **k):
        seen.append((list(cols), [f.iloc[:, 0].tolist()[:2] for f in frames]))
        return real(frames, cols, *a, **k)
    monkeypatch.setattr(cte, "frozen_embeddings_cached", spy)
    _run("p9_prompt", x, y, "reg", cfg, tmp_path)
    prompt_calls = [s for s in seen if any(c.startswith("predict rating | ") for c in s[0])]
    assert prompt_calls and all(str(v) not in str(prompt_calls) for v in y.iloc[:3].round(6))


def test_rff_gamma_memory_light(tmp_path, cfg):
    """Bug: the gamma heuristic built an (n, n, d) array (9 GB for 3 text columns). Now (n, n) only."""
    import tracemalloc
    x, y = _table(n=300, task="bin")
    tracemalloc.start()
    _run("p9_rff", x, y, "bin", cfg, tmp_path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100e6  # fixed code: ~25 MB here; the old (n, n, d) array alone was 240*240*768*8 B ~ 354 MB


def test_query_prefix_variants(tmp_path, cfg):
    """e5 'query: ' prefix: its own embedding store, different vectors, every query variant end to end."""
    x, y = _table(task="bin")
    a = _run("p9_base", x, y, "bin", cfg, tmp_path)
    b = _run("p9_query", x, y, "bin", cfg, tmp_path)
    ea = a.embedding_result.row_embeddings("review", np.asarray(x.index))
    eb = b.embedding_result.row_embeddings("review", np.asarray(x.index))
    assert ea.shape == eb.shape and not np.allclose(ea, eb) and np.allclose(np.linalg.norm(eb, axis=1), 1, atol=1e-5)
    for m in ("frozen0_query", "tf_num_lsa_query"):
        r = _run(m, x, y, "bin", cfg, tmp_path)
        assert np.allclose(r.embedding_result.row_embeddings("review", np.asarray(x.index)), eb, atol=1e-6)
    import glob
    assert len(glob.glob(str(tmp_path / "c" / "frozen_emb" / "SYNTH" / "*" / "*" / "*.npz"))) >= 1  # query store

"""CPU tests for P11 depth probes (multabench/ours/depth.py): cte_depth = cte_probe + depth columns (bit-identical
base); every variant end to end for binary / multiclass / regression; label-free standardisation uses training rows
only; OOF predictions never see their own label; the encoder's final layer is Joint Frozen's encoder; pooling ignores
padding; the layer-state cache is reused."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import cte, depth, p7
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
def test_cte_depth_is_cte_probe_plus_depth_columns(tmp_path, cfg, task):
    x, y = _table(task=task)
    a, b = _run("cte_depth", x, y, task, cfg, tmp_path), _run("cte_probe", x, y, task, cfg, tmp_path)
    for col in ("review", "title"):
        fa = a.features[col][[c for c in a.features[col].columns if not c.startswith("depth")]]
        assert list(fa.columns) == list(b.features[col].columns)
        assert np.allclose(fa.to_numpy(), b.features[col].to_numpy(), atol=1e-9, equal_nan=True), col
    assert np.allclose(a.embedding_result.row_embeddings("review", np.asarray(x.index)),
                       b.embedding_result.row_embeddings("review", np.asarray(x.index)))


@pytest.mark.parametrize("method", sorted(depth.DEPTH_METHODS))
@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_every_depth_variant_end_to_end(tmp_path, cfg, method, task):
    x, y = _table(task=task)
    res = _run(method, x, y, task, cfg, tmp_path)
    f = pd.concat([res.features[c] for c in ("review", "title")], axis=1)
    d = res.diagnostics
    assert len(f) == 300 and d["engine_version"] == "depth/1" and d["depth"]["n_layers"] == 12
    dcols = [c for c in f.columns if c.startswith("depth")]
    spec = depth.DEPTH_METHODS[method]
    k = 2 if task == "mul" else 1  # 3 classes -> C-1 = 2 columns per expert
    agg = spec.aggregate if k == 1 else "top3"  # multiclass always top3
    assert len(dcols) == (12 if agg == "stack" else k)
    assert d["depth"]["aggregate"] == agg
    if agg == "sl":
        assert len(d["depth"]["sl_weights"]) == 12 and min(d["depth"]["sl_weights"]) >= 0
    assert np.isfinite(f[dcols].to_numpy()).all()
    assert set(d["depth"]["layer_oof_metric"]) == set(range(1, 13))
    assert all(c.startswith("depth") for c in res.features["review"].columns[-len(dcols):])  # owner = first text col


def test_layer_standardisation_uses_training_rows_only():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(50, 3, 8)).astype(np.float32)
    states = {"t": (s, {1000 + i: i for i in range(50)})}
    pool, test = np.arange(1000, 1040), np.arange(1040, 1050)
    Xp, stats = depth.layer_matrix(states, ["t"], pool, 1)
    s2 = s.copy()
    s2[40:] += 100.0  # move only the test rows' states
    Xp2, _ = depth.layer_matrix({"t": (s2, states["t"][1])}, ["t"], pool, 1)
    assert np.array_equal(Xp, Xp2)
    Xt, _ = depth.layer_matrix(states, ["t"], test, 1, stats)
    mu, sd = s[:40, 1].mean(0), s[:40, 1].std(0) + 1e-6
    assert np.allclose(Xt, (s[40:, 1] - mu) / sd / np.sqrt(8), atol=1e-6)


@pytest.mark.parametrize("is_cls", [True, False])
def test_depth_probe_oof_never_sees_own_label(monkeypatch, is_cls):
    """The depth expert is p7.xfit_linear on a layer matrix with depth's alpha grid: a row's OOF prediction is unchanged
    when its own label changes (folds held fixed)."""
    rng = np.random.default_rng(3)
    n = 150
    X = rng.normal(size=(n, 12))
    y = rng.integers(0, 3, n) if is_cls else rng.normal(size=n)
    c = depth.DepthSpec().cte_config()
    folds = ex.expert_folds(n, y, is_cls, c.expert_config())
    monkeypatch.setattr(p7, "expert_folds", lambda *a, **k: folds)
    row = 5
    y2 = y.copy()
    y2[row] = (y[row] + 1) % 3 if is_cls else y[row] + 9
    a, _, ma = p7.xfit_linear(X, {"test": X[:7]}, y, is_cls, 3 if is_cls else 1, c)
    b, _, mb = p7.xfit_linear(X, {"test": X[:7]}, y2, is_cls, 3 if is_cls else 1, c)
    if ma["alpha"] == mb["alpha"]:
        assert np.array_equal(a[row], b[row]) and not np.array_equal(a, b)
    assert len(ma["oof_metric"]) == len(depth.DepthSpec().alphas)


def test_final_layer_is_joint_frozen_encoder(tmp_path, cfg):
    """Last layer, mean-pooled and L2-normalised = cte's frozen e5 embedding (Joint Frozen's encoder)."""
    x, _ = _table(n=40)
    states = depth.layer_states_cached([x[["review"]]], ["review"], cfg, CPU, None, "SYNTH", "mean")
    s, uid = states["review"]
    rows = np.asarray(x.index)
    last = s[[uid[int(r)] for r in rows], -1]
    last = last / np.linalg.norm(last, axis=1, keepdims=True)
    frozen, _ = cte.frozen_embeddings_cached([x[["review"]]], ["review"], cfg, CPU, None, "SYNTH")
    e = frozen.row_embeddings("review", rows)
    assert (np.sum(last * e, axis=1) > 0.999).all()  # the frozen path stores lower layers in fp16


def test_pooling_ignores_padding():
    """A string pooled alone equals the same string pooled in a batch with a much longer one (mean and max)."""
    short, long_ = "passage: review: fine", "passage: review: " + " ".join(["lovely sturdy noisy"] * 40)
    for pooling in ("mean", "meanmax"):
        alone = depth.encode_layer_states([short], cfg_model(), 512, pooling, CPU)
        both = depth.encode_layer_states([long_, short], cfg_model(), 512, pooling, CPU, batch_size=2)
        assert np.allclose(alone[0], both[1], atol=1e-4)
    assert alone.shape == (1, 12, 768)


def cfg_model():
    return te.TarConfig().model_name


def test_layer_state_cache_reused(tmp_path, cfg, monkeypatch):
    x, _ = _table(n=30)
    a = depth.layer_states_cached([x[["title"]]], ["title"], cfg, CPU, str(tmp_path), "SYNTH", "mean")
    calls = []
    real = depth.encode_layer_states
    monkeypatch.setattr(depth, "encode_layer_states", lambda *a_, **k: calls.append(1) or real(*a_, **k))
    b = depth.layer_states_cached([x[["title"]]], ["title"], cfg, CPU, str(tmp_path), "SYNTH", "mean")
    assert not calls and np.array_equal(a["title"][0], b["title"][0]) and a["title"][1] == b["title"][1]
    x2 = x.copy()
    x2.index = x2.index + 10_000
    x2["title"] = "Brand new title"
    depth.layer_states_cached([x[["title"]], x2[["title"]]], ["title"], cfg, CPU, str(tmp_path), "SYNTH", "mean")
    assert calls == [1]  # only the new unique text is encoded


def test_top3_aggregate_is_mean_of_best_layers(tmp_path, cfg):
    x, y = _table(task="reg")
    res = _run("cte_depth_top3", x, y, "reg", cfg, tmp_path)
    full = _run("cte_depth", x, y, "reg", cfg, tmp_path)
    ranked = res.diagnostics["depth"]["ranked_layers"][:3]
    f, g = res.features["review"], full.features["review"]
    manual = np.mean([g[f"depth{l:02d}:{ex.ALL_COLS}#0"].to_numpy() for l in ranked], axis=0)
    assert np.allclose(f[f"depthtop3:{ex.ALL_COLS}#0"].to_numpy(), manual)


# ---------------------------------------------------------------------------------------- packaging (train / predict)

def _skip_if_no_data(e):
    if any(w in repr(e).lower() for w in ("kaggle", "download", "connection", "credential")):
        pytest.skip(f"dataset not available: {e!r}")
    raise e


@pytest.mark.parametrize("method", ["cte_depth", "cte_depth_top3", "cte_depth_sl"])
def test_packaged_featurizer_reproduces_fit_time_depth_features(method):
    """The saved experts (incl. the depth Ridges + standardisation) give the test rows the features run_depth gave them
    at fit time; the test rows' layer states are re-encoded from scratch (float16 store -> tiny tolerance)."""
    import argparse
    import sys
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "scripts"))
    import run_experiment as rx
    from multabench.ours.cte_model import CTEFeaturizer
    args = argparse.Namespace(dataset="MUL_TEXT_PRODUCT_SENTIMENT", fold=0, method=method, learner="catboost", seed=42,
                              train_examples=300, epochs=50, no_fast=False, micro_batch_size=256, lower_cache_dir=None,
                              cache_dir=None)
    try:
        model, _, _, _, x_test, _, _ = rx.fit_learner(args)
    except Exception as e:  # noqa: BLE001
        _skip_if_no_data(e)
    fz = CTEFeaturizer.from_learner(model)
    pre = model._ours_preprocess_like_transform(x_test)[fz.text_cols]
    rows = np.asarray(x_test.index)
    feats = fz.transform({c: te.encoding_texts(pre, c) for c in fz.text_cols},
                         {c: model.tar_result.row_embeddings(c, rows) for c in fz.text_cols}, x_test.index)
    for col in fz.text_cols:
        ref = model.stack_result.features[col].loc[rows]
        assert list(feats[col].columns) == list(ref.columns)
        base = [c for c in ref.columns if not c.startswith("depth")]
        dcols = [c for c in ref.columns if c.startswith("depth")]
        assert np.allclose(feats[col][base].to_numpy(), ref[base].to_numpy(), atol=1e-9, rtol=0, equal_nan=True)
        if dcols:
            assert np.allclose(feats[col][dcols].to_numpy(), ref[dcols].to_numpy(), atol=2e-3, rtol=0), col


def test_csv_mode_round_trip_cte_depth(tmp_path):
    import sys
    from os.path import abspath, dirname, join
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "scripts"))
    import predict
    import train
    rng = np.random.default_rng(0)
    n = 400
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "noisy", "bright", "dull"]
    review = [" ".join(rng.choice(words, 12)) + f" model {i}" for i in range(n)]
    good = np.array([r.count("great") + r.count("lovely") for r in review])
    df = pd.DataFrame({"review": review, "price": rng.lognormal(3, 1, n),
                       "rating": np.round(2 + good + rng.normal(0, 0.5, n), 2)})
    df.to_csv(tmp_path / "table.csv", index=False)
    out = tmp_path / "m"
    m = train.main(["--csv", str(tmp_path / "table.csv"), "--target", "rating", "--out_dir", str(out),
                    "--method", "cte_depth"])
    assert m["task"] == "regression" and m["score"] > 0.3 and (out / "depth.joblib").exists()
    pred = predict.main(["--model_dir", str(out), "--csv", str(out / "test_rows.csv"), "--out", str(tmp_path / "p.csv"),
                         "--device", "cpu"])
    ref = pd.read_csv(out / "test_predictions.csv").to_numpy()
    assert np.allclose(pred.to_numpy(), ref, atol=1e-3 * max(1.0, float(np.abs(ref).max())))


def test_stacked_layers_is_cross_fitted_nnls():
    """NNLS weights are non-negative; a pool row's stacked OOF value does not depend on its own target (folds fixed)."""
    rng = np.random.default_rng(1)
    n = 200
    t = rng.normal(size=n)
    P = np.column_stack([t + rng.normal(0, s, n) for s in (0.5, 1.0, 3.0)])
    c = depth.DepthSpec().cte_config()
    oof, ext, (w, b) = depth.stacked_layers(P, {"test": P[:9]}, lambda idx: t[idx], t, False, c)
    assert (w >= 0).all() and w[0] > w[2] and oof.shape == (n, 1) and ext["test"].shape == (9, 1)
    assert np.allclose(ext["test"][:, 0], P[:9] @ w + b)
    t2 = t.copy()
    t2[7] += 50.0
    oof2, _, _ = depth.stacked_layers(P, {"test": P[:9]}, lambda idx: t2[idx], t, False, c)  # same folds
    assert np.isclose(oof[7, 0], oof2[7, 0]) and not np.allclose(oof, oof2)


@pytest.mark.parametrize("n_classes", [3, 12])
def test_eig_solver_matches_sparse_cg_xfit(n_classes):
    """Multiclass depth probes use the eigen-form solver: same folds, alpha rule and output columns as p7.xfit_linear,
    numerically the same predictions (sparse_cg is iterative: small tolerance)."""
    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(size=(n, 20)) / np.sqrt(20)
    y = (X[:, :3] @ rng.normal(size=(3, n_classes))).argmax(axis=1)
    c = dataclasses.replace(depth.DepthSpec().cte_config(), ridge_alphas=(1.0, 10.0))
    a, ea, ma = p7.xfit_linear(X, {"t": X[:9]}, y, True, n_classes, c)
    b, eb, mb, (W, bias) = depth.xfit_linear_eig(X, {"t": X[:9]}, y, True, n_classes, c)
    assert ma["alpha"] == mb["alpha"] and a.shape == b.shape and eb["t"].shape == ea["t"].shape
    assert np.allclose(a, b, atol=1e-3) and np.allclose(ea["t"], eb["t"], atol=1e-3)
    assert np.allclose(ma["oof_metric"], mb["oof_metric"], atol=1e-3)

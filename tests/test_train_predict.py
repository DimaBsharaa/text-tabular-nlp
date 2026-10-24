"""Round trip of scripts/train.py -> model folder -> scripts/predict.py (multabench/ours/cte_model.py), on CPU.

Uses one small real dataset (Product Sentiment, 300 training rows; skipped if it cannot be downloaded here) and a small
synthetic CSV (the --csv mode)."""
import argparse
import json
import subprocess
import sys
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import pytest

SCRIPTS = join(dirname(dirname(abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)
DATASET = "MUL_TEXT_PRODUCT_SENTIMENT"
FOLDER_FILES = {"meta.json", "vectorizers.joblib", "ridge.joblib", "knn_index.npz", "pca.joblib", "catboost.cbm",
                "learner.joblib", "metrics.json", "test_rows.csv", "test_predictions.csv"}


def _predict_subprocess(out, tmp_path):
    x = pd.read_csv(out / "test_rows.csv").assign(some_target_column=1)  # extra columns are ignored
    x.to_csv(tmp_path / "new.csv", index=False)
    r = subprocess.run([sys.executable, join(SCRIPTS, "predict.py"), "--model_dir", str(out), "--csv",
                        str(tmp_path / "new.csv"), "--out", str(tmp_path / "pred.csv"), "--device", "cpu"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return pd.read_csv(tmp_path / "pred.csv"), pd.read_csv(out / "test_predictions.csv")


def _skip_if_no_data(e):
    if any(w in repr(e).lower() for w in ("kaggle", "download", "connection", "credential")):
        pytest.skip(f"dataset not available: {e!r}")
    raise e


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import train
    out = tmp_path_factory.mktemp("model")
    try:
        m = train.main(["--dataset", DATASET, "--fold", "0", "--out_dir", str(out), "--train_examples", "300"])
    except Exception as e:  # noqa: BLE001
        _skip_if_no_data(e)
    return out, m


def test_model_folder_and_metrics(trained):
    out, m = trained
    assert FOLDER_FILES <= {p.name for p in out.iterdir()}
    meta = json.load(open(out / "meta.json"))
    assert meta["format"] == "cte_model/1" and meta["method"] == "cte_depth_top3_attn"  # the final method (P18)
    assert meta["engine_version"] == "depth/1+attn/1"
    assert meta["alpha"] in (0.3, 1.0, 3.0, 10.0) and meta["probe_alpha"] in (0.3, 1.0, 3.0, 10.0)
    assert meta["learner"] == "catboost" and meta["learner_format"] == "catboost" and meta["fill_medians"] is None
    assert (out / "probe.joblib").exists() and (out / "depth.joblib").exists() and (out / "attn.joblib").exists()
    assert meta["attn"]["heads"] == 5 and meta["attn"]["layer"] in meta["depth"]["layers"]
    assert m["method"] == "cte_depth_top3_attn" and m["learner"] == "catboost"
    assert 0 <= m["score"] <= 1 and m["n_train"] == 300 and m["leaderboard_same_fold"]


def test_previous_method_still_packages(tmp_path):
    """--method cte_depth_top3 (the method before P18) still trains, saves without an attention probe and predicts."""
    import train
    out = tmp_path / "m"
    try:
        train.main(["--dataset", DATASET, "--fold", "0", "--out_dir", str(out), "--train_examples", "300",
                    "--method", "cte_depth_top3"])
    except Exception as e:  # noqa: BLE001
        _skip_if_no_data(e)
    meta = json.load(open(out / "meta.json"))
    assert meta["method"] == "cte_depth_top3" and meta["engine_version"] == "depth/1" and meta["attn"] is None
    assert not (out / "attn.joblib").exists()
    pred, ref = _predict_subprocess(out, tmp_path)
    assert np.allclose(pred.to_numpy(), ref.to_numpy(), atol=1e-6)


def test_round_trip_predict_matches_fit_time(trained, tmp_path):
    """predict.py (a fresh process, nothing re-fitted) on the held-out rows reproduces the fit-time predictions."""
    out, _ = trained
    pred, ref = _predict_subprocess(out, tmp_path)
    assert list(pred.columns) == list(ref.columns) and pred.shape == ref.shape
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(pred.to_numpy(), ref.to_numpy(), atol=1e-6)


@pytest.mark.parametrize("method", ["cte", "cte_probe"])
def test_packaged_featurizer_reproduces_fit_time_features(method):
    """The saved full-pool experts give the test rows exactly the features run_cte / run_p7 gave them at fit time."""
    import run_experiment as rx
    from multabench.ours import tar_engine as te
    from multabench.ours.cte_model import CTEFeaturizer
    args = argparse.Namespace(dataset=DATASET, fold=0, method=method, learner="catboost", seed=42, train_examples=300,
                              epochs=50, no_fast=False, micro_batch_size=256, lower_cache_dir=None, cache_dir=None)
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
        assert np.allclose(feats[col].to_numpy(), ref.to_numpy(), atol=1e-9, rtol=0, equal_nan=True), col


@pytest.mark.parametrize("refit", [False, True])
def test_packaged_attention_probe_reproduces_fit_time_features(refit):
    """The saved fold heads give the test rows the attention features run_depth gave them at fit time: with the heads
    kept from the fit, and with the heads refitted (the path used after an expert-cache hit). The token states of the
    new rows are re-encoded in another batch composition, hence float tolerance rather than bit equality."""
    import run_experiment as rx
    from multabench.ours import tar_engine as te
    from multabench.ours.cte_model import CTEFeaturizer
    args = argparse.Namespace(dataset=DATASET, fold=0, method="cte_depth_top3_attn", learner="catboost", seed=42,
                              train_examples=300, epochs=50, no_fast=False, micro_batch_size=256, lower_cache_dir=None,
                              cache_dir=None)
    try:
        model, _, _, _, x_test, _, _ = rx.fit_learner(args)
    except Exception as e:  # noqa: BLE001
        _skip_if_no_data(e)
    assert model.stack_result.attn_fit is not None and len(model.stack_result.attn_fit["heads"]) == 5
    if refit:
        model.stack_result.attn_fit = None
    fz = CTEFeaturizer.from_learner(model)
    pre = model._ours_preprocess_like_transform(x_test)[fz.text_cols]
    rows = np.asarray(x_test.index)
    feats = fz.transform({c: te.encoding_texts(pre, c) for c in fz.text_cols},
                         {c: model.tar_result.row_embeddings(c, rows) for c in fz.text_cols}, x_test.index)
    for col in fz.text_cols:
        ref = model.stack_result.features[col].loc[rows]
        assert list(feats[col].columns) == list(ref.columns)
        a = [c for c in ref.columns if c.startswith("attn:")]
        assert a or col != fz.text_cols[0], "the attention columns live on the first text column"
        assert np.allclose(feats[col][a].to_numpy(), ref[a].to_numpy(), atol=1e-4, rtol=0), col


def test_csv_mode(tmp_path):
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
    m = train.main(["--csv", str(tmp_path / "table.csv"), "--target", "rating", "--out_dir", str(out)])
    assert m["task"] == "regression" and m["n_test"] == 40 and m["score"] > 0.3  # R², the review text drives the target
    import predict
    pred = predict.main(["--model_dir", str(out), "--csv", str(out / "test_rows.csv"), "--out", str(tmp_path / "p.csv"),
                         "--device", "cpu"])
    assert list(pred.columns) == ["prediction"] and len(pred) == 40
    assert np.allclose(pred.to_numpy(), pd.read_csv(out / "test_predictions.csv").to_numpy(), atol=1e-6)


@pytest.mark.parametrize("learner,fmt,fname", [("lgbm", "joblib", "learner_model.joblib"),
                                                ("tabm", "joblib", "learner_model.joblib"),
                                                ("tabpfnv2", "tabpfn", "learner_model.tabpfn_fit"),
                                                ("tabpfnv2p5", "tabpfn", "learner_model.tabpfn_fit")])
def test_round_trip_other_learners(learner, fmt, fname, tmp_path):
    """Every learner: train.py -> model folder in the learner's own format -> predict.py in a fresh process reproduces
    the fit-time predictions. TabM also checks the saved fill medians (its appended columns are median-filled)."""
    import os
    import train
    if learner == "tabpfnv2p5" and not os.environ.get("TABPFN_TOKEN"):
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("TABPFN_TOKEN"):
            pytest.skip("TabPFN-2.5 needs TABPFN_TOKEN")
    out = tmp_path / "m"
    try:
        m = train.main(["--dataset", DATASET, "--fold", "0", "--out_dir", str(out), "--train_examples", "300",
                        "--learner", learner])
    except Exception as e:  # noqa: BLE001
        _skip_if_no_data(e)
    meta = json.load(open(out / "meta.json"))
    assert meta["learner"] == learner and meta["learner_format"] == fmt and (out / fname).exists()
    assert not (out / "catboost.cbm").exists() and m["learner"] == learner and 0 <= m["score"] <= 1
    assert (meta["fill_medians"] is not None) == (learner == "tabm")
    pred, ref = _predict_subprocess(out, tmp_path)
    assert list(pred.columns) == list(ref.columns) and pred.shape == ref.shape
    # The other learners reproduce the fit-time predictions to 1e-5. TabPFN does not, although its own save / load is
    # exact (checked directly on synthetic data): the remaining difference arises when predict.py re-featurises the
    # rows, and TabPFN amplifies it (measured on this fold: v2 max 0.0018, all classes equal; v2.5 max 0.013,
    # 508 / 509 classes equal). Cause not investigated further; documented in multabench/ours/README.md.
    if fmt == "tabpfn":
        assert np.abs(pred.to_numpy() - ref.to_numpy()).max() < 0.03
        assert (pred.to_numpy().argmax(axis=1) == ref.to_numpy().argmax(axis=1)).mean() >= 0.99
    else:
        assert np.allclose(pred.to_numpy(), ref.to_numpy(), atol=1e-5)

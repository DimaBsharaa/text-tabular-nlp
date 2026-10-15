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
    assert meta["format"] == "cte_model/1" and meta["engine_version"] == "cte/3" and meta["alpha"] in (0.3, 1.0, 3.0, 10.0)
    assert m["method"] == "cte" and 0 <= m["score"] <= 1 and m["n_train"] == 300 and m["leaderboard_same_fold"]


def test_round_trip_predict_matches_fit_time(trained, tmp_path):
    """predict.py (a fresh process, nothing re-fitted) on the held-out rows reproduces the fit-time predictions."""
    out, _ = trained
    x = pd.read_csv(out / "test_rows.csv").assign(some_target_column=1)  # extra columns are ignored
    x.to_csv(tmp_path / "new.csv", index=False)
    r = subprocess.run([sys.executable, join(SCRIPTS, "predict.py"), "--model_dir", str(out), "--csv",
                        str(tmp_path / "new.csv"), "--out", str(tmp_path / "pred.csv"), "--device", "cpu"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    pred, ref = pd.read_csv(tmp_path / "pred.csv"), pd.read_csv(out / "test_predictions.csv")
    assert list(pred.columns) == list(ref.columns) and pred.shape == ref.shape
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(pred.to_numpy(), ref.to_numpy(), atol=1e-6)


def test_packaged_featurizer_reproduces_fit_time_features():
    """The saved full-pool experts give the test rows exactly the CTE features run_cte gave them at fit time."""
    import run_experiment as rx
    from multabench.ours import tar_engine as te
    from multabench.ours.cte_model import CTEFeaturizer
    args = argparse.Namespace(dataset=DATASET, fold=0, method="cte", learner="catboost", seed=42, train_examples=300,
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

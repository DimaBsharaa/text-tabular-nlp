"""Smoke test of scripts/train.py + scripts/predict.py on one small real dataset (Product Sentiment, 300 rows).

Skipped when the dataset is not available locally (it is downloaded through kagglehub with Kaggle credentials)."""
import subprocess
import sys
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import pytest

SCRIPTS = join(dirname(dirname(abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)
DATASET = "MUL_TEXT_PRODUCT_SENTIMENT"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import train
    out = tmp_path_factory.mktemp("model")
    try:
        metrics = train.main(["--dataset", DATASET, "--fold", "0", "--out_dir", str(out), "--train_examples", "300"])
    except Exception as e:  # noqa: BLE001 - dataset not downloadable here
        if "kaggle" in repr(e).lower() or "download" in repr(e).lower() or "connection" in repr(e).lower():
            pytest.skip(f"dataset not available: {e!r}")
        raise
    return out, metrics


def test_train_writes_model_and_metrics(trained):
    out, m = trained
    assert (out / "model.joblib").exists() and (out / "metrics.json").exists()
    assert m["method"] == "cte" and m["engine_version"] == "cte/3" and 0 <= m["score"] <= 1
    assert m["n_train"] == 300 and m["feature_columns"]


def test_predict_on_new_rows_matches_fit_time_test_predictions(trained, tmp_path):
    """The test rows, given to predict.py as a new CSV, get the same predictions as at fit time (where their CTE
    features came from the same full-training-pool experts)."""
    import joblib
    out, m = trained
    model = joblib.load(out / "model.joblib")
    x_test, y_test = model.ours_test
    fit_time = np.asarray(model.predict(x_test))  # features precomputed at fit time for these rows
    csv = tmp_path / "new.csv"
    x_test.assign(target_should_be_ignored=y_test.to_numpy()).to_csv(csv, index=False)
    # a fresh process, as a user would run it (the learner class must unpickle without make_learner being called)
    r = subprocess.run([sys.executable, join(SCRIPTS, "predict.py"), "--model_dir", str(out), "--csv", str(csv),
                        "--out", str(tmp_path / "pred.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    pred = pd.read_csv(tmp_path / "pred.csv")
    classes = [str(c) for c in model.target_transformer.classes_]
    assert list(pred.columns) == [f"proba_{c}" for c in classes] and len(pred) == len(x_test)
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(pred.to_numpy(), fit_time, atol=1e-5)

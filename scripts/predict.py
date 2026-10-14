"""Predict with a model saved by scripts/train.py on new rows from a CSV (same feature columns as the dataset).

    python scripts/predict.py --model_dir models/book_price_f0 --csv new_rows.csv --out predictions.csv

Output: the input rows' predictions (regression: column "prediction"; classification: one "proba_<class>" column
per class), in the input order. A target column in the CSV, if present, is ignored (never read by the model).
New rows get CTE features exactly like test rows (multabench/ours/inference.py); nothing is re-fitted on them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("WANDB_MODE", "disabled")


def main(argv=None):
    import joblib
    import pandas as pd
    from multabench.ours.inference import predict_new
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache_dir", default=None, help="optional frozen-embedding cache (e.g. the one used in training)")
    a = p.parse_args(argv)
    model = joblib.load(join(a.model_dir, "model.joblib"))
    cols = json.load(open(join(a.model_dir, "metrics.json")))["feature_columns"]
    x = pd.read_csv(a.csv)
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise SystemExit(f"CSV is missing feature columns: {missing}")
    pred = predict_new(model, x[cols], cache_dir=a.cache_dir)
    pred.to_csv(a.out, index=False)
    print(f"{len(pred)} predictions -> {a.out}")
    return pred


if __name__ == "__main__":
    main()

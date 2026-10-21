"""Predict with a model folder saved by scripts/train.py on new rows from a CSV.

    python scripts/predict.py --model_dir models/book_price_f0 --csv new_rows.csv --out predictions.csv

The CSV needs the training feature columns (listed in <model_dir>/meta.json); extra columns, such as the target, are
ignored. Output, in the input order: regression -> column "prediction"; classification -> one "proba_<class>" column
per class. Nothing is re-fitted: the saved TF-IDF vectorizers, Ridge, kNN index, probes, PCA and learner (any of the 5,
whichever train.py used) are applied (multabench/ours/cte_model.py). The frozen e5-small-v2 encoder is loaded from
Hugging Face; a TabPFN folder also downloads the TabPFN weights again (TabPFN-2.5: TABPFN_TOKEN needed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("WANDB_MODE", "disabled")


def main(argv=None):
    import pandas as pd
    import torch
    from multabench.ours.cte_model import predict_folder
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    a = p.parse_args(argv)
    t0 = time.time()
    cols = json.load(open(join(a.model_dir, "meta.json")))["feature_columns"]
    x = pd.read_csv(a.csv)
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise SystemExit(f"CSV is missing feature columns: {missing}")
    device = torch.device("cuda" if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    pred = predict_folder(a.model_dir, x, device)
    pred.to_csv(a.out, index=False)
    print(f"{len(pred)} predictions -> {a.out} ({time.time() - t0:.0f} s)")
    return pred


if __name__ == "__main__":
    main()

"""Train our method (CTE + CatBoost, engine cte/3) end to end and save a model folder. No fine-tuning.

MulTaBench dataset + fold (same loading, split and code as the benchmark runs, scripts/run_experiment.py):
    python scripts/train.py --dataset REG_TEXT_BOOK_PRICE --fold 0 --out_dir models/book_price_f0
Any CSV with a target column (the same protocol: 10% test capped at 2,000, train capped at 10,000, chosen by --fold):
    python scripts/train.py --csv my_table.csv --target price --out_dir models/my_table

Writes <out_dir>/: the model folder (multabench/ours/cte_model.py: vectorizers, ridge, kNN index, PCA, CatBoost,
learner state, meta.json), metrics.json (test-split score; for MulTaBench also the leaderboard's numbers for the same
fold), test_rows.csv (the held-out rows, features only) and test_predictions.csv (their predictions).
Then: python scripts/predict.py --model_dir <out_dir> --csv new_rows.csv --out predictions.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from os.path import abspath, basename, dirname, join, splitext

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path[:0] = [REPO_ROOT, dirname(abspath(__file__))]
os.environ.setdefault("WANDB_MODE", "disabled")


def _task(y, task: str):
    import pandas as pd
    from multabench.datasets.objects import SupervisedTask
    if task == "auto":
        numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
        integral = numeric and bool((y.dropna() % 1 == 0).all())
        task = "regression" if numeric and not (integral and y.nunique() <= 20) else (
            "binary" if y.nunique() == 2 else "multiclass")
    return {"regression": SupervisedTask.REGRESSION, "binary": SupervisedTask.BINARY,
            "multiclass": SupervisedTask.MULTICLASS}[task]


def fit_csv(a, args):
    """CSV mode: the benchmark's subsampling + test split, then our learner (same as run_experiment.fit_learner)."""
    import pandas as pd
    from tabstar.preprocessing.splits import split_to_test
    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.ours.learners import make_learner
    from multabench.ours.tar_engine import TarConfig, set_seed
    import torch
    t0 = time.time()
    set_seed(args.seed)
    df = pd.read_csv(a.csv)
    if a.target not in df.columns:
        raise SystemExit(f"target column {a.target!r} not in {a.csv}")
    df = df[df[a.target].notna()]
    x, y = df.drop(columns=[a.target]), df[a.target]
    problem = _task(y, a.task)
    is_cls = problem.name != "REGRESSION"
    x, y = subsample_dataset(x=x, y=y, is_cls=is_cls, train_examples=args.train_examples, fold=args.fold)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=is_cls, fold=args.fold,
                                                     train_examples=args.train_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args._device = device
    model = make_learner("catboost")(problem_type=problem, device=device, verbose=False, tune_e5=False,
                                     pca_components=30, no_pca=False)
    cfg = TarConfig(seed=args.seed, eval_batch_size=256 if device.type == "cuda" else 16)
    model.configure_ours(variant="cte", cfg=cfg, cache_dir=args.cache_dir, dataset=args.dataset, fold=args.fold,
                         test=(x_test, y_test))
    model.fit(x_train, y_train)
    return model, type(model), x_train, y_train, x_test, y_test, t0


def main(argv=None) -> dict:
    import numpy as np
    import pandas as pd
    import run_experiment as rx
    from multabench.ours.cte_model import save_model
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="MulTaBench dataset id, e.g. REG_TEXT_BOOK_PRICE")
    src.add_argument("--csv", help="a CSV file with feature columns and a target column")
    p.add_argument("--target", help="--csv: name of the target column")
    p.add_argument("--task", default="auto", choices=("auto", "binary", "multiclass", "regression"))
    p.add_argument("--fold", type=int, default=0, help="which of the 5 benchmark splits (test rows) to hold out")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None, help="frozen-embedding / expert cache (default <out_dir>/_cache)")
    p.add_argument("--train_examples", type=int, default=rx.DEFAULTS["train_examples"])
    p.add_argument("--seed", type=int, default=rx.DEFAULTS["seed"])
    a = p.parse_args(argv)
    if a.csv and not a.target:
        p.error("--csv needs --target")
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.dataset or "CSV_" + splitext(basename(a.csv))[0]
    args = argparse.Namespace(dataset=name, fold=a.fold, method="cte", learner="catboost", seed=a.seed,
                              train_examples=a.train_examples, epochs=rx.DEFAULTS["epochs"], no_fast=False,
                              micro_batch_size=256, lower_cache_dir=None,
                              cache_dir=a.cache_dir or join(a.out_dir, "_cache"))
    model, learner_cls, x_train, y_train, x_test, y_test, t0 = (rx.fit_learner(args) if a.dataset else fit_csv(a, args))
    t_fit = time.time() - t0
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    fit_time_pred = np.asarray(model.predict(x_test))
    ref = rx.reference_scores(a.dataset, learner_cls.MODEL_NAME, a.fold) if a.dataset else {}
    feature_columns = [str(c) for c in x_train.columns]
    out = {"source": a.dataset or a.csv, "target": a.target, "fold": a.fold, "method": "cte", "learner": "catboost",
           "task": "classification" if model.is_cls else "regression", "score": float(metrics.score),
           "metric": "AUROC (binary) / macro one-vs-rest AUROC (multiclass) / R2 (regression)",
           "metrics": json.loads(json.dumps(asdict(metrics), default=float)), "leaderboard_same_fold": ref,
           "n_train": int(len(y_train)), "n_test": int(len(y_test)), "train_examples": a.train_examples,
           "seconds_fit": round(t_fit, 1), "git": rx.git_info()}
    save_model(model, a.out_dir, {"feature_columns": feature_columns, "source": out["source"], "fold": a.fold})
    x_test.to_csv(join(a.out_dir, "test_rows.csv"), index=False)
    labels = [str(k) for k in model.target_transformer.classes_] if model.is_cls else []
    if model.is_cls:
        probs = np.column_stack([1 - fit_time_pred, fit_time_pred]) if fit_time_pred.ndim == 1 else fit_time_pred
        pd.DataFrame(probs, columns=[f"proba_{k}" for k in labels]).to_csv(join(a.out_dir, "test_predictions.csv"), index=False)
    else:
        pd.DataFrame({"prediction": fit_time_pred}).to_csv(join(a.out_dir, "test_predictions.csv"), index=False)
    out["seconds_total"] = round(time.time() - t0, 1)
    json.dump(out, open(join(a.out_dir, "metrics.json"), "w"), indent=1)
    print(f"cte catboost {out['source']} fold {a.fold}: score {out['score']:.4f}"
          + (f" (leaderboard Joint TAR {ref['joint_tar']:.4f})" if ref.get("joint_tar") is not None else "")
          + f" | {out['seconds_total']:.0f} s -> {a.out_dir}")
    return out


if __name__ == "__main__":
    main()

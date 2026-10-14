"""Train our method end to end on one MulTaBench dataset + fold and save the fitted model (no fine-tuning needed).

    python scripts/train.py --dataset REG_TEXT_BOOK_PRICE --fold 0 --out_dir models/book_price_f0
    -> models/book_price_f0/model.joblib (the fitted learner: CTE features + CatBoost) and metrics.json (test-split
       score of this fold, with the leaderboard's Joint TAR / Joint Frozen for the same fold)

Same data loading, split, method and learner as scripts/run_experiment.py (shared code: run_experiment.fit_learner).
Default method: cte (P5 v2, engine cte/3) with CatBoost. Then predict on new rows with scripts/predict.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from os.path import abspath, dirname, join

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path[:0] = [REPO_ROOT, dirname(abspath(__file__))]
os.environ.setdefault("WANDB_MODE", "disabled")

PREDICTABLE = ("cte", "cte_lex", "cte_nbr", "cte_num", "catboost_native_text", "cte_plus_native", "cte_nb", "cte_residual")


def main(argv=None) -> dict:
    import joblib
    import run_experiment as rx
    from multabench.ours.variants import engine_version
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="MulTaBench dataset id, e.g. REG_TEXT_BOOK_PRICE")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--method", default="cte", choices=PREDICTABLE)
    p.add_argument("--learner", default="catboost", choices=("catboost", "lgbm"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None, help="frozen-embedding / expert cache (default <out_dir>/_cache)")
    p.add_argument("--train_examples", type=int, default=rx.DEFAULTS["train_examples"])
    p.add_argument("--seed", type=int, default=rx.DEFAULTS["seed"])
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    args = argparse.Namespace(dataset=a.dataset, fold=a.fold, method=a.method, learner=a.learner, seed=a.seed,
                              train_examples=a.train_examples, epochs=rx.DEFAULTS["epochs"], no_fast=False,
                              micro_batch_size=256, lower_cache_dir=None,
                              cache_dir=a.cache_dir or join(a.out_dir, "_cache"))
    model, learner_cls, x_train, y_train, x_test, y_test, t0 = rx.fit_learner(args)
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    ref = rx.reference_scores(a.dataset, learner_cls.MODEL_NAME, a.fold)
    out = {"dataset": a.dataset, "fold": a.fold, "method": a.method, "learner": a.learner,
           "engine_version": engine_version(a.method), "score": float(metrics.score),
           "metrics": json.loads(json.dumps(asdict(metrics), default=float)), "leaderboard_same_fold": ref,
           "n_train": int(len(y_train)), "n_test": int(len(y_test)), "train_examples": a.train_examples,
           "seconds": round(time.time() - t0, 1), "feature_columns": list(x_train.columns), "git": rx.git_info()}
    model.ours_cache_dir = None  # the saved model must not depend on a local cache path
    joblib.dump(model, join(a.out_dir, "model.joblib"), compress=3)
    json.dump(out, open(join(a.out_dir, "metrics.json"), "w"), indent=1)
    print(f"{a.method} {a.learner} {a.dataset} fold {a.fold}: score {out['score']:.4f} "
          f"(leaderboard Joint TAR {ref.get('joint_tar')}) -> {join(a.out_dir, 'model.joblib')}")
    return out


if __name__ == "__main__":
    main()

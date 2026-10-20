"""Leakage check: train the method with the TRAINING labels randomly permuted; score on the true dev-test labels.

With no leakage, every label-aware part (experts, probes, the learner) sees only noise, so the score must fall to
chance: AUROC ~ 0.5 (binary / macro-OvR) and R^2 <= ~0 (regression). If it stayed high, label information would be
reaching the model through some other path (e.g. test rows inside an expert fit). Uses dev splits of fold 0's training
rows (scripts/dev_eval.py), so the real test rows are never used.
    python scripts/check_label_shuffle.py --datasets BIN_TEXT_KICKSTARTER_FUNDING --method cte_depth_top3 \
        --out .explore/shuffle.csv --cache_dir .explore/dev/_cache
"""
import argparse
import os
import sys
import time
from os.path import abspath, dirname

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path[:0] = [REPO_ROOT, dirname(abspath(__file__))]
os.environ.setdefault("WANDB_MODE", "disabled")


def main():
    import numpy as np
    import pandas as pd
    import torch
    from dev_eval import dev_split, load_train_rows
    from multabench.ours.learners import make_learner
    from multabench.ours.tar_engine import TarConfig, set_seed
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--method", default="cte_depth_top3")
    p.add_argument("--out", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--perm_seeds", nargs="+", type=int, default=[123], help="one shuffled run per permutation seed")
    p.add_argument("--skip_true", action="store_true", help="only the shuffled runs")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in a.datasets:
        ds, x_train, y_train = load_train_rows(dataset, 0)
        x_tr, x_te, y_tr, y_te = dev_split(x_train, y_train, ds.is_cls, 0)
        runs = ([] if a.skip_true else [(False, None)]) + [(True, s_) for s_ in a.perm_seeds]
        for shuffled, perm_seed in runs:
            t0 = time.time()
            set_seed(42)
            y_fit = y_tr
            if shuffled:  # permute the training labels only; the dev-test labels stay true
                y_fit = pd.Series(np.random.default_rng(perm_seed).permutation(y_tr.to_numpy()), index=y_tr.index, name=y_tr.name)
            model = make_learner("catboost")(problem_type=ds.task_type, device=device, verbose=False, tune_e5=False,
                                             pca_components=30, no_pca=False)
            model.configure_ours(variant=a.method, cfg=TarConfig(seed=42, eval_batch_size=16), cache_dir=a.cache_dir,
                                 dataset=dataset, fold=900 + (0 if not shuffled else perm_seed), test=(x_te, y_te))  # separate cache keys
            model.fit(x_tr, y_fit)
            score = float(model.score_all_metrics(X=x_te, y=y_te).score)
            row = {"dataset": dataset, "method": a.method, "train_labels": "shuffled" if shuffled else "true",
                   "perm_seed": perm_seed,
                   "score": score, "seconds": round(time.time() - t0, 1)}
            pd.DataFrame([row]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
            print(f"{dataset} {row['train_labels']:<9} {score:.4f} ({row['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

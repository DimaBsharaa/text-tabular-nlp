"""P7 development evaluation: compare methods on dev splits of a fold's TRAINING rows only (test rows never used).

For each dataset: upstream loading + subsampling + split_to_test(fold) -> training rows. For each seed: an 80/20
(stratified for classification) split of the training rows into dev-train / dev-test; our CatBoost learner is fit on
dev-train with the method's text features and scored on dev-test with the benchmark metric. One CSV row per
(dataset, fold, seed, method); finished rows are skipped (resumable).
    python scripts/dev_eval.py --datasets REG_TEXT_BOOK_PRICE MUL_TEXT_PRODUCT_SENTIMENT --methods frozen0 tf_num cte \
        --seeds 0 1 --out .explore/dev/results.csv --cache_dir .explore/dev/_cache
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from os.path import abspath, dirname

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path[:0] = [REPO_ROOT, dirname(abspath(__file__))]
os.environ.setdefault("WANDB_MODE", "disabled")


def load_train_rows(dataset: str, fold: int, train_examples: int = 10_000):
    from tabstar.preprocessing.splits import split_to_test
    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.datasets.utils import dataset_from_name
    ds = download_dataset(dataset_id=dataset_from_name(name=dataset), multimodal_state=MultimodalState.ALL,
                          target_override=None)
    x, y = subsample_dataset(x=ds.x, y=ds.y, is_cls=ds.is_cls, train_examples=train_examples, fold=fold)
    x_train, _, y_train, _ = split_to_test(x=x, y=y, is_cls=ds.is_cls, fold=fold, train_examples=train_examples)
    return ds, x_train, y_train  # the test split is discarded here on purpose


def dev_split(x, y, is_cls: bool, seed: int, frac: float = 0.2, cv: int = 0):
    """seed-th dev split of the training rows: a random 80/20 split (P7), or with cv=K the seed-th fold of a K-fold
    partition (P11: every training row is dev-test exactly once over the K splits, so the K scores are less noisy)."""
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
    if cv:
        strat = is_cls and y.astype(str).value_counts().min() >= cv
        kf = (StratifiedKFold if strat else KFold)(n_splits=cv, shuffle=True, random_state=0)
        tr, te_ = list(kf.split(x, y.astype(str) if strat else None))[seed]
        return x.iloc[tr], x.iloc[te_], y.iloc[tr], y.iloc[te_]
    try:
        return train_test_split(x, y, test_size=frac, random_state=seed, stratify=y if is_cls else None)
    except ValueError:  # a class too rare to stratify
        return train_test_split(x, y, test_size=frac, random_state=seed)


def main(argv=None) -> None:
    import pandas as pd
    import torch
    from multabench.ours.learners import make_learner
    from multabench.ours.tar_engine import TarConfig, set_seed
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--cv", type=int, default=0, help="K > 0: dev split = the seed-th fold of a K-fold partition "
                                                     "of the training rows (seeds 0..K-1)")
    a = p.parse_args(argv)
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out)
        cv_col = d.cv if "cv" in d else [0] * len(d)  # P7 CSVs have no cv column: random 80/20 splits
        done = set(zip(d.dataset, d.fold, d.seed, d.method, cv_col))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in a.datasets:
        ds, x_train, y_train = load_train_rows(dataset, a.fold)
        for seed in a.seeds:
            x_tr, x_te, y_tr, y_te = dev_split(x_train, y_train, ds.is_cls, seed, cv=a.cv)
            for method in a.methods:
                if (dataset, a.fold, seed, method, a.cv) in done:
                    continue
                t0 = time.time()
                set_seed(42)
                model = make_learner("catboost")(problem_type=ds.task_type, device=device, verbose=False, tune_e5=False,
                                                 pca_components=30, no_pca=False)
                cfg = TarConfig(seed=42, eval_batch_size=256 if device.type == "cuda" else 16)
                model.configure_ours(variant=method, cfg=cfg, cache_dir=a.cache_dir, dataset=dataset,
                                     fold=100 * (a.fold + 1) + (50 if a.cv else 0) + seed,  # dev cache keys only
                                     test=(x_te, y_te))
                model.fit(x_tr, y_tr)
                score = float(model.score_all_metrics(X=x_te, y=y_te).score)
                stack = getattr(model, "stack_result", None)
                row = {"dataset": dataset, "fold": a.fold, "seed": seed, "cv": a.cv, "method": method, "score": score,
                       "n_dev_train": len(y_tr), "n_dev_test": len(y_te), "seconds": round(time.time() - t0, 1),
                       "importance": None if stack is None else str(model.stack_feature_importance()["group_share"]),
                       "depth_layer_oof": None if stack is None or "depth" not in stack.diagnostics
                       else str(stack.diagnostics["depth"]["layer_oof_metric"])}
                pd.DataFrame([row]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
                print(f"{dataset} seed {seed} {method:<18} {score:.4f}  ({row['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

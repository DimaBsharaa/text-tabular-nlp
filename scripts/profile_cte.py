"""Time per stage of one CTE job (P5 v2 amendment 1): wraps the expensive calls with timers, then runs the job
in-process with scripts/run_experiment.py's own argument parser. Experts run serially here (timers are per process).
    OURS_EXPERT_JOBS=1 python scripts/profile_cte.py --report out.json -- --method cte --learner catboost \
        --dataset REG_TEXT_BOOK_PRICE --fold 0 --out .explore/prof --cache_dir .explore/prof/_cache
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
from collections import defaultdict
from os.path import abspath, dirname

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path[:0] = [REPO_ROOT, dirname(abspath(__file__))]
TIMES: dict = defaultdict(lambda: [0.0, 0])


def wrap(owner, attr: str, label: str) -> None:
    if not hasattr(owner, attr):
        return
    fn = getattr(owner, attr)

    @functools.wraps(fn)
    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            TIMES[label][0] += time.perf_counter() - t0
            TIMES[label][1] += 1
    setattr(owner, attr, timed)


def install() -> None:
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
    from sklearn.neighbors import NearestNeighbors
    from multabench.ours import cte, experts
    wrap(TfidfVectorizer, "fit_transform", "tfidf fit_transform")
    wrap(TfidfVectorizer, "transform", "tfidf transform")
    wrap(LogisticRegression, "fit", "LogisticRegression fit")
    wrap(RidgeCV, "fit", "RidgeCV fit")
    wrap(Ridge, "fit", "Ridge fit")
    wrap(NearestNeighbors, "kneighbors", "kNN (sklearn)")
    wrap(cte, "knn_topk", "kNN top-k (matrix product)")
    wrap(experts, "rank_gauss", "rank-Gauss target")
    wrap(cte, "rank_gauss", "rank-Gauss target")
    wrap(cte, "numeracy_matrix", "numeracy features")
    wrap(cte, "frozen_embeddings_cached", "frozen embeddings")
    wrap(cte, "compute_cte_features", "experts total")
    wrap(cte, "tfidf_matrices", "tfidf fit+transform (once)")
    wrap(cte, "ridge_scores", "Ridge fit+predict")
    wrap(cte, "neighbour_predict", "neighbour expert")
    wrap(experts, "compute_expert_features", "experts total")
    wrap(PCA, "fit", "PCA fit")
    for c in (CatBoostClassifier, CatBoostRegressor):
        wrap(c, "fit", "CatBoost fit")
    for c in (LGBMClassifier, LGBMRegressor):
        wrap(c, "fit", "LightGBM fit")


def main() -> int:
    argv = sys.argv[1:]
    report = None
    if "--report" in argv:
        i = argv.index("--report")
        report, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    argv = [a for a in argv if a != "--"]
    install()
    import run_experiment
    sys.argv = ["run_experiment.py", *argv]
    t0 = time.perf_counter()
    rc = run_experiment.main()
    total = time.perf_counter() - t0
    rows = {k: {"seconds": round(v[0], 1), "calls": v[1]} for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1][0])}
    print(f"\n{'stage':<26}{'seconds':>10}{'calls':>7}")
    for k, v in rows.items():
        print(f"{k:<26}{v['seconds']:>10.1f}{v['calls']:>7}")
    print(f"{'whole job (wall)':<26}{total:>10.1f}")
    if report:
        os.makedirs(dirname(abspath(report)), exist_ok=True)
        json.dump({"argv": argv, "stages": rows, "total_seconds": round(total, 1), "cpu_count": os.cpu_count()},
                  open(report, "w"), indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())

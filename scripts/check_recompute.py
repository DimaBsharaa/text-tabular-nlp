"""Independent recomputation of the headline claims, using NO project code (only json, pandas, scipy).

Reads the raw per-job result JSONs of a Colab export and the official upstream leaderboard CSVs, then recomputes,
for a method and learner: per dataset, mean over folds 0-4 of (our test score - leaderboard Joint TAR score of the
same learner and fold); the mean / median of those 20 differences; wins; and a two-sided Wilcoxon signed-rank p-value.
If the project's analysis code had a bug, these numbers would disagree with reports/*.json.
    python scripts/check_recompute.py --method cte_depth_top3 --learner catboost   # reads the committed results/raw/
"""
import argparse
import glob
import json
import os

import pandas as pd
from scipy.stats import wilcoxon

LEADERBOARD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "multabench", "leaderboard",
                           "results", "text")
LEARNER_PREFIX = {"catboost": "CatBoost", "lgbm": "LightGBM", "tabm": "TabM", "tabpfnv2": "TabPFN-v2", "tabpfnv2p5": "TabPFN-v2p5"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(os.path.dirname(LEADERBOARD), "..", "..", "..", "results", "raw"),
                   help="a results root with <method>/<learner>/<dataset>/fold<k>.json (default: the committed results/raw)")
    p.add_argument("--method", required=True)
    p.add_argument("--learner", default="catboost")
    a = p.parse_args()
    ours = []
    for path in glob.glob(os.path.join(a.results, a.method, a.learner, "*", "fold[0-4].json")):
        r = json.load(open(path))
        ours.append({"dataset": r["dataset"], "fold": int(r["fold"]), "ours": float(r["score"])})
    ours = pd.DataFrame(ours)
    lb = []
    for path in glob.glob(os.path.join(LEADERBOARD, "*.csv")):
        d = pd.read_csv(path)
        d = d[(d.multimodal_state == "ft") & d.model.str.startswith(LEARNER_PREFIX[a.learner] + " ")]
        lb += [{"dataset": os.path.basename(path)[:-4], "fold": int(f), "tar": float(s)} for f, s in zip(d.fold, d.test_score)]
    lb = pd.DataFrame(lb)
    m = ours.merge(lb, on=["dataset", "fold"])
    complete = m.groupby("dataset").fold.nunique() == 5
    m = m[m.dataset.isin(complete[complete].index)]
    d = (m.ours - m.tar).groupby(m.dataset).mean()
    print(f"{a.method} / {a.learner}: {len(ours)} result files, {d.size} datasets with all 5 folds")
    print(f"  mean diff {d.mean():+.4f} | median {d.median():+.4f} | wins {(d > 0).sum()}/{d.size} | "
          f"two-sided Wilcoxon p = {wilcoxon(d).pvalue:.4f}")


if __name__ == "__main__":
    main()

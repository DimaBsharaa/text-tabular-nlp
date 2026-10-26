"""Builds results/ from reports/tables/fold_scores.csv: the final method's scores next to the MulTaBench leaderboard's
Joint TAR and Joint Frozen scores of the same learner, dataset and fold.
    python scripts/make_results.py
Writes results/scores_per_fold.csv, results/scores_per_dataset.csv and results/summary.csv (one row per learner: mean
scores, mean / median gain, wins, two-sided Wilcoxon p over datasets, 95% bootstrap CI of the mean gain), and the final
method's own scores without baselines: results/results_summary.csv, results_per_dataset.csv and results_per_fold.csv.
"""
from __future__ import annotations

import os
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from summarize_results import our_scores  # the final method's own scores (same code as for new runs)

REPO_ROOT = dirname(dirname(abspath(__file__)))
METHOD = "cte_depth_top3_attn"
LEARNERS = {"catboost": "CatBoost", "lgbm": "LightGBM", "tabm": "TabM", "tabpfnv2": "TabPFN-v2",
            "tabpfnv2p5": "TabPFN-2.5"}


def main() -> dict:
    fs = pd.read_csv(join(REPO_ROOT, "reports", "tables", "fold_scores.csv"))
    twenty = sorted(fs.loc[fs.method == "tar_repro", "dataset"].unique())  # the 20 MulTaBench text datasets
    d = fs[(fs.method == METHOD) & fs.learner.isin(LEARNERS) & fs.dataset.isin(twenty)]
    d = d.dropna(subset=["lb_joint_tar"])  # TabPFN: no leaderboard baseline for Spotify Genres / Wine Review
    folds = (d.rename(columns={"score": "ours", "lb_joint_tar": "joint_tar", "lb_joint_frozen": "joint_frozen"})
             [["learner", "dataset", "task", "fold", "ours", "joint_tar", "joint_frozen"]]
             .assign(learner=lambda x: x.learner.map(LEARNERS))
             .sort_values(["learner", "dataset", "fold"]))
    folds["gain_vs_tar"] = folds.ours - folds.joint_tar
    per_ds = folds.groupby(["learner", "dataset", "task"], as_index=False)[
        ["ours", "joint_tar", "joint_frozen", "gain_vs_tar"]].mean()
    rng = np.random.default_rng(42)
    rows = []
    for learner in LEARNERS.values():
        g = per_ds[per_ds.learner == learner]
        gains = g.gain_vs_tar.to_numpy()
        boot = rng.choice(gains, size=(10_000, len(gains)), replace=True).mean(axis=1)
        rows.append({"learner": learner, "datasets": len(g), "ours": g.ours.mean(), "joint_tar": g.joint_tar.mean(),
                     "joint_frozen": g.joint_frozen.mean(), "mean_gain": gains.mean(), "median_gain": np.median(gains),
                     "ci95_low": np.quantile(boot, 0.025), "ci95_high": np.quantile(boot, 0.975),
                     "wins": int((gains > 0).sum()), "wilcoxon_p": wilcoxon(gains).pvalue})
    summary = pd.DataFrame(rows)
    out = join(REPO_ROOT, "results")
    os.makedirs(out, exist_ok=True)
    folds.round(6).to_csv(join(out, "scores_per_fold.csv"), index=False)
    per_ds.round(6).to_csv(join(out, "scores_per_dataset.csv"), index=False)
    summary.round(6).to_csv(join(out, "summary.csv"), index=False)
    our_scores(folds, per_ds, out)
    print(summary.round(4).to_string(index=False))
    return {"folds": folds, "per_dataset": per_ds, "summary": summary}


if __name__ == "__main__":
    main()

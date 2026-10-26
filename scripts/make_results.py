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


TASKS = {"BIN": ("binary classification", "AUROC"), "MUL": ("multiclass classification", "AUROC (macro one-vs-rest)"),
         "REG": ("regression", "R2")}


def dataset_name(d: str) -> str:
    n = d.split("_TEXT_", 1)[1].replace("_", " ").title()
    return n.replace("Us ", "US ").replace("Scimagojr", "SCImago")


def our_scores(folds: pd.DataFrame, per_ds: pd.DataFrame, out: str) -> None:
    """The final method's own scores only (no baselines): per learner, per dataset and per run."""
    learners = list(LEARNERS.values())
    label = lambda df: df.assign(dataset_name=df.dataset.map(dataset_name), task_type=df.task.map(lambda t: TASKS[t][0]),
                                 metric=df.task.map(lambda t: TASKS[t][1])).rename(columns={"dataset": "dataset_id"})
    wide = (label(per_ds).pivot_table(index=["dataset_id", "dataset_name", "task_type", "metric"], columns="learner",
                                      values="ours").reindex(columns=learners).reset_index()
            .sort_values(["task_type", "dataset_name"]))
    wide.round(4).to_csv(join(out, "results_per_dataset.csv"), index=False)
    runs = label(folds)[["dataset_id", "dataset_name", "task_type", "metric", "learner", "fold", "ours"]]
    runs = runs.rename(columns={"ours": "score"}).assign(learner=lambda x: pd.Categorical(x.learner, learners))
    runs.sort_values(["task_type", "dataset_name", "learner", "fold"]).round(6).to_csv(
        join(out, "results_per_fold.csv"), index=False)
    rows = []
    for learner in learners:
        g = per_ds[per_ds.learner == learner]
        cls, reg = g[g.task != "REG"], g[g.task == "REG"]
        rows.append({"learner": learner, "datasets": len(g), "mean_score": g.ours.mean(),
                     "mean_AUROC_classification": cls.ours.mean(), "classification_datasets": len(cls),
                     "mean_R2_regression": reg.ours.mean(), "regression_datasets": len(reg)})
    pd.DataFrame(rows).round(3).to_csv(join(out, "results_summary.csv"), index=False)


if __name__ == "__main__":
    main()

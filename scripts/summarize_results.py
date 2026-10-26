"""Our results tables from the runs of scripts/run_experiment.py: the final method's AUROC / R2 only (no baselines).
    python scripts/run_experiment.py --jobs_file jobs/p18.txt --out my_results/
    python scripts/summarize_results.py --out my_results/
Reads <out>/<method>/<learner>/<dataset>/fold<k>.json and writes, into <out> (or --to):
  results_summary.csv      one row per learner: mean score over the datasets, mean AUROC (classification), mean R2
  results_per_dataset.csv  one row per dataset: task type, metric and the fold-averaged score of each learner
  results_per_fold.csv     one row per run (dataset, learner, fold)
On the committed runs (--out results/raw) it reproduces results/results_*.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
from os.path import join

import pandas as pd

LEARNERS = {"catboost": "CatBoost", "lgbm": "LightGBM", "tabm": "TabM", "tabpfnv2": "TabPFN-v2",
            "tabpfnv2p5": "TabPFN-2.5"}
TASKS = {"BIN": ("binary classification", "AUROC"), "MUL": ("multiclass classification", "AUROC (macro one-vs-rest)"),
         "REG": ("regression", "R2")}


def dataset_name(d: str) -> str:
    n = d.split("_TEXT_", 1)[1].replace("_", " ").title()
    return n.replace("Us ", "US ").replace("Scimagojr", "SCImago")


def our_scores(folds: pd.DataFrame, per_ds: pd.DataFrame, out: str) -> None:
    """Writes the three tables. folds: learner (display name), dataset, task, fold, ours; per_ds: its fold means."""
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
        if g.empty:
            continue
        cls, reg = g[g.task != "REG"], g[g.task == "REG"]
        rows.append({"learner": learner, "datasets": len(g), "mean_score": g.ours.mean(),
                     "mean_AUROC_classification": cls.ours.mean(), "classification_datasets": len(cls),
                     "mean_R2_regression": reg.ours.mean(), "regression_datasets": len(reg)})
    pd.DataFrame(rows).round(3).to_csv(join(out, "results_summary.csv"), index=False)


def summarize(out: str, method: str = "cte_depth_top3_attn", to: str | None = None) -> pd.DataFrame:
    paths = sorted(glob.glob(join(out, method, "*", "*", "fold*.json")))
    if not paths:
        raise SystemExit(f"no runs of {method} under {out} (expected {out}/{method}/<learner>/<dataset>/fold<k>.json)")
    runs = [json.load(open(p)) for p in paths]
    folds = pd.DataFrame([{"learner": LEARNERS[r["learner"]], "dataset": r["dataset"], "task": r["task"],
                           "fold": r["fold"], "ours": r["score"]} for r in runs])
    per_ds = folds.groupby(["learner", "dataset", "task"], as_index=False)["ours"].mean()
    our_scores(folds, per_ds, to or out)
    return per_ds


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="the --out folder given to run_experiment.py")
    p.add_argument("--method", default="cte_depth_top3_attn")
    p.add_argument("--to", default=None, help="where to write the three tables (default: --out)")
    a = p.parse_args()
    per_ds = summarize(a.out, a.method, a.to)
    print(f"{len(per_ds)} (learner, dataset) pairs from the runs in {a.out}; tables written to {a.to or a.out}")
    print(pd.read_csv(join(a.to or a.out, "results_summary.csv")).to_string(index=False))

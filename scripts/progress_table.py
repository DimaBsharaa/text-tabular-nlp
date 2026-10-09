"""Progress table for a jobs file: which jobs are done, their scores vs upstream's CSV, TAR diagnostics, runtimes.

Only needs pandas (runs in Colab's own Python as well as in .venv):
    python scripts/progress_table.py --jobs_file jobs/pilot_tar_repro.txt --out results
"""
from __future__ import annotations

import argparse
import json
from os.path import exists

import pandas as pd

from run_experiment import DEFAULTS, read_jobs, result_path, run_name


def progress_table(jobs_file: str, out: str, **overrides) -> pd.DataFrame:
    """One row per job. overrides: the same result-changing settings passed to run_experiment (e.g. epochs=1)."""
    args = argparse.Namespace(**{**DEFAULTS, **overrides})
    rows = []
    for method, learner, dataset, fold in read_jobs(jobs_file):
        path = result_path(out, run_name(method, args), learner, dataset, fold)
        row = {"method": method, "learner": learner, "dataset": dataset, "fold": fold, "status": "pending"}
        if exists(path):
            r = json.load(open(path))
            hq = (r.get("tar") or {}).get("head_quality", {})
            tr = (r.get("tar") or {}).get("train", {})
            row.update({
                "status": "done", "score": r["score"], "csv_tar": r["reference"].get("joint_tar"),
                "diff_vs_csv_tar": r.get("diff_vs_csv_joint_tar"),
                "head_e5_train": hq.get("e5_train", {}).get("score"), "head_test": hq.get("test", {}).get("score"),
                "leak_gap": r.get("leakage_gap_head_train_minus_test"),
                "epochs": tr.get("epochs_run"), "best_epoch": tr.get("best_epoch"),
                "cache_hit": (r.get("tar") or {}).get("cache_hit"),
                "min_total": round(r["runtime_s"]["total"] / 60, 1),
                "min_tar_train": None if r["runtime_s"].get("tar_train") is None else round(r["runtime_s"]["tar_train"] / 60, 1),
                "fast": tr.get("fast_path"), "engine": ((r.get("tar") or {}).get("config") or {}).get("engine_version"),
                "git": r["git"]["commit"][:7] + ("*" if r["git"]["dirty"] else ""),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def per_dataset_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Fold-averaged score vs upstream CSV TAR, over finished folds only."""
    done = table[table["status"] == "done"]
    if done.empty:
        return pd.DataFrame()
    return done.groupby(["method", "learner", "dataset"]).agg(
        folds_done=("fold", "count"), score=("score", "mean"), csv_tar=("csv_tar", "mean"),
        diff_vs_csv_tar=("diff_vs_csv_tar", "mean"), leak_gap=("leak_gap", "mean"),
        min_total=("min_total", "sum")).reset_index()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs_file", required=True)
    p.add_argument("--out", required=True)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=int, default=v)
    a = p.parse_args()
    t = progress_table(a.jobs_file, a.out, **{k: getattr(a, k) for k in DEFAULTS})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(per_dataset_summary(t).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{(t.status == 'done').sum()} / {len(t)} jobs done")

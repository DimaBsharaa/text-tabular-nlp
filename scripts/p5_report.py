"""P5 analysis, written before any P5 full-run result. Reuses full_report's tested statistics.

C2 primary : CatBoost tar_experts vs leaderboard Joint TAR (`ft`); two-sided Wilcoxon over 20 datasets on fold-averaged
             differences; pass = p < 0.05 AND mean > 0.
C3         : CatBoost frozen_experts vs leaderboard Joint TAR (same test); also reported vs leaderboard Joint Frozen.
Secondary  : LightGBM / TabM for tar_experts and frozen_experts vs `ft`, Holm-corrected over these 4.
Ablations  : tar_experts_lex, tar_experts_nbr, tar_experts_lex_insample (CatBoost) vs `ft` and vs tar_experts.
Context    : tar_experts vs our tar_repro (CatBoost, results_full, same GPU type) if --tar_repro_out is given.
Runtime    : GPU/CPU hours per method (sum of job times).
    python scripts/p5_report.py --out MyDrive/.../results_p5 [--tar_repro_out MyDrive/.../results_full]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from os.path import abspath, dirname, join

import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402

REPO_ROOT = dirname(dirname(abspath(__file__)))
ABLATIONS = ["tar_experts_lex", "tar_experts_nbr", "tar_experts_lex_insample"]


def leaderboard_state(learner: str, state: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(join(REPO_ROOT, "multabench", "leaderboard", "results", "text", "*.csv")):
        df = pd.read_csv(path)
        df = df[(df.multimodal_state == state) & (df.model.str.rsplit(" ", n=1).str[0] == fr.MODEL_NAMES[learner])]
        rows += [{"dataset": os.path.basename(path)[:-4], "fold": int(r.fold), "score": float(r.test_score)}
                 for r in df.itertuples()]
    return pd.DataFrame(rows, columns=["dataset", "fold", "score"])


def runtime_hours(out: str, run: str, learner: str) -> float | None:
    t = [json.load(open(p))["runtime_s"]["total"] for p in glob.glob(join(out, run, learner, "*", "fold[0-9].json"))]
    return round(sum(t) / 3600, 2) if t else None


def comparison(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    d, excl = fr.fold_averaged_differences(a, b)
    return {**fr.compare(d), "excluded_datasets": excl, "per_dataset": {k: round(float(v), 5) for k, v in d.items()}}


def p5_report(out: str, tar_repro_out: str | None = None) -> dict:
    ft = {lr: leaderboard_state(lr, "ft") for lr in ("catboost", "lgbm", "tabm")}
    mine = lambda run, lr="catboost": fr.ours(out, run, lr)
    rep = {"C2": comparison(mine("tar_experts"), ft["catboost"]),
           "C3": comparison(mine("frozen_experts"), ft["catboost"]),
           "C3_vs_joint_frozen": comparison(mine("frozen_experts"), leaderboard_state("catboost", "all")),
           "secondary": {}, "ablations": {}, "runtime_hours": {}}
    for k in ("C2", "C3"):
        rep[k]["passes"] = bool(rep[k].get("p_value") is not None and rep[k]["p_value"] < fr.ALPHA
                                and (rep[k]["mean_diff"] or 0) > 0)
    for run in ("tar_experts", "frozen_experts"):
        for lr in ("lgbm", "tabm"):
            rep["secondary"][f"{run}/{lr}"] = comparison(mine(run, lr), ft[lr])
    adj = fr.holm({k: v.get("p_value") for k, v in rep["secondary"].items()})
    for k, v in rep["secondary"].items():
        v["p_holm"] = adj.get(k)
    for run in ABLATIONS:
        rep["ablations"][run] = {"vs_leaderboard_ft": comparison(mine(run), ft["catboost"]),
                                 "vs_tar_experts": comparison(mine(run), mine("tar_experts"))}
    if tar_repro_out:
        rep["C2_vs_tar_repro"] = comparison(mine("tar_experts"), fr.ours(tar_repro_out, "tar_repro", "catboost"))
    for run in ["tar_experts", "frozen_experts"] + ABLATIONS:
        for lr in ("catboost", "lgbm", "tabm"):
            h = runtime_hours(out, run, lr)
            if h is not None:
                rep["runtime_hours"][f"{run}/{lr}"] = h
    gpus = set()
    for p in glob.glob(join(out, "*", "*", "*", "fold[0-9].json")):
        g = json.load(open(p)).get("gpu")
        if g:
            gpus.add(g)
    rep["gpu_types"] = sorted(gpus)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tar_repro_out", default=None)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p5"))
    a = p.parse_args()
    rep = p5_report(a.out, a.tar_repro_out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p5_report.json"), "w"), indent=1)
    for k in ("C2", "C3", "C3_vs_joint_frozen"):
        v = rep[k]
        print(f"{k:<20} n={v['n']} mean={v.get('mean_diff')} CI={v.get('ci95')} p={v.get('p_value')} "
              f"wins={v.get('wins')}" + (f" -> {'PASS' if v['passes'] else 'no pass'}" if "passes" in v else ""))
    for k, v in rep["secondary"].items():
        print(f"secondary {k:<22} n={v['n']} mean={v.get('mean_diff')} p={v.get('p_value')} p_holm={v.get('p_holm')}")
    for k, v in rep["ablations"].items():
        print(f"ablation {k:<26} vs ft mean={v['vs_leaderboard_ft'].get('mean_diff')} | vs tar_experts "
              f"mean={v['vs_tar_experts'].get('mean_diff')}")
    print("runtime hours:", rep["runtime_hours"], "| GPU types:", rep["gpu_types"])


if __name__ == "__main__":
    main()

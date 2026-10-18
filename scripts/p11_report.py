"""P11 analysis (PROGRESS.md P11), written before any P11 result. C1-C6 stay as reported.

C7 (confirmation on the 36 text datasets in multabench/leaderboard/results/text_source/): cte_probe (CatBoost) vs
    leaderboard Joint Frozen (C7a) and Joint TAR (C7b); two-sided Wilcoxon on fold-averaged differences; Holm over the
    two; pass = Holm p < 0.05 AND positive median. Datasets without all 5 folds are excluded and listed.
C8: cte_probe (LightGBM) vs leaderboard Joint TAR (LightGBM) on the 20 MulTaBench datasets; pass = p < 0.05 AND
    positive mean. Also: Holm over all pre-registered tests C1-C8.
    python scripts/p11_report.py --out <export>/results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from os.path import abspath, basename, dirname, join

import numpy as np
import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from p5_report import comparison, leaderboard_state  # noqa: E402

REPO_ROOT = dirname(dirname(abspath(__file__)))
SOURCE_DIR = join(REPO_ROOT, "multabench", "leaderboard", "results", "text_source")
EARLIER = {"C1": 0.409, "C2": 0.0583, "C3_nb": 0.0484, "C3_res": 0.0637, "C4": 1.907e-5, "C5": 0.002325, "C6": 0.01531}


def source_leaderboard(learner: str, state: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(join(SOURCE_DIR, "*.csv")):
        df = pd.read_csv(path)
        df = df[(df.multimodal_state == state) & (df.model.str.rsplit(" ", n=1).str[0] == fr.MODEL_NAMES[learner])]
        rows += [{"dataset": basename(path)[:-4], "fold": int(r.fold), "score": float(r.test_score)} for r in df.itertuples()]
    return pd.DataFrame(rows, columns=["dataset", "fold", "score"])


def complete(df: pd.DataFrame) -> pd.DataFrame:
    n = df.groupby("dataset").fold.nunique()
    return df[df.dataset.isin(n[n == 5].index)]


def p11_report(out: str) -> dict:
    names = sorted(basename(p)[:-4] for p in glob.glob(join(SOURCE_DIR, "*.csv")))
    mine = fr.ours(out, "cte_probe", "catboost")
    mine36 = complete(mine[mine.dataset.isin(names)])
    done = sorted(mine36.dataset.unique())
    rep = {"label": "P11 - claims C7 (36 untouched datasets) and C8 (LightGBM); C1-C6 unchanged",
           "n_datasets_complete": len(done), "missing": sorted(set(names) - set(done)),
           "C7a_vs_joint_frozen": comparison(mine36, source_leaderboard("catboost", "all")),
           "C7b_vs_joint_tar": comparison(mine36, source_leaderboard("catboost", "ft")),
           "C8_lgbm_vs_joint_tar": comparison(fr.ours(out, "cte_probe", "lgbm"), leaderboard_state("lgbm", "ft"))}
    adj = fr.holm({k: rep[k].get("p_value") for k in ("C7a_vs_joint_frozen", "C7b_vs_joint_tar")})
    for k in ("C7a_vs_joint_frozen", "C7b_vs_joint_tar"):
        c = rep[k]
        c["p_holm"] = adj.get(k)
        c["passes"] = None if not c["n"] else bool(c["p_holm"] < fr.ALPHA and (c.get("median_diff") or 0) > 0)
    c8 = rep["C8_lgbm_vs_joint_tar"]
    c8["passes"] = None if c8["n"] != 20 else bool(c8["p_value"] < fr.ALPHA and (c8.get("mean_diff") or 0) > 0)
    allp = dict(EARLIER, C7a=rep["C7a_vs_joint_frozen"].get("p_value"), C7b=rep["C7b_vs_joint_tar"].get("p_value"),
                C8=c8.get("p_value"))
    rep["holm_all_preregistered"] = {k: (None if v is None else round(float(v), 6))
                                     for k, v in fr.holm({k: v for k, v in allp.items() if v is not None}).items()}
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p11"))
    a = p.parse_args()
    rep = p11_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p11_report.json"), "w"), indent=1)
    print(rep["label"], "| complete datasets:", rep["n_datasets_complete"], "| missing:", rep["missing"])
    for k in ("C7a_vs_joint_frozen", "C7b_vs_joint_tar", "C8_lgbm_vs_joint_tar"):
        c = rep[k]
        print(f"{k:<22} n={c['n']} mean={c.get('mean_diff')} median={c.get('median_diff')} CI={c.get('ci95')} "
              f"p={c.get('p_value')} p_holm={c.get('p_holm')} wins={c.get('wins')} -> passes={c.get('passes')}")
    print("Holm over all pre-registered tests:", rep["holm_all_preregistered"])


if __name__ == "__main__":
    main()

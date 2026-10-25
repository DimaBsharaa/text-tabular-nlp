"""P5 v2 (CTE) analysis, written before any CTE full-run result. Reuses full_report's tested statistics.

C2 primary : CatBoost cte vs leaderboard Joint TAR (`ft`), two-sided Wilcoxon over 20 datasets on fold-averaged
             differences; pass = p < 0.05 AND positive mean. Also on 16 datasets (without Book Price, Product
             Sentiment, Baby Products, Video Games) and on the strict 14 (also without Fake Job Postings and Data
             Scientist Salary, explored at fold 0). Amendment 2: held-out C2_heldout13 / C2_strict11 also
             exclude Jigsaw, Wine and Spotify Genres (profiled at fold 0).
Secondary  : cte vs leaderboard Joint Frozen (`all`); LightGBM cte vs `ft` (Holm over the secondary learner tests);
             ablations cte_lex / cte_nbr / cte_num / cte_lex_insample (CatBoost) vs `ft` and vs cte.
Also       : runtime per method, mean CatBoost importance share per feature group, expert OOF-vs-test gaps.
    python scripts/cte_report.py --out MyDrive/.../results_cte
"""
from __future__ import annotations
import os as _os
import sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # archive/scripts -> repo
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "scripts"), _os.path.dirname(_os.path.abspath(__file__))]

import argparse
import glob
import json
import os
import sys
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from p5_report import comparison, leaderboard_state, runtime_hours  # noqa: E402

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))
EXCLUDED_16 = ["REG_TEXT_BOOK_PRICE", "MUL_TEXT_PRODUCT_SENTIMENT", "REG_TEXT_BABIES_PRICES", "REG_TEXT_VIDEO_GAMES_SALES"]
EXCLUDED_14 = EXCLUDED_16 + ["BIN_TEXT_FAKE_JOB_POSTING", "MUL_TEXT_DATA_SCIENTIST_SALARY"]
# P5 v2 amendment 2: the held-out report also excludes the datasets profiled at fold 0
PROFILED = ["BIN_TEXT_JIGSAW_TOXICITY", "MUL_TEXT_WINE_REVIEW", "MUL_TEXT_SPOTIFY_GENRES"]
EXCLUDED_HELDOUT13 = EXCLUDED_16 + PROFILED
EXCLUDED_STRICT11 = EXCLUDED_14 + PROFILED
ABLATIONS = ["cte_lex", "cte_nbr", "cte_num", "cte_lex_insample"]


def _subset(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    return df[~df.dataset.isin(exclude)]


def _with_pass(c: dict) -> dict:
    c["passes"] = bool(c.get("p_value") is not None and c["p_value"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)
    return c


def group_shares(out: str, run: str, learner: str = "catboost") -> dict:
    shares = []
    for p in glob.glob(join(out, run, learner, "*", "fold[0-9].json")):
        g = ((json.load(open(p)).get("stacking") or {}).get("catboost_importance") or {}).get("group_share")
        if g:
            shares.append(g)
    return {k: round(float(np.mean([s.get(k, 0.0) for s in shares])), 4) for k in sorted({k for s in shares for k in s})}


def cte_report(out: str) -> dict:
    ft = leaderboard_state("catboost", "ft")
    mine = fr.ours(out, "cte", "catboost")
    rep = {"C2": _with_pass(comparison(mine, ft)),
           "C2_16": _with_pass(comparison(_subset(mine, EXCLUDED_16), _subset(ft, EXCLUDED_16))),
           "C2_14": _with_pass(comparison(_subset(mine, EXCLUDED_14), _subset(ft, EXCLUDED_14))),
           "C2_heldout13": _with_pass(comparison(_subset(mine, EXCLUDED_HELDOUT13), _subset(ft, EXCLUDED_HELDOUT13))),
           "C2_strict11": _with_pass(comparison(_subset(mine, EXCLUDED_STRICT11), _subset(ft, EXCLUDED_STRICT11))),
           "vs_joint_frozen": comparison(mine, leaderboard_state("catboost", "all")),
           "secondary": {"lgbm": comparison(fr.ours(out, "cte", "lgbm"), leaderboard_state("lgbm", "ft"))},
           "ablations": {}, "runtime_hours": {}, "group_share_catboost": group_shares(out, "cte")}
    adj = fr.holm({k: v.get("p_value") for k, v in rep["secondary"].items()})
    for k, v in rep["secondary"].items():
        v["p_holm"] = adj.get(k)
    for run in ABLATIONS:
        rep["ablations"][run] = {"vs_leaderboard_ft": comparison(fr.ours(out, run, "catboost"), ft),
                                 "vs_cte": comparison(fr.ours(out, run, "catboost"), mine)}
    for run in ["cte"] + ABLATIONS:
        for lr in ("catboost", "lgbm"):
            h = runtime_hours(out, run, lr)
            if h is not None:
                rep["runtime_hours"][f"{run}/{lr}"] = h
    rep["gpu_types"] = sorted({json.load(open(p)).get("gpu") for p in glob.glob(join(out, "*", "*", "*", "fold[0-9].json"))} - {None})
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "archive", "reports", "cte"))
    a = p.parse_args()
    rep = cte_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "cte_report.json"), "w"), indent=1)
    for k in ("C2", "C2_heldout13", "C2_strict11", "C2_16", "C2_14", "vs_joint_frozen"):
        v = rep[k]
        print(f"{k:<16} n={v['n']} mean={v.get('mean_diff')} CI={v.get('ci95')} p={v.get('p_value')} wins={v.get('wins')}"
              + (f" -> {'PASS' if v['passes'] else 'no pass'}" if "passes" in v else ""))
    for k, v in rep["secondary"].items():
        print(f"secondary {k:<6} n={v['n']} mean={v.get('mean_diff')} p={v.get('p_value')} p_holm={v.get('p_holm')}")
    for k, v in rep["ablations"].items():
        print(f"ablation {k:<18} vs ft mean={v['vs_leaderboard_ft'].get('mean_diff')} | vs cte "
              f"mean={v['vs_cte'].get('mean_diff')}")
    print("group share (CatBoost cte):", rep["group_share_catboost"])
    print("runtime hours:", rep["runtime_hours"], "| GPU types:", rep["gpu_types"])


if __name__ == "__main__":
    main()

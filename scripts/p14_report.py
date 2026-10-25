"""P14 analysis, written before any P14 result. C1-C8 stay as reported.

C9: cte_depth_top3 (CatBoost) vs leaderboard Joint TAR on the 20 datasets (folds 0-4); pass = p < 0.05 AND positive
    mean. Secondary: vs our cte_probe (paired). Final method = cte_depth_top3 iff C9 passes AND that paired mean > 0,
    else cte_probe.
C7 (amended): cte_probe and cte_depth_top3 on the 36 untouched text_source datasets, folds 0-2, each vs leaderboard
    Joint Frozen and Joint TAR (fold-averaged over folds 0-2 on BOTH sides); Holm over the 4; pass = Holm p < 0.05 AND
    positive median. Datasets without folds 0-2 on both sides are excluded and listed.
Also: Holm over all pre-registered tests C1-C9.
    python scripts/p14_report.py --out <export>/results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

import pandas as pd

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from p5_report import comparison, leaderboard_state  # noqa: E402
from p11_report import EARLIER, source_leaderboard  # noqa: E402

REPO_ROOT = dirname(dirname(abspath(__file__)))
C7_FOLDS = (0, 1, 2)
EARLIER_ALL = dict(EARLIER, C8=0.01068878173828125)


def compare_folds(a: pd.DataFrame, b: pd.DataFrame, folds=C7_FOLDS) -> dict:
    m = a.merge(b, on=["dataset", "fold"], suffixes=("_a", "_b"))
    m = m[m.fold.isin(folds)]
    ok = m.groupby("dataset").fold.nunique() == len(folds)
    keep = ok[ok].index
    d = m[m.dataset.isin(keep)].assign(diff=lambda x: x.score_a - x.score_b).groupby("dataset")["diff"].mean()
    out = fr.compare(d)
    out["per_dataset"] = {k: round(float(v), 6) for k, v in d.items()}
    out["excluded_datasets"] = sorted((set(a.dataset) | set(b.dataset)) - set(keep))
    return out


def p14_report(out: str) -> dict:
    lb_ft = leaderboard_state("catboost", "ft")
    depth20, probe20 = fr.ours(out, "cte_depth_top3", "catboost"), fr.ours(out, "cte_probe", "catboost")
    src_all, src_ft = source_leaderboard("catboost", "all"), source_leaderboard("catboost", "ft")
    rep = {"label": "P14 - claims C9 (20 datasets) and C7 (36 untouched, folds 0-2); C1-C8 unchanged",
           "C9": comparison(depth20, lb_ft), "C9_vs_our_cte_probe": comparison(depth20, probe20), "C7": {}}
    c9 = rep["C9"]
    c9["passes"] = None if c9["n"] != 20 else bool(c9["p_value"] < fr.ALPHA and (c9.get("mean_diff") or 0) > 0)
    paired = rep["C9_vs_our_cte_probe"].get("mean_diff")
    rep["final_method"] = ("cte_depth_top3" if c9["passes"] and paired is not None and paired > 0 else
                           "cte_probe" if c9["passes"] is not None else "pending")
    for m in ("cte_probe", "cte_depth_top3"):
        mine = fr.ours(out, m, "catboost")
        for ref, lb in (("joint_frozen", src_all), ("joint_tar", src_ft)):
            rep["C7"][f"{m}|vs_{ref}"] = compare_folds(mine[mine.dataset.isin(src_all.dataset.unique())], lb)
    adj = fr.holm({k: v.get("p_value") for k, v in rep["C7"].items() if v.get("p_value") is not None})
    for k, c in rep["C7"].items():
        c["p_holm"] = adj.get(k)
        c["passes"] = None if not c["n"] else bool(c["p_holm"] is not None and c["p_holm"] < fr.ALPHA
                                                   and (c.get("median_diff") or 0) > 0)
    allp = dict(EARLIER_ALL, C9=c9.get("p_value"), **{f"C7:{k}": v.get("p_value") for k, v in rep["C7"].items()})
    rep["holm_all_preregistered"] = {k: round(float(v), 6) for k, v in
                                     fr.holm({k: v for k, v in allp.items() if v is not None}).items()}
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p14"))
    a = p.parse_args()
    rep = p14_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p14_report.json"), "w"), indent=1)
    print(rep["label"], "| final method:", rep["final_method"])
    for k, c in [("C9", rep["C9"]), ("C9_vs_our_cte_probe", rep["C9_vs_our_cte_probe"])] + list(rep["C7"].items()):
        print(f"{k:<34} n={c['n']} mean={c.get('mean_diff')} median={c.get('median_diff')} p={c.get('p_value')} "
              f"p_holm={c.get('p_holm')} wins={c.get('wins')} passes={c.get('passes')} excluded={len(c.get('excluded_datasets') or [])}")
    print("Holm over all pre-registered tests:", rep["holm_all_preregistered"])


if __name__ == "__main__":
    main()

"""P6 analysis, written before any P6 result. Never changes claim C2 (scripts/cte_report.py).

Claim C3 (PROGRESS.md, pre-registered 2026-10-14): cte_plus_native, cte_nb, cte_residual (CatBoost) each vs leaderboard
Joint TAR (`ft`); two-sided Wilcoxon over 20 datasets on fold-averaged differences, Holm over these 3 tests; pass =
Holm-adjusted p < 0.05 AND positive mean. Also the held-out 13 (Holm over the same 3, secondary), and
catboost_native_text vs `ft` as a baseline (no correction, no pass/fail).

Exploratory (P6 pre-registration):

For each P6 variant (CatBoost): vs leaderboard Joint TAR (`ft`) and vs our `cte` CatBoost, paired over datasets on
fold-averaged scores; two-sided Wilcoxon, Holm over the 8 tests; bootstrap CI and wins (same code as C1/C2).
Printed on all 20 datasets and on the held-out 13 (cte_report.EXCLUDED_HELDOUT13).
    python scripts/p6_report.py --out MyDrive/.../results_cte
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from cte_report import EXCLUDED_HELDOUT13, _subset  # noqa: E402
from p5_report import comparison, leaderboard_state, runtime_hours  # noqa: E402

REPO_ROOT = dirname(dirname(abspath(__file__)))
P6_VARIANTS = ["catboost_native_text", "cte_plus_native", "cte_nb", "cte_residual"]
C3_VARIANTS = ["cte_plus_native", "cte_nb", "cte_residual"]


def c3(out: str, exclude: list[str]) -> dict:
    ft = _subset(leaderboard_state("catboost", "ft"), exclude)
    res = {v: comparison(_subset(fr.ours(out, v, "catboost"), exclude), ft) for v in C3_VARIANTS}
    adj = fr.holm({v: c.get("p_value") for v, c in res.items()})
    for v, c in res.items():
        c["p_holm"] = adj.get(v)
        c["passes"] = bool(c["p_holm"] is not None and c["p_holm"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)
    return res


def p6_report(out: str) -> dict:
    ft = leaderboard_state("catboost", "ft")
    cte = fr.ours(out, "cte", "catboost")
    rep = {"label": "P6 / claim C3 - does not change claim C2", "C3": c3(out, []),
           "C3_heldout13": c3(out, EXCLUDED_HELDOUT13),
           "baseline_native_text_vs_ft": comparison(fr.ours(out, "catboost_native_text", "catboost"), ft),
           "all20": {}, "heldout13": {}, "runtime_hours": {}}
    for subset, excl in (("all20", []), ("heldout13", EXCLUDED_HELDOUT13)):
        for v in P6_VARIANTS:
            mine = _subset(fr.ours(out, v, "catboost"), excl)
            rep[subset][v] = {"vs_leaderboard_ft": comparison(mine, _subset(ft, excl)),
                              "vs_cte": comparison(mine, _subset(cte, excl))}
        adj = fr.holm({f"{v}|{k}": rep[subset][v][k].get("p_value") for v in P6_VARIANTS
                       for k in ("vs_leaderboard_ft", "vs_cte")})
        for v in P6_VARIANTS:
            for k in ("vs_leaderboard_ft", "vs_cte"):
                rep[subset][v][k]["p_holm"] = adj.get(f"{v}|{k}")
    for v in P6_VARIANTS:
        h = runtime_hours(out, v, "catboost")
        if h is not None:
            rep["runtime_hours"][v] = h
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p6"))
    a = p.parse_args()
    rep = p6_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p6_report.json"), "w"), indent=1)
    print(rep["label"])
    for k in ("C3", "C3_heldout13"):
        for v, c in rep[k].items():
            print(f"{k:<13} {v:<16} n={c['n']} mean={c.get('mean_diff')} CI={c.get('ci95')} p={c.get('p_value')} "
                  f"p_holm={c.get('p_holm')} wins={c.get('wins')} -> {'PASS' if c['passes'] else 'no pass'}")
    b = rep["baseline_native_text_vs_ft"]
    print(f"baseline catboost_native_text vs ft: n={b['n']} mean={b.get('mean_diff')} CI={b.get('ci95')} "
          f"p={b.get('p_value')} wins={b.get('wins')} (baseline, not a claim)")
    print("--- exploratory (P6 pre-registration)")
    for subset in ("all20", "heldout13"):
        for v, d in rep[subset].items():
            for k, c in d.items():
                print(f"{subset:<9} {v:<21} {k:<18} n={c['n']} mean={c.get('mean_diff')} CI={c.get('ci95')} "
                      f"p={c.get('p_value')} p_holm={c.get('p_holm')} wins={c.get('wins')}")
    print("runtime hours:", rep["runtime_hours"])


if __name__ == "__main__":
    main()

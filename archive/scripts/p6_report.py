"""P6 analysis, written before any P6 result. Never changes claim C2 (scripts/cte_report.py).

Claim C3 (PROGRESS.md, pre-registered 2026-10-14; P6 amendment 2, 2026-10-15): cte_nb and cte_residual (CatBoost)
each vs leaderboard Joint TAR (`ft`); two-sided Wilcoxon over 20 datasets on fold-averaged differences, Holm over these
2 tests; pass = Holm-adjusted p < 0.05 AND positive mean. Also the held-out 13 (Holm over the same 2, secondary).
catboost_native_text / cte_plus_native were dropped for runtime (amendment 2); finished catboost_native_text jobs are
shown only as a partial descriptive side table (no test).

Exploratory (P6 pre-registration):

For each P6 variant (CatBoost): vs leaderboard Joint TAR (`ft`) and vs our `cte` CatBoost, paired over datasets on
fold-averaged scores; two-sided Wilcoxon, Holm over the 8 tests; bootstrap CI and wins (same code as C1/C2).
Printed on all 20 datasets and on the held-out 13 (cte_report.EXCLUDED_HELDOUT13).
    python scripts/p6_report.py --out MyDrive/.../results_cte
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

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from cte_report import EXCLUDED_HELDOUT13, _subset  # noqa: E402
from p5_report import comparison, leaderboard_state, runtime_hours  # noqa: E402

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))
P6_VARIANTS = ["cte_nb", "cte_residual"]  # amendment 2: the native-text variants were dropped
C3_VARIANTS = ["cte_nb", "cte_residual"]


def native_side_table(out: str) -> list[dict]:
    """Finished catboost_native_text jobs (partial; amendment 2): per-job score vs the leaderboard, descriptive only."""
    rows = []
    for p in sorted(glob.glob(join(out, "catboost_native_text", "catboost", "*", "fold[0-9].json"))):
        r = json.load(open(p))
        ref = (r.get("reference") or {})
        rows.append({"dataset": r["dataset"], "fold": r["fold"], "score": r["score"],
                     "lb_joint_tar": ref.get("joint_tar"), "lb_joint_frozen": ref.get("joint_frozen"),
                     "feature_calcers": (r.get("native_text") or {}).get("feature_calcers"),
                     "engine_version": r.get("engine_version"), "job_seconds": (r.get("runtime_s") or {}).get("total")})
    return rows


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
           "native_text_partial": native_side_table(out),
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
    p.add_argument("--out", required=True)  # an export's results/ folder (or the Drive results root)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "archive", "reports", "p6"))
    a = p.parse_args()
    rep = p6_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p6_report.json"), "w"), indent=1)
    print(rep["label"])
    for k in ("C3", "C3_heldout13"):
        for v, c in rep[k].items():
            print(f"{k:<13} {v:<16} n={c['n']} mean={c.get('mean_diff')} CI={c.get('ci95')} p={c.get('p_value')} "
                  f"p_holm={c.get('p_holm')} wins={c.get('wins')} -> {'PASS' if c['passes'] else 'no pass'}")
    nat = rep["native_text_partial"]
    print(f"catboost_native_text (dropped, partial, descriptive only): {len(nat)} finished jobs, "
          f"{len({r['dataset'] for r in nat})} datasets")
    print("--- exploratory (P6 pre-registration)")
    for subset in ("all20", "heldout13"):
        for v, d in rep[subset].items():
            for k, c in d.items():
                print(f"{subset:<9} {v:<21} {k:<18} n={c['n']} mean={c.get('mean_diff')} CI={c.get('ci95')} "
                      f"p={c.get('p_value')} p_holm={c.get('p_holm')} wins={c.get('wins')}")
    print("runtime hours:", rep["runtime_hours"])


if __name__ == "__main__":
    main()

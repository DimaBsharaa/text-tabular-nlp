"""P10 analysis, written before any P10 result. C1-C5 stay as reported.

C6: p9_query (cte_probe with the e5 "query:" prefix, CatBoost) vs leaderboard Joint TAR (`ft`), 20 datasets,
    fold-averaged, two-sided Wilcoxon; pass = p < 0.05 AND positive mean.
Secondary: p9_query vs our cte_probe (paired); held-out 13 vs Joint TAR.
Descriptive (19 datasets, no Spotify): frozen0_query vs leaderboard Joint Frozen, vs our frozen0, vs Joint TAR.
    python scripts/p10_report.py --out <export>/results
"""
from __future__ import annotations
import os as _os
import sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # archive/scripts -> repo
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "scripts"), _os.path.dirname(_os.path.abspath(__file__))]

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from cte_report import EXCLUDED_HELDOUT13, _subset  # noqa: E402
from p5_report import comparison, leaderboard_state, runtime_hours  # noqa: E402

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))
NO_SPOTIFY = ["MUL_TEXT_SPOTIFY_GENRES"]


def p10_report(out: str) -> dict:
    lb_ft, lb_all = leaderboard_state("catboost", "ft"), leaderboard_state("catboost", "all")
    q, probe = fr.ours(out, "p9_query", "catboost"), fr.ours(out, "cte_probe", "catboost")
    fq, fz = fr.ours(out, "frozen0_query", "catboost"), fr.ours(out, "frozen0", "catboost")
    ns = lambda d: _subset(d, NO_SPOTIFY)
    rep = {"label": "P10 - claim C6; C1-C5 unchanged", "C6": comparison(q, lb_ft),
           "C6_vs_our_cte_probe": comparison(q, probe),
           "C6_heldout13": comparison(_subset(q, EXCLUDED_HELDOUT13), _subset(lb_ft, EXCLUDED_HELDOUT13)),
           "frozen_query_vs_lb_frozen": comparison(ns(fq), ns(lb_all)),
           "frozen_query_vs_our_frozen0": comparison(ns(fq), ns(fz)),
           "frozen_query_vs_lb_tar": comparison(ns(fq), ns(lb_ft)), "runtime_hours": {}}
    c = rep["C6"]
    c["status"] = None if c["n"] == 20 else "pending"
    c["passes"] = None if c["n"] != 20 else bool(c["p_value"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)
    for m in ("p9_query", "frozen0_query"):
        h = runtime_hours(out, m, "catboost")
        if h is not None:
            rep["runtime_hours"][m] = h
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "archive", "reports", "p10"))
    a = p.parse_args()
    rep = p10_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p10_report.json"), "w"), indent=1)
    print(rep["label"])
    for k, c in rep.items():
        if isinstance(c, dict) and "n" in c:
            extra = f" -> {'PENDING' if c.get('status') else ('PASS' if c['passes'] else 'no pass')}" if k == "C6" else ""
            print(f"{k:<28} n={c['n']} mean={c.get('mean_diff')} CI={c.get('ci95')} p={c.get('p_value')} wins={c.get('wins')}{extra}")
    print("runtime hours:", rep["runtime_hours"])


if __name__ == "__main__":
    main()

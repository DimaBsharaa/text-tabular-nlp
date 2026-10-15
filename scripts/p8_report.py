"""P8 analysis (PROGRESS.md P8, approved 2026-10-16), written before any P8 result. C2 / C3 stay as reported.

C4 (label-free): tf_num_lsa (CatBoost) vs leaderboard Joint Frozen (`all`), 20 datasets, fold-averaged, two-sided
    Wilcoxon. Environment control (secondary): tf_num_lsa vs our frozen0, and frozen0 vs leaderboard Joint Frozen, on the
    19 datasets without Spotify Genres. Descriptive: tf_num_lsa vs Joint TAR.
C5 (label-aware): cte_probe (CatBoost) vs leaderboard Joint TAR (`ft`), 20 datasets, same test. Secondary: held-out 13;
    cte_probe vs our cte (paired).
Holm over {C4, C5}; pass = Holm-adjusted p < 0.05 AND positive mean. A claim whose results are missing is "pending".
    python scripts/p8_report.py --out <export>/results
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
NO_SPOTIFY = ["MUL_TEXT_SPOTIFY_GENRES"]


def p8_report(out: str) -> dict:
    lb_all, lb_ft = leaderboard_state("catboost", "all"), leaderboard_state("catboost", "ft")
    tf, fz = fr.ours(out, "tf_num_lsa", "catboost"), fr.ours(out, "frozen0", "catboost")
    pr, cte = fr.ours(out, "cte_probe", "catboost"), fr.ours(out, "cte", "catboost")
    rep = {"label": "P8 - claims C4 / C5; C2 and C3 unchanged",
           "C4": comparison(tf, lb_all), "C5": comparison(pr, lb_ft),
           "C4_env_control_vs_our_frozen0": comparison(_subset(tf, NO_SPOTIFY), _subset(fz, NO_SPOTIFY)),
           "env_frozen0_vs_leaderboard_frozen": comparison(_subset(fz, NO_SPOTIFY), _subset(lb_all, NO_SPOTIFY)),
           "C4_descriptive_vs_joint_tar": comparison(tf, lb_ft),
           "C5_heldout13": comparison(_subset(pr, EXCLUDED_HELDOUT13), _subset(lb_ft, EXCLUDED_HELDOUT13)),
           "C5_vs_our_cte": comparison(pr, cte), "runtime_hours": {}}
    complete = {k: rep[k]["n"] == 20 for k in ("C4", "C5")}
    adj = fr.holm({k: rep[k].get("p_value") for k in ("C4", "C5") if complete[k]})
    for k in ("C4", "C5"):
        c = rep[k]
        c["status"] = "pending" if not complete[k] else None
        c["p_holm"] = adj.get(k)
        c["passes"] = None if not complete[k] else bool(c["p_holm"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)
    if not all(complete.values()):
        rep["note"] = "Holm over the completed claims only; recompute when both are complete"
    for m in ("tf_num_lsa", "frozen0", "cte_probe"):
        h = runtime_hours(out, m, "catboost")
        if h is not None:
            rep["runtime_hours"][m] = h
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p8"))
    a = p.parse_args()
    rep = p8_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p8_report.json"), "w"), indent=1)
    print(rep["label"])
    for k, c in rep.items():
        if isinstance(c, dict) and "n" in c:
            extra = (f" p_holm={c.get('p_holm')} -> {'PENDING' if c.get('status') else ('PASS' if c['passes'] else 'no pass')}"
                     if k in ("C4", "C5") else "")
            print(f"{k:<36} n={c['n']} mean={c.get('mean_diff')} CI={c.get('ci95')} p={c.get('p_value')} wins={c.get('wins')}{extra}")
    print("runtime hours:", rep["runtime_hours"], rep.get("note", ""))


if __name__ == "__main__":
    main()

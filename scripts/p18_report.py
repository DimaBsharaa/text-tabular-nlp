"""P18 analysis, written before any P18 result. C1-C12 stay as reported.

C13: cte_depth_top3_attn (the final method + the cross-fitted attention probe) vs the leaderboard Joint TAR of
     the SAME learner, for each of the 5 learners; the 20 datasets (TabPFN: the 18 with a leaderboard baseline),
     folds 0-4, fold-averaged; two-sided Wilcoxon over datasets; Holm over the 5 learners; a learner passes if
     Holm p < 0.05 AND the mean gain is positive.
Secondary (reported regardless of outcome): cte_depth_top3_attn vs our cte_depth_top3, paired, same learner and
     folds (Wilcoxon, mean, wins); the current method's own per-learner result under the same Holm-over-5 reading.
Final-method rule (fixed now): the final method becomes cte_depth_top3_attn iff
     (a) C13 passes for at least as many learners as cte_depth_top3 does under the same Holm-over-5 reading, AND
     (b) its paired mean vs cte_depth_top3 is positive for at least 4 of the 5 learners.
     Otherwise cte_depth_top3 stays final and P18 is reported as an analysis.
Also reported: Holm over all pre-registered tests (C1-C13).
    python scripts/p18_report.py --out <results root with cte_depth_top3_attn> [--baseline_out results/raw]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

sys.path.insert(0, dirname(abspath(__file__)))
import full_report as fr  # noqa: E402
from p5_report import comparison, leaderboard_state  # noqa: E402
from p16_report import EARLIER_P  # noqa: E402  (raw p-values of the pre-registered tests C1-C10)

REPO_ROOT = dirname(dirname(abspath(__file__)))
LEARNERS = ("catboost", "lgbm", "tabm", "tabpfnv2", "tabpfnv2p5")
NEW, CUR = "cte_depth_top3_attn", "cte_depth_top3"


def _n_full(learner: str) -> int:
    return 18 if learner.startswith("tabpfn") else 20


def _holm_pass(block: dict) -> None:
    adj = fr.holm({k: c.get("p_value") for k, c in block.items() if c["n"] == _n_full(k)})
    for k, c in block.items():
        c["p_holm"] = adj.get(k)
        c["passes"] = None if c["n"] != _n_full(k) else bool(c["p_holm"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)


def p18_report(out: str, baseline_out: str) -> dict:
    lb = {lr: leaderboard_state(lr, "ft") for lr in LEARNERS}
    rep = {"label": "P18 - claim C13 (attention probe, 5 learners); C1-C12 unchanged",
           "C13": {lr: comparison(fr.ours(out, NEW, lr), lb[lr]) for lr in LEARNERS},
           "current_method_same_reading": {lr: comparison(fr.ours(baseline_out, CUR, lr), lb[lr]) for lr in LEARNERS},
           "vs_current_method": {lr: comparison(fr.ours(out, NEW, lr), fr.ours(baseline_out, CUR, lr))
                                 for lr in LEARNERS}}
    _holm_pass(rep["C13"])
    _holm_pass(rep["current_method_same_reading"])
    n_new = [c["passes"] for c in rep["C13"].values()]
    n_cur = [c["passes"] for c in rep["current_method_same_reading"].values()]
    pos = [(c.get("mean_diff") or 0) > 0 for c in rep["vs_current_method"].values() if c["n"]]
    rep["learners_passing_new"] = None if None in n_new else sum(n_new)
    rep["learners_passing_current"] = None if None in n_cur else sum(n_cur)
    rep["learners_ahead_of_current"] = sum(pos)
    if rep["learners_passing_new"] is None or rep["learners_passing_current"] is None:
        rep["final_method"] = None  # incomplete runs: no decision
    else:
        rule = rep["learners_passing_new"] >= rep["learners_passing_current"] and sum(pos) >= 4
        rep["final_method"] = NEW if rule else CUR
    # Holm over every pre-registered test: C1-C10 (EARLIER_P), C11 and C12 from their reports, C13 from this one
    p16 = json.load(open(join(REPO_ROOT, "reports", "p16", "p16_report.json")))
    p17 = json.load(open(join(REPO_ROOT, "reports", "p17", "p17_report.json")))
    allp = dict(EARLIER_P, **{f"C11_{k}": c["p_value"] for k, c in p16["C11"].items()},
                **{f"C12_{k}": c["p_value"] for k, c in p17["C12"].items()},
                **{f"C13_{k}": c.get("p_value") for k, c in rep["C13"].items() if c["n"] == _n_full(k)})
    rep["holm_all_preregistered"] = {k: round(float(v), 6) for k, v in fr.holm(allp).items()}
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--baseline_out", default=join(REPO_ROOT, "results", "raw"))
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p18"))
    a = p.parse_args()
    rep = p18_report(a.out, a.baseline_out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p18_report.json"), "w"), indent=1)
    print(rep["label"])
    for block in ("C13", "current_method_same_reading", "vs_current_method"):
        print(f"-- {block}")
        for k, c in rep[block].items():
            print(f"{k:<11} n={c['n']} mean={c.get('mean_diff')} median={c.get('median_diff')} p={c.get('p_value')} "
                  f"p_holm={c.get('p_holm')} wins={c.get('wins')} passes={c.get('passes')}")
    print("learners passing: new", rep["learners_passing_new"], "| current", rep["learners_passing_current"],
          "| ahead of current:", rep["learners_ahead_of_current"], "| FINAL METHOD:", rep["final_method"])
    print(f"Holm over all {len(rep['holm_all_preregistered'])} pre-registered tests:", rep["holm_all_preregistered"])


if __name__ == "__main__":
    main()

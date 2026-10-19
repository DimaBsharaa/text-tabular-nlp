"""P15 analysis (PROGRESS.md P15), written before any P15 result. C1-C9 stay as reported.

C10: cte_depth_top3 with LightGBM and with TabM, each vs the leaderboard Joint TAR of the same learner (20 datasets,
     folds 0-4, two-sided Wilcoxon), Holm over the 2; pass = Holm p < 0.05 AND positive mean.
"≥ 3 of 5 learners" (original C1 draft) is met iff C9 (CatBoost) and both C10 tests pass.
Also: the cross-learner table (our CatBoost vs every learner's Joint TAR) and Holm over all pre-registered tests.
    python scripts/p15_report.py --out <export>/results
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
from p14_report import EARLIER_ALL  # noqa: E402

REPO_ROOT = dirname(dirname(abspath(__file__)))
C9_P = 0.0023250579833984375  # E38


def p15_report(out: str) -> dict:
    rep = {"label": "P15 - claim C10 (learner robustness of cte_depth_top3); C1-C9 unchanged", "C10": {}}
    for learner in ("lgbm", "tabm"):
        rep["C10"][learner] = comparison(fr.ours(out, "cte_depth_top3", learner), leaderboard_state(learner, "ft"))
    adj = fr.holm({k: c.get("p_value") for k, c in rep["C10"].items() if c["n"] == 20})
    for k, c in rep["C10"].items():
        c["p_holm"] = adj.get(k)
        c["passes"] = None if c["n"] != 20 else bool(c["p_holm"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)
    rep["three_of_five"] = (None if any(c["passes"] is None for c in rep["C10"].values())
                            else bool(all(c["passes"] for c in rep["C10"].values())))
    ours_cb = fr.ours(out, "cte_depth_top3", "catboost")
    rep["cross_learner_catboost_vs"] = {l: comparison(ours_cb, leaderboard_state(l, "ft"))
                                        for l in ("catboost", "lgbm", "tabm", "tabpfnv2", "tabpfnv2p5")}
    allp = dict(EARLIER_ALL, C9=C9_P, **{f"C10_{k}": c.get("p_value") for k, c in rep["C10"].items()})
    rep["holm_all_preregistered"] = {k: round(float(v), 6) for k, v in
                                     fr.holm({k: v for k, v in allp.items() if v is not None}).items()}
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p15"))
    a = p.parse_args()
    rep = p15_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p15_report.json"), "w"), indent=1)
    print(rep["label"], "| 3 of 5 learners:", rep["three_of_five"])
    for k, c in list(rep["C10"].items()) + [(f"CatBoost vs {l} Joint TAR", c) for l, c in rep["cross_learner_catboost_vs"].items()]:
        print(f"{k:<30} n={c['n']} mean={c.get('mean_diff')} p={c.get('p_value')} p_holm={c.get('p_holm')} wins={c.get('wins')} passes={c.get('passes')}")
    print("Holm over all pre-registered tests:", rep["holm_all_preregistered"])


if __name__ == "__main__":
    main()

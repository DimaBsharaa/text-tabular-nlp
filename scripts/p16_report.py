"""P16 analysis, written before any P16 result. C1-C10 stay as reported.

C11: cte_depth_top3 with TabPFN-v2 and TabPFN-2.5 as the learner, each vs the leaderboard Joint TAR of the same
     learner (18 datasets: no TabPFN baseline for Spotify / Wine), folds 0-4, two-sided Wilcoxon, Holm over the 2;
     pass = Holm p < 0.05 AND positive mean. Reports how many of the 5 learners pass (C9 CatBoost, C10 LightGBM / TabM).
    python scripts/p16_report.py --out <export>/results
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
EARLIER_PASSES = {"catboost": True, "lgbm": True, "tabm": True}  # C9, C10
# raw two-sided Wilcoxon p-values of the earlier pre-registered tests, from reports/p14 and reports/p15
EARLIER_P = dict(EARLIER_ALL, C9=0.0023250579833984375,
                 **{"C7:cte_probe|vs_joint_frozen": 0.01918059220770374,
                    "C7:cte_probe|vs_joint_tar": 0.024941622250480577,
                    "C7:cte_depth_top3|vs_joint_frozen": 0.27855437609832734,
                    "C7:cte_depth_top3|vs_joint_tar": 0.016752589581301436,
                    "C10_lgbm": 0.005580902099609375, "C10_tabm": 0.0007076263427734375})


def p16_report(out: str) -> dict:
    rep = {"label": "P16 - claim C11 (TabPFN learners); C1-C10 unchanged", "C11": {}}
    for learner in ("tabpfnv2", "tabpfnv2p5"):
        rep["C11"][learner] = comparison(fr.ours(out, "cte_depth_top3", learner), leaderboard_state(learner, "ft"))
    adj = fr.holm({k: c.get("p_value") for k, c in rep["C11"].items() if c["n"] == 18})
    for k, c in rep["C11"].items():
        c["p_holm"] = adj.get(k)
        c["passes"] = None if c["n"] != 18 else bool(c["p_holm"] < fr.ALPHA and (c.get("mean_diff") or 0) > 0)
    passes = dict(EARLIER_PASSES, **{k: c["passes"] for k, c in rep["C11"].items()})
    rep["learners_passing"] = None if any(v is None for v in passes.values()) else sum(passes.values())
    rep["per_learner_pass"] = passes
    allp = dict(EARLIER_P, **{f"C11_{k}": c.get("p_value") for k, c in rep["C11"].items() if c["n"] == 18})
    rep["holm_all_preregistered"] = {k: round(float(v), 6) for k, v in fr.holm(allp).items()}
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--report_dir", default=join(REPO_ROOT, "reports", "p16"))
    a = p.parse_args()
    rep = p16_report(a.out)
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump(rep, open(join(a.report_dir, "p16_report.json"), "w"), indent=1)
    print(rep["label"], "| learners passing (of 5):", rep["learners_passing"])
    for k, c in rep["C11"].items():
        print(f"{k:<11} n={c['n']} mean={c.get('mean_diff')} median={c.get('median_diff')} p={c.get('p_value')} "
              f"p_holm={c.get('p_holm')} wins={c.get('wins')} passes={c.get('passes')}")
    print(f"Holm over all {len(rep['holm_all_preregistered'])} pre-registered tests:", rep["holm_all_preregistered"])


if __name__ == "__main__":
    main()

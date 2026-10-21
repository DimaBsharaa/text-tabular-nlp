"""scripts/full_report.py on synthetic result JSONs (no torch, no data)."""
import json
import os
import sys
from os.path import abspath, dirname, join

import numpy as np

sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "scripts"))
sys.path.insert(1, join(dirname(dirname(abspath(__file__))), "archive", "scripts"))  # reports of earlier claims
import full_report as fr  # noqa: E402


def _write(out, run, learner, scores, gpu="NVIDIA A100-SXM4-40GB"):  # noqa: E302
    for (ds, fold), s in scores.items():
        d = join(out, run, learner, ds)
        os.makedirs(d, exist_ok=True)
        json.dump({"dataset": ds, "fold": fold, "score": s, "gpu": gpu, "runtime_s": {"total": 60.0}},
                  open(join(d, f"fold{fold}.json"), "w"))


def _lb(learner, offset=0.0, drop=()):
    lb = fr.leaderboard(learner)
    return {(r.dataset, r.fold): r.score + offset for r in lb.itertuples() if r.dataset not in drop}


def test_leaderboard_has_all_datasets():
    assert fr.leaderboard("catboost").dataset.nunique() == 20
    assert fr.leaderboard("tabpfnv2").dataset.nunique() == 18


def test_primary_passes_with_a_consistent_gain_and_fails_with_none(tmp_path):
    out = str(tmp_path)
    rng = np.random.default_rng(0)
    _write(out, "frozen_xfit_pred", "catboost", {k: v + 0.01 + rng.normal(0, 0.002) for k, v in _lb("catboost").items()})
    _write(out, "tar_repro", "catboost", _lb("catboost"))
    rep = fr.full_report(out)
    p = rep["primary"]["catboost"]
    assert p["n"] == 20 and p["passes"] and p["wins"] == 20 and p["ci95"][0] > 0 and p["rank_biserial_r"] == 1.0
    assert rep["vs_tar_repro"]["n"] == 20 and rep["same_gpu_type"]
    out2 = str(tmp_path / "null")
    _write(out2, "frozen_xfit_pred", "catboost", _lb("catboost"))  # identical to the leaderboard -> no pass
    assert not fr.full_report(out2)["primary"]["catboost"]["passes"]


def test_incomplete_datasets_are_excluded_and_reported(tmp_path):
    out = str(tmp_path)
    sc = _lb("catboost", 0.01)
    sc.pop(("REG_TEXT_BOOK_PRICE", 4))
    _write(out, "frozen_xfit_pred", "catboost", sc)
    p = fr.full_report(out)["primary"]["catboost"]
    assert p["n"] == 19 and p["excluded_datasets"] == ["REG_TEXT_BOOK_PRICE"]


def test_holm_and_mixed_gpu_detection(tmp_path):
    adj = fr.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)
    out = str(tmp_path)
    _write(out, "frozen_xfit_pred", "catboost", _lb("catboost", 0.01))
    _write(out, "tar_repro", "catboost", _lb("catboost"), gpu="NVIDIA L4")
    assert not fr.full_report(out)["same_gpu_type"]


def test_p5_report_on_synthetic_results(tmp_path):
    import p5_report as p5
    out = str(tmp_path)
    rng = np.random.default_rng(0)
    for run, off in (("tar_experts", 0.01), ("frozen_experts", 0.0), ("tar_experts_lex", 0.005),
                     ("tar_experts_nbr", 0.004), ("tar_experts_lex_insample", -0.01)):
        sc = {k: v + off + rng.normal(0, 0.002) for k, v in _lb("catboost").items()}
        _write(out, run, "catboost", sc)
    rep = p5.p5_report(out)
    assert rep["C2"]["passes"] and rep["C2"]["n"] == 20 and not rep["C3"]["passes"]
    assert rep["ablations"]["tar_experts_lex_insample"]["vs_tar_experts"]["mean_diff"] < 0
    assert rep["secondary"]["tar_experts/lgbm"]["n"] == 0


def test_cte_report_on_synthetic_results(tmp_path):
    import cte_report as cr
    out = str(tmp_path)
    rng = np.random.default_rng(0)
    for run, off in (("cte", 0.01), ("cte_lex", 0.006), ("cte_nbr", 0.003), ("cte_num", 0.0),
                     ("cte_lex_insample", -0.02)):
        _write(out, run, "catboost", {k: v + off + rng.normal(0, 0.002) for k, v in _lb("catboost").items()})
    rep = cr.cte_report(out)
    assert rep["C2"]["n"] == 20 and rep["C2"]["passes"]
    assert rep["C2_16"]["n"] == 16 and rep["C2_14"]["n"] == 14
    assert rep["C2_heldout13"]["n"] == 13 and rep["C2_strict11"]["n"] == 11
    assert "BIN_TEXT_JIGSAW_TOXICITY" not in rep["C2_heldout13"]["per_dataset"]
    assert "REG_TEXT_BOOK_PRICE" not in rep["C2_16"]["per_dataset"]
    assert rep["ablations"]["cte_lex_insample"]["vs_cte"]["mean_diff"] < 0
    assert rep["secondary"]["lgbm"]["n"] == 0


def test_p6_report_on_synthetic_results(tmp_path):
    import p6_report as pr
    out = str(tmp_path)
    rng = np.random.default_rng(1)
    for run, off in (("cte", 0.01), ("cte_nb", 0.011), ("cte_residual", -0.004)):
        _write(out, run, "catboost", {k: v + off + rng.normal(0, 0.002) for k, v in _lb("catboost").items()})
    rep = pr.p6_report(out)
    assert set(rep["C3"]) == {"cte_nb", "cte_residual"} and "does not change claim C2" in rep["label"]
    nb, res = rep["C3"]["cte_nb"], rep["C3"]["cte_residual"]
    assert nb["n"] == 20 and nb["passes"] and nb["p_holm"] >= nb["p_value"]
    assert not res["passes"] and res["mean_diff"] < 0
    assert rep["C3_heldout13"]["cte_nb"]["n"] == 13 and rep["native_text_partial"] == []
    assert set(rep["all20"]) == {"cte_nb", "cte_residual"} and rep["all20"]["cte_nb"]["vs_cte"]["n"] == 20


def test_p8_report_on_synthetic_results(tmp_path):
    import p8_report as pr
    out = str(tmp_path)
    rng = np.random.default_rng(2)
    from p5_report import leaderboard_state
    lb_all = {(r.dataset, r.fold): r.score for r in leaderboard_state("catboost", "all").itertuples()}
    assert len(lb_all) == 100
    for run, off in (("tf_num_lsa", 0.02), ("frozen0", 0.0)):
        _write(out, run, "catboost", {k: v + off + rng.normal(0, 0.001) for k, v in lb_all.items()})
    rep = pr.p8_report(out)
    assert rep["C4"]["n"] == 20 and rep["C4"]["passes"] and rep["C4"]["p_holm"] == rep["C4"]["p_value"]
    assert rep["C5"]["status"] == "pending" and rep["C5"]["passes"] is None and "note" in rep
    assert rep["C4_env_control_vs_our_frozen0"]["n"] == 19 and rep["env_frozen0_vs_leaderboard_frozen"]["n"] == 19
    for run, off in (("cte_probe", -0.002), ("cte", 0.0)):
        _write(out, run, "catboost", {k: v + off + rng.normal(0, 0.001) for k, v in _lb("catboost").items()})
    rep = pr.p8_report(out)
    assert rep["C5"]["n"] == 20 and not rep["C5"]["passes"] and rep["C4"]["p_holm"] >= rep["C4"]["p_value"]
    assert rep["C5_heldout13"]["n"] == 13 and rep["C5_vs_our_cte"]["n"] == 20


def test_p10_report_on_synthetic_results(tmp_path):
    import p10_report as pr
    from p5_report import leaderboard_state
    out = str(tmp_path)
    rng = np.random.default_rng(5)
    lb_all = {(r.dataset, r.fold): r.score for r in leaderboard_state("catboost", "all").itertuples()}
    for run, off in (("p9_query", 0.02), ("cte_probe", 0.012)):
        _write(out, run, "catboost", {k: v + off + rng.normal(0, 0.001) for k, v in _lb("catboost").items()})
    for run, off in (("frozen0_query", 0.01), ("frozen0", 0.0)):
        _write(out, run, "catboost", {k: v + off + rng.normal(0, 0.001) for k, v in lb_all.items()})
    rep = pr.p10_report(out)
    assert rep["C6"]["n"] == 20 and rep["C6"]["passes"] and rep["C6_vs_our_cte_probe"]["mean_diff"] > 0
    assert rep["C6_heldout13"]["n"] == 13 and rep["frozen_query_vs_lb_frozen"]["n"] == 19
    assert rep["frozen_query_vs_our_frozen0"]["mean_diff"] > 0


def test_p11_report_on_synthetic_results(tmp_path):
    import p11_report as pr
    out = str(tmp_path)
    rng = np.random.default_rng(6)
    src = {(r.dataset, r.fold): r.score for r in pr.source_leaderboard("catboost", "all").itertuples()}
    assert len({d for d, _ in src}) == 36 and len(src) == 180
    partial = {k: v + 0.01 + rng.normal(0, 0.001) for k, v in src.items() if not (k[0].startswith("REG_TEXT_FOOD_BEER") and k[1] == 4)}
    _write(out, "cte_probe", "catboost", partial)
    _write(out, "cte_probe", "lgbm", {k: v + 0.01 + rng.normal(0, 0.001) for k, v in _lb("lgbm").items()})
    rep = pr.p11_report(out)
    assert rep["n_datasets_complete"] == 35 and rep["missing"] == ["REG_TEXT_FOOD_BEER_RATINGS"]
    assert rep["C7a_vs_joint_frozen"]["n"] == 35 and rep["C7a_vs_joint_frozen"]["passes"]
    assert rep["C8_lgbm_vs_joint_tar"]["n"] == 20 and rep["C8_lgbm_vs_joint_tar"]["passes"]
    assert rep["holm_all_preregistered"]["C5"] >= 0.002325


def test_p14_report_on_synthetic_results(tmp_path):
    import p14_report as pr
    from p11_report import source_leaderboard
    out = str(tmp_path)
    rng = np.random.default_rng(7)
    _write(out, "cte_depth_top3", "catboost", {k: v + 0.02 + rng.normal(0, 0.001) for k, v in _lb("catboost").items()})
    _write(out, "cte_probe", "catboost", {k: v + 0.015 + rng.normal(0, 0.001) for k, v in _lb("catboost").items()})
    rep = pr.p14_report(out)
    assert rep["C9"]["passes"] and rep["final_method"] == "cte_depth_top3" and rep["C7"]["cte_probe|vs_joint_frozen"]["n"] == 0
    src = {(r.dataset, r.fold): r.score for r in source_leaderboard("catboost", "all").itertuples() if r.fold in (0, 1, 2)}
    for m, off in (("cte_depth_top3", 0.01), ("cte_probe", 0.008)):
        _write(out, m, "catboost", {k: v + off + rng.normal(0, 0.001) for k, v in src.items()})
    rep = pr.p14_report(out)
    c = rep["C7"]["cte_depth_top3|vs_joint_frozen"]
    assert c["n"] == 36 and c["passes"] and c["p_holm"] >= c["p_value"]
    assert len(rep["C7"]) == 4 and "C9" in rep["holm_all_preregistered"]


def test_p15_report_on_synthetic_results(tmp_path):
    import p15_report as pr
    out = str(tmp_path)
    rng = np.random.default_rng(8)
    _write(out, "cte_depth_top3", "lgbm", {k: v + 0.015 + rng.normal(0, 0.001) for k, v in _lb("lgbm").items()})
    rep = pr.p15_report(out)
    assert rep["C10"]["lgbm"]["n"] == 20 and rep["C10"]["tabm"]["n"] == 0 and rep["three_of_five"] is None
    _write(out, "cte_depth_top3", "tabm", {k: v + 0.012 + rng.normal(0, 0.001) for k, v in _lb("tabm").items()})
    rep = pr.p15_report(out)
    assert rep["C10"]["lgbm"]["passes"] and rep["C10"]["tabm"]["passes"] and rep["three_of_five"] is True
    assert "C10_tabm" in rep["holm_all_preregistered"]


def test_p16_report_on_synthetic_results(tmp_path):
    import p16_report as pr
    out = str(tmp_path)
    rng = np.random.default_rng(9)
    for learner, off in (("tabpfnv2", 0.02), ("tabpfnv2p5", -0.001)):
        _write(out, "cte_depth_top3", learner, {k: v + off + rng.normal(0, 0.002) for k, v in _lb(learner).items()})
    rep = pr.p16_report(out)
    assert rep["C11"]["tabpfnv2"]["n"] == 18 and rep["C11"]["tabpfnv2"]["passes"]
    assert rep["C11"]["tabpfnv2p5"]["passes"] is False and rep["learners_passing"] == 4
    holm = rep["holm_all_preregistered"]  # C1-C8 (C3 x2), C9, C7 x4, C10 x2, C11 x2
    assert len(holm) == 17 and holm["C1"] >= 0.409 and all(0 < v <= 1 for v in holm.values())


def test_p17_report_on_synthetic_results(tmp_path):
    import p17_report as pr
    out = str(tmp_path)
    rng = np.random.default_rng(17)
    for run, off in (("cte_depth_top3", 0.02), ("cte_depth_top3_xs43", 0.019), ("cte_depth_top3_xs44", -0.002)):
        _write(out, run, "lgbm", {k: v + off + rng.normal(0, 0.002) for k, v in _lb("lgbm").items()})
    rep = pr.p17_report(out)
    assert rep["C12"]["seed43"]["n"] == 20 and rep["C12"]["seed43"]["passes"]
    assert rep["C12"]["seed44"]["passes"] is False and rep["C12_passes"] is False
    assert rep["seed42_C10"]["n"] == 20 and rep["seed_spread"]["datasets"] == 20

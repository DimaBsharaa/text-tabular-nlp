"""scripts/full_report.py on synthetic result JSONs (no torch, no data)."""
import json
import os
import sys
from os.path import abspath, dirname, join

import numpy as np

sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "scripts"))
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
    for run, off in (("cte", 0.01), ("catboost_native_text", -0.01), ("cte_plus_native", 0.012), ("cte_nb", 0.011),
                     ("cte_residual", 0.009)):
        _write(out, run, "catboost", {k: v + off + rng.normal(0, 0.002) for k, v in _lb("catboost").items()})
    rep = pr.p6_report(out)
    assert set(rep["all20"]) == set(pr.P6_VARIANTS) and "does not change claim C2" in rep["label"]
    nat = rep["all20"]["catboost_native_text"]
    assert nat["vs_leaderboard_ft"]["n"] == 20 and nat["vs_leaderboard_ft"]["mean_diff"] < 0
    assert nat["vs_cte"]["mean_diff"] < 0 and nat["vs_cte"]["p_holm"] >= nat["vs_cte"]["p_value"]
    assert rep["heldout13"]["cte_nb"]["vs_cte"]["n"] == 13
    assert set(rep["C3"]) == {"cte_plus_native", "cte_nb", "cte_residual"}
    assert rep["C3"]["cte_plus_native"]["passes"] and rep["C3"]["cte_plus_native"]["p_holm"] >= rep["C3"]["cte_plus_native"]["p_value"]
    assert rep["C3_heldout13"]["cte_nb"]["n"] == 13 and "passes" not in rep["baseline_native_text_vs_ft"]
    assert rep["baseline_native_text_vs_ft"]["mean_diff"] < 0

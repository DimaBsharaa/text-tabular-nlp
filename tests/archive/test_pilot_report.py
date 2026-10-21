"""scripts/pilot_report.py on synthetic result JSONs (no torch, no data)."""
import json
import sys
from os.path import abspath, dirname, join

import numpy as np

sys.path.insert(0, join(dirname(dirname(dirname(abspath(__file__)))), "archive", "scripts"))
import pilot_report as pr  # noqa: E402


def _write(out, run, scores, csv=0.5, head=(0.45, 0.40), seed=42):
    for (ds, fold), s in scores.items():
        d = join(out, run, "catboost", ds)
        import os
        os.makedirs(d, exist_ok=True)
        r = {"dataset": ds, "fold": fold, "score": s, "reference": {"joint_tar": csv}, "engine_version": "tar_engine/2",
             "config": {"seed": seed}, "git": {"commit": "abcdef1"},
             "tar": {"head_quality": {"e5_train": {"score": head[0]}, "test": {"score": head[1]},
                                      "learner_val": {"score": head[1]}, "e5_val": {"score": head[1]}}}}
        json.dump(r, open(join(d, f"fold{fold}.json"), "w"))


def _scores(offset=0.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return {(ds, f): 0.5 + offset + rng.normal(0, noise) for ds in pr.PILOT_DATASETS for f in pr.PILOT_FOLDS}


def test_engine_bias_and_leakage(tmp_path):
    _write(str(tmp_path), "tar_repro", _scores(offset=-0.01, noise=0.002), csv=0.5)
    base = pr.load_runs(str(tmp_path), "tar_repro")
    assert len(base) == 18
    per_ds, overall = pr.engine_bias(base)
    assert abs(overall["mean_diff"] + 0.01) < 0.003 and overall["n_negative"] == 18
    assert overall["wilcoxon_jobs"]["n"] == 18 and overall["wilcoxon_jobs"]["p_value"] < 0.001
    lk_ds, lk = pr.leakage_summary(base)
    assert np.isclose(lk["mean_gap"], 0.05) and len(lk_ds) == 6


def test_advance_rule_and_aa(tmp_path):
    out = str(tmp_path)
    _write(out, "tar_repro", _scores(seed=1))
    _write(out, "tar_repro__s43", _scores(noise=0.0005, seed=2), seed=43)   # A/A: tiny noise -> must fail
    _write(out, "variant_good", _scores(offset=0.01, seed=3))               # +0.01 everywhere -> passes
    base = pr.load_runs(out, "tar_repro")
    _, aa = pr.advance_rule(pr.load_runs(out, "tar_repro__s43"), base)
    assert aa["complete"] and not aa["passes"]
    _, good = pr.advance_rule(pr.load_runs(out, "variant_good"), base)
    assert good["passes"] and good["wins"] == 6 and abs(good["mean_gain"] - 0.01) < 1e-9
    cand = pr.aa_threshold_candidates(None, aa)
    assert cand["candidate_2x_mean_abs"] == 2 * aa["mean_abs_gain"]


def test_incomplete_never_passes_and_old_engine_files_ignored(tmp_path):
    out = str(tmp_path)
    full = _scores()
    _write(out, "tar_repro", full)
    _write(out, "v", {k: v + 0.05 for k, v in list(full.items())[:12]})  # only 12 of 18 jobs
    json.dump({}, open(join(out, "tar_repro", "catboost", pr.PILOT_DATASETS[0], "fold0.tar_engine-1.json"), "w"))
    base = pr.load_runs(out, "tar_repro")
    assert len(base) == 18
    _, s = pr.advance_rule(pr.load_runs(out, "v"), base)
    assert not s["complete"] and not s["passes"]


def test_control_is_mean_of_seeds_and_requires_both(tmp_path):
    out = str(tmp_path)
    a, b = _scores(seed=1), _scores(seed=2)
    _write(out, "tar_repro", a)
    _write(out, "tar_repro__s43", dict(list(b.items())[:17]), seed=43)  # one pair missing
    c = pr.load_control(out, ["tar_repro", "tar_repro__s43"])
    assert len(c) == 17
    row = c.iloc[0]
    k = (row.dataset, row.fold)
    assert np.isclose(row.score, (a[k] + b[k]) / 2)
    _write(out, "v", {k: v + 0.02 for k, v in a.items()})
    _, s = pr.advance_rule(pr.load_runs(out, "v"), c)
    assert not s["complete"] and not s["passes"]  # incomplete control -> never passes

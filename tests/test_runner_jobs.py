"""run_experiment.py jobs runner: result validity across engine versions, per-job timeout, process_errors log."""
import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

import pytest

sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "scripts"))
import run_experiment as rx  # noqa: E402


def _write(path, **r):
    os.makedirs(dirname(path), exist_ok=True)
    json.dump(r, open(path, "w"))
    return path


def test_existing_result_ok_rules(tmp_path):
    p = lambda m, k: str(tmp_path / m / "catboost" / "D" / f"fold{k}.json")
    # native text: no tar block -> the top-level engine_version decides (bug fix; was always "stale")
    assert rx.existing_result_ok(_write(p("catboost_native_text", 0), method="catboost_native_text",
                                        engine_version="p6/3", tar=None, native_text={"feature_calcers": ["BoW"]}))
    # amendment 2: a p6/2 native result that was already BoW-only has the p6/3 configuration -> kept
    assert rx.existing_result_ok(_write(p("catboost_native_text", 1), method="catboost_native_text",
                                        engine_version="p6/2", tar=None, native_text={"feature_calcers": ["BoW"]}))
    # ... one with NaiveBayes / BM25 is renamed and re-run
    q = _write(p("catboost_native_text", 2), method="catboost_native_text", engine_version="p6/2", tar=None,
               native_text={"feature_calcers": ["BoW", "NaiveBayes", "BM25"]})
    assert not rx.existing_result_ok(q) and not os.path.exists(q) and os.path.exists(q[:-5] + ".p6-2.json")
    # cte_nb / cte_residual are untouched by amendment 2: p6/2 stays current
    assert rx.existing_result_ok(_write(p("cte_nb", 0), method="cte_nb", engine_version="p6/2",
                                        tar={"config": {"engine_version": "p6/2"}}))
    # CTE (claim C2) results unchanged
    assert rx.existing_result_ok(_write(p("cte", 0), method="cte", engine_version="cte/3",
                                        tar={"config": {"engine_version": "cte/3"}}))


FAKE_JOB = '''import argparse, json, os, sys, time
a = argparse.ArgumentParser(); [a.add_argument(k) for k in ("--method", "--learner", "--dataset", "--fold", "--seed",
    "--out", "--cache_dir", "--train_examples", "--epochs", "--micro_batch_size", "--lower_cache_dir")]
a.add_argument("--no_fast", action="store_true"); a.add_argument("--force", action="store_true")
x, _ = a.parse_known_args()
if x.dataset == "SLOW":
    time.sleep(60)
p = os.path.join(x.out, x.method, x.learner, x.dataset, f"fold{x.fold}.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump({"method": x.method, "engine_version": "cte/3", "score": 0.5, "reference": {"joint_tar": 0.4}}, open(p, "w"))
'''


def test_parallel_timeout_logged_not_retried(tmp_path, monkeypatch):
    fake = tmp_path / "fake_job.py"
    fake.write_text(FAKE_JOB)
    monkeypatch.setattr(rx, "__file__", str(fake))  # the runner launches abspath(__file__) per job
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("cte catboost FAST 0\ncte catboost SLOW 0\n")
    out = tmp_path / "out"
    args = argparse.Namespace(jobs_file=str(jobs), seed=42, filter=[], out=str(out), cache_dir=str(out / "_cache"),
                              train_examples=10000, epochs=50, micro_batch_size=256, no_fast=False,
                              lower_cache_dir=None, max_lower_cache_gb=40.0, force=False, stream=False, max_jobs=None,
                              parallel_jobs="2", job_timeout_min=0.05, retry_timeouts=False)  # 3 s limit
    assert rx.run_jobs(args) == 0  # a timeout is reported as missing, not as a failure
    slow = out / "cte" / "catboost" / "SLOW" / "fold0.json"
    assert (out / "cte" / "catboost" / "FAST" / "fold0.json").exists() and not slow.exists()
    errs = [json.loads(line) for line in open(out / "_logs" / "process_errors.jsonl")]
    assert [(e["status"], e["dataset"]) for e in errs] == [("timeout", "SLOW")] and errs[0]["seconds"] < 30
    assert os.path.exists(rx.timeout_marker(str(slow)))
    # second run: FAST is done, SLOW is not retried
    import time
    t0 = time.time()
    assert rx.run_jobs(args) == 0 and time.time() - t0 < 20
    assert len(open(out / "_logs" / "process_errors.jsonl").readlines()) == 1


def test_timeout_only_in_parallel_mode():
    args = argparse.Namespace(parallel_jobs="1")
    assert rx.parallel_plan(args, [("tar_repro", "catboost", "D", 0, 42)]) == (1, {})


def test_auto_parallel_for_every_p8_p10_jobs_file(monkeypatch):
    """Every CPU-bound jobs file must run in parallel ('frozen0_query' was missed once: 28 h instead of 4.6 h)."""
    monkeypatch.setattr(rx.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(rx.shutil, "which", lambda _: None)
    root = dirname(dirname(abspath(__file__)))
    for f in ("archive/jobs/p8_A.txt", "archive/jobs/p8_B.txt", "archive/jobs/p10.txt", "archive/jobs/p11.txt", "jobs/p14_A.txt", "jobs/p14_B.txt", "jobs/p15.txt",
              "jobs/p17.txt",
              "archive/jobs/cte_full.txt"):
        assert rx.parallel_plan(argparse.Namespace(parallel_jobs="auto"), rx.read_jobs(join(root, f)))[0] == 6, f
    # GPU-bound jobs (the attention probe, the TabPFN learners) share one GPU: one at a time unless asked otherwise
    for f in ("jobs/p16.txt", "jobs/p18.txt", "jobs/p18_check.txt", "jobs/p18_a100.txt"):
        assert rx.parallel_plan(argparse.Namespace(parallel_jobs="auto"), rx.read_jobs(join(root, f)))[0] == 1, f


def test_dataset_lock_serialises_parallel_loads(tmp_path):
    """E35: parallel jobs of one dataset must not read a half-downloaded file. Two threads: the second waits."""
    import threading
    import time
    order = []

    def job(i):
        with rx.dataset_lock("SYNTH_LOCK_TEST"):
            order.append(("in", i))
            time.sleep(0.3)
            order.append(("out", i))
    ts = [threading.Thread(target=job, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert order[0][0] == "in" and order[1][0] == "out" and order[2][0] == "in" and order[3][0] == "out"

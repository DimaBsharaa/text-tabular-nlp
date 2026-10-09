"""tar_repro is frozen at commit 96c1f5c (the code the tar_repro pilot ran on).

If this test fails, something changed tar_repro's method — its training texts, labels, E5 split, tokenization,
epoch-1 batch order, config or config hash — and tar_repro results would no longer be comparable with the pilot.
Implement variants alongside tar_repro instead of changing it. Golden values: tests/golden/tar_repro_96c1f5c.json.
"""
import argparse
import json
import sys
from os.path import abspath, dirname, join

HERE = dirname(abspath(__file__))
sys.path.insert(0, join(HERE, "golden"))
sys.path.insert(0, join(dirname(HERE), "scripts"))

from tar_repro_fingerprint import fingerprint  # noqa: E402

GOLDEN = json.load(open(join(HERE, "golden", "tar_repro_96c1f5c.json")))


def test_tar_repro_fingerprint_unchanged_since_96c1f5c():
    now = fingerprint()
    assert now["engine_version"] == GOLDEN["engine_version"] == "tar_engine/2"
    for key in ("config_result_dict_cuda", "config_result_dict_cpu", "config_hash_cuda", "config_hash_cpu",
                "bucketing_probe", "random_batches_probe"):
        assert now[key] == GOLDEN[key], f"tar_repro {key} changed"
    for task, gold in GOLDEN["tasks"].items():
        for field, value in gold.items():
            assert now["tasks"][task][field] == value, f"tar_repro {task}/{field} changed since 96c1f5c"


def test_run_experiment_defaults_still_select_frozen_tar_repro():
    import run_experiment as rx
    from multabench.ours.tar_engine import TarConfig

    assert "tar_repro" in rx.METHODS
    args = argparse.Namespace(**rx.DEFAULTS, no_fast=False)
    assert rx.run_name("tar_repro", args) == "tar_repro"  # results land where the pilot's are
    cfg = TarConfig()
    assert (rx.DEFAULTS["epochs"], rx.DEFAULTS["seed"], rx.DEFAULTS["train_examples"]) == (cfg.epochs, cfg.seed, 10_000)

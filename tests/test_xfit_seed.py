"""P17: the cross-fitting fold seed (method names *_xs<seed>) reaches every fold assignment and nothing else changes."""
import os
import subprocess
import sys
from os.path import abspath, dirname, join

import numpy as np

REPO = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(REPO, "scripts"))

PROBE = ("from multabench.ours import cte, p7; "
         "print(cte.CTE_CONFIG.seed, p7.xfit_linear.__defaults__[-1].seed, cte.CTE_CONFIG.expert_config().seed)")


def _seeds(env_seed=None):
    env = {k: v for k, v in os.environ.items() if k != "OURS_XFIT_SEED"}
    if env_seed is not None:
        env["OURS_XFIT_SEED"] = str(env_seed)
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    return [int(v) for v in r.stdout.split()]


def test_default_seed_is_42():
    assert _seeds() == [42, 42, 42]


def test_env_seed_reaches_the_probe_and_the_experts():
    assert _seeds(43) == [43, 43, 43]


def test_folds_change_with_the_seed():
    from multabench.ours.experts import ExpertConfig, expert_folds
    y = np.arange(200) % 3
    f42 = expert_folds(200, y, True, ExpertConfig(seed=42))
    f43 = expert_folds(200, y, True, ExpertConfig(seed=43))
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(f42, f43))


def test_names_and_engine():
    import run_experiment as rx
    from multabench.ours.variants import engine_version
    assert rx.split_xfit("cte_depth_top3_xs43") == ("cte_depth_top3", 43)
    assert rx.split_xfit("cte_depth_top3") == ("cte_depth_top3", None)
    assert engine_version("cte_depth_top3_xs44") == engine_version("cte_depth_top3") == "depth/1"
    assert {"cte_depth_top3_xs43", "cte_depth_top3_xs44"} <= set(rx.METHODS)

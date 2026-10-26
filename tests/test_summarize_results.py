"""summarize_results.py: from the committed raw runs it reproduces the committed results tables exactly."""
import filecmp
import sys
from os.path import abspath, dirname, join

import pytest

ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(ROOT, "scripts"))
import summarize_results as sr  # noqa: E402


def test_reproduces_committed_results(tmp_path):
    sr.summarize(join(ROOT, "results", "raw"), to=str(tmp_path))
    for f in ("results_summary.csv", "results_per_dataset.csv", "results_per_fold.csv"):
        assert filecmp.cmp(join(ROOT, "results", f), tmp_path / f, shallow=False), f


def test_no_runs_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="no runs"):
        sr.summarize(str(tmp_path))

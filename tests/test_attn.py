"""E52 attention probe: leakage, output columns, determinism, and that it reads a single decisive token."""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from multabench.ours import attn, cte

DEV = torch.device("cpu")
SPEC = attn.AttnSpec(hidden=16, epochs=15, patience=3, batch_size=64)


def _states(n: int, width: int = 24, seed: int = 0, signal: np.ndarray = None):
    """Synthetic token states for one column: row r has 8-20 random tokens; rows with signal[r] = 1 get one extra
    token along a fixed direction (the 'decisive word'), which a mean pool dilutes."""
    rng = np.random.RandomState(seed)
    direction = rng.randn(width)
    segs = []
    for r in range(n):
        t = rng.randn(rng.randint(8, 21), width)
        if signal is not None and signal[r]:
            t[rng.randint(len(t))] += 3 * direction
        segs.append(t.astype(np.float16))
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in segs])]).astype(np.int64)
    rows = np.arange(n) + 1000  # row indices are not positions
    return {"col": (np.concatenate(segs), offsets, {int(r): i for i, r in enumerate(rows)})}, rows


def _split(rows, n_pool):
    return rows[:n_pool], {"test": rows[n_pool:]}


def test_reads_a_single_decisive_token():
    n = 600
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, n)
    states, rows = _states(n, signal=y)
    pool, extra = _split(rows, 500)
    f, ext, meta = attn.xfit_attention(states, ["col"], pool, extra, y[:500], True, 2, cte.CTE_CONFIG, DEV, SPEC)
    assert f.shape == (500, 1) and ext["test"].shape == (100, 1)
    assert meta["oof_metric"] > 0.85
    assert roc_auc_score(y[500:], ext["test"][:, 0]) > 0.85


@pytest.mark.parametrize("task,n_classes,cols", [("bin", 2, 1), ("mul", 4, 3), ("reg", 1, 1)])
def test_output_columns(task, n_classes, cols):
    n = 300
    rng = np.random.RandomState(2)
    states, rows = _states(n)
    y = rng.randn(n) if task == "reg" else rng.randint(0, n_classes, n)
    pool, extra = _split(rows, 250)
    f, ext, _ = attn.xfit_attention(states, ["col"], pool, extra, y[:250], task != "reg", n_classes,
                                    cte.CTE_CONFIG, DEV, SPEC)
    assert f.shape == (250, cols) and ext["test"].shape == (50, cols)
    assert np.isfinite(f).all() and np.isfinite(ext["test"]).all()


def test_out_of_fold_feature_ignores_own_label():
    """Regression (unstratified folds): changing row r's label must not change r's own feature."""
    n = 300
    rng = np.random.RandomState(3)
    states, rows = _states(n)
    y = rng.randn(n)
    pool, extra = _split(rows, 250)
    f1, _, _ = attn.xfit_attention(states, ["col"], pool, extra, y[:250], False, 1, cte.CTE_CONFIG, DEV, SPEC)
    y2 = y[:250].copy()
    y2[17] += 50.0
    f2, _, _ = attn.xfit_attention(states, ["col"], pool, extra, y2, False, 1, cte.CTE_CONFIG, DEV, SPEC)
    assert f1[17, 0] == pytest.approx(f2[17, 0], abs=1e-6)
    assert not np.allclose(f1, f2)  # other rows' heads did see the change


def test_deterministic():
    n = 250
    rng = np.random.RandomState(4)
    states, rows = _states(n)
    y = rng.randint(0, 2, n)
    pool, extra = _split(rows, 200)
    a = attn.xfit_attention(states, ["col"], pool, extra, y[:200], True, 2, cte.CTE_CONFIG, DEV, SPEC)
    b = attn.xfit_attention(states, ["col"], pool, extra, y[:200], True, 2, cte.CTE_CONFIG, DEV, SPEC)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1]["test"], b[1]["test"])


def test_saved_heads_reproduce_extra_rows_exactly():
    """Packaged-model path: predict_heads with the kept fold heads gives the val/test rows exactly the features
    xfit_attention gave them (the mean of the same fold heads, same standardisation)."""
    n = 300
    rng = np.random.RandomState(5)
    states, rows = _states(n)
    y = rng.randint(0, 3, n)
    pool, extra = _split(rows, 240)
    f, ext, _, fit = attn.xfit_attention(states, ["col"], pool, extra, y[:240], True, 3, cte.CTE_CONFIG, DEV, SPEC,
                                         keep_heads=True)
    assert len(fit["heads"]) == cte.CTE_CONFIG.n_folds
    assert np.array_equal(attn.predict_heads(states, ["col"], extra["test"], fit, DEV), ext["test"])
    f2, ext2, _ = attn.xfit_attention(states, ["col"], pool, extra, y[:240], True, 3, cte.CTE_CONFIG, DEV, SPEC)
    assert np.array_equal(f, f2) and np.array_equal(ext["test"], ext2["test"])  # keep_heads changes nothing

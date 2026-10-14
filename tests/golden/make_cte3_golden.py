"""Writes tests/golden/cte3_features.npz: cte/3 expert features (P5 v2 after amendments 1+2) on small synthetic data,
generated from a clean checkout of f1514bc. tests/test_p6.py checks that later code (P6) reproduces them, so P5's
code path stays unchanged.
    python tests/golden/make_cte3_golden.py
"""
import sys
from os.path import abspath, dirname, join

import numpy as np
import torch

sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
from multabench.ours import cte  # noqa: E402

OUT = join(dirname(abspath(__file__)), "cte3_features.npz")
CASES = {"bin": (True, 2), "mul": (True, 4), "many": (True, 12), "reg": (False, 1)}


def synthetic(task: str, n: int = 160):
    rng = np.random.default_rng(11)
    words = ["great", "awful", "fine", "broken", "lovely", "cheap", "sturdy", "noisy", "bright", "dull"]
    texts = {c: [" ".join(rng.choice(words, 6)) + f" {i % 13}" for i in range(n)] for c in ("a", "b")}
    embs = {c: cte._normalise(rng.normal(size=(n, 16))) for c in ("a", "b")}
    is_cls, n_classes = CASES[task]
    y = rng.integers(0, n_classes, n) if is_cls else rng.lognormal(size=n)
    return texts, embs, y, is_cls, n_classes


def features(task: str) -> dict:
    texts, embs, y, is_cls, n_classes = synthetic(task)
    pool = {c: v[:130] for c, v in texts.items()}
    ext_t = {"test": {c: v[130:] for c, v in texts.items()}}
    pool_e = {c: v[:130] for c, v in embs.items()}
    ext_e = {"test": {c: v[130:] for c, v in embs.items()}}
    out = {}
    for kind, mode in (("lex", "oof"), ("lex", "insample"), ("nbr", "oof")):
        f = cte.compute_cte_features(kind, pool, pool_e if kind == "nbr" else None, y[:130], is_cls, n_classes, ext_t,
                                     ext_e if kind == "nbr" else None, mode, cte.CTE_CONFIG, torch.device("cpu"), {})
        for split, d in f.items():
            for name, a in d.items():
                out[f"{task}|{kind}|{mode}|{split}|{name}"] = np.asarray(a, dtype=float)
    return out


def all_features() -> dict:
    return {k: v for task in CASES for k, v in features(task).items()}


if __name__ == "__main__":
    np.savez_compressed(OUT, **all_features())
    print("wrote", OUT)

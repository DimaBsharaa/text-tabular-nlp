"""Exact fingerprint of a tiny tar_repro TRAINING run (CPU, 1 thread, fp32): embeddings, head logits, val losses.

Complements tar_repro_fingerprint.py (inputs/config): this one proves the training numerics are bit-for-bit the
same. Golden file tests/golden/tar_repro_96c1f5c_training.json was produced by running this against a git-worktree
checkout of 96c1f5c. Both the fast path and the full-model path are covered.

    PYTHONPATH=<checkout> .venv/bin/python tests/golden/tar_repro_training_fingerprint.py > tests/golden/tar_repro_96c1f5c_training.json
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile

import numpy as np


def _h(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def training_fingerprint() -> dict:
    import torch
    torch.set_num_threads(1)
    from tar_repro_fingerprint import synthetic_tables
    from multabench.ours import tar_engine as te

    out = {"engine_version": te.ENGINE_VERSION, "runs": {}}
    tables = synthetic_tables()
    for task in ("cls", "reg"):
        x, y, is_cls = tables[task]
        x_fit, y_fit = x.iloc[:220], y.iloc[:220]
        extra = {"test": (x.iloc[220:], y.iloc[220:])}
        for fast in (True, False):
            cfg = te.TarConfig(epochs=2, batch_size=32, micro_batch_size=32, eval_batch_size=16, fast=fast,
                               bucket_chunk_batches=2, lower_cache_dir=tempfile.mkdtemp())
            res = te.run_tar(x_fit, y_fit, ["brand", "review", "title"], is_cls, extra, cfg, torch.device("cpu"),
                             None, "SYNTH", 0)
            out["runs"][f"{task}_{'fast' if fast else 'full'}"] = {
                "val_losses": [h["val_loss"] for h in res.diagnostics["train"]["history"]],
                "emb": {c: _h(res.columns[c].emb) for c in res.text_cols},
                "logits": {c: _h(res.columns[c].logits) for c in res.text_cols},
                "head_quality": {k: v["score"] for k, v in res.diagnostics["head_quality"].items()},
            }
    return out


if __name__ == "__main__":
    print(json.dumps(training_fingerprint(), indent=1, sort_keys=True))

"""Fingerprint of the tar_repro method: everything that determines its results, hashed.

The golden file tests/golden/tar_repro_96c1f5c.json was produced by running this function against a checkout of
commit 96c1f5c (the code the tar_repro pilot runs on). tests/test_tar_repro_frozen.py recomputes it with the current
code and requires equality, so any change to tar_repro's training texts, labels, split, tokenization, batch order or
config fails loudly.

Regenerate (only if the tar_repro method is deliberately changed — that also requires a new ENGINE_VERSION):
    git worktree add /tmp/wt 96c1f5c
    PYTHONPATH=/tmp/wt .venv/bin/python tests/golden/tar_repro_fingerprint.py > tests/golden/tar_repro_96c1f5c.json
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def _h(obj) -> str:
    if isinstance(obj, np.ndarray):
        data = obj.tobytes() + str(obj.dtype).encode() + str(obj.shape).encode()
    else:
        data = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def synthetic_tables() -> dict:
    """Deterministic inputs covering cls / reg, missing-value placeholders, empty strings, long and short cells."""
    rng = np.random.default_rng(1234)
    n = 260
    idx = pd.RangeIndex(5000, 5000 + n)
    review = [f"{rng.choice(['great', 'bad', 'ok'])} product {i} " + "word " * int(rng.integers(0, 120)) for i in range(n)]
    review[3], review[7] = "", "   padded   "
    x = pd.DataFrame({"review": review, "title": rng.choice(["Phone", "Lamp", "MISSING_VALUE", "Chair"], n),
                      "brand": rng.choice(["acme", "globex", "initech"], n)}, index=idx).astype(str)
    y_cls = pd.Series(rng.choice(["neg", "neu", "pos"], n), index=idx)
    y_reg = pd.Series(rng.lognormal(3, 1, n), index=idx)
    return {"cls": (x, y_cls, True), "reg": (x, y_reg, False)}


def fingerprint() -> dict:
    import torch
    from transformers import AutoTokenizer

    from multabench.ours import tar_engine as te

    cfg = te.TarConfig()  # the defaults run_experiment uses for tar_repro (epochs 50, seed 42, fast path)
    out = {"engine_version": te.ENGINE_VERSION,
           "config_result_dict_cuda": cfg.result_dict(torch.device("cuda")),
           "config_result_dict_cpu": cfg.result_dict(torch.device("cpu")),
           "config_hash_cuda": te.config_hash(cfg, "tar_repro", torch.device("cuda")),
           "config_hash_cpu": te.config_hash(cfg, "tar_repro", torch.device("cpu")),
           "tasks": {}}
    # Batching probe big enough to span several 50-batch chunks (the real datasets have up to ~80k cells).
    probe_lengths = np.random.default_rng(7).integers(3, 513, 30_000)
    out["bucketing_probe"] = _h([b.tolist() for b in te.bucketed_batches(
        probe_lengths, cfg.batch_size, torch.Generator().manual_seed(cfg.seed), cfg.bucket_chunk_batches)])
    out["random_batches_probe"] = _h([b.tolist() for b in te.random_batches(
        30_000, cfg.batch_size, torch.Generator().manual_seed(cfg.seed))])
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    cols = ["brand", "review", "title"]
    for name, (x, y, is_cls) in synthetic_tables().items():
        d = te.prepare_e5_data(x, y, cols, is_cls, cfg.n_bins)
        strings = list(dict.fromkeys(d["train_texts"] + d["val_texts"]))
        tokens = te.TokenStore(strings, tok, cfg.max_length)
        train_uids = tokens.uids(d["train_texts"])
        gen = torch.Generator().manual_seed(cfg.seed)
        batches = te.bucketed_batches(tokens.lengths[train_uids], cfg.batch_size, gen, cfg.bucket_chunk_batches)
        out["tasks"][name] = {
            "e5_train_rows": _h(np.asarray(d["x_tr"].index, dtype=np.int64)),
            "e5_val_rows": _h(np.asarray(d["x_va"].index, dtype=np.int64)),
            "classes": d["classes"],
            "train_texts": _h(d["train_texts"]), "val_texts": _h(d["val_texts"]),
            "train_labels": _h(np.asarray(d["train_labels"], dtype=np.int64)),
            "val_labels": _h(np.asarray(d["val_labels"], dtype=np.int64)),
            "token_ids": _h(tokens.ids), "epoch1_batches": _h([b.tolist() for b in batches]),
            "lower_cache_key": te.LowerCache.key(cfg, tokens),
            "first_train_text": d["train_texts"][0],
        }
    return out


if __name__ == "__main__":
    print(json.dumps(fingerprint(), indent=1, sort_keys=True))

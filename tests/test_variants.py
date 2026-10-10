"""CPU tests for the TAR variants (docs/method_spec.md section 6) and the structured offset module."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import tar_engine as te
from multabench.ours.structured_offset import fit_structured_offset, oof_folds
from multabench.ours.variants import run_variant

CPU = torch.device("cpu")
BASE = te.TarConfig(epochs=1, micro_batch_size=16, eval_batch_size=16)


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(BASE, lower_cache_dir=str(tmp_path / "lower"))


def _table(n=260, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(2000, 2000 + n)
    price = rng.lognormal(3, 1, n)
    brand = rng.choice(["acme", "globex", "initech"], n)
    y_cls = pd.Series(np.where(price > np.median(price), "high", "low"), index=idx)
    y_reg = pd.Series(np.log(price) + rng.normal(0, 0.3, n), index=idx)
    x = pd.DataFrame({"review": [f"{rng.choice(['great', 'bad'])} item {i}" for i in range(n)],
                      "title": rng.choice(["Phone", "Lamp", "Chair"], n),
                      "price": price, "brand": brand}, index=idx)
    x["price"] = x["price"].where(rng.random(n) > 0.05)  # some missing numerics
    return x, y_cls, y_reg


def _split(x, y):
    return x.iloc[:220], y.iloc[:220], {"test": (x.iloc[220:], y.iloc[220:])}


# (1) OOF: every row's offset comes from a model that never saw its label
class _MemorizingModel:
    """Predicts the exact label of any row it was trained on (by row id), else the training mean."""
    def __init__(self, is_cls, cat_cols):
        self.table = {}

    def fit(self, X, y):
        self.table = dict(zip(X["rid"].tolist(), y))
        self.mean = float(np.mean(y))

    def predict(self, X):
        return np.array([self.table.get(r, self.mean) for r in X["rid"]])


def test_oof_never_uses_own_label():
    n = 100
    x = pd.DataFrame({"rid": np.arange(n, dtype=float)}, index=pd.RangeIndex(n))
    y = pd.Series(np.random.default_rng(0).normal(size=n), index=x.index)  # unique labels
    so = fit_structured_offset(x, y, ["rid"], [], False, [], model_factory=_MemorizingModel)
    assert not np.isclose(so.oof, y.to_numpy()).any(), "some row's OOF prediction used its own label"
    folds = oof_folds(n, None)
    covered = np.sort(np.concatenate([te_ for _, te_ in folds]))
    assert np.array_equal(covered, np.arange(n))
    assert all(len(set(tr) & set(te_)) == 0 for tr, te_ in folds)
    # the full fit (diagnostics only) does memorize: it is never used for pool rows
    assert np.allclose(so.predict_full(x), y.to_numpy())


# (2) with a zero head, logits equal log(offset)
def test_zero_head_gives_log_offset(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tokens = te.TokenStore(["passage: review: good", "passage: review: bad item"], tok, cfg.max_length)
    model = te.build_model(cfg, d_output=3, device=CPU).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    p = np.array([[0.7, 0.2, 0.1], [0.0005, 0.5, 0.4995]])
    log_off = np.log(np.clip(p, 1e-3, 1))
    source = te.make_source(model, tokens, cfg, CPU)
    logits = te._logits(model, source, np.array([0, 1]), CPU, False).float() + torch.tensor(log_off, dtype=torch.float32)
    assert np.allclose(logits.detach().numpy(), log_off, atol=1e-6)


# (7) end to end, (8) no structured columns == tar_repro
@pytest.mark.parametrize("task", ["cls", "reg"])
def test_offset_end_to_end(tmp_path, cfg, task):
    x, y_cls, y_reg = _table()
    y = y_cls if task == "cls" else y_reg
    x_fit, y_fit, extra = _split(x, y)
    res = run_variant("offset", x_fit, y_fit, ["review", "title"], ["price", "brand"], ["brand"], task == "cls",
                      extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    dg = res.diagnostics
    assert res.config["engine_version"] == "variants/1" and res.config["method"] == "offset"
    for k in ("struct_only_quality", "combined_quality", "lift", "head_quality"):
        assert set(dg[k]) == {"e5_train", "e5_val", "test"}
    assert dg["offset_info"]["has_features"] and dg["offset_info"]["top_features"][0] in ("price", "brand")
    # the price feature determines y_cls: the structured model alone must be strong on test
    if task == "cls":
        assert dg["struct_only_quality"]["test"] > 0.9
    emb = res.row_embeddings("review", extra["test"][0].index, texts=te.encoding_texts(extra["test"][0], "review"))
    assert emb.shape == (40, 384)
    again = run_variant("offset", x_fit, y_fit, ["review", "title"], ["price", "brand"], ["brand"], task == "cls",
                        extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    assert again.cache_hit


def test_offset_reuses_tar_repro_lower_cache(tmp_path, cfg):
    x, y_cls, _ = _table()
    x_fit, y_fit, extra = _split(x, y_cls)
    cols = ["review", "title"]
    te.run_tar(x_fit[cols], y_fit, cols, True, {k: (v[0][cols], v[1]) for k, v in extra.items()}, cfg, CPU, None, "SYNTH", 0)
    import os
    before = set(os.listdir(cfg.lower_cache_dir))
    run_variant("offset", x_fit, y_fit, cols, ["price", "brand"], ["brand"], True, extra, cfg, CPU, None, "SYNTH", 0)
    assert set(os.listdir(cfg.lower_cache_dir)) == before, "offset built a new lower-layer cache"


@pytest.mark.parametrize("task", ["cls", "reg"])
def test_no_structured_columns_equals_tar_repro(tmp_path, cfg, task):
    x, y_cls, y_reg = _table()
    y = y_cls if task == "cls" else y_reg
    cols = ["review", "title"]
    x_fit, y_fit, extra = _split(x[cols], y)
    tar = te.run_tar(x_fit, y_fit, cols, task == "cls", extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    off = run_variant("offset", x_fit, y_fit, cols, [], [], task == "cls", extra, cfg, CPU, str(tmp_path / "tar"),
                      "SYNTH", 0)
    assert off.diagnostics["offset_delegated"] and off.cache_key == tar.cache_key
    for c in cols:
        assert np.array_equal(off.columns[c].emb, tar.columns[c].emb)
    assert off.config["engine_version"] == "variants/1"

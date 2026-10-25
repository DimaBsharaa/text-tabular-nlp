"""CPU tests for the TAR variants and the structured offset module."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours import tar_engine as te
from multabench.ours.archive.structured_offset import fit_structured_offset, oof_folds
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


# ------------------------------------------------------------------ prefix variants

# (6) the numeric encoding is smooth: nearby values -> nearby encodings / tokens
def test_ple_is_smooth_and_monotone():
    from multabench.ours.archive.feature_tokens import ple_edges, ple_encode
    vals = np.random.default_rng(0).lognormal(0, 1, 1000)
    edges = ple_edges(vals)
    x = np.linspace(vals.min() - 1, vals.max() + 1, 20001)
    enc = ple_encode(x, edges)
    step = np.abs(np.diff(enc, axis=0)).sum(axis=1)
    assert step.max() < 0.05, "a tiny change in x caused a jump in the encoding"
    assert (np.diff(enc, axis=0) >= -1e-7).all(), "encoding is not monotone"
    assert np.allclose(enc[0], 0) and np.allclose(enc[-1, :len(edges) - 1], 1)
    assert np.allclose(ple_encode(np.array([np.nan]), edges), 0)


def _prefix_setup(cfg):
    """A fresh model + lower cache + feature tokens on a tiny table, shared by tests 4/5."""
    from transformers import AutoTokenizer
    from multabench.ours.archive.feature_tokens import FeatureSpec, FeatureTokenizer, PrefixSource, RowFeatures, frozen_name_vectors
    x, _, _ = _table(60)
    strings = list(dict.fromkeys(f"passage: review: {t}" for t in x["review"]))
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tokens = te.TokenStore(strings, tok, cfg.max_length)
    model = te.build_model(cfg, d_output=2, device=CPU).eval()
    with torch.no_grad():  # non-trivial LoRA, so the comparison is meaningful
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    cache = te.LowerCache.get_or_build(model, tokens, cfg, CPU)
    specs = [FeatureSpec("price", "num", x["price"]), FeatureSpec("brand", "cat", x["brand"])]
    feats = RowFeatures(specs, x[["price", "brand"]])
    model.feature_tokenizer = FeatureTokenizer(specs, frozen_name_vectors(model, tok, ["price", "brand"], cfg, CPU), 384)
    uids = tokens.uids([f"passage: review: {t}" for t in x["review"]])
    src = PrefixSource(model, cache, cfg, uids, np.arange(len(x)), feats)
    return model, cache, tokens, src, uids, x


# (5) with all feature tokens masked out, prefix embeddings equal TAR's
def test_masked_feature_tokens_equal_tar(cfg):
    model, cache, tokens, src, uids, x = _prefix_setup(cfg)
    src.features_visible = False
    emb_prefix, _ = te.encode(model, src, np.arange(len(x)), cfg, CPU)
    emb_tar, _ = te.encode(model, te._CachedLowerSource(model, cache, cfg), uids, cfg, CPU)
    assert np.abs(emb_prefix - emb_tar).max() < 1e-5
    src.features_visible = True
    emb_seen, _ = te.encode(model, src, np.arange(len(x)), cfg, CPU)
    assert np.abs(emb_seen - emb_tar).max() > 1e-4, "visible feature tokens should change the embeddings"


def test_feature_tokens_shape_missing_and_rare_categories(cfg):
    from tabstar.preprocessing.feat_types import MISSING_VALUE  # upstream's placeholder for missing categoricals
    from multabench.ours.archive.feature_tokens import FeatureSpec, RowFeatures
    s = pd.Series(["a"] * 5 + ["b"] * 2 + [MISSING_VALUE] * 4)
    spec = FeatureSpec("c", "cat", s)
    assert spec.vocab == {"a": 1}  # "b" seen only twice -> "other"; the placeholder never gets its own category
    ids, miss = spec.encode(pd.Series(["a", "b", "zzz", MISSING_VALUE]))
    assert ids.tolist() == [1, 0, 0, 0] and miss.tolist() == [False, False, False, True]
    model, cache, tokens, src, uids, x = _prefix_setup(cfg)
    toks = model.feature_tokenizer(src.feats, np.arange(4), CPU)
    assert toks.shape == (4, 2, 384)


# (4) prefix reuses tar_repro's lower-layer cache; (7) prefix variants end to end
@pytest.mark.parametrize("method,task", [("prefix", "cls"), ("prefix", "reg"), ("prefix_offset", "cls"),
                                         ("prefix_offset", "reg")])
def test_prefix_variants_end_to_end_and_cache_reuse(tmp_path, cfg, method, task):
    import os
    x, y_cls, y_reg = _table()
    y = y_cls if task == "cls" else y_reg
    x_fit, y_fit, extra = _split(x, y)
    cols = ["review", "title"]
    tar = te.run_tar(x_fit[cols], y_fit, cols, task == "cls", {k: (v[0][cols], v[1]) for k, v in extra.items()},
                     cfg, CPU, None, "SYNTH", 0)
    before = set(os.listdir(cfg.lower_cache_dir))
    res = run_variant(method, x_fit, y_fit, cols, ["price", "brand"], ["brand"], task == "cls", extra, cfg, CPU,
                      str(tmp_path / "tar"), "SYNTH", 0)
    assert set(os.listdir(cfg.lower_cache_dir)) == before, "prefix built a new lower-layer cache"
    assert res.config["method"] == method and res.diagnostics["feature_info"]["top_features"]
    test_x = extra["test"][0]
    emb = res.row_embeddings("review", test_x.index, texts=te.encoding_texts(test_x, "review"))
    assert emb.shape == (40, 384) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    # embeddings are per row: two rows with the same review text but different structure may differ
    assert len(res.columns["title"].emb) == len(x)
    if method == "prefix_offset":
        assert "combined_quality" in res.diagnostics
    assert run_variant(method, x_fit, y_fit, cols, ["price", "brand"], ["brand"], task == "cls", extra, cfg, CPU,
                       str(tmp_path / "tar"), "SYNTH", 0).cache_hit


# ------------------------------------------------------------------ context variants

# (3) the verbalizer is deterministic and respects token caps
def test_verbalizer_deterministic_and_capped():
    from transformers import AutoTokenizer
    from multabench.ours.archive.context_text import CONTEXT_MAX_TOKENS, Verbalizer, context_ids, full_string
    x, _, _ = _table(100)
    x["brand"] = x["brand"].astype(object)
    x.loc[x.index[0], "brand"] = "a very long category value that goes on and on"
    v1 = Verbalizer(["price", "brand"], ["brand"], x)
    v2 = Verbalizer(["price", "brand"], ["brand"], x.sample(frac=1, random_state=3))  # row order must not matter
    c1, c2 = v1.contexts(x), v2.contexts(x)
    assert c1 == c2
    long_val = "a very long category value that goes on and on"
    assert c1[0].endswith(f"brand={long_val[:30]}") and len(long_val[:30]) == 30  # categoricals cut at 30 chars
    assert all(("decile" in c) or ("price=missing" in c) for c in c1)
    tok = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
    long_ctx = "; ".join(f"feature_{i}=value number {i}" for i in range(60))
    long_txt = "review: " + "word " * 1000
    ids = context_ids(tok, long_ctx, long_txt, 512)
    assert len(ids) == 512
    ctx_len = len(tok(long_ctx, add_special_tokens=False)["input_ids"][:CONTEXT_MAX_TOKENS])
    assert ctx_len == CONTEXT_MAX_TOKENS
    # without caps hit, piecewise tokens == tokenizing the whole string at once
    short = context_ids(tok, c1[1], "review: great item", 512)
    assert short == tok(full_string(c1[1], "review: great item"))["input_ids"]


@pytest.mark.parametrize("method,task", [("context", "cls"), ("context", "reg"), ("context_offset", "cls"),
                                         ("context_offset", "reg")])
def test_context_variants_end_to_end(tmp_path, cfg, method, task):
    x, y_cls, y_reg = _table()
    y = y_cls if task == "cls" else y_reg
    x_fit, y_fit, extra = _split(x, y)
    cols = ["review", "title"]
    res = run_variant(method, x_fit, y_fit, cols, ["price", "brand"], ["brand"], task == "cls", extra, cfg, CPU,
                      str(tmp_path / "tar"), "SYNTH", 0)
    fi = res.diagnostics["feature_info"]
    assert fi["context_example"].startswith("passage: ") and " | " in fi["context_example"]
    assert "decile" in fi["context_example"] or "missing" in fi["context_example"]
    test_x = extra["test"][0]
    emb = res.row_embeddings("review", test_x.index, texts=te.encoding_texts(test_x, "review"))
    assert emb.shape == (40, 384)
    if method == "context_offset":
        assert "combined_quality" in res.diagnostics
    assert run_variant(method, x_fit, y_fit, cols, ["price", "brand"], ["brand"], task == "cls", extra, cfg, CPU,
                       str(tmp_path / "tar"), "SYNTH", 0).cache_hit

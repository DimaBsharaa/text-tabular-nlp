"""CPU tests for multabench/ours/tar_engine.py on a tiny synthetic text table (downloads e5-small-v2 once).

    uv run pytest tests/test_tar_engine.py -q
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours.tar_engine import (TarConfig, _load, bucketed_batches, encoding_texts, random_batches, run_tar,
                                        training_texts)

CPU = torch.device("cpu")
BASE = TarConfig(epochs=1, micro_batch_size=8, eval_batch_size=16)


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(BASE, lower_cache_dir=str(tmp_path / "lower"))


def _table(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n), index=pd.RangeIndex(1000, 1000 + n))
    words = np.where(y == 1, "great excellent loved it", "awful terrible broken")
    extra = rng.choice(["", " and it arrived late", " the box was damaged and the manual was missing pages"], n)
    x = pd.DataFrame({"review": [f"{w} item {i}{e}" for i, (w, e) in enumerate(zip(words, extra))],
                      "title": rng.choice(["Phone", "Lamp", "Chair"], n)}, index=y.index)
    return x.astype(str), y


def _parts(x, y):
    return x.iloc[:100], y.iloc[:100], {"learner_val": (x.iloc[100:110], y.iloc[100:110]),
                                        "test": (x.iloc[110:], y.iloc[110:])}


@pytest.mark.parametrize("fast", [True, False])
def test_binary_outputs_and_cache_roundtrip(tmp_path, cfg, fast):
    cfg = dataclasses.replace(cfg, fast=fast)
    x, y = _table(140)
    x_fit, y_fit, extra = _parts(x, y)
    res = run_tar(x_fit, y_fit, ["review", "title"], True, extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    assert res.diagnostics["train"]["fast_path"] is fast
    assert set(res.splits) == {"e5_train", "e5_val", "learner_val", "test"}
    assert len(res.splits["e5_train"]) + len(res.splits["e5_val"]) == 100
    assert set(res.diagnostics["head_quality"]) == set(res.splits)
    test_rows = extra["test"][0].index
    emb = res.row_embeddings("review", test_rows, texts=encoding_texts(extra["test"][0], "review"))
    assert emb.shape == (30, 384) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    assert res.head_probs(test_rows).shape == (30, 2)

    again = run_tar(x_fit, y_fit, ["review", "title"], True, extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    assert again.cache_hit and again.cache_key == res.cache_key
    assert np.array_equal(again.row_embeddings("review", test_rows), emb)
    assert _load(str(tmp_path / "tar" / res.cache_key)).diagnostics["head_quality"] == res.diagnostics["head_quality"]


def test_fast_and_old_path_have_different_cache_keys(tmp_path, cfg):
    x, y = _table(140)
    x_fit, y_fit, extra = _parts(x, y)
    a = run_tar(x_fit, y_fit, ["review"], True, extra, cfg, CPU, str(tmp_path / "tar"), "SYNTH", 0)
    b = run_tar(x_fit, y_fit, ["review"], True, extra, dataclasses.replace(cfg, fast=False), CPU,
                str(tmp_path / "tar"), "SYNTH", 0)
    assert a.cache_key != b.cache_key and not b.cache_hit


def test_text_mismatch_is_caught(cfg):
    x, y = _table(140)
    x_fit, y_fit, extra = _parts(x, y)
    res = run_tar(x_fit, y_fit, ["review"], True, extra, cfg, CPU, None, "SYNTH", 0)
    with pytest.raises(ValueError, match="mismatch"):
        res.row_embeddings("review", x_fit.index[:2], texts=["something else", "entirely"])


def test_regression_bins_and_training_text_format(cfg):
    # Upstream needs >= 200 fit rows here: the stratified 10% E5 val split must hold all 20 bins.
    x, _ = _table(260)
    y = pd.Series(np.linspace(0, 1, 260), index=x.index)
    res = run_tar(x.iloc[:220], y.iloc[:220], ["review"], False, {"test": (x.iloc[220:], y.iloc[220:])}, cfg, CPU,
                  None, "SYNTH", 0)
    assert res.head["d_output"] == 20 and len(res.head["bin_means"]) == 20
    assert training_texts(x.iloc[:1], ["title", "review"])[0] == f"passage: title: {x.iloc[0]['title']}"


def test_rows_must_be_disjoint(cfg):
    x, y = _table(140)
    with pytest.raises(AssertionError):
        run_tar(x.iloc[:100], y.iloc[:100], ["review"], True, {"test": (x.iloc[90:], y.iloc[90:])}, cfg, CPU,
                None, "SYNTH", 0)


def test_bucketed_batches_same_steps_and_cells():
    lengths = np.random.default_rng(0).integers(3, 300, 1000)
    gen_a, gen_b = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    rb = random_batches(1000, 256, gen_a)
    bb = bucketed_batches(lengths, 256, gen_b, chunk_batches=2)
    assert len(rb) == len(bb) == 4
    assert sorted(np.concatenate(bb).tolist()) == list(range(1000))
    assert sorted(len(b) for b in bb) == [232, 256, 256, 256]
    pad = lambda bs: sum(int(lengths[b].max()) * len(b) - int(lengths[b].sum()) for b in bs)
    assert pad(bb) < pad(rb)  # the point of bucketing: less padding


# ------------------------------------------------------------------ fast path == old path

def _model_with_nonzero_lora(cfg, d_output=2):
    """Fresh model whose LoRA B matrices are random (fresh LoRA is a no-op, which would make parity trivial)."""
    from multabench.ours.tar_engine import build_model, set_seed
    set_seed(0)
    model = build_model(cfg, d_output=d_output, device=CPU)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    return model.eval()


def test_fast_path_embeddings_match_full_model(cfg):
    from transformers import AutoTokenizer
    from multabench.e5.constants import format_e5_passage
    from multabench.ours.tar_engine import encode_strings

    x, _ = _table(60)
    strings = [format_e5_passage("review", t) for t in encoding_texts(x, "review")]
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = _model_with_nonzero_lora(cfg)
    old_emb, old_logits = encode_strings(model, tok, strings, dataclasses.replace(cfg, fast=False), CPU)
    new_emb, new_logits = encode_strings(model, tok, strings, cfg, CPU)
    assert np.abs(new_emb - old_emb).max() < 1e-3
    assert np.abs(new_logits - old_logits).max() < 1e-2


def test_fast_training_matches_full_model_without_dropout(cfg):
    """Same init, same (random) batch order, dropout off everywhere: one epoch on each path -> same LoRA weights."""
    from transformers import AutoTokenizer
    from multabench.ours.tar_engine import (TokenStore, _trainable_state, build_model, make_source, prepare_e5_data,
                                            set_seed, train_tar)

    x, y = _table(140)
    d = prepare_e5_data(x.iloc[:120], y.iloc[:120], ["review", "title"], True, n_bins=20)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tokens = TokenStore(list(dict.fromkeys(d["train_texts"] + d["val_texts"])), tok, cfg.max_length)
    states, losses = {}, {}
    for fast in (False, True):
        c = dataclasses.replace(cfg, fast=fast, lora_dropout=0.0, bucket_chunk_batches=0, batch_size=32,
                                micro_batch_size=32)
        set_seed(c.seed)
        model = build_model(c, d_output=2, device=CPU)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        source = make_source(model, tokens, c, CPU)
        assert source.fast is fast
        log = train_tar(c, model, source, tokens.uids(d["train_texts"]), d["train_labels"],
                        tokens.uids(d["val_texts"]), d["val_labels"], CPU)
        states[fast], losses[fast] = _trainable_state(model), log["history"][0]["val_loss"]
    assert max(float((states[True][k] - states[False][k]).abs().max()) for k in states[False]) < 1e-3
    assert abs(losses[True] - losses[False]) < 1e-3


# ------------------------------------------------------------------ parity with upstream

def test_same_finetuning_inputs_as_upstream(monkeypatch):
    """Upstream's fit_text_encoders_tuned (with the fine-tune itself stubbed out) must see the same texts/labels/split."""
    import multabench.e5.e5_finetune as up
    from multabench.baselines.preprocessing.text_embeddings import fit_text_encoders_tuned
    from multabench.ours.tar_engine import prepare_e5_data

    captured = {}

    def fake_finetune(train_texts, train_y, val_texts, val_y, device, tokenizer, **kw):
        captured.update(train_texts=train_texts, train_y=train_y, val_texts=val_texts, val_y=val_y)
        return up.get_vanilla_e5(device)

    monkeypatch.setattr(up, "finetune_e5_with_lora", fake_finetune)
    for is_cls, n in ((True, 140), (False, 260)):
        x, y = _table(n)
        if not is_cls:
            y = pd.Series(np.linspace(0, 1, n), index=x.index)
        cols = ["review", "title"]
        fit_text_encoders_tuned(x=x, text_features_list=cols, device=CPU, y=y, e5_train_kwargs={}, is_cls=is_cls,
                                d_output=2)
        ours = prepare_e5_data(x, y, cols, is_cls, n_bins=20)
        # TextLabelDataset adds the prefix and maps empty strings to " " at __getitem__ time.
        assert ours["train_texts"] == ["passage: " + (str(t).strip() or " ") for t in captured["train_texts"]]
        assert ours["val_texts"] == ["passage: " + (str(t).strip() or " ") for t in captured["val_texts"]]
        assert [ours["classes"][i] for i in ours["train_labels"]] == [str(v) for v in captured["train_y"]]
        assert [ours["classes"][i] for i in ours["val_labels"]] == [str(v) for v in captured["val_y"]]


@pytest.mark.parametrize("fast", [False, True])
def test_matches_upstream_encoding(cfg, fast):
    """Fresh LoRA (B = 0, a no-op): our encoder (either path) must equal upstream's padded-to-512 encode_texts_with_e5."""
    from transformers import AutoTokenizer
    from multabench.e5.constants import format_e5_passage
    from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
    from multabench.ours.tar_engine import build_model, encode_strings

    x, _ = _table(40)
    texts = encoding_texts(x, "review")
    vanilla, tok = get_vanilla_e5(CPU)
    theirs = encode_texts_with_e5(texts=texts, col_name="review", model=vanilla, tokenizer=tok, device=CPU)
    c = dataclasses.replace(cfg, fast=fast)
    ours, _ = encode_strings(build_model(c, d_output=2, device=CPU), AutoTokenizer.from_pretrained(c.model_name),
                             [format_e5_passage("review", t) for t in texts], c, CPU)
    assert np.abs(ours - theirs).max() < (1e-3 if fast else 1e-5)

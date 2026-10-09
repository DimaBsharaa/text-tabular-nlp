"""CPU tests for multabench/ours/tar_engine.py on a tiny synthetic text table (downloads e5-small-v2 once).

    uv run pytest tests/test_tar_engine.py -q
"""
import numpy as np
import pandas as pd
import pytest
import torch

from multabench.ours.tar_engine import TarConfig, _load, encoding_texts, run_tar, training_texts

CPU = torch.device("cpu")
CFG = TarConfig(epochs=1, micro_batch_size=8, eval_batch_size=16)


def _table(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.integers(0, 2, n), index=pd.RangeIndex(1000, 1000 + n))
    words = np.where(y == 1, "great excellent loved it", "awful terrible broken")
    x = pd.DataFrame({"review": [f"{w} item {i}" for i, w in enumerate(words)],
                      "title": rng.choice(["Phone", "Lamp", "Chair"], n)}, index=y.index)
    return x.astype(str), y


def _parts(x, y):
    return x.iloc[:100], y.iloc[:100], {"learner_val": (x.iloc[100:110], y.iloc[100:110]),
                                        "test": (x.iloc[110:], y.iloc[110:])}


def test_binary_outputs_and_cache_roundtrip(tmp_path):
    x, y = _table(140)
    x_fit, y_fit, extra = _parts(x, y)
    res = run_tar(x_fit, y_fit, ["review", "title"], True, extra, CFG, CPU, str(tmp_path), "SYNTH", 0)
    assert set(res.splits) == {"e5_train", "e5_val", "learner_val", "test"}
    assert len(res.splits["e5_train"]) + len(res.splits["e5_val"]) == 100
    assert set(res.diagnostics["head_quality"]) == set(res.splits)
    test_rows = extra["test"][0].index
    emb = res.row_embeddings("review", test_rows, texts=encoding_texts(extra["test"][0], "review"))
    assert emb.shape == (30, 384) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    assert res.head_probs(test_rows).shape == (30, 2)

    again = run_tar(x_fit, y_fit, ["review", "title"], True, extra, CFG, CPU, str(tmp_path), "SYNTH", 0)
    assert again.cache_hit and again.cache_key == res.cache_key
    assert np.array_equal(again.row_embeddings("review", test_rows), emb)
    assert _load(str(tmp_path / res.cache_key)).diagnostics["head_quality"] == res.diagnostics["head_quality"]


def test_text_mismatch_is_caught(tmp_path):
    x, y = _table(140)
    x_fit, y_fit, extra = _parts(x, y)
    res = run_tar(x_fit, y_fit, ["review"], True, extra, CFG, CPU, None, "SYNTH", 0)
    with pytest.raises(ValueError, match="mismatch"):
        res.row_embeddings("review", x_fit.index[:2], texts=["something else", "entirely"])


def test_regression_bins_and_training_text_format():
    # Upstream needs >= 200 fit rows here: the stratified 10% E5 val split must hold all 20 bins.
    x, _ = _table(260)
    y = pd.Series(np.linspace(0, 1, 260), index=x.index)
    res = run_tar(x.iloc[:220], y.iloc[:220], ["review"], False, {"test": (x.iloc[220:], y.iloc[220:])}, CFG, CPU,
                  None, "SYNTH", 0)
    assert res.head["d_output"] == 20 and len(res.head["bin_means"]) == 20
    assert training_texts(x.iloc[:1], ["title", "review"])[0] == f"passage: title: {x.iloc[0]['title']}"


def test_rows_must_be_disjoint():
    x, y = _table(140)
    with pytest.raises(AssertionError):
        run_tar(x.iloc[:100], y.iloc[:100], ["review"], True, {"test": (x.iloc[90:], y.iloc[90:])}, CFG, CPU,
                None, "SYNTH", 0)


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


def test_dynamic_padding_matches_upstream_encoding():
    """With LoRA freshly initialized (B = 0, i.e. a no-op) our encode() must equal upstream's padded-to-512 encoder."""
    from multabench.e5.constants import format_e5_passage
    from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
    from multabench.ours.tar_engine import build_model, encode
    from transformers import AutoTokenizer

    x, _ = _table(40)
    texts = encoding_texts(x, "review")
    vanilla, tok = get_vanilla_e5(CPU)
    theirs = encode_texts_with_e5(texts=texts, col_name="review", model=vanilla, tokenizer=tok, device=CPU)
    ours, _ = encode(build_model(CFG, d_output=2, device=CPU), AutoTokenizer.from_pretrained(CFG.model_name),
                     [format_e5_passage("review", t) for t in texts], CFG, CPU)
    assert np.abs(ours - theirs).max() < 1e-5

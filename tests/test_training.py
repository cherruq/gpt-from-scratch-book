from dataclasses import replace

import numpy as np
import pytest

from examples.training import gradient_report, sample_window
from src.numpy_gpt.data import load_corpus
from src.numpy_gpt.generate import distribution, generate
from src.numpy_gpt.model import Config, GPT
from src.numpy_gpt.train import TrainingConfig, evaluate, load_checkpoint, train


def test_windows_and_train_only_vocab(tmp_path):
    x, y = sample_window()
    np.testing.assert_array_equal(x, [[0, 1, 2], [1, 2, 3], [5, 6, 7]])
    np.testing.assert_array_equal(y, [[1, 2, 3], [2, 3, 4], [6, 7, 0]])
    a, b = tmp_path / "train", tmp_path / "validation"
    a.write_text("春风春雨\n")
    b.write_text("春风冬雪\n")
    corpus = load_corpus(a, b, 3)
    assert corpus.tokenizer.encode("冬") == [corpus.tokenizer.unk]
    b.write_text(a.read_text())
    with pytest.raises(ValueError, match="overlap"):
        load_corpus(a, b, 3)


def test_numerical_training_and_resume(tmp_path):
    silent = lambda _: None
    config = TrainingConfig(steps=12, log_every=12)
    full, optimizer, rng = train(config, tmp_path / "full.npz", report=silent)
    path = tmp_path / "split.npz"
    train(replace(config, steps=5), path, report=silent)
    half, _, _, _ = load_checkpoint(path)
    resumed, resumed_optimizer, resumed_rng = train(config, path, resume=True, report=silent)
    for name, value in full.state_dict().items():
        np.testing.assert_array_equal(value, resumed.parameters[name].data)
        np.testing.assert_array_equal(optimizer.m[name], resumed_optimizer.m[name])
        np.testing.assert_array_equal(optimizer.v[name], resumed_optimizer.v[name])
    assert rng.bit_generator.state == resumed_rng.bit_generator.state
    corpus = config.corpus()
    assert evaluate(full, corpus.validation, 4) < evaluate(half, corpus.validation, 4)
    with pytest.raises(ValueError, match="differs"):
        train(replace(config, batch_size=2), path, resume=True, report=silent)


def test_sampler_and_eos():
    row = np.log([0.5, 0.3, 0.2])
    np.testing.assert_allclose(distribution(row, top_k=2), [0.625, 0.375, 0])
    np.testing.assert_allclose(distribution(row, top_p=0.75), [0.625, 0.375, 0])
    np.testing.assert_allclose(distribution(row, temperature=0.5), [25/38, 9/38, 4/38])
    np.testing.assert_array_equal(distribution([1., 1.], 0), [1, 0])
    np.testing.assert_array_equal(distribution(row, top_p=0.01), [1, 0, 0])
    for args in ({"temperature": -1}, {"top_p": 0}, {"top_k": 0}):
        with pytest.raises(ValueError):
            distribution(row, **args)
    model = GPT(Config(vocab=8))
    first = generate(model, [0], 1, temperature=0)[-1]
    assert generate(model, [0], 8, temperature=0, eos=first) == [0, first]
    assert len(generate(model, [0, 1, 2, 3, 4], 4, seed=42)) == 9
    with pytest.raises(ValueError, match="learned position"):
        model.forward([[0, 1, 2, 3, 4]])
    assert generate(model, [0], 10, seed=42) == generate(model, [0], 10, seed=42)


def test_model_gradient_and_parameter_update():
    metrics = gradient_report()
    assert metrics["error"] < 5e-8
    assert metrics["parameter_delta"] != 0
    assert metrics["nll_after"] < metrics["nll_before"]


def test_byte_bpe_training_checkpoint_roundtrip(tmp_path):
    from src.numpy_gpt.data import restore_tokenizer

    config = TrainingConfig(tokenizer="bpe", vocab=280, context=8, channels=4,
                            layers=1, hidden=8, steps=2, log_every=2)
    checkpoint = tmp_path / "byte.npz"
    model, _, _ = train(config, checkpoint, report=lambda _: None)
    restored, optimizer, _, metadata = load_checkpoint(checkpoint)
    tokenizer = restore_tokenizer(metadata["tokenizer"])
    encoded = tokenizer.encode("春风\nprint('雪')", boundaries=True)
    assert tokenizer.decode(encoded) == "春风\nprint('雪')"
    assert tokenizer.vocab == model.config.vocab == 280
    np.testing.assert_array_equal(restored.forward([encoded[:8]], track=False).data,
                                  model.forward([encoded[:8]], track=False).data)
    assert optimizer.steps == 2

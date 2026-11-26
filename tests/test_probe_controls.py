import json
from pathlib import Path

import numpy as np

from examples.probe_controls import TASK_LABELS, balanced_labelings, prepare_samples, run_controls
from src.numpy_gpt.train import read_config


def test_probe_controls_keep_scene_balance_and_tokens():
    config = read_config("configs/quick.json")
    tokenizer = config.corpus().tokenizer
    train, evaluation = prepare_samples(config)
    assert len(train.prompts) == 56 and len(set(train.prompts)) == 8
    assert np.count_nonzero(TASK_LABELS[train.scenes] == 1) == 28

    for samples in evaluation.values():
        assert len(set(samples.prompts)) == 8
        assert np.count_nonzero(TASK_LABELS[samples.scenes] == 1) == 4
        for prompt in samples.prompts:
            ids = tokenizer.encode(prompt)
            assert len(ids) == 11 and tokenizer.unk not in ids

    novel = set(evaluation["template_1"].prompts + evaluation["template_2"].prompts)
    assert len(novel) == 16 and novel.isdisjoint(train.prompts)


def test_probe_label_controls_are_balanced_and_unique():
    maps = balanced_labelings()
    assert maps.shape == (8, 68)
    np.testing.assert_array_equal(maps.sum(axis=0), np.zeros(68))
    assert len(np.unique(maps, axis=1).T) == 68
    assert not np.any(np.all(maps == TASK_LABELS[:, None], axis=0))
    assert not np.any(np.all(maps == -TASK_LABELS[:, None], axis=0))


def test_probe_controls_reproduce_report():
    observed = run_controls()
    expected = json.loads(Path("reports/probe-controls.json").read_text())
    assert observed == expected
    assert observed["trained"]["original"]["correct"] == 7
    assert observed["untrained"]["original"]["correct"] == 5
    assert observed["trained"]["new_combined"]["correct"] == 8
    assert observed["untrained"]["new_combined"]["correct"] == 9
    counts = observed["trained"]["original"]["arbitrary_histogram"]
    assert counts == {"4": 2, "5": 12, "6": 22, "7": 26, "8": 6}
    assert sum(counts.values()) == 68

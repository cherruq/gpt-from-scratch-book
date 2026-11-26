"""Data coverage, a failure control, and resume diagnostics for chapters 29–30."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.autograd.tensor import Tensor
from src.numpy_gpt.data import split_targets, windows
from src.numpy_gpt.model import Config, GPT, cross_entropy
from src.numpy_gpt.optim import AdamW
from src.numpy_gpt.train import TrainingConfig, evaluate, load_checkpoint, train, train_step


def seen_windows(corpus):
    train_rows = {tuple(row) for row in corpus.train}
    return np.array([tuple(row) in train_rows for row in corpus.validation])


# region window_audit
def window_audit():
    record = [0, 3, 4, 5, 6, 1]
    x, y = split_targets(windows([record], context=2))
    validation_y = split_targets(windows([record], context=2, stride=2))[1]

    config = TrainingConfig()
    corpus = config.corpus()
    val_text = [line for line in Path(config.validation).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    possible_targets = sum(len(corpus.tokenizer.encode(line, boundaries=True)) - 1
                           for line in val_text)
    evaluated_targets = corpus.validation.shape[0] * config.context
    return {
        "toy_inputs": x.tolist(), "toy_targets": y.tolist(), "toy_validation": validation_y.tolist(),
        "train_windows": len(corpus.train), "validation_windows": len(corpus.validation),
        "validation_total_targets": possible_targets, "validation_evaluated": evaluated_targets,
        "validation_tail": possible_targets - evaluated_targets,
        "validation_windows_in_train": int(seen_windows(corpus).sum()),
        "vocab": corpus.tokenizer.vocab,
    }
# endregion window_audit


# region clipping
def clipping():
    weight = Tensor([2., -1.], True)
    weight.grad = np.array([3., 4.])
    opt = AdamW({"w": weight}, lr=0.1, decay=0.2, max_norm=1.)
    norm = opt.step()
    return {"raw_norm": norm, "m": opt.m["w"].tolist(), "v": opt.v["w"].tolist(),
            "updated": weight.data.tolist()}
# endregion clipping


# region target_control
def target_control():
    batch = np.array([[0, 1, 2, 3, 0], [4, 5, 6, 7, 4]])
    ids, true_targets = split_targets(batch)
    rows = {}
    for name, targets in (("next_token", true_targets), ("copy_input", ids)):
        model = GPT(Config(vocab=8), seed=42)
        opt = AdamW(model.parameters, lr=0.025)
        start = float(cross_entropy(model.forward(ids, track=False), true_targets).data)
        for _ in range(80):
            if name == "next_token":
                train_step(model, opt, batch)
            else:
                model.zero_grad()
                cross_entropy(model.forward(ids), targets).backward()
                opt.step()
        logits = model.forward(ids, track=False)
        rows[name] = {"initial": start, "own_target_nll": float(cross_entropy(logits, targets).data),
                      "next_token_nll": float(cross_entropy(logits, true_targets).data),
                      "predictions": logits.data.argmax(axis=-1).tolist()}
    return rows
# endregion target_control


# region restore_probe
def restore_probe():
    config = TrainingConfig(steps=6)
    corpus = config.corpus()
    path_source = ("continuous.npz", "interrupted.npz")
    with TemporaryDirectory(prefix="gpt-restore-lab-") as work:
        continuous, interrupted = (Path(work) / name for name in path_source)
        quiet = lambda _: None
        full, full_opt, full_rng = train(config, continuous, report=quiet)
        train(replace(config, steps=3), interrupted, report=quiet)
        _, _, restored_rng, _ = load_checkpoint(interrupted)
        restored_batch = restored_rng.integers(len(corpus.train), size=config.batch_size)
        fresh_batch = np.random.default_rng(config.seed).integers(len(corpus.train),
                                                                 size=config.batch_size)
        resumed, resumed_opt, resumed_rng = train(config, interrupted, resume=True, report=quiet)
    same_weights = all(np.array_equal(full.parameters[key].data, resumed.parameters[key].data)
                       for key in full.parameters)
    same_moments = all(np.array_equal(a[key], b[key])
                       for a, b in ((full_opt.m, resumed_opt.m), (full_opt.v, resumed_opt.v))
                       for key in a)
    return {"same_weights": same_weights, "same_moments": same_moments,
            "same_rng": full_rng.bit_generator.state == resumed_rng.bit_generator.state,
            "first_restored_batch": restored_batch.tolist(), "fresh_seed_batch": fresh_batch.tolist()}
# endregion restore_probe


# region validation_groups
def validation_groups():
    config = TrainingConfig()
    corpus = config.corpus()
    seen = seen_windows(corpus)
    with TemporaryDirectory(prefix="gpt-validation-lab-") as work:
        model, _, _ = train(config, Path(work) / "fit.npz", report=lambda _: None)
    return {
        "seen_tokens": int(seen.sum()) * config.context,
        "unseen_tokens": int((~seen).sum()) * config.context,
        "seen_nll": evaluate(model, corpus.validation[seen], config.batch_size),
        "unseen_nll": evaluate(model, corpus.validation[~seen], config.batch_size),
        "overall_nll": evaluate(model, corpus.validation, config.batch_size),
    }
# endregion validation_groups


def training_audit():
    return {"clipping": clipping(), "targets": target_control(), "resume": restore_probe(),
            "validation_groups": validation_groups()}


LABS = {29: window_audit, 30: training_audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=int, choices=LABS)
    print(json.dumps(LABS[parser.parse_args().chapter](), indent=2))

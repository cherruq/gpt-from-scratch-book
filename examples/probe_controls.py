"""Chapter 39 controls: arbitrary scene labels, initialization and new templates."""

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.make_corpus import SCENES
from src.numpy_gpt.model import GPT
from src.numpy_gpt.train import read_config, train


ORIGINAL = "今早我走到{scene}，看见了"
NEW_TEMPLATES = ("午后我来到{scene}，看见了", "傍晚我离开{scene}，看见了")
TASK_LABELS = np.array([1, 1, 1, 1, -1, -1, -1, -1])
PENALTY = 0.1


@dataclass(frozen=True)
class Samples:
    prompts: tuple[str, ...]
    scenes: np.ndarray


def prepare_samples(config):
    def read_records(path):
        return Path(path).read_text(encoding="utf-8").splitlines()

    def rows(records):
        prompts, scenes = [], []
        for record in records:
            scene = record[5:7]
            prompt = ORIGINAL.format(scene=scene)
            if scene not in SCENES or not record.startswith(prompt):
                raise ValueError("Expected the original scene template")
            prompts.append(prompt)
            scenes.append(SCENES.index(scene))
        return Samples(tuple(prompts), np.array(scenes))

    train_records = read_records(config.train)
    validation_records = read_records(config.validation)
    evaluation = {"original": rows(validation_records)}
    for number, template in enumerate(NEW_TEMPLATES, start=1):
        prompts = tuple(template.format(scene=scene) for scene in SCENES)
        if any(prompt in record for prompt in prompts for record in train_records + validation_records):
            raise ValueError("An evaluation template already occurs in the original corpus")
        evaluation[f"template_{number}"] = Samples(prompts, np.arange(len(SCENES)))
    return rows(train_records), evaluation


# region readout
def scene_features(model, tokenizer, samples):
    encoded = [[tokenizer.bos, *tokenizer.encode(prompt)] for prompt in samples.prompts]
    expected = len(ORIGINAL.format(scene=SCENES[0])) + 1
    if expected > model.config.context or any(
        len(ids) != expected or tokenizer.unk in ids for ids in encoded
    ):
        raise ValueError("All prompts must fit in context at the same length with known characters")
    layer = f"block.{model.config.layers - 1}"

    def observe(name, value):
        if name == layer:
            last_states.append(value[:, -1, :])

    last_states = []
    model.forward(np.asarray(encoded), track=False, observer=observe)
    return np.column_stack((last_states[0], np.ones(len(encoded))))


def fit_readout(features, labels):
    gram = features.T @ features
    return np.linalg.solve(gram + PENALTY * np.eye(gram.shape[0]), features.T @ labels)


def correct_count(features, weights, labels):
    prediction = np.where(features @ weights >= 0, 1, -1)
    return np.count_nonzero(prediction == labels, axis=0)
# endregion readout


# region control_labels
def balanced_labelings():
    """All 4–4 maps except the task itself and its sign reversal."""
    labelings = []
    for positive in combinations(range(len(SCENES)), len(SCENES) // 2):
        labels = np.full(len(SCENES), -1)
        labels[list(positive)] = 1
        if not np.array_equal(labels, TASK_LABELS) and not np.array_equal(labels, -TASK_LABELS):
            labelings.append(labels)
    return np.array(labelings).T
# endregion control_labels


# region evaluate_conditions
def evaluate_conditions(model, tokenizer, training, evaluation):
    train_x = scene_features(model, tokenizer, training)
    labels = np.column_stack((TASK_LABELS, balanced_labelings()))
    weights = fit_readout(train_x, labels[training.scenes])

    results, novel_counts = {}, []
    for name, samples in evaluation.items():
        features = scene_features(model, tokenizer, samples)
        counts = correct_count(features, weights, labels[samples.scenes])
        results[name] = summarise(counts, len(samples.prompts))
        if name != "original":
            novel_counts.append(counts)
    results["new_combined"] = summarise(
        np.sum(novel_counts, axis=0),
        sum(len(samples.prompts) for name, samples in evaluation.items() if name != "original"),
    )
    return results


def summarise(counts, total):
    arbitrary = counts[1:]
    hist = np.bincount(arbitrary, minlength=total + 1)
    return {
        "correct": int(counts[0]),
        "total": total,
        "arbitrary_mean_correct": float(arbitrary.mean()),
        "arbitrary_median_correct": float(np.median(arbitrary)),
        "arbitrary_min_correct": int(arbitrary.min()),
        "arbitrary_max_correct": int(arbitrary.max()),
        "arbitrary_histogram": {str(i): int(n) for i, n in enumerate(hist) if n},
    }
# endregion evaluate_conditions


def run_controls():
    config = read_config("configs/quick.json")
    corpus = config.corpus()
    samples, evaluation = prepare_samples(config)
    spec = config.model_config(corpus.tokenizer.vocab)
    initial = GPT(spec, seed=config.seed)
    with TemporaryDirectory(prefix="gpt-probe-controls-") as work:
        trained, _, _ = train(config, Path(work) / "probe.npz", report=lambda _: None)
    return {
        "initialization_seed": config.seed, "training_steps": config.steps,
        "ridge_penalty": PENALTY, "scenes": list(SCENES), "train_rows": len(samples.prompts),
        "train_unique_prefixes": len(set(samples.prompts)),
        "new_templates": list(NEW_TEMPLATES), "arbitrary_labelings": balanced_labelings().shape[1],
        "trained": evaluate_conditions(trained, corpus.tokenizer, samples, evaluation),
        "untrained": evaluate_conditions(initial, corpus.tokenizer, samples, evaluation),
    }


if __name__ == "__main__":
    print(json.dumps(run_controls(), ensure_ascii=False, indent=2))

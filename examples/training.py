"""M6 data, sampler and diagnostic experiments."""

import argparse

import numpy as np

from src.numpy_gpt.data import split_targets, windows
from src.numpy_gpt.generate import distribution
from src.numpy_gpt.model import Config, GPT, cross_entropy
from src.numpy_gpt.optim import AdamW
from src.numpy_gpt.train import train_step


# region sample_window
def sample_window():
    rows = windows([[0, 1, 2, 3, 4], [5, 6, 7, 0]], context=3)
    return split_targets(rows)
# endregion sample_window


# region sample_distribution
def sample_distribution():
    scores = np.log([0.5, 0.3, 0.2])
    return [distribution(scores, **options) for options in
            ({}, {"top_k": 2}, {"top_p": 0.75}, {"temperature": 0.5}, {"temperature": 0})]
# endregion sample_distribution


# region gradient_report
def gradient_report():
    model = GPT(Config(vocab=8))
    ids, targets = [[1, 1, 2]], [[1, 2, 3]]

    def objective(track):
        return cross_entropy(model.forward(ids, track=track), targets)

    objective(True).backward()
    parameter = model.parameters["token"]
    index = (1, 0)
    start = parameter.data[index]
    analytic = float(parameter.grad[index])
    try:
        parameter.data[index] = start + 1e-6
        plus = objective(False).data
        parameter.data[index] = start - 1e-6
        minus = objective(False).data
    finally:
        parameter.data[index] = start
    numeric = float((plus - minus) / 2e-6)
    update = AdamW(model.parameters)
    loss, gradnorm = train_step(model, update, np.array([[1, 1, 2, 3]]))
    return {"analytic": analytic, "numeric": numeric, "error": abs(analytic - numeric),
            "parameter_delta": float(parameter.data[index] - start),
            "nll_before": loss, "nll_after": float(objective(False).data), "grad_norm": gradnorm}
# endregion gradient_report


EXPERIMENTS = {29: sample_window, 31: sample_distribution, 32: gradient_report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=int, choices=EXPERIMENTS)
    args = parser.parse_args()
    print(EXPERIMENTS[args.chapter]())

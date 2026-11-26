"""Chapter 10: gather valid positions before stable cross-entropy."""

import argparse
import json

import numpy as np

from examples.math_basics import central_difference
from src.autograd.tensor import Tensor
from src.numpy_gpt.model import cross_entropy


# region masked_loss
def masked_loss(logits, targets, valid):
    """Inputs [B,T,V], [B,T], [B,T]; the denominator is the count of True."""
    targets, valid = np.asarray(targets), np.asarray(valid)
    if logits.data.ndim != 3 or targets.shape != logits.data.shape[:2] or valid.shape != targets.shape:
        raise ValueError("Expected logits [B,T,V], targets and mask [B,T]")
    if valid.dtype != np.bool_ or not valid.any():
        raise ValueError("Expected a boolean mask with at least one valid target")
    return cross_entropy(logits[valid], targets[valid])
# endregion masked_loss


def hand_case():
    scores = np.log(np.array([[[1, 2, 3], [1, 1, 1]], [[1, 1, 2], [1, 4, 1]]]))
    scores[0, 1] = -np.inf
    logits = Tensor(scores, requires_grad=True)
    targets = np.array([[1, -1], [2, 1]])
    valid = np.array([[True, False], [True, True]])
    loss = masked_loss(logits, targets, valid)
    loss.backward()
    batch_means = np.array([
        masked_loss(Tensor(scores[i:i + 1]), targets[i:i + 1], valid[i:i + 1]).data.item()
        for i in range(2)
    ])
    return {"mean_nll": loss.data.item(), "gradients": logits.grad.tolist(),
            "valid_counts": valid.sum(axis=1).tolist(), "batch_means": batch_means.tolist(),
            "unweighted_mean": float(batch_means.mean())}


# region loss_check
def check(batch, time, vocab, seed):
    if min(batch, time, vocab) < 1:
        raise ValueError("Use positive dimensions")
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(batch, time, vocab))
    targets = rng.integers(vocab, size=(batch, time))
    valid = rng.random(size=targets.shape) > 0.4
    valid[0, 0] = True
    direction = rng.normal(size=scores.shape)
    logits = Tensor(scores, requires_grad=True)
    loss = masked_loss(logits, targets, valid)
    loss.backward()

    def objective(step):
        rows = (scores + step * direction)[valid]
        return (np.logaddexp.reduce(rows, axis=-1)
                - rows[np.arange(len(rows)), targets[valid]]).mean()

    predicted = np.sum(logits.grad * direction)
    numeric = central_difference(objective, 0.)
    translated = masked_loss(Tensor(scores + 1000), targets, valid).data
    return {"shape": list(scores.shape), "valid_count": int(valid.sum()),
            "difference_error": float(abs(numeric - predicted)),
            "translation_error": float(abs(translated - loss.data)),
            "max_row_gradient_sum": float(np.max(np.abs(logits.grad.sum(axis=-1))))}
# endregion loss_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--time", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps({"hand": hand_case(), "check": check(**vars(args))}, indent=2))

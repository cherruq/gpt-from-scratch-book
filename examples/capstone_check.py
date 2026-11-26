"""Exercise verifier for a reader's independently written small decoder."""

import argparse
from importlib import import_module
import json

import numpy as np


CONFIG = {"vocab": 8, "context": 4, "channels": 16, "heads": 4, "hidden": 48, "layers": 2}
ATOL = 1e-10


class CheckFailed(Exception):
    pass


# region context_batch
def context_batch(token_seed=None):
    """Paired prefixes ending in the same cue but requiring different answers."""
    rows = np.array([[0, 2, 6, 4], [1, 2, 6, 5],
                     [0, 3, 6, 4], [1, 3, 6, 5]])
    if token_seed is not None:
        renaming = np.random.default_rng(token_seed).permutation(CONFIG["vocab"])
        rows = renaming[rows]
    return rows[:, :-1], rows[:, 1:]


def mean_nll(logits, targets):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    target_scores = np.take_along_axis(shifted, targets[..., None], axis=-1)[..., 0]
    return float(np.mean(np.logaddexp.reduce(shifted, axis=-1) - target_scores))
# endregion context_batch


def read_logits(model, ids):
    argument = ids.copy()
    z = np.asarray(model.forward(argument))
    expected = (*ids.shape, CONFIG["vocab"])
    if z.shape != expected or z.dtype != np.float64 or not np.all(np.isfinite(z)):
        raise CheckFailed(f"forward: expected finite float64 logits of shape {expected}")
    if not np.array_equal(argument, ids):
        raise CheckFailed("forward: modified the input IDs")
    return z.copy()


# region prefix_check
def check_prefix(model, ids):
    full = read_logits(model, ids)
    differences = [np.max(np.abs(read_logits(model, ids[:1]) - full[:1]))]
    for length in range(1, ids.shape[1]):
        altered = ids.copy()
        altered[:, length:] = (altered[:, length:] + 1) % CONFIG["vocab"]
        unchanged = full[:, :length]
        prefix = read_logits(model, ids[:, :length])
        with_other_suffix = read_logits(model, altered)[:, :length]
        differences.extend((np.max(np.abs(prefix - unchanged)),
                            np.max(np.abs(with_other_suffix - unchanged))))
    error = float(max(differences))
    if error > ATOL:
        raise CheckFailed(f"causality or batch independence: prefix error {error:.3g}")
    return error
# endregion prefix_check


# region verify_greedy
def check_generation(model, prompt):
    steps = 4
    oracle = prompt.copy()
    for _ in range(steps):
        window = np.array([oracle[-CONFIG["context"]:]])
        oracle.append(int(np.argmax(read_logits(model, window)[0, -1])))

    first = oracle[len(prompt)]
    for eos, expected in ((None, oracle), (first, [*prompt, first])):
        argument = prompt.copy()
        actual = list(model.generate(argument, steps, eos=eos))
        if argument != prompt or actual != expected:
            raise CheckFailed("generate: expected greedy last-row IDs, context crop and EOS")
    return oracle
# endregion verify_greedy


# region verify_training
def verify(factory, *, seed=42, token_seed=None, steps=160):
    if steps < 1:
        raise ValueError("At least one training step is required")
    ids, targets = context_batch(token_seed)
    probe = np.column_stack((ids, targets[:, -1]))
    model = factory(seed=seed, **CONFIG)
    fresh = factory(seed=seed, **CONFIG)
    initial = read_logits(model, ids)
    if np.max(np.abs(read_logits(fresh, ids) - initial)) > ATOL:
        raise CheckFailed("initialization: same seed did not reproduce logits")
    before_prefix_error = check_prefix(model, probe)
    before = mean_nll(initial, targets)
    first = float(model.train_step(ids.copy(), targets.copy()))
    if not np.isclose(first, before, atol=ATOL, rtol=0):
        raise CheckFailed(f"train_step: returned {first}, expected pre-update mean NLL {before}")
    if np.max(np.abs(read_logits(model, ids) - initial)) <= ATOL:
        raise CheckFailed("train_step: first update left every logit unchanged")
    for _ in range(steps - 1):
        if not np.isfinite(model.train_step(ids.copy(), targets.copy())):
            raise CheckFailed("train_step: received nonfinite training loss")

    logits = read_logits(model, ids)
    after = mean_nll(logits, targets)
    final_rows = logits[:, -1, :]
    last_nll = mean_nll(final_rows, targets[:, -1])
    winners = final_rows.argmax(axis=-1)
    if after >= before or last_nll >= 0.2 or not np.array_equal(winners, targets[:, -1]):
        raise CheckFailed(f"context task: NLL {before:.4f} → {after:.4f}, "
                          f"last-position NLL {last_nll:.4f}, predicted {winners.tolist()}, "
                          f"expected {targets[:, -1].tolist()}")
    after_prefix_error = check_prefix(model, probe)
    output = check_generation(model, ids[0].tolist())
    return {
        "seed": seed, "token_seed": token_seed, "steps": steps, "config": CONFIG,
        "initial_nll": before, "final_nll": after, "last_position_nll": last_nll,
        "last_token_only_nll_lower_bound": float(np.log(2)),
        "predictions": winners.tolist(), "targets": targets[:, -1].tolist(),
        "max_prefix_error": max(before_prefix_error, after_prefix_error),
        "generation": output,
    }
# endregion verify_training


# region reference_adapter
def reference_model(*, seed, **config):
    from src.numpy_gpt.generate import generate
    from src.numpy_gpt.model import Config, GPT
    from src.numpy_gpt.optim import AdamW
    from src.numpy_gpt.train import train_step

    model = GPT(Config(**config), seed=seed)
    optimizer = AdamW(model.parameters, lr=0.01)

    class Adapter:
        def forward(self, ids):
            return model.forward(ids, track=False).data

        def train_step(self, ids, targets):
            if not np.array_equal(ids[:, 1:], targets[:, :-1]):
                raise ValueError("Expected shifted labels")
            batch = np.column_stack((ids, targets[:, -1]))
            loss, _ = train_step(model, optimizer, batch)
            return loss

        def generate(self, prompt, max_new_tokens, *, eos=None):
            return generate(model, prompt, max_new_tokens, temperature=0., eos=eos)

    return Adapter()
# endregion reference_adapter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", metavar="MODULE:FACTORY", help="import your local model factory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-seed", type=int, help="permute the eight IDs before training")
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    factory = reference_model
    if args.candidate:
        module, separator, name = args.candidate.partition(":")
        if not separator or not module or not name:
            parser.error("Use --candidate MODULE:FACTORY")
        factory = getattr(import_module(module), name)
    try:
        result = verify(factory, seed=args.seed, token_seed=args.token_seed, steps=args.steps)
    except CheckFailed as error:
        parser.exit(1, f"FAIL {error}\n")
    print(json.dumps(result, indent=2))

"""Deterministic filtering followed by categorical sampling."""

import numpy as np


def distribution(logits, temperature=1.0, top_k=None, top_p=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0 or not np.all(np.isfinite(logits)):
        raise ValueError("Expected a finite nonempty logits row")
    if not np.isfinite(temperature) or temperature < 0 or not (0 < top_p <= 1):
        raise ValueError("Invalid temperature or top_p")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    order = np.argsort(-logits, kind="stable")
    probabilities = np.zeros_like(logits)
    if temperature == 0:
        probabilities[order[0]] = 1.0
        return probabilities
    order = order[:top_k]
    with np.errstate(over="ignore"):
        weights = np.exp((logits[order] - logits[order[0]]) / temperature)
    weights /= weights.sum()
    if top_p < 1:
        cut = min(np.searchsorted(weights.cumsum(), top_p, side="left") + 1, len(weights))
        order, weights = order[:cut], weights[:cut]
    probabilities[order] = weights / weights.sum()
    return probabilities


def generate(model, prompt, length, *, seed=42, temperature=1.0, top_k=None,
             top_p=1.0, eos=None, use_cache=False):
    sequence = list(prompt)
    if not sequence or length < 0:
        raise ValueError("Expected nonempty prompt and nonnegative generation length")
    rng = np.random.default_rng(seed)
    cache = None
    context = model.config.context
    for _ in range(length):
        if use_cache:
            if cache is not None and cache[0][0].shape[2] < context:
                inputs = sequence[-1:]
            else:
                inputs, cache = sequence[-context:], None
            logits, cache = model.forward([inputs], track=False, cache=cache, use_cache=True)
        else:
            logits = model.forward([sequence[-context:]], track=False)
        probs = distribution(logits.data[0, -1], temperature, top_k, top_p)
        token = int(rng.choice(probs.size, p=probs))
        sequence.append(token)
        if token == eos:
            break
    return sequence

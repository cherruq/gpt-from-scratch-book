"""M3. Counting, word embeddings, and positional representations."""

import numpy as np


# region bigram
def bigram(records, vocabulary, alpha=1.0):
    counts = np.full((vocabulary, vocabulary), alpha, dtype=np.float64)
    for ids in records:
        np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return counts / counts.sum(axis=-1, keepdims=True)
# endregion bigram


# region unigram
def unigram(records, vocabulary, alpha=1.0):
    counts = np.full(vocabulary, alpha, dtype=np.float64)
    for ids in records:
        np.add.at(counts, ids[1:], 1)
    return counts / counts.sum()
# endregion unigram


def cooccurrences(records, vocabulary, radius=1):
    counts = np.zeros((vocabulary, vocabulary), dtype=np.int64)
    pairs = []
    for record in records:
        for i, center in enumerate(record):
            for j in range(max(0, i - radius), min(len(record), i + radius + 1)):
                if i != j:
                    pairs.append((center, record[j]))
                    counts[center, record[j]] += 1
    return counts, np.asarray(pairs, dtype=int).reshape(-1, 2)


# region skipgram
def skipgram(pairs, vocabulary, *, width=4, steps=120, seed=42):
    rng = np.random.default_rng(seed)
    source = rng.normal(0, 0.1, (vocabulary, width))
    target = rng.normal(0, 0.1, (vocabulary, width))
    negative_weights = np.bincount(pairs[:, 1], minlength=vocabulary) ** 0.75
    negative_weights /= negative_weights.sum()
    negatives = rng.choice(vocabulary, size=(len(pairs), 3), p=negative_weights)
    centers = pairs[:, 0]
    outputs = np.column_stack((pairs[:, 1], negatives))
    labels = np.array([1., 0., 0., 0.])
    losses = []
    for step in range(steps + 1):
        c = source[centers]
        o = target[outputs]
        scores = np.einsum("nd,nkd->nk", c, o)
        losses.append(float(np.mean(np.logaddexp(0, scores) - scores * labels)))
        if step == steps:
            break
        p = np.exp(-np.logaddexp(0, -scores))
        g = (p - labels) / scores.size
        ds, dt = np.zeros_like(source), np.zeros_like(target)
        np.add.at(ds, centers, np.einsum("nk,nkd->nd", g, o))
        np.add.at(dt, outputs, g[:, :, None] * c[:, None, :])
        source -= 1.0 * ds
        target -= 1.0 * dt
    return source, target, losses
# endregion skipgram


# region positions
def sinusoidal(length, channels):
    if channels % 2:
        raise ValueError("This paired implementation requires an even width")
    position = np.arange(length)[:, None]
    frequencies = 10_000 ** (-np.arange(0, channels, 2) / channels)
    angle = position * frequencies
    result = np.empty((length, channels))
    result[:, 0::2] = np.sin(angle)
    result[:, 1::2] = np.cos(angle)
    return result
# endregion positions

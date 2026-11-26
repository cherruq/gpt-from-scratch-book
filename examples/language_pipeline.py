"""M3 lab. Follow short BPE records through cooccurrence, bigram and position."""

import argparse
import json

import numpy as np

from src.autograd.tensor import Tensor
from src.math_basics.language import bigram, cooccurrences, sinusoidal, skipgram, unigram
from src.numpy_gpt.model import Config, GPT, attention_weights
from src.tokenizer.bpe import ByteBPE


def counted_pairs(matrix):
    return [[int(i), int(j), int(matrix[i, j])] for i, j in zip(*matrix.nonzero())]


# region context_pairs
def context_report(records, vocab, radius, seed):
    counts, pairs = cooccurrences(records, vocab, radius)
    _, _, losses = skipgram(pairs, vocab, seed=seed)
    frequencies = np.bincount(pairs[:, 1], minlength=vocab)
    mass = frequencies ** 0.75
    noise = mass / mass.sum()
    active = np.flatnonzero(frequencies)
    return {"counts": counted_pairs(counts), "directed_pairs": len(pairs),
            "noise": [[int(i), float(noise[i])] for i in active],
            "steps": len(losses) - 1, "initial_loss": losses[0], "final_loss": losses[-1]}
# endregion context_pairs


# region repeated_negative
def repeated_negative():
    center = np.array([1.0, 0.0])
    table = np.array([[0.0, 1.0], [0.0, -1.0]])
    outputs = np.array([0, 0, 1])
    labels = np.array([1.0, 0.0, 0.0])

    def loss(u, v):
        scores = v[outputs] @ u
        return float((np.logaddexp(0, scores) - labels * scores).mean())

    scores = table[outputs] @ center
    errors = (np.exp(-np.logaddexp(0, -scores)) - labels) / len(labels)
    center_grad = errors @ table[outputs]
    table_grad = np.zeros_like(table)
    np.add.at(table_grad, outputs, errors[:, None] * center)

    # Perturb shared rows themselves so repeated indices move together.
    step = 1e-5
    max_error = 0.0
    for parameter, gradient in ((center, center_grad), (table, table_grad)):
        for index in np.ndindex(parameter.shape):
            before = parameter[index]
            parameter[index] = before + step
            plus = loss(center, table)
            parameter[index] = before - step
            minus = loss(center, table)
            parameter[index] = before
            max_error = max(max_error, abs((plus - minus) / (2 * step) - gradient[index]))
    return {"loss": loss(center, table), "scores": scores.tolist(),
            "score_grad": errors.tolist(), "center_grad": center_grad.tolist(),
            "table_grad": table_grad.tolist(), "difference_max_error": float(max_error)}
# endregion repeated_negative


# region next_token_report
def transition_counts(records, vocab):
    counts = np.zeros((vocab, vocab), dtype=np.int64)
    for ids in records:
        np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return counts


def bigram_report(train, held_out, vocab, alpha):
    counts = transition_counts(train, vocab)
    conditional = bigram(train, vocab, alpha)
    marginal = unigram(train, vocab, alpha)
    prefixes = np.concatenate([ids[:-1] for ids in held_out])
    targets = np.concatenate([ids[1:] for ids in held_out])
    p = conditional[prefixes, targets]
    seen_counts = counts[prefixes, targets]
    context_counts = counts.sum(axis=1)[prefixes]
    has_zero = np.any(seen_counts == 0)
    mle_nll = None if has_zero else float(-np.log(seen_counts / context_counts).mean())
    smooth_nll = float(-np.log(p).mean())
    return {"counts": counted_pairs(counts), "targets": len(targets),
            "held_out": [{"from": int(i), "to": int(j), "count": int(n), "p": float(prob)}
                         for i, j, n, prob in zip(prefixes, targets, seen_counts, p)],
            "unseen_transitions": int(np.count_nonzero(seen_counts == 0)),
            "unseen_contexts": int(np.count_nonzero(context_counts == 0)),
            "mle_nll": mle_nll, "unigram_nll": float(-np.log(marginal[targets]).mean()),
            "bigram_nll": smooth_nll, "bigram_ppl": float(np.exp(smooth_nll))}
# endregion next_token_report


# region position_control
def position_control(ids):
    """Fixed query and two visible slots, using the book's attention operator."""
    q = Tensor([[[[np.sqrt(2), 0.0]]]])   # [B=1, H=1, Tq=1, D=2]
    tokens = np.array([[0.0, 0.0], [0.0, 1.0]])
    slots = np.array([[0.0, 0.0], [np.log(3), 0.0]])
    value_projection = np.diag([0.0, 1.0])
    results = []
    for positions in (np.zeros_like(slots), slots):
        for order in (np.array([0, 1]), np.array([1, 0])):
            keys = tokens[order] + positions
            values = keys @ value_projection
            weights = attention_weights(q, Tensor(keys[None, None]))
            output = (weights @ Tensor(values[None, None])).data[0, 0, 0]
            results.append({"ids": ids[order].tolist(), "positions": bool(positions.any()),
                            "weights": weights.data[0, 0, 0].tolist(), "output": output.tolist()})
    return results


def position_report(ids, vocab, seed):
    model = GPT(Config(vocab=vocab, context=len(ids), layers=1), seed=seed)
    observed = {}
    model.forward(ids[None, :], track=False, observer=observed.__setitem__)
    token_rows = model.parameters["token"].data[ids]
    slots = model.parameters["position"].data
    embedding = observed["embedding"][0]
    np.testing.assert_array_equal(embedding, token_rows + slots)
    repeated = None
    _, first, frequency = np.unique(ids, return_index=True, return_counts=True)
    if np.any(frequency >= 2):
        token = ids[first[frequency >= 2].min()]
        a, b = np.flatnonzero(ids == token)[:2]
        np.testing.assert_array_equal(token_rows[a], token_rows[b])
        delta = embedding[b] - embedding[a]
        np.testing.assert_allclose(delta, slots[b] - slots[a], atol=1e-15, rtol=0)
        repeated = {"id": int(token), "slots": [int(a), int(b)],
                    "token_delta": (token_rows[b] - token_rows[a]).tolist(),
                    "embedding_delta": delta.tolist()}
    p = sinusoidal(4, 4)
    return {"repeat": repeated, "control": position_control(ids[:2]),
            "sinusoidal_one_step": [float(p[0] @ p[1]), float(p[2] @ p[3])]}
# endregion position_control


def experiment(train, held_out, *, merges=2, radius=1, alpha=0.01, seed=42):
    if not train or not held_out:
        raise ValueError("Provide at least one training and one held-out record")
    if merges < 0 or radius < 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Expected nonnegative merges, positive radius and finite positive alpha")
    tokenizer = ByteBPE.train(train, vocab=ByteBPE.first_merge + merges)
    train_ids = [np.array(tokenizer.encode(text, boundaries=True)) for text in train]
    held_out_ids = [np.array(tokenizer.encode(text, boundaries=True)) for text in held_out]
    return {"train": train, "held_out": held_out, "vocab": tokenizer.vocab,
            "merges": [list(pair) for pair in tokenizer.merges],
            "radius": radius, "alpha": alpha, "seed": seed,
            "train_ids": [ids.tolist() for ids in train_ids],
            "held_out_ids": [ids.tolist() for ids in held_out_ids],
            "context": context_report(train_ids, tokenizer.vocab, radius, seed),
            "negative": repeated_negative(),
            "bigram": bigram_report(train_ids, held_out_ids, tokenizer.vocab, alpha),
            "positions": position_report(train_ids[0], tokenizer.vocab, seed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", help="training record; repeat for multiple records")
    parser.add_argument("--eval", action="append", help="held-out record; repeat for multiple records")
    parser.add_argument("--merges", type=int, default=2)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = experiment(args.train or ["春春", "春"], args.eval or ["春春春", ""],
                            merges=args.merges, radius=args.radius, alpha=args.alpha, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))

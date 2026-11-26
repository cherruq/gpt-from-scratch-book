"""M3 experiments. Run python -m examples.language 17..21."""

import argparse

import numpy as np

from src.autograd.tensor import Tensor
from src.math_basics.language import bigram, cooccurrences, sinusoidal, skipgram, unigram
from src.numpy_gpt.generate import distribution
from src.numpy_gpt.model import cross_entropy
from src.numpy_gpt.optim import AdamW
from src.tokenizer.bpe import ByteBPE


# region tokenizer
def tokenizer():
    toy = ByteBPE.train(["abab"], vocab=260)
    samples = ["春风，晴雪。", "for i in range(3):\n    print(i)\n", "你好🐍\t"]
    learned = ByteBPE.train(samples, vocab=280)
    decoded = [learned.decode(learned.encode(s, boundaries=True)) for s in samples]
    assert decoded == samples
    return toy.merges, toy.encode("abab"), learned.vocab
# endregion tokenizer


# region embedding
def embedding():
    ids = np.array([1, 1, 0])
    table = Tensor([[1., 2.], [3., 4.], [5., 6.]], True)
    rows = table[ids]
    np.testing.assert_array_equal(rows.data, np.eye(3)[ids] @ table.data)
    (rows * [[1, 1], [2, 3], [4, 5]]).sum().backward()
    return rows.data, table.grad
# endregion embedding


# region word2vec
def word2vec():
    records = [[0, 2, 0, 2], [1, 2, 1, 2], [3, 4, 3, 4]]
    counts, pairs = cooccurrences(records, 5)
    vectors, _, losses = skipgram(pairs, 5)
    unit = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    return counts, (losses[0], losses[-1]), unit @ unit.T
# endregion word2vec


# region counts_model
def counts_model():
    records = [np.array([0, 1, 0, 1]), np.array([0, 1])]
    targets = np.array([1, 0, 1])
    prefixes = np.array([0, 1, 0])
    one, two = unigram(records, 2), bigram(records, 2)
    counts_ppl = np.exp(-np.log(two[prefixes, targets]).mean())
    rng = np.random.default_rng(42)
    embedding = Tensor(rng.normal(0, 0.2, (2, 2)), True)
    weight = Tensor(rng.normal(0, 0.2, (2, 2)), True)
    parameters = {"embedding": embedding, "weight": weight}
    model = AdamW(parameters, lr=0.05, decay=0)
    initial = None
    for _ in range(60):
        for p in parameters.values():
            p.grad = None
        loss = cross_entropy(embedding[prefixes] @ weight, targets)
        initial = float(loss.data) if initial is None else initial
        loss.backward()
        model.step()
    final = float(cross_entropy(embedding[prefixes] @ weight, targets).data)
    greedy = 0
    sample = [greedy]
    for _ in range(4):
        greedy = int(distribution(embedding.data[greedy] @ weight.data, 0).argmax())
        sample.append(greedy)
    return one, two, counts_ppl, initial, final, sample
# endregion counts_model


# region position_demo
def position_demo():
    p = sinusoidal(4, 4)
    return p[0], p[1], float(p[0] @ p[1]), float(p[2] @ p[3])
# endregion position_demo


EXPERIMENTS = {17: tokenizer, 18: embedding, 19: word2vec, 20: counts_model, 21: position_demo}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=int, choices=EXPERIMENTS)
    args = parser.parse_args()
    print(EXPERIMENTS[args.chapter]())

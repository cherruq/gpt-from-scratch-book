"""Chapter 1 / M0: one complete forward, training and generation cycle."""

import numpy as np

from src.numpy_gpt.generate import generate
from src.numpy_gpt.model import Config, GPT, cross_entropy, softmax
from src.numpy_gpt.optim import AdamW


def run():
    vocabulary = "春夏秋冬风雨晴雪"
    x = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = np.array([[1, 2, 3, 0], [5, 6, 7, 4]])
    model = GPT(Config(vocab=len(vocabulary)), seed=42)
    optimizer = AdamW(model.parameters, lr=0.025)
    initial_logits = model.forward(x, track=False)
    initial = float(cross_entropy(initial_logits, targets).data)
    for _ in range(80):
        model.zero_grad()
        loss = cross_entropy(model.forward(x), targets)
        loss.backward()
        optimizer.step()
    final_logits = model.forward(x, track=False)
    final = float(cross_entropy(final_logits, targets).data)
    sample = generate(model, [0], 7, temperature=0)
    first_row = softmax(final_logits).data[0, 0]
    return initial_logits.shape, initial, final, first_row, "".join(vocabulary[i] for i in sample)


if __name__ == "__main__":
    shape, initial, final, row, text = run()
    print(f"logits shape = {shape}")
    print(f"loss: {initial:.6f} -> {final:.6f}")
    print("first position p =", np.round(row, 6))
    print("greedy =", text)

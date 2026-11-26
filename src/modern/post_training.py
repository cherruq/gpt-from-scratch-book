"""A two-prompt RLVR bandit and a numerically stable DPO pair loss."""

import numpy as np


def dpo_loss(policy_log_odds, reference_log_odds, beta=0.1):
    return np.logaddexp(0, -beta * (policy_log_odds - reference_log_odds))


# region rlvr
def parity_policy(steps=45, batch=32, seed=42):
    prompts = np.array([[2, 3], [3, 3]])
    correct = prompts.sum(axis=-1) % 2
    logits = np.zeros((len(prompts), 2))
    rng = np.random.default_rng(seed)
    rows = np.arange(len(prompts))
    for _ in range(steps):
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p = exp / exp.sum(axis=-1, keepdims=True)
        baseline = p[rows, correct]
        actions = np.stack([rng.choice(2, size=batch, p=row) for row in p])
        rewards = (actions == correct[:, None]).astype(float)
        score_gradient = np.eye(2)[actions] - p[:, None, :]
        logits += ((rewards - baseline[:, None])[..., None] * score_gradient).mean(axis=1)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = exp / exp.sum(axis=-1, keepdims=True)
    return correct, p[rows, correct]
# endregion rlvr

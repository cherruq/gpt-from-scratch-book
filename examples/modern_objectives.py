"""M8. Audit routing, streamed attention and small post-training objectives."""

import json

import numpy as np

from src.autograd.tensor import Tensor
from src.modern.architecture import mixture
from src.modern.efficiency import cache_bytes, online_attention
from src.modern.post_training import dpo_loss
from src.numpy_gpt.model import cross_entropy


# region routing_budget
def routing_budget():
    hidden = Tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    router = Tensor(np.eye(2) * np.log(3.), requires_grad=True)
    experts = [Tensor(np.eye(2)), Tensor(2 * np.eye(2))]
    result, auxiliary, load = mixture(hidden, router, experts)
    result.sum().backward()
    task_gradient = router.grad.copy()
    router.grad = None
    auxiliary.backward()

    channels, heads, width = 16, 4, 4
    resources = []
    for name, kv in (("MHA", 4), ("GQA", 2), ("MQA", 1)):
        resources.append({"name": name, "query_heads": heads, "kv_heads": kv,
                          "projection_weights": channels * width * (2 * heads + 2 * kv),
                          "cache_bytes": cache_bytes(2, 1, kv, 128, width, 2)})
    return {"load": load.tolist(), "auxiliary": float(auxiliary.data),
            "task_router_gradient": task_gradient.tolist(), "aux_router_gradient": router.grad.tolist(),
            "resources": resources}
# endregion routing_budget


# region streaming_check
def streaming_check():
    scores = np.log([1., 3., 6.])
    values = np.array([[0.], [2.], [4.]])
    prefix_weights = np.exp(scores[:2] - scores[1])
    mass = float(prefix_weights.sum())
    accumulator = float((prefix_weights @ values[:2])[0])
    prefix_scale = float(np.exp(scores[1] - scores[2]))
    streamed = float(online_attention(scores, values, block=2)[0])
    shifted = float(online_attention(scores + 1000, values, block=2)[0])

    target = np.array([.6, .3, .1])
    draft = np.array([.3, .5, .2])
    accepted = np.minimum(target, draft)
    residual = np.maximum(target - draft, 0.)
    return {"old_mass": mass, "old_accumulator": accumulator, "rescale": prefix_scale,
            "new_mass": mass * prefix_scale + 1., "new_accumulator": accumulator * prefix_scale + values[2, 0],
            "streamed": streamed, "shifted": shifted,
            "acceptance": float(accepted.sum()), "accepted_mass": accepted.tolist(),
            "rejection_distribution": (residual / residual.sum()).tolist(),
            "combined": (accepted + residual).tolist()}
# endregion streaming_check


# region policy_objectives
def clipped_surrogate(ratio, advantage, epsilon=.2):
    return np.minimum(ratio * advantage, np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)


def relative_advantages(rewards):
    centered = rewards - rewards.mean()
    scale = centered.std()
    return centered / scale if scale > 0 else np.zeros_like(centered)


def policy_objectives():
    # Inputs [BOS, question, answer], targets [question, answer, EOS].
    logits = Tensor(np.zeros((3, 4)), requires_grad=True)
    labels = np.array([1, 2, 3])
    valid = np.array([False, True, True])
    loss = cross_entropy(logits[valid], labels[valid])
    loss.backward()

    ratio = np.array([1.3, .7, 1.3, .7])
    advantage = np.array([1., 1., -1., -1.])
    ppo = clipped_surrogate(ratio, advantage)
    step = 1e-5
    slope = (clipped_surrogate(ratio + step, advantage)
             - clipped_surrogate(ratio - step, advantage)) / (2 * step)
    group = relative_advantages(np.array([1., 0., 1., 0.]))
    margin = 1.
    beta = .1
    dpo_grad = -beta / (1 + np.exp(beta * margin))
    dpo_diff = (dpo_loss(margin + step, 0., beta) - dpo_loss(margin - step, 0., beta)) / (2 * step)
    return {"sft_loss": float(loss.data), "sft_gradients": logits.grad.tolist(),
            "ppo": ppo.tolist(), "ppo_mean": float(ppo.mean()), "ppo_slopes": slope.tolist(),
            "group_advantage": group.tolist(), "equal_group_advantage": relative_advantages(np.ones(4)).tolist(),
            "dpo_loss": float(dpo_loss(margin, 0., beta)), "dpo_slope": float(dpo_grad),
            "dpo_slope_error": float(abs(dpo_grad - dpo_diff))}
# endregion policy_objectives


def experiment():
    return {"routing": routing_budget(), "efficiency": streaming_check(), "policy": policy_objectives()}


if __name__ == "__main__":
    print(json.dumps(experiment(), indent=2, allow_nan=False))

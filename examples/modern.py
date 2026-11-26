"""M8 reproducible inference, modern architecture, and analysis experiments."""

import argparse
from pathlib import Path

import numpy as np

from examples.preview import run as preview
from src.autograd.tensor import Tensor
from src.modern.architecture import mixture, modern_block
from src.modern.efficiency import cache_bytes, forward_flops, online_attention, quantize_int8, speculative_choice
from src.modern.post_training import dpo_loss, parity_policy
from src.numpy_gpt.data import restore_tokenizer
from src.numpy_gpt.model import Config, GPT, softmax
from src.numpy_gpt.train import load_checkpoint


# region cached
def cached():
    model = GPT(Config(vocab=8))
    ids = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    prefill, cache = model.forward(ids[:, :2], track=False, use_cache=True)
    chunks = [prefill.data]
    for t in range(2, 4):
        step, cache = model.forward(ids[:, t:t+1], track=False, cache=cache, use_cache=True)
        chunks.append(step.data)
    reference = model.forward(ids, track=False).data
    error = np.max(np.abs(np.concatenate(chunks, axis=1) - reference))
    memory = sum(k.nbytes + v.nbytes for k, v in cache)
    assert memory == cache_bytes(2, 2, 2, 4, 2, 8)
    return {"cache_bytes": memory, "logits_max_abs": float(error)}
# endregion cached


# region modern_parts
def modern_parts():
    x = Tensor([[[1., 0., 0., 1.], [0., 1., 1., 0.]]], True)
    rng = np.random.default_rng(42)
    shapes = {"q": (4, 4), "k": (4, 2), "v": (4, 2), "out": (4, 4),
              "gate": (4, 6), "up": (4, 6), "down": (6, 4)}
    parameters = {name: Tensor(rng.normal(0, .1, shape), True) for name, shape in shapes.items()}
    parameters.update({name: Tensor(np.ones(4), True) for name in ("attention_gain", "ffn_gain")})
    out = modern_block(x, parameters, 2, 1)
    out.sum().backward()
    simple = Tensor([[[2., 0.]]])
    router = Tensor([[1., 0.], [0., 1.]])
    chosen, load_loss, load = mixture(simple, router, [Tensor(np.eye(2)), Tensor(2 * np.eye(2))])
    return {"block_shape": out.shape, "block_input_gradient_finite": bool(np.isfinite(x.grad).all()),
            "moe": chosen.data.tolist(), "router_auxiliary": float(load_loss.data), "load": load.tolist()}
# endregion modern_parts


# region efficiency
def efficiency():
    scores = np.array([1000., 999., 998., -np.inf])
    values = np.array([[2., 0.], [0., 4.], [2., 6.], [9., 9.]])
    online = online_attention(scores, values, 2)
    dense = softmax(Tensor(scores)).data @ values
    weight = np.array([[1., 0.], [-0.4, 0.]])
    quantized, scale = quantize_int8(weight)
    rng = np.random.default_rng(42)
    draws = [speculative_choice(np.array([.6, .3, .1]), np.array([.3, .4, .3]), rng)
             for _ in range(5_000)]
    frequency = np.bincount([value for value, _ in draws], minlength=3) / len(draws)
    return {"attention": online.tolist(), "online_error": float(np.max(np.abs(online - dense))),
            "quantization_error": float(np.max(np.abs(weight - quantized * scale))),
            "decode_frequency": frequency.tolist(),
            "forward_flops_config_larger": forward_flops(4, 128, 128, 512, 4, 124)}
# endregion efficiency


def post_training():
    correct, probabilities = parity_policy()
    return {"answers": correct.tolist(), "probability_of_correct": probabilities.tolist(),
            "dpo_zero_margin": float(dpo_loss(0, 0)), "dpo_policy_log_odds_1": float(dpo_loss(1, 0))}


# region interpretation
def interpretation(checkpoint):
    model, _, _, info = load_checkpoint(checkpoint)
    tokenizer = restore_tokenizer(info["tokenizer"])

    def observe(text):
        states = {}
        prefix = [tokenizer.bos, *tokenizer.encode(text)]
        array = np.array([prefix[-model.config.context:]])
        logits = model.forward(array, track=False, observer=states.__setitem__)
        return array, logits, states

    def features(path):
        samples, labels = [], []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            scene = line[5:7]
            prefix = line[:line.index("看见了") + 3]
            _, _, state = observe(prefix)
            samples.append(state["block.1"][0, -1])
            labels.append(1 if scene in {"山间", "河边", "村口", "庭院"} else -1)
        features = np.asarray(samples)
        return np.column_stack((features, np.ones(len(features)))), np.asarray(labels)

    train_x, train_y = features("data/train.txt")
    valid_x, valid_y = features("data/validation.txt")
    readout = np.linalg.solve(train_x.T @ train_x + 0.1 * np.eye(train_x.shape[1]), train_x.T @ train_y)
    accuracy = float(np.mean(np.where(valid_x @ readout >= 0, 1, -1) == valid_y))
    x, logits, state = observe("今早我走到山间，看见了")
    _, alt_logits, _ = observe("今早我走到窗前，看见了")
    target = tokenizer.encode("春")[0]
    probability = lambda logits: float(softmax(logits).data[0, -1, target])
    baseline_probability, alternative_probability = probability(logits), probability(alt_logits)
    attention = state["block.0.attention"][0, 0, -1]
    for term in ("out.weight", "out.bias"):
        model.parameters["block.0." + term].data.fill(0)
    ablated = model.forward(x, track=False)
    tokens = [tokenizer.decode([int(i)]) or "BOS" for i in x[0]]
    return {"probe_accuracy": accuracy, "probe_validation_count": len(valid_y),
            "candidate": "春", "original": baseline_probability, "other_context": alternative_probability,
            "ablation": probability(ablated), "attention_row": attention.tolist(), "attention_tokens": tokens}
# endregion interpretation


# region capstone
def capstone():
    shape, initial, final, _, generated = preview()
    cache = cached()
    if tuple(shape) != (2, 4, 8) or final >= initial * 0.05 or cache["logits_max_abs"] > 1e-12:
        raise AssertionError("Miniature project acceptance criteria failed")
    return {"logits_shape": shape, "loss_before": initial, "loss_after": final,
            "generated": generated, **cache}
# endregion capstone


if __name__ == "__main__":
    import json
    functions = {35: cached, 36: modern_parts, 37: efficiency, 38: post_training, 39: interpretation, 40: capstone}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=int, choices=functions)
    parser.add_argument("--checkpoint", default="runs/quick.npz", help="Chapter 39's trained NumPy checkpoint")
    args = parser.parse_args()
    result = interpretation(args.checkpoint) if args.chapter == 39 else functions[args.chapter]()
    print(json.dumps(result, ensure_ascii=False, indent=2))

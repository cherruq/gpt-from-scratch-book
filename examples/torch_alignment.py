"""Chapter 33 / M7 independent reference verification."""

import json

import numpy as np
import torch

from src.numpy_gpt.model import Config, GPT, cross_entropy
from src.numpy_gpt.optim import AdamW
from src.torch_gpt.model import TorchGPT, objective


# region align
def align():
    cfg = Config(vocab=8)
    reference = GPT(cfg)
    replica = TorchGPT(cfg, seed=7)
    replica.import_numpy(reference.state_dict())
    ids = np.array([[0, 1, 2, 3], [4, 1, 6, 7]])
    targets = np.array([[1, 2, 3, 0], [1, 6, 7, 4]])
    n_logits = reference.forward(ids)
    t_logits = replica(torch.from_numpy(ids))
    n_loss = cross_entropy(n_logits, targets)
    t_loss = objective(t_logits, torch.from_numpy(targets))
    n_loss.backward()
    t_loss.backward()

    def error(a, b):
        return float(np.max(np.abs(a - b)))

    results = {
        "logits_max_abs": error(n_logits.data, t_logits.detach().numpy()),
        "loss_abs": abs(float(n_loss.data) - float(t_loss.detach())),
        "gradient_max_abs": max(error(reference.parameters[name].grad, value.grad.numpy())
                                for name, value in replica.named_weights()),
        "gradient_floor": 1e-11,
    }
    adamw = AdamW(reference.parameters, lr=0.003, max_norm=None)
    torch_adamw = torch.optim.AdamW(replica.parameters(), lr=0.003, weight_decay=0.01, foreach=False)
    adamw.step()
    torch_adamw.step()
    results["update_max_abs"] = max(
        error(value, replica.numpy_state()[name]) for name, value in reference.state_dict().items()
    )
    return results
# endregion align


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(align(), indent=2))

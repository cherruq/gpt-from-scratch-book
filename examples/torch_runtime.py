"""CPU one-step comparisons for bfloat16 autocast and optional two-rank Gloo."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from src.numpy_gpt.model import Config
from src.torch_gpt.model import TorchGPT


def batch():
    ids = torch.tensor([[0, 1, 2, 3], [4, 1, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 0], [1, 6, 7, 4]])
    valid = torch.tensor([[True, False, False, False], [True, True, True, False]])
    return ids, targets, valid


def flat_gradients(model):
    return torch.cat([param.grad.flatten() for _, param in model.named_weights()])


def max_difference(left, right):
    return float((left - right).abs().max())


# region amp
def amp_report():
    ids, targets, valid = batch()

    def step(enabled):
        model = TorchGPT(Config(vocab=8), dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, foreach=False)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
            logits = model(ids)
            loss = F.cross_entropy(logits[valid], targets[valid])
        loss.backward()
        gradient = flat_gradients(model)
        optimizer.step()
        weights = torch.cat([p.detach().reshape(-1) for _, p in model.named_weights()])
        return logits.detach().float(), loss.detach(), gradient, weights, str(logits.dtype)

    logits, loss, grad, params, _ = step(False)
    amp_logits, amp_loss, amp_grad, amp_params, dtype = step(True)
    return {"logits_dtype": dtype, "loss_dtype": str(amp_loss.dtype),
            "gradient_dtype": str(amp_grad.dtype), "weights_dtype": str(amp_params.dtype),
            "nll_fp32": float(loss), "nll_autocast": float(amp_loss),
            "logits_max_abs": max_difference(logits, amp_logits),
            "gradient_max_abs": max_difference(grad, amp_grad),
            "weights_max_abs_after_step": max_difference(params, amp_params)}
# endregion amp


# region ddp
def rank_worker(rank, rendezvous, output):
    # Gloo writes its peer banners to stdout. Keep the parent process's JSON clean.
    os.dup2(2, 1)
    torch.set_num_threads(1)
    if sys.platform == "linux":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        ids, targets, valid = (array[rank:rank + 1] for array in batch())
        model = DistributedDataParallel(TorchGPT(Config(vocab=8)))
        local_count = valid.sum()
        global_count = local_count.clone()
        dist.all_reduce(global_count)
        gradients = {}
        for method, scale in (("local_means", 1 / local_count.to(torch.float64)),
                              ("global_mean", dist.get_world_size() / global_count.to(torch.float64))):
            model.zero_grad(set_to_none=True)
            local_sum = F.cross_entropy(model(ids)[valid], targets[valid], reduction="sum")
            (local_sum * scale).backward()
            gradients[method] = flat_gradients(model.module)
        if rank == 0:
            torch.save(gradients, output)
    finally:
        dist.destroy_process_group()


def ddp_report():
    with TemporaryDirectory(prefix="gpt-book-gloo-") as folder:
        root = Path(folder).resolve()
        path = root / "gradients.pt"
        torch.multiprocessing.spawn(rank_worker, args=((root / "rendezvous").as_uri(), path), nprocs=2)
        distributed = torch.load(path, weights_only=True)
    ids, targets, valid = batch()
    model = TorchGPT(Config(vocab=8))
    F.cross_entropy(model(ids)[valid], targets[valid]).backward()
    reference = flat_gradients(model)
    return {"backend": "gloo", "valid_targets_per_rank": valid.sum(dim=1).tolist(),
            "local_means_max_abs": max_difference(reference, distributed["local_means"]),
            "global_mean_max_abs": max_difference(reference, distributed["global_mean"])}
# endregion ddp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gloo", action="store_true", help="also run a two-rank local CPU collective")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = {"amp": amp_report()}
    if args.gloo:
        if not dist.is_available() or not dist.is_gloo_available():
            parser.error("The Gloo experiment requires a PyTorch build with Gloo")
        output["ddp"] = ddp_report()
    print(json.dumps(output, indent=2, allow_nan=False))

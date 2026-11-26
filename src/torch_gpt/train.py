"""Train and generate using the PyTorch baseline and shared corpus format."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.numpy_gpt.data import restore_tokenizer, split_targets, tokenizer_state
from src.numpy_gpt.generate import generate
from src.numpy_gpt.model import Config, GPT
from src.numpy_gpt.train import read_config
from src.torch_gpt.model import TorchGPT, objective


def as_tensors(rows, device):
    return tuple(torch.as_tensor(array.copy(), device=device) for array in split_targets(rows))


@torch.inference_mode()
def evaluate(model, windows, batch_size, device):
    total, count = 0.0, 0
    for start in range(0, len(windows), batch_size):
        x, y = as_tensors(windows[start:start + batch_size], device)
        total += float(objective(model(x), y)) * y.numel()
        count += y.numel()
    return total / count


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as file:
        temporary = Path(file.name)
        try:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


# region torch_fit
def fit(config, path, *, device="cpu", dtype=torch.float32, sdpa=False, resume=False, report=print):
    corpus = config.corpus()
    spec = config.model_config(corpus.tokenizer.vocab)
    model = TorchGPT(spec, config.seed, device=device, dtype=dtype, sdpa=sdpa)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01, foreach=False)
    rng = np.random.default_rng(config.seed)
    settings = {"shape": asdict(spec), "fingerprint": corpus.fingerprint, "batch_size": config.batch_size,
                "lr": config.lr, "dtype": str(dtype), "sdpa": sdpa, "format": 1}
    start = 0
    if resume:
        state = torch.load(path, weights_only=True, map_location=device)
        if state["settings"] != settings:
            raise ValueError("Checkpoint settings differ")
        model.load_state_dict(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = json.loads(state["rng"])
        start = state["step"]
    if config.steps < start:
        raise ValueError("Requested final step precedes the checkpoint")

    def log(step):
        nll = evaluate(model, corpus.validation, config.batch_size, device)
        report(json.dumps({"step": step, "validation_nll": nll, "validation_ppl": math.exp(nll),
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "validation_tokens": corpus.validation.size - len(corpus.validation)}))

    def checkpoint(step):
        save(path, {"settings": settings, "step": step, "parameters": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "rng": json.dumps(rng.bit_generator.state),
                    "tokenizer": tokenizer_state(corpus.tokenizer)})

    log(start)
    for step in range(start + 1, config.steps + 1):
        idx = rng.integers(len(corpus.train), size=config.batch_size)
        x, y = as_tensors(corpus.train[idx], device)
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(x), y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % config.log_every == 0 or step == config.steps:
            report(json.dumps({"step": step, "train_batch_nll": float(loss.detach()),
                               "gradient_norm": float(grad_norm)}))
            log(step)
            checkpoint(step)
    if config.steps == start and not resume:
        checkpoint(start)
    return model
# endregion torch_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("fit")
    training.add_argument("--config")
    training.add_argument("--checkpoint", default="runs/torch.pt")
    training.add_argument("--device", default="cpu")
    training.add_argument("--threads", type=int, default=1)
    training.add_argument("--steps", type=int)
    training.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    training.add_argument("--sdpa", action="store_true")
    training.add_argument("--resume", action="store_true")
    sampling = sub.add_parser("generate")
    sampling.add_argument("--checkpoint", default="runs/torch.pt")
    sampling.add_argument("--prompt", required=True)
    sampling.add_argument("--length", type=int, default=24)
    sampling.add_argument("--temperature", type=float, default=0.8)
    sampling.add_argument("--top-k", type=int)
    sampling.add_argument("--top-p", type=float, default=1.0)
    sampling.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "fit":
        torch.set_num_threads(args.threads)
        fit(read_config(args.config, args.steps), args.checkpoint, device=args.device,
            dtype=getattr(torch, args.dtype), sdpa=args.sdpa, resume=args.resume)
    else:
        torch.set_num_threads(1)
        state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        config = Config(**state["settings"]["shape"])
        replica = TorchGPT(config)
        replica.load_state_dict(state["parameters"])
        model = GPT(config)
        model.load_state_dict(replica.numpy_state())
        tokenizer = restore_tokenizer(state["tokenizer"])
        prefix = [tokenizer.bos, *tokenizer.encode(args.prompt)]
        ids = generate(model, prefix, args.length, seed=args.seed, temperature=args.temperature,
                       top_k=args.top_k, top_p=args.top_p, eos=tokenizer.eos, use_cache=True)
        options = {"errors": "replace"} if state["tokenizer"]["kind"] == "bpe" else {}
        print(tokenizer.decode(ids, **options))


if __name__ == "__main__":
    main()

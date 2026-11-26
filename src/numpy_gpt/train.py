"""CPU training, checkpoint resume and validation for the NumPy baseline."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from src.numpy_gpt.data import load_corpus, restore_tokenizer, split_targets, tokenizer_state
from src.numpy_gpt.generate import generate
from src.numpy_gpt.model import Config, GPT, cross_entropy
from src.numpy_gpt.optim import AdamW


@dataclass(frozen=True)
class TrainingConfig:
    train: str = "data/train.txt"
    validation: str = "data/validation.txt"
    tokenizer: str = "char"
    vocab: int = 320
    context: int = 16
    channels: int = 16
    heads: int = 4
    hidden: int = 48
    layers: int = 2
    batch_size: int = 4
    steps: int = 120
    log_every: int = 30
    lr: float = 0.005
    seed: int = 42

    def __post_init__(self):
        if self.batch_size <= 0 or self.steps < 0 or self.log_every <= 0:
            raise ValueError("Invalid training batch size or step count")

    def model_config(self, vocab):
        return Config(vocab, self.context, self.channels, self.heads, self.hidden, self.layers)

    def corpus(self):
        return load_corpus(self.train, self.validation, self.context, self.tokenizer, self.vocab)


# region train_step
def train_step(model, optimizer, batch):
    x, y = split_targets(batch)
    model.zero_grad()
    loss = cross_entropy(model.forward(x), y)
    if not np.isfinite(loss.data):
        raise FloatingPointError("Non-finite loss")
    loss.backward()
    norm = optimizer.step()
    return float(loss.data), norm
# endregion train_step


def evaluate(model, windows, batch_size):
    total, count = 0.0, 0
    for start in range(0, len(windows), batch_size):
        x, y = split_targets(windows[start:start + batch_size])
        total += float(cross_entropy(model.forward(x, track=False), y).data) * y.size
        count += y.size
    return total / count


# region checkpoint
def save_checkpoint(path, model, optimizer, rng, corpus, batch_size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"format": 1, "config": asdict(model.config), "optimizer": optimizer.options(),
                "steps": optimizer.steps, "rng": rng.bit_generator.state,
                "tokenizer": tokenizer_state(corpus.tokenizer), "corpus": corpus.fingerprint,
                "batch_size": batch_size}
    arrays = {"metadata": np.array(json.dumps(metadata))}
    for group, weights in (("p", model.state_dict()), ("m", optimizer.m), ("v", optimizer.v)):
        arrays.update({group + "." + name: values for name, values in weights.items()})
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as file:
        temporary = Path(file.name)
        try:
            np.savez(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].item())
        if metadata["format"] != 1:
            raise ValueError("Unsupported checkpoint format")
        model = GPT(Config(**metadata["config"]))
        names = model.parameters.keys()
        model.load_state_dict({name: archive["p." + name] for name in names})
        optimizer = AdamW(model.parameters, **metadata["optimizer"])
        for name in names:
            optimizer.m[name][...] = archive["m." + name]
            optimizer.v[name][...] = archive["v." + name]
        optimizer.steps = metadata["steps"]
    rng = np.random.default_rng()
    rng.bit_generator.state = metadata["rng"]
    return model, optimizer, rng, metadata
# endregion checkpoint


def train(config, checkpoint, *, resume=False, report=print):
    corpus = config.corpus()
    spec = config.model_config(corpus.tokenizer.vocab)
    if resume:
        model, optimizer, rng, metadata = load_checkpoint(checkpoint)
        if (spec != model.config or metadata["corpus"] != corpus.fingerprint
                or config.batch_size != metadata["batch_size"] or config.lr != optimizer.lr):
            raise ValueError("Current model, corpus, batch size or rate differs from checkpoint")
    else:
        model, rng = GPT(spec, seed=config.seed), np.random.default_rng(config.seed)
        optimizer = AdamW(model.parameters, lr=config.lr)
    if config.steps < optimizer.steps:
        raise ValueError("Requested final step precedes the checkpoint")

    def validation_log():
        nll = evaluate(model, corpus.validation, config.batch_size)
        report(json.dumps({"step": optimizer.steps, "validation_nll": nll,
                           "validation_ppl": math.exp(nll), "parameters": model.parameter_count(),
                           "validation_tokens": corpus.validation.size - len(corpus.validation)}))

    validation_log()
    start = optimizer.steps
    for step in range(start + 1, config.steps + 1):
        selection = rng.integers(len(corpus.train), size=config.batch_size)
        loss, norm = train_step(model, optimizer, corpus.train[selection])
        if step % config.log_every == 0 or step == config.steps:
            report(json.dumps({"step": step, "train_batch_nll": loss, "gradient_norm": norm}))
            validation_log()
            save_checkpoint(checkpoint, model, optimizer, rng, corpus, config.batch_size)
    if config.steps == start and not resume:
        save_checkpoint(checkpoint, model, optimizer, rng, corpus, config.batch_size)
    return model, optimizer, rng


def read_config(path, steps=None):
    config = json.loads(Path(path).read_text()) if path else {}
    if steps is not None:
        config["steps"] = steps
    return TrainingConfig(**config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    learn = sub.add_parser("fit")
    learn.add_argument("--config")
    learn.add_argument("--checkpoint", default="runs/numpy.npz")
    learn.add_argument("--steps", type=int)
    learn.add_argument("--resume", action="store_true")
    sample = sub.add_parser("generate")
    sample.add_argument("--checkpoint", default="runs/numpy.npz")
    sample.add_argument("--prompt", required=True)
    sample.add_argument("--length", type=int, default=24)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top-k", type=int)
    sample.add_argument("--top-p", type=float, default=1.0)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--cache", action="store_true")
    args = parser.parse_args()
    if args.command == "fit":
        train(read_config(args.config, args.steps), args.checkpoint, resume=args.resume)
    else:
        model, _, _, metadata = load_checkpoint(args.checkpoint)
        tokenizer = restore_tokenizer(metadata["tokenizer"])
        prefix = [tokenizer.bos, *tokenizer.encode(args.prompt)]
        output = generate(model, prefix, args.length, seed=args.seed, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p, eos=tokenizer.eos, use_cache=args.cache)
        decoded = tokenizer.decode(output, **({"errors": "replace"} if metadata["tokenizer"]["kind"] == "bpe" else {}))
        print(decoded)


if __name__ == "__main__":
    main()

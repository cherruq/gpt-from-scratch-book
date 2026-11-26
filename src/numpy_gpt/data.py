"""Separate corpora, fit a training-only vocabulary, and build next-token windows."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from src.tokenizer.bpe import ByteBPE
from src.tokenizer.char import Characters


# region windows
def windows(records, context, stride=1):
    if context < 1 or stride < 1:
        raise ValueError("context and stride must be positive")
    rows = [
        np.lib.stride_tricks.sliding_window_view(np.asarray(ids, dtype=np.int64), context + 1)[::stride]
        for ids in records if len(ids) >= context + 1
    ]
    if not rows:
        raise ValueError("No record is long enough for this context")
    return np.concatenate(rows, axis=0)


def split_targets(batch):
    return batch[:, :-1], batch[:, 1:]
# endregion windows


@dataclass(frozen=True)
class Corpus:
    train: np.ndarray
    validation: np.ndarray
    tokenizer: Characters | ByteBPE
    fingerprint: str


def tokenizer_state(tokenizer):
    if isinstance(tokenizer, Characters):
        return {"kind": "char", "symbols": tokenizer.symbols}
    return {"kind": "bpe", "merges": [list(pair) for pair in tokenizer.merges]}


def restore_tokenizer(state):
    if state["kind"] == "char":
        return Characters(state["symbols"])
    if state["kind"] == "bpe":
        return ByteBPE(tuple(tuple(pair) for pair in state["merges"]))
    raise ValueError("Unknown tokenizer kind")


def load_corpus(train_path, validation_path, context, tokenizer="char", vocab=320):
    contents = [Path(path).read_bytes() for path in (train_path, validation_path)]
    train, validation = ([line for line in content.decode("utf-8").splitlines() if line.strip()]
                         for content in contents)
    if not train or not validation or set(train) & set(validation):
        raise ValueError("Need nonempty train and validation files with no exact record overlap")
    if tokenizer == "char":
        model = Characters.train(train)
    elif tokenizer == "bpe":
        model = ByteBPE.train(train, vocab)
    else:
        raise ValueError("Use char or bpe tokenizer")

    def encoded(records):
        return [model.encode(text, boundaries=True) for text in records]

    metadata = [context, tokenizer_state(model), *(hashlib.sha256(c).hexdigest() for c in contents)]
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return Corpus(windows(encoded(train), context), windows(encoded(validation), context, stride=context),
                  model, fingerprint)

"""Character vocabulary for the small CPU training experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Characters:
    symbols: str
    bos = 0
    eos = 1
    unk = 2

    @classmethod
    def train(cls, texts):
        return cls("".join(sorted(set("".join(texts)))))

    @property
    def vocab(self):
        return len(self.symbols) + 3

    def encode(self, text, *, boundaries=False):
        lookup = {c: i for i, c in enumerate(self.symbols, 3)}
        ids = [lookup.get(c, self.unk) for c in text]
        return [self.bos, *ids, self.eos] if boundaries else ids

    def decode(self, ids):
        symbols = ["", "", "�", *self.symbols]
        if any(i < 0 or i >= len(symbols) for i in ids):
            raise ValueError("ID outside vocabulary")
        return "".join(symbols[i] for i in ids)

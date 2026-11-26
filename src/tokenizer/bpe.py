"""M3. Byte-level BPE over records, with a stable merge order."""

from collections import Counter
from dataclasses import dataclass
import operator


# region merge
def merge_pair(ids, pair, replacement):
    merged = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            merged.append(replacement)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged
# endregion merge


@dataclass(frozen=True)
class ByteBPE:
    merges: tuple[tuple[int, int], ...] = ()
    bos = 256
    eos = 257
    first_merge = 258

    @property
    def vocab(self):
        return self.first_merge + len(self.merges)

    @classmethod
# region train_bpe
    def train(cls, texts, vocab=320):
        if vocab < cls.first_merge:
            raise ValueError("Vocabulary must leave room for all bytes, BOS and EOS")
        records = [list(text.encode("utf-8")) for text in texts]
        merges = []
        for replacement in range(cls.first_merge, vocab):
            counts = Counter(pair for ids in records for pair in zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            records = [merge_pair(ids, pair, replacement) for ids in records]
            merges.append(pair)
        return cls(tuple(merges))
# endregion train_bpe

    def encode(self, text, *, boundaries=False):
        ids = list(text.encode("utf-8"))
        for replacement, pair in enumerate(self.merges, self.first_merge):
            ids = merge_pair(ids, pair, replacement)
        return [self.bos, *ids, self.eos] if boundaries else ids

    def decode(self, ids, *, errors="strict"):
        vocabulary = [bytes([i]) for i in range(256)] + [b"", b""]
        for left, right in self.merges:
            vocabulary.append(vocabulary[left] + vocabulary[right])
        ids = [operator.index(i) for i in ids]
        if any(i < 0 or i >= self.vocab for i in ids):
            raise ValueError("ID outside vocabulary")
        return b"".join(vocabulary[i] for i in ids).decode("utf-8", errors=errors)

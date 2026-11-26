"""M3 lab. Follow a UTF-8 string through BPE, lookup and scatter-add."""

import argparse
import json

import numpy as np

from src.autograd.tensor import Tensor
from src.tokenizer.bpe import ByteBPE, merge_pair


# region utf8_spans
def utf8_spans(text):
    spans = []
    offset = 0
    for char in text:
        raw = char.encode("utf-8")
        end = offset + len(raw)
        spans.append({"char": char, "code_point": f"U+{ord(char):04X}",
                      "byte_span": [offset, end], "hex": raw.hex(" ")})
        offset = end
    return spans


def token_spans(tokenizer, text, ids, code_points):
    widths = [1] * 256 + [0, 0]  # BOS and EOS carry no text bytes.
    for left, right in tokenizer.merges:
        widths.append(widths[left] + widths[right])
    raw = text.encode("utf-8")
    boundaries = {0} | {span["byte_span"][1] for span in code_points}
    spans = []
    offset = 0
    for token in ids:
        end = offset + widths[token]
        spans.append({"id": token, "byte_span": [offset, end],
                      "hex": raw[offset:end].hex(" "),
                      "aligned_to_code_points": offset in boundaries and end in boundaries})
        offset = end
    assert offset == len(raw)
    return spans
# endregion utf8_spans


# region trace_merges
def trace_merges(tokenizer, texts):
    records = [list(text.encode("utf-8")) for text in texts]
    initial = records
    steps = []
    for new_id, pair in enumerate(tokenizer.merges, tokenizer.first_merge):
        matches = sum((a, b) == pair for ids in records for a, b in zip(ids, ids[1:]))
        merged = [merge_pair(ids, pair, new_id) for ids in records]
        replacements = sum(map(len, records)) - sum(map(len, merged))
        steps.append({"pair": list(pair), "id": new_id, "matches": matches,
                      "replacements": replacements, "records": merged})
        records = merged
    return {"initial": initial, "steps": steps}, records
# endregion trace_merges


# region embedding_check
def embedding_check(ids, vocab):
    """Compare Tensor lookup to one-hot multiplication in both directions."""
    flat = ids.ravel()
    positions = np.arange(1, flat.size + 1, dtype=np.float64)
    upstream = np.stack((positions, positions + 1), axis=-1)
    values = np.arange(vocab, dtype=np.float64)
    table = Tensor(np.stack((values, -values), axis=-1), requires_grad=True)
    rows = table[ids]
    (rows * upstream.reshape(rows.data.shape)).sum().backward()

    one_hot = np.zeros((flat.size, vocab))
    one_hot[np.arange(flat.size), flat] = 1
    np.testing.assert_array_equal(rows.data.reshape(-1, 2), one_hot @ table.data)
    np.testing.assert_array_equal(table.grad, one_hot.T @ upstream)

    unique, counts = np.unique(flat, return_counts=True)
    row = unique[np.argmax(counts)]
    unhit = np.ones(vocab, dtype=bool)
    unhit[unique] = False
    # The checked loss is linear. A binary-exact step also avoids cancellation here.
    step = 0.125
    plus, minus = table.data.copy(), table.data.copy()
    plus[row, 0] += step
    minus[row, 0] -= step
    loss_plus = float(np.sum(plus[flat] * upstream))
    loss_minus = float(np.sum(minus[flat] * upstream))
    difference = (loss_plus - loss_minus) / (2 * step)
    np.testing.assert_equal(difference, table.grad[row, 0])
    return {"shape": list(rows.data.shape), "rows": rows.data.reshape(-1, 2).tolist(),
            "upstream": upstream.tolist(),
            "hits": [{"id": int(token), "count": int(count), "gradient": table.grad[token].tolist()}
                     for token, count in zip(unique, counts)],
            "unhit_rows": int(unhit.sum()),
            "max_unhit_gradient": float(np.max(np.abs(table.grad[unhit]), initial=0)),
            "difference": {"id": int(row), "channel": 0, "step": step,
                           "losses": [loss_plus, loss_minus], "value": difference}}
# endregion embedding_check


def experiment(text, training, max_merges):
    if max_merges < 0:
        raise ValueError("Maximum merge count must be nonnegative")
    training = list(training)
    tokenizer = ByteBPE.train(training, vocab=ByteBPE.first_merge + max_merges)
    training_trace, _ = trace_merges(tokenizer, training)
    encoding_trace, (encoded,) = trace_merges(tokenizer, [text])
    ids = tokenizer.encode(text, boundaries=True)
    assert ids == [tokenizer.bos, *encoded, tokenizer.eos]
    decoded = tokenizer.decode(ids)
    assert decoded == text
    code_points = utf8_spans(text)
    return {"text": text, "training": training, "max_merges": max_merges,
            "code_points": code_points, "vocab": tokenizer.vocab,
            "training_trace": training_trace, "encoding_trace": encoding_trace,
            "ids": ids, "token_spans": token_spans(tokenizer, text, ids, code_points),
            "decoded": decoded, "embedding": embedding_check(np.array([ids]), tokenizer.vocab)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="春春", help="text to encode and embed")
    parser.add_argument("--train", action="append", help="training record, repeat for multiple records")
    parser.add_argument("--merges", type=int, default=2, help="maximum number of BPE merges")
    args = parser.parse_args()
    if args.merges < 0:
        parser.error("--merges must be nonnegative")
    print(json.dumps(experiment(args.text, args.train or ["春"], args.merges),
                     ensure_ascii=False, indent=2))

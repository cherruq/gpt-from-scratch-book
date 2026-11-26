"""Measure CPU prefill and decode separately; run with one BLAS thread."""

from statistics import median
from time import perf_counter

import numpy as np

from src.numpy_gpt.model import Config, GPT


# region latency
def latency(repetitions=10):
    model = GPT(Config(vocab=32, context=32, channels=16, heads=4, hidden=48, layers=2))
    ids = (np.arange(32) % model.config.vocab)[None, :]
    result = {}
    for enabled in (False, True):
        prefill_ms, decode_ms = [], []
        for repetition in range(repetitions + 1):
            start = perf_counter()
            if enabled:
                _, cache = model.forward(ids[:, :16], track=False, use_cache=True)
            else:
                model.forward(ids[:, :16], track=False)
            prefill = (perf_counter() - start) * 1000
            times = []
            for position in range(16, 32):
                start = perf_counter()
                if enabled:
                    _, cache = model.forward(ids[:, position:position+1], track=False,
                                             cache=cache, use_cache=True)
                else:
                    model.forward(ids[:, :position+1], track=False)
                times.append((perf_counter() - start) * 1000)
            if repetition:
                prefill_ms.append(prefill)
                decode_ms.extend(times)
        result["cache" if enabled else "recompute"] = {
            "median_prefill_ms": median(prefill_ms), "median_decode_ms": median(decode_ms),
            "p95_decode_ms": float(np.percentile(decode_ms, 95)),
        }
    return result
# endregion latency


if __name__ == "__main__":
    import json
    print(json.dumps(latency(), indent=2))

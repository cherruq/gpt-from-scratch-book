"""M4. Isolate queries, keys and values in a two-slot attention example."""

import argparse
import json

import numpy as np

from src.autograd.tensor import Tensor
from src.numpy_gpt.model import attention_weights


# region project_and_read
def project_and_read(logit_gap=np.log(3)):
    x = np.array([[1., 0., 0., 2.], [0., 1., 4., 0.]])
    identity = np.eye(4)
    q = x @ (identity[:, :2] * np.sqrt(2) * logit_gap)
    k = x @ identity[:, :2]
    v = x @ identity[:, [3, 2]]

    def read(query, keys, values):
        scores = (query @ keys.T) / np.sqrt(keys.shape[-1])
        weights = attention_weights(Tensor(query[None, None]), Tensor(keys[None, None]))
        output = weights @ Tensor(values[None, None])
        return {"scores": scores.tolist(), "weights": weights.data[0, 0].tolist(),
                "output": output.data[0, 0].tolist()}

    return {
        "settings": {"logit_gap": float(logit_gap)}, "x": x.tolist(),
        "q": q.tolist(), "k": k.tolist(), "v": v.tolist(),
        "fixed": (np.array([[.25, .75]]) @ v).tolist(),
        "base": read(q[:1], k, v),
        "query": read(q[1:], k, v),
        "keys": read(q[:1], k[::-1], v),
        "values": read(q[:1], k, v[::-1]),
        "paired": read(q[:1], k[::-1], v[::-1]),
        "two_queries": read(q, k, v),
    }
# endregion project_and_read


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logit-gap", type=float, default=float(np.log(3)))
    args = parser.parse_args()
    if not np.isfinite(args.logit_gap) or abs(args.logit_gap) > np.finfo(float).max / np.sqrt(2):
        parser.error("--logit-gap must remain finite after multiplying by sqrt(2)")
    print(json.dumps(project_and_read(args.logit_gap), indent=2, allow_nan=False))

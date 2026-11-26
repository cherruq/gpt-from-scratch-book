import shutil
import subprocess
import json
import numpy as np
import pytest

from examples.math_basics import EXPERIMENTS, central_difference
from src.autograd.tensor import Tensor
from src.numpy_gpt.generate import distribution


def test_layout_and_geometry():
    assert EXPERIMENTS[2]() == ((128, 32, 8), (128, 16, 32, 8), [34, 45, 56, 67], [24, 25])
    vector, rank, norm = EXPERIMENTS[3]()
    np.testing.assert_array_equal(vector, [2, -1, 2, -1])
    assert rank == 2
    assert norm == pytest.approx(np.sqrt(10))
    score, cosine, projection, distance = EXPERIMENTS[4]()
    assert (score, cosine, distance) == pytest.approx((4, 0.8, np.sqrt(2)))
    np.testing.assert_allclose(projection, [0.8, 1.6])


def test_matrices_and_activations():
    y, rank = EXPERIMENTS[5]()
    np.testing.assert_array_equal(y, [[4, 6, 3], [10, 12, 5]])
    assert rank == 2
    relu, sigmoid, gelu, silu = EXPERIMENTS[6]()
    np.testing.assert_array_equal(relu, [0, 0, 1])
    np.testing.assert_allclose(sigmoid, [0.26894142, 0.5, 0.73105858])
    np.testing.assert_allclose(gelu, [-0.158808, 0, 0.841192], atol=1e-6)
    np.testing.assert_allclose(silu, [-0.26894142, 0, 0.73105858])


def test_scalar_gradient_and_matrix_vjp():
    loss, estimated, analytical = EXPERIMENTS[7]()
    assert loss == pytest.approx(np.log(2))
    assert estimated == pytest.approx(analytical, abs=1e-9)
    y, dx, dw, db = EXPERIMENTS[8]()
    np.testing.assert_array_equal(y, [[2, 5], [6, 9]])
    np.testing.assert_array_equal(dx, [[2, 0], [-4, 6]])
    np.testing.assert_array_equal(dw, [[-2, 6], [-2, 8]])
    np.testing.assert_array_equal(db, [0, 2])
    x = Tensor([[1., 2.], [3., 4.]], True)
    w = Tensor([[2., -1.], [0., 3.]], True)
    g = [[1., 0.], [-1., 2.]]
    ((x @ w) * g).sum().backward()
    np.testing.assert_array_equal(x.grad, dx)
    np.testing.assert_array_equal(w.grad, dw)
    def objective(z):
        perturbed = np.array([[z, -1.], [0., 3.]])
        return np.sum((x.data @ perturbed) * g)
    assert central_difference(objective, 2) == pytest.approx(dw[0, 0], abs=1e-9)


def test_probability_cost_and_update():
    joint, p, var = EXPERIMENTS[9]()
    np.testing.assert_allclose(joint, [[0.125, 0.375], [0.45, 0.05]])
    assert p == pytest.approx(0.425)
    assert var == pytest.approx(0.244375)
    probs, h, loss, pp, grad = EXPERIMENTS[10]()
    np.testing.assert_allclose(probs, [1/6, 1/3, 1/2])
    np.testing.assert_allclose([h, loss, pp], [1.0114042647, np.log(3), 3])
    np.testing.assert_allclose(grad, [1/6, -2/3, 1/2])
    weight, traces = EXPERIMENTS[11]()
    assert weight == pytest.approx(1.86)
    assert traces[0.1][-1] == pytest.approx(0.8192)
    assert traces[1.0][-1] == 2
    assert traces[1.1][-1] == pytest.approx(4.1472)


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js required for demo checks")
def test_browser_demo_temperature_matches_numpy():
    script = """
    const {probabilities} = require('./assets/demos.js');
    console.log(JSON.stringify([0.1, 1, 3].map(t => probabilities([0,2,1,-1], t))));
    """
    rows = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    expected = [distribution([0, 2, 1, -1], t) for t in (0.1, 1, 3)]
    np.testing.assert_allclose(rows, expected, atol=1e-15)


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js required for demo checks")
def test_bpe_embedding_attention_and_descent_demonstrations():
    from examples.language import embedding
    from examples.transformer import scaled
    from src.tokenizer.bpe import ByteBPE

    script = """
    const demos = require('./assets/demos.js');
    console.log(JSON.stringify({
        merged: demos.mergePair(demos.mergePair([97,98,97,98], [97,98], 258), [258,258], 259),
        gradient: demos.embeddingData(1).gradient,
        attention: demos.attentionData(),
        trajectory: demos.trajectory(0.1, 5),
        angle: demos.angleValues(60),
    }));
    """
    actual = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert actual["merged"] == ByteBPE.train(["abab"], 260).encode("abab")
    np.testing.assert_array_equal(actual["gradient"], embedding()[1])
    np.testing.assert_allclose(actual["attention"]["weights"], scaled()[0], atol=1e-15)
    np.testing.assert_allclose(actual["attention"]["output"], scaled()[1], atol=1e-15)
    np.testing.assert_allclose(actual["trajectory"], EXPERIMENTS[11]()[1][0.1], atol=1e-14)
    np.testing.assert_allclose(actual["angle"], [0.5, np.sqrt(3)/2], atol=1e-14)

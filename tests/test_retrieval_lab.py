import numpy as np
import pytest

from examples.retrieval_lab import project_and_read


def test_project_and_read_matches_hand_controls():
    run = project_and_read()
    np.testing.assert_array_equal(run["k"], np.eye(2))
    np.testing.assert_array_equal(run["v"], [[2, 0], [0, 4]])
    np.testing.assert_array_equal(run["fixed"], [[0.5, 3]])
    for name, weight, output in [
        ("base", [.75, .25], [1.5, 1]),
        ("query", [.25, .75], [.5, 3]),
        ("keys", [.25, .75], [.5, 3]),
        ("values", [.75, .25], [.5, 3]),
        ("paired", [.25, .75], [1.5, 1]),
    ]:
        np.testing.assert_allclose(run[name]["weights"], [weight], atol=1e-15, rtol=0)
        np.testing.assert_allclose(run[name]["output"], [output], atol=1e-15, rtol=0)
    np.testing.assert_allclose(run["two_queries"]["output"], [[1.5, 1], [.5, 3]], atol=1e-15)


@pytest.mark.parametrize("gap", [-2.0, 0.0, 2.0, 100.0])
def test_gap_changes_weights_without_changing_values(gap):
    run = project_and_read(gap)
    p = 1 / (1 + np.exp(-gap))
    assert run["base"]["weights"][0][0] == pytest.approx(p, abs=1e-15, rel=0)
    np.testing.assert_allclose(run["base"]["output"], run["paired"]["output"], atol=1e-15, rtol=0)
    if gap == 0:
        np.testing.assert_array_equal(run["base"]["output"], [[1, 2]])

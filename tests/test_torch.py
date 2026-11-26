from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from examples.torch_alignment import align
from examples.torch_runtime import amp_report
from src.numpy_gpt.model import Config
from src.numpy_gpt.train import TrainingConfig
from src.torch_gpt.model import TorchGPT
from src.torch_gpt.train import fit


def test_reference_alignment():
    errors = align()
    for key in ("logits_max_abs", "loss_abs", "gradient_max_abs", "update_max_abs"):
        assert errors[key] < errors["gradient_floor"]


def test_sdpa_optimization_matches_reference():
    config = Config(vocab=8)
    explicit, optimized = TorchGPT(config), TorchGPT(config, sdpa=True)
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    torch.testing.assert_close(explicit(ids), optimized(ids), atol=1e-12, rtol=1e-10)


def test_cpu_autocast_keeps_loss_gradients_and_optimizer_in_fp32():
    report = amp_report()
    assert report["logits_dtype"] == "torch.bfloat16"
    for key in ("loss_dtype", "gradient_dtype", "weights_dtype"):
        assert report[key] == "torch.float32"
    assert abs(report["nll_fp32"] - report["nll_autocast"]) < 0.002
    assert max(report["logits_max_abs"], report["gradient_max_abs"],
               report["weights_max_abs_after_step"]) < 0.01


def test_torch_resume(tmp_path):
    config = TrainingConfig(steps=3, log_every=3)
    checkpoint = tmp_path / "partial.pt"
    silent = lambda _: None
    fit(replace(config, steps=1), checkpoint, report=silent)
    actual = fit(config, checkpoint, resume=True, report=silent)
    expected = fit(config, tmp_path / "full.pt", report=silent)
    for (_, a), (_, b) in zip(actual.named_weights(), expected.named_weights(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

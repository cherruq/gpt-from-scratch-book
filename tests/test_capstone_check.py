from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from examples import capstone_check as project


@pytest.mark.parametrize("seed,token_seed,losses,generation", [
    (42, None, (2.139205, 0.232363, 0.001164), [0, 2, 6, 4, 4, 5, 5]),
    (17, 21, (2.117518, 0.232530, 0.001627), [5, 4, 7, 6, 1, 7, 3]),
])
def test_reference_passes_context_and_generation(seed, token_seed, losses, generation):
    result = project.verify(project.reference_model, seed=seed, token_seed=token_seed)
    assert result["predictions"] == result["targets"]
    measured = [result[key] for key in ("initial_nll", "final_nll", "last_position_nll")]
    assert measured == pytest.approx(losses, abs=5e-7, rel=0)
    assert result["last_token_only_nll_lower_bound"] == pytest.approx(np.log(2))
    assert result["final_nll"] > np.log(2) / 3
    assert result["max_prefix_error"] < 1e-10
    assert result["generation"] == generation


def test_prefix_invariance_detects_future_information():
    class ReadsFuture:
        def forward(self, ids):
            z = np.zeros((*ids.shape, project.CONFIG["vocab"]))
            z[..., 0] = ids.sum(axis=1, keepdims=True)
            return z

    ids, _ = project.context_batch()
    with pytest.raises(project.CheckFailed, match="causality"):
        project.check_prefix(ReadsFuture(), ids)


def test_context_task_rejects_a_trainable_bigram():
    class Bigram:
        def __init__(self, *, seed, vocab, **config):
            self.weight = np.random.default_rng(seed).normal(scale=0.01, size=(vocab, vocab))

        def forward(self, ids):
            return self.weight[ids]

        def train_step(self, ids, targets):
            z = self.forward(ids)
            loss = project.mean_nll(z, targets)
            p = np.exp(z - z.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            grad = p - np.eye(self.weight.shape[0])[targets]
            np.add.at(self.weight, ids, -0.5 * grad / targets.size)
            return loss

    with pytest.raises(project.CheckFailed, match="context task"):
        project.verify(Bigram, steps=80)


def test_generation_validator_catches_ignored_eos():
    class NoStop:
        def forward(self, ids):
            return np.zeros((*ids.shape, project.CONFIG["vocab"]))

        def generate(self, prompt, max_new_tokens, *, eos):
            return prompt + [0] * max_new_tokens

    with pytest.raises(project.CheckFailed, match="EOS"):
        project.check_generation(NoStop(), [5, 2, 6])


def test_candidate_import_works_in_a_standalone_directory(tmp_path):
    script = tmp_path / "capstone_check.py"
    shutil.copyfile(project.__file__, script)
    (tmp_path / "exercise.py").write_text(
        "import numpy as np\n"
        "class Decoder:\n"
        "    def forward(self, ids): return np.zeros((*ids.shape, 1))\n"
        "def build(**settings): return Decoder()\n"
    )
    run = subprocess.run([sys.executable, str(script), "--candidate", "exercise:build"],
                         cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert run.returncode == 1
    assert run.stderr.startswith("FAIL forward: expected finite float64 logits")

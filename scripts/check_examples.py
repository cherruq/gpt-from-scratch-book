"""Run every chapter's command and every shipped source snapshot in isolated directories."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

from build import ROOT
from milestones import snapshot_releases


def validation_rows(output):
    rows = [json.loads(line) for line in output.splitlines()]
    return {row["step"]: row for row in rows if "validation_nll" in row}


def check_validation(output, report, atol):
    observed = validation_rows(output)
    reference = validation_rows((ROOT / "reports" / report).read_text())
    if observed.keys() != reference.keys():
        raise ValueError(f"{report}: validation steps differ")
    for step, row in observed.items():
        want = reference[step]
        for key in ("parameters", "validation_tokens"):
            if row[key] != want[key]:
                raise ValueError(f"{report} step {step}: {key} differs")
        if not math.isclose(row["validation_nll"], want["validation_nll"], rel_tol=0, abs_tol=atol):
            raise ValueError(f"{report} step {step}: validation NLL differs")
        if not math.isclose(row["validation_ppl"], math.exp(row["validation_nll"]), rel_tol=1e-12):
            raise ValueError(f"{report} step {step}: perplexity differs")


def check_probe(output):
    observed = json.loads(output)
    reference = json.loads((ROOT / "reports" / "interpretability.json").read_text())
    for key in ("candidate", "probe_accuracy", "probe_validation_count", "attention_tokens"):
        if observed[key] != reference[key]:
            raise ValueError(f"Chapter 39 replay differs for {key}")
    for key in ("original", "other_context", "ablation"):
        if not math.isclose(observed[key], reference[key], rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Chapter 39 replay differs for {key}")
    for actual, expected in zip(observed["attention_row"], reference["attention_row"], strict=True):
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Chapter 39 replay differs for the attention row")


def check_language_value(actual, expected, path):
    if type(actual) is not type(expected):
        raise ValueError(f"{path}: value type differs")
    if isinstance(expected, dict) and actual.keys() == expected.keys():
        for key, value in expected.items():
            check_language_value(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list) and len(actual) == len(expected):
        for index, value in enumerate(expected):
            check_language_value(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        return
    elif actual != expected:
        raise ValueError(f"{path}: replay differs from report")


def check(report_path, include_torch):
    records = []
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

    def run(label, args, workdir=ROOT):
        start = perf_counter()
        try:
            result = subprocess.run([sys.executable, *args], cwd=workdir, env=env,
                                    text=True, encoding="utf-8", capture_output=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{label} timed out: {args}") from error
        if result.returncode:
            raise RuntimeError(f"{label} failed: {args}\n{result.stderr}")
        records.append({"case": label, "seconds": round(perf_counter() - start, 3),
                        "command": ["python", *args], "stdout": result.stdout.strip()})
        return result.stdout

    with TemporaryDirectory(prefix="gpt-book-replay-") as workspace:
        root = Path(workspace)
        run("01", ["-m", "examples.preview"])
        groups = [("math_basics", 2, 11), ("networks", 12, 16), ("language", 17, 21),
                  ("transformer", 22, 28)]
        for module, first, last in groups:
            for chapter in range(first, last + 1):
                run(f"{chapter:02d}", ["-m", f"examples.{module}", str(chapter)])
        for chapter in (7, 8, 15):
            run(f"{chapter:02d}.worked", ["-m", "examples.gradient_labs", str(chapter)])
        run("10.masked_loss", ["-m", "examples.loss_lab"])
        run("11.learning_step", ["-m", "examples.training_bridge"])
        run("14.step_size", ["-m", "examples.training_bridge", "--rate", "0.3"])
        run("16.normalization", ["-m", "examples.norm_lab"])
        tokenizer_report = json.loads((ROOT / "reports" / "tokenizer-embedding.json").read_text())
        for label, args in (
            ("17.overlap", ["--train", "aaaa", "--text", "aaaa", "--merges", "1"]),
            ("17.merge_rank", ["--train", "ab", "--train", "ab", "--train", "ba",
                               "--text", "baba春🐍", "--merges", "2"]),
            ("18.split_bytes", ["--merges", "1"]),
            ("18.embedding", []),
        ):
            actual = run(label, ["-m", "examples.tokenizer_embedding", *args])
            if json.loads(actual) != tokenizer_report[label]:
                raise ValueError(f"{label}: tokenizer or embedding replay differs from the report")
        language_report = json.loads((ROOT / "reports" / "language-pipeline.json").read_text())
        for label, args in (
            ("19.context", ["--radius", "2"]),
            ("20.unseen", ["--train", "ab", "--train", "ba", "--eval", "aa", "--eval", "🐍",
                           "--merges", "0", "--radius", "2", "--alpha", "0.1", "--seed", "7"]),
            ("21.positions", []),
        ):
            actual = run(label, ["-m", "examples.language_pipeline", *args])
            check_language_value(json.loads(actual), language_report[label], label)
        run("22.retrieval", ["-m", "examples.retrieval_lab"])
        run("22.uniform", ["-m", "examples.retrieval_lab", "--logit-gap", "0"])
        for chapter in range(23, 29):
            run(f"{chapter}.worked", ["-m", "examples.attention_labs", str(chapter)])
        for chapter in (29, 31, 32):
            run(str(chapter), ["-m", "examples.training", str(chapter)])
        for chapter in (29, 30):
            run(f"{chapter}.worked", ["-m", "examples.training_labs", str(chapter)])
        numpy = str(root / "cpu.npz")
        fit = ["-m", "src.numpy_gpt.train", "fit", "--config", "configs/quick.json", "--checkpoint", numpy]
        numpy_fit = run("30.fit", [*fit, "--steps", "60"])
        numpy_resume = run("30.resume", [*fit, "--resume"])
        check_validation(numpy_fit + numpy_resume, "numpy-quick.jsonl", atol=1e-8)
        run("30.generate", ["-m", "src.numpy_gpt.train", "generate", "--checkpoint", numpy,
                            "--prompt", "今早我", "--top-k", "8", "--cache"])
        if include_torch:
            run("33", ["-m", "examples.torch_alignment"])
            runtime_args = ["--gloo"] if sys.platform == "linux" else []
            runtime = json.loads(run("33.cpu_runtime", ["-m", "examples.torch_runtime", *runtime_args]))
            if runtime_args:
                ddp = runtime["ddp"]
                if not (ddp["global_mean_max_abs"] < 1e-12 < ddp["local_means_max_abs"]):
                    raise ValueError("DDP weighting did not reproduce the global mean gradient")
            torch = str(root / "cpu.pt")
            torch_fit = run("34.fit", ["-m", "src.torch_gpt.train", "fit", "--config", "configs/larger.json",
                                       "--checkpoint", torch, "--sdpa"])
            check_validation(torch_fit, "torch-larger.jsonl", atol=1e-5)
            run("34.generate", ["-m", "src.torch_gpt.train", "generate", "--checkpoint", torch,
                                "--prompt", "今早我", "--top-k", "8"])
        for chapter in range(35, 41):
            output = run(str(chapter), ["-m", "examples.modern", str(chapter), "--checkpoint", numpy])
            if chapter == 39:
                check_probe(output)
        controls = run("39.controls", ["-m", "examples.probe_controls"])
        run("38.objectives", ["-m", "examples.modern_objectives"])
        expected_controls = json.loads((ROOT / "reports" / "probe-controls.json").read_text())
        if json.loads(controls) != expected_controls:
            raise ValueError("Chapter 39 control-task replay differs from the report")
        run("40.context", ["-m", "examples.capstone_check"])
        run("40.renamed", ["-m", "examples.capstone_check", "--seed", "17", "--token-seed", "21"])
        run("35.latency", ["-m", "examples.benchmark"])
        run("35.long_context_smoke", ["-m", "examples.kv_benchmark"])
        for stage, version, _, command in snapshot_releases():
            if stage == "M7" and not include_torch:
                continue
            target = root / version
            with ZipFile(ROOT / "dist" / "downloads" / f"{version}.zip") as archive:
                metadata = json.loads(archive.read("snapshot.json"))
                if metadata["version"] != version or metadata["command"] != ["python", *command]:
                    raise ValueError(f"Unexpected {version} manifest")
                for name, checksum in metadata["sha256"].items():
                    if sha256(archive.read(name)).hexdigest() != checksum:
                        raise ValueError(f"{name} failed its checksum in {version}")
                archive.extractall(target)
            run(version, command, target)

    result = {"python": sys.version.split()[0], "torch_cases": include_torch, "passed": len(records),
              "cases": records}
    if report_path:
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"通过 {len(records)} 个章节命令和快照入口" + ("" if include_torch else "（省略可选 PyTorch）"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    check(args.report, args.torch)

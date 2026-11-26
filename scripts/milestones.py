"""Freeze deterministic, runnable code distributions for M0 through M8."""

from hashlib import sha256
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


MILESTONES = (
    ("M0", ("src/autograd/tensor.py", "src/numpy_gpt/model.py", "src/numpy_gpt/optim.py",
            "src/numpy_gpt/generate.py", "examples/preview.py", "examples/math_basics.py"),
     ("-m", "examples.preview")),
    ("M1", ("src/math_basics/mlp.py", "src/math_basics/normalization.py",
            "src/autograd/scalar.py", "examples/networks.py"), ("-m", "examples.networks", "13")),
    ("M2", (), ("-m", "examples.networks", "15")),
    ("M3", ("src/tokenizer/bpe.py", "src/tokenizer/char.py",
            "src/math_basics/language.py", "examples/language.py"), ("-m", "examples.language", "17")),
    ("M4", ("examples/transformer.py",), ("-m", "examples.transformer", "23")),
    ("M5", (), ("-m", "examples.transformer", "28")),
    ("M6", ("src/numpy_gpt/data.py", "src/numpy_gpt/train.py", "examples/training.py",
            "configs/quick.json", "scripts/make_corpus.py", "data/README.md",
            "data/train.txt", "data/validation.txt"),
     ("-m", "src.numpy_gpt.train", "fit", "--steps", "1")),
    ("M7", ("src/torch_gpt/model.py", "src/torch_gpt/train.py", "examples/torch_alignment.py",
            "configs/larger.json"), ("-m", "examples.torch_alignment")),
    ("M8", ("src/modern/architecture.py", "src/modern/efficiency.py", "src/modern/post_training.py",
            "examples/modern.py", "examples/benchmark.py"), ("-m", "examples.modern", "40")),
)


REVISIONS = {
    "M0": (("examples/gradient_labs.py",), ("-m", "examples.gradient_labs", "7")),
    "M2": ((), ("-m", "examples.gradient_labs", "15")),
    "M4": (("examples/attention_labs.py",), ("-m", "examples.attention_labs", "25")),
    "M5": ((), ("-m", "examples.attention_labs", "28")),
    "M6": (("examples/training_labs.py",), ("-m", "examples.training_labs", "30")),
    "M8": ((), ("-m", "examples.training_labs", "30")),
}


FOLLOWUPS = {
    "M0": ((("examples/loss_lab.py",), ("-m", "examples.loss_lab")),),
    "M1": ((("examples/training_bridge.py",), ("-m", "examples.training_bridge")),),
    "M2": ((("examples/norm_lab.py",), ("-m", "examples.norm_lab")),),
    "M3": (
        (("examples/tokenizer_embedding.py",), ("-m", "examples.tokenizer_embedding")),
        (("examples/language_pipeline.py",), ("-m", "examples.language_pipeline")),
    ),
    "M4": ((("examples/retrieval_lab.py",), ("-m", "examples.retrieval_lab")),),
    "M7": ((("examples/torch_runtime.py",), ("-m", "examples.torch_runtime")),),
    "M8": (
        (("examples/probe_controls.py",), ("-m", "examples.probe_controls")),
        (("examples/kv_benchmark.py",), ("-m", "examples.kv_benchmark")),
        (("examples/loss_lab.py", "examples/norm_lab.py", "examples/capstone_check.py"),
         ("-m", "examples.capstone_check")),
        (("examples/modern_objectives.py",), ("-m", "examples.modern_objectives")),
    ),
}


def snapshot_releases():
    names = {"requirements.txt"}
    revised = set()
    for stage, new_sources, command in MILESTONES:
        names.update(new_sources)
        yield stage, f"{stage}-v1", sorted(names), command
        if stage in REVISIONS:
            files, replay = REVISIONS[stage]
            revised.update(files)
            yield stage, f"{stage}-v2", sorted(names | revised), replay
        added = set()
        next_version = 3 if stage in REVISIONS else 2
        for revision, (files, replay) in enumerate(FOLLOWUPS.get(stage, ()), start=next_version):
            added.update(files)
            yield stage, f"{stage}-v{revision}", sorted(names | revised | added), replay


def package_milestones(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for stage, version, names, command in snapshot_releases():
        description = f"# {version}\n\npython -m pip install -r requirements.txt\n"
        if stage in {"M7", "M8"}:
            description += "PyTorch 查验需要另装 CPU 版 torch==2.10.0+cpu。\n"
        description += f"\n复现命令：python {' '.join(command)}\n"
        members = {name: (root / name).read_bytes() for name in names}
        manifest = {"version": version, "command": ["python", *command],
                    "sha256": {name: sha256(data).hexdigest() for name, data in members.items()}}
        members["README.md"] = description.encode()
        members["snapshot.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()

        with ZipFile(destination / f"{version}.zip", "w") as output:
            for name, data in members.items():
                info = ZipInfo(name, (2026, 11, 26, 0, 0, 0))
                output.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)

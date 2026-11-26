"""Bundle the offline edition, source, experiments and evidence for delivery."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from build import ROOT
from check_book import check


FILES = (
    ".gitignore", "AGENTS.md", "DECISIONS.md", "PLAN.md", "PROGRESS.md", "README.md",
    "TEXTBOOK_SPEC.md", "package.json", "package-lock.json", "requirements.txt",
)
DIRECTORIES = ("assets", "book", "configs", "data", "dist", "examples", "reports", "scripts", "src", "tests")


def bundle(root=ROOT):
    check(root / "dist", full=True)
    members = [root / name for name in FILES]
    for directory in DIRECTORIES:
        members.extend(path for path in (root / directory).rglob("*")
                       if path.is_file() and not path.is_symlink()
                       and "__pycache__" not in path.relative_to(root).parts
                       and path.suffix != ".pyc")
    release = root / "release"
    release.mkdir(exist_ok=True)
    archive = release / "gpt-from-scratch-book.zip"
    with TemporaryDirectory(prefix=".package-", dir=release) as scratch:
        staged = Path(scratch) / archive.name
        manifest = {}
        with ZipFile(staged, "w") as output:
            def add(name, content):
                entry = ZipInfo(name, (2026, 11, 26, 0, 0, 0))
                output.writestr(entry, content, compress_type=ZIP_DEFLATED, compresslevel=9)

            for path in sorted(members):
                name = path.relative_to(root).as_posix()
                content = path.read_bytes()
                add(name, content)
                manifest[name] = sha256(content).hexdigest()
            add("release-manifest.json", (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        digest = sha256(staged.read_bytes()).hexdigest()
        sums = staged.with_name("SHA256SUMS")
        sums.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        staged.replace(archive)
        sums.replace(release / "SHA256SUMS")
    print(f"交付包 {archive}，共 {len(manifest)} 个原始文件")


if __name__ == "__main__":
    bundle()

"""Check local links, required chapter sections, math rendering and demo registration."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from build import ROOT, collected_chapters


REQUIRED = ("学习目标", "前置知识", "本章在 GPT 中的位置", "常见错误",
            "练习", "参考答案", "小结与下一章")
PLACEHOLDER = re.compile(r"\b(?:TODO|FIXME|TBD)\b|等待补充|（占位）")


class Page(HTMLParser):
    def __init__(self, content):
        super().__init__(convert_charrefs=True)
        self.ids, self.duplicates, self.references, self.demos = set(), set(), [], set()
        self.feed(content)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            target = attrs["id"]
            if target in self.ids:
                self.duplicates.add(target)
            self.ids.add(target)
        for name in ("href", "src"):
            if attrs.get(name):
                self.references.append((tag, name, attrs[name]))
        if "data-demo" in attrs:
            self.demos.add(attrs["data-demo"])


def check(destination, full):
    chapters = collected_chapters()
    pages = {p.name: p.read_text() for p in destination.glob("*.html")}
    parsed = {name: Page(content) for name, content in pages.items()}
    errors = []
    if full and set(chapters) != set(range(1, 42)):
        errors.append("Full edition requires chapters 01–40 and appendix 41")
    expected = {"index.html", *(chapter.link for chapter in chapters.values())}
    if pages.keys() != expected:
        errors.append(f"Unexpected or absent pages: {sorted(pages.keys() ^ expected)}")
    for chapter in chapters.values():
        name = chapter.link
        for label in REQUIRED if chapter.number <= 40 else ():
            if label not in chapter.raw:
                errors.append(f"{name}: missing {label}")
        if chapter.number <= 40 and "::source" not in chapter.raw:
            errors.append(f"{name}: missing an executable snippet")
        if chapter.number <= 40:
            prerequisite = re.search(r"^## 前置知识\n(.*?)(?=^## |\Z)", chapter.raw, re.M | re.S)
            if prerequisite and any(int(n) >= chapter.number
                                    for n in re.findall(r"\[\[(\d{2})", prerequisite[1])):
                errors.append(f"{name}: prerequisite must precede this chapter")
            for category in ("基础", "推导", "工程", "开放探索"):
                if f"**{category}。" not in chapter.raw:
                    errors.append(f"{name}: missing exercise category {category}")
        if PLACEHOLDER.search(chapter.raw):
            errors.append(f"{name}: unfinished placeholder")
    demo_file = destination / "assets" / "demos.js"
    registered = set(re.findall(r"^        ['\"]?([a-z0-9-]+)['\"]?\(host", demo_file.read_text(), re.M))
    formulas = 0
    for name, page in parsed.items():
        html = pages[name]
        formulas += html.count('class="katex"')
        if re.search(r'data-book-equation|class="katex-error"|::(?:source|demo)|\[\[\d{2}', html):
            errors.append(f"{name}: unresolved directive or math rendering error")
        if page.duplicates:
            errors.append(f"{name}: duplicate IDs {sorted(page.duplicates)}")
        if page.demos - registered:
            errors.append(f"{name}: unknown demos {sorted(page.demos - registered)}")
        for tag, attr, target in page.references:
            url = urlsplit(target)
            if url.scheme or url.netloc:
                if tag != "a" or attr != "href" or url.scheme not in ("http", "https"):
                    errors.append(f"{name}: remote resource {target}")
                continue
            relative = unquote(url.path) or name
            path = (destination / relative).resolve()
            if not path.is_relative_to(destination.resolve()) or not path.is_file():
                errors.append(f"{name}: missing or nonlocal target {target}")
            if relative in parsed and url.fragment and unquote(url.fragment) not in parsed[relative].ids:
                errors.append(f"{name}: missing anchor {target}")
    katex = destination / "assets" / "katex"
    for url in re.findall(r"url\(([^)]+)\)", (katex / "katex.min.css").read_text()):
        if not (katex / url.strip("'\"")).is_file():
            errors.append(f"Missing bundled KaTeX asset: {url}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"校验通过：{len(pages)} 个 HTML 文件、{formulas} 次公式渲染，本地链接及交互挂载有效")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    check(args.out, args.full)

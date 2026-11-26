"""Build the offline edition from chapter sources and executable examples."""

import argparse
from dataclasses import dataclass, field
from html import escape, unescape
import json
from pathlib import Path
import re
import shutil
import subprocess

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from milestones import package_milestones


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "book" / "chapters"
PARTS = [
    (1, "先看到完整目标"),
    (3, "GPT 所需数学"),
    (12, "神经网络如何学习"),
    (17, "语言的数值表示"),
    (22, "Attention 与 Transformer"),
    (29, "从零训练并运行 GPT"),
    (35, "从教学 GPT 到现代 LLM"),
    (41, "附录"),
]
MATH = re.compile(r"^\$\$ ([a-z][a-z0-9-]*)\n(.*?)\n\$\$ *$|\\\((.*?)\\\)",
                  re.MULTILINE | re.DOTALL)
REFERENCE = re.compile(r"\[\[(\d{2})(?:#([a-z][a-z0-9-]*))?(?:\|([^\]]+))?\]\]")
SOURCE = re.compile(r"^::source ([a-zA-Z0-9_/]+\.py)(?: ([a-zA-Z0-9_]+))?$", re.M)
DEMO = re.compile(r"^::demo ([a-z][a-z0-9-]*) (.+)$", re.M)
FENCE = re.compile(r"^ *(\x60{3,}|~{3,})[^\n]*\n.*?^ *\1 *$", re.M | re.S)
FORMATTER = HtmlFormatter(cssclass="highlight")


@dataclass
class Chapter:
    number: int
    title: str
    raw: str
    equations: dict = field(default_factory=dict)
    html: str = ""
    toc: str = ""

    @property
    def link(self):
        return f"{self.number:02d}.html"


def collected_chapters():
    chapters = {}
    for path in sorted(CHAPTERS.glob("*.md")):
        number = int(path.stem.split("-", 1)[0])
        if number in chapters:
            raise ValueError(f"Duplicate chapter number in {path.name}")
        heading, raw = path.read_text().split("\n", 1)
        if not heading.startswith("# "):
            raise ValueError(f"Missing title in {path.name}")
        chapters[number] = Chapter(number, heading[2:], raw)
    return chapters


def outside_fences(raw, transform):
    spans, start = [], 0
    for match in FENCE.finditer(raw):
        spans.extend((transform(raw[start:match.start()]), match[0]))
        start = match.end()
    spans.append(transform(raw[start:]))
    return "".join(spans)


def render_equations(chapters):
    equations = []
    for chapter in chapters.values():
        def replace(match):
            label, display, inline = match.groups()
            marker = f'<span data-book-equation="{len(equations)}"></span>'
            equations.append(dict(tex=display if label else inline,
                                  display=bool(label), source=chapter.link))
            if label:
                if label in chapter.equations:
                    raise ValueError(f"{chapter.link}: repeated equation {label}")
                number = f"{chapter.number}.{len(chapter.equations) + 1}"
                chapter.equations[label] = number
                return (f'<div class="equation" id="eq-{label}">{marker}'
                        f'<a href="#eq-{label}" aria-label="公式 {number}">({number})</a></div>')
            return marker

        chapter.raw = outside_fences(chapter.raw, lambda raw: MATH.sub(replace, raw))
    result = subprocess.run(["node", str(ROOT / "scripts/render_math.cjs")],
                            input=json.dumps(equations), encoding="utf-8",
                            capture_output=True, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def code_embed(match):
    path, region = match.groups()
    absolute = ROOT / path
    code = absolute.read_text()
    if region:
        limits = re.search(rf"^# region {region}\n(.*?)^# endregion {region}$", code, re.M | re.S)
        if not limits:
            raise ValueError(f"Missing region {region} in {path}")
        code = limits[1]
    return (f'<figure class="code-example"><figcaption>{escape(path)}'
            f'{f" · {escape(region)}" if region else ""}</figcaption>'
            f'{highlight(code, PythonLexer(), FORMATTER)}</figure>')


def render_chapter(chapter, chapters, equations):
    def reference(match):
        target, label, caption = match.groups()
        number = int(target)
        if number not in chapters:
            if label:
                raise ValueError(f"{chapter.link}: invalid equation target {target}#{label}")
            return f"第 {number} 章（{caption}）" if caption else f"第 {number} 章"
        page = chapters[number]
        if label and label not in page.equations:
            raise ValueError(f"{chapter.link}: missing equation {target}#{label}")
        suffix = f"#eq-{label}" if label else ""
        default = f"式 ({page.equations[label]})" if label else f"第 {number} 章"
        return f"[{caption or default}]({page.link}{suffix})"

    def expand(raw):
        raw = REFERENCE.sub(reference, raw)
        raw = SOURCE.sub(code_embed, raw)
        return DEMO.sub(lambda m: (f'<figure class="experiment" data-demo="{m[1]}">'
                                   f'<figcaption>{escape(m[2])}</figcaption><div class="demo"></div>'
                                   f'</figure>'), raw)

    engine = markdown.Markdown(extensions=["extra", "toc", "codehilite"],
                               extension_configs={"toc": {"toc_depth": "2-3"}})
    chapter.html = engine.convert(outside_fences(chapter.raw, expand))
    chapter.html = re.sub(r'<span data-book-equation="(\d+)"></span>',
                          lambda m: equations[int(m[1])], chapter.html)
    chapter.toc = engine.toc


def part_name(number):
    return next(title for first, title in reversed(PARTS) if first <= number)


def navigation(chapters, current=None, *, landing=False):
    items = []
    for first, title in PARTS:
        next_first = next((start for start, _ in PARTS if start > first), 42)
        group = [chapter for n, chapter in chapters.items() if first <= n < next_first]
        if not group:
            continue
        links = []
        for chapter in group:
            active = ' aria-current="page"' if chapter.number == current else ''
            links.append(f'<li><a href="{chapter.link}"{active}>'
                         f'<span>{chapter.number:02d}</span>{escape(chapter.title)}</a></li>')
        heading = "h2" if landing else "p"
        items.append(f'<section><{heading}>{title}</{heading}><ol>{"".join(links)}</ol></section>')
    return "".join(items)


def layout(title, main, chapters, current=None, toc=""):
    number = current or ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · 从零实现 GPT</title>
<link rel="stylesheet" href="assets/katex/katex.min.css">
<link rel="stylesheet" href="assets/highlight.css">
<link rel="stylesheet" href="assets/book.css">
<script src="assets/search-index.js" defer></script>
<script src="assets/experiment-data.js" defer></script>
<script src="assets/demos.js" defer></script>
<script src="assets/book.js" defer></script>
</head>
<body data-chapter="{number}">
<a class="skip" href="#content">跳到正文</a>
<header class="top"><div class="top-inner">
<a class="brand" href="index.html"><span aria-hidden="true">∑</span> 从零实现 GPT</a>
<div class="toolbar" hidden><button type="button" id="open-search">搜索</button>
<button type="button" id="theme" aria-label="切换明暗主题">主题</button></div>
</div></header>
<dialog id="search"><form method="dialog"><label for="query">搜索全书</label>
<button aria-label="关闭搜索">关闭</button></form>
<input id="query" type="search" placeholder="输入章节标题、术语或公式名称">
<div id="search-results" aria-live="polite"></div></dialog>
<div class="layout">
<aside class="library"><details open><summary>全书目录</summary>
<nav aria-label="全书目录">{navigation(chapters, current)}</nav></details></aside>
<main id="content">{main}</main>
<aside class="outline">{('<nav aria-label="本章目录"><p>本章目录</p>' + toc + '</nav>') if toc else ''}</aside>
</div>
</body></html>
"""


def chapter_main(chapter, previous, following):
    footer = []
    for label, neighbor in (("上一篇", previous), ("下一篇", following)):
        if neighbor is not None:
            footer.append(f'<a href="{neighbor.link}"><small>{label}</small>{escape(neighbor.title)}</a>')
    chapters = "".join(footer)
    return f"""<article>
<header class="chapter-header"><p class="eyebrow">{part_name(chapter.number)} / 第 {chapter.number:02d} 章</p>
<h1>{escape(chapter.title)}</h1></header>
<details class="mobile-outline"><summary>本章目录</summary>{chapter.toc.replace(' id="', ' data-unused-id="')}</details>
<div class="prose">{chapter.html}</div>
<label class="progress" hidden><input type="checkbox" id="completed"> 标记为已读</label>
<nav class="neighbors" aria-label="相邻章节">{chapters}</nav>
</article>"""


def search_text(html):
    html = re.sub(r'<(?:figure|pre|div class="equation")[^>]*>.*?</(?:figure|pre|div)>',
                  " ", html, flags=re.S)
    html = re.sub(r"<[^>]*>", " ", html)
    return " ".join(unescape(html).split())


def build(destination):
    chapters = collected_chapters()
    if not chapters:
        raise ValueError("No chapters to publish")
    equations = render_equations(chapters)
    for chapter in chapters.values():
        render_chapter(chapter, chapters, equations)

    destination.mkdir(parents=True, exist_ok=True)
    package_milestones(ROOT, destination / "downloads")
    static = destination / "assets"
    shutil.copytree(ROOT / "assets", static, dirs_exist_ok=True)
    katex = ROOT / "node_modules" / "katex"
    local_math = static / "katex"
    shutil.copytree(katex / "dist" / "fonts", local_math / "fonts", dirs_exist_ok=True)
    shutil.copyfile(katex / "dist" / "katex.min.css", local_math / "katex.min.css")
    shutil.copyfile(katex / "LICENSE", local_math / "LICENSE")
    (static / "highlight.css").write_text(FORMATTER.get_style_defs(".highlight"), encoding="utf-8")
    index = []
    ordered = list(chapters.values())
    for i, chapter in enumerate(ordered):
        previous = ordered[i - 1] if i else None
        following = ordered[i + 1] if i + 1 < len(ordered) else None
        main = chapter_main(chapter, previous, following)
        page = layout(chapter.title, main, chapters, chapter.number, chapter.toc)
        (destination / chapter.link).write_text(page, encoding="utf-8")
        index.append(dict(id=chapter.number, title=chapter.title, href=chapter.link,
                          text=search_text(chapter.html)))

    start = ordered[0].link
    hero = f"""<article>
<header class="hero"><p class="eyebrow">数学 / 神经网络 / 大语言模型</p>
<h1>从零实现<br><em>GPT</em></h1>
<p>面向有编程经验的读者。沿着下一 Token 预测的计算路径，从数组、点积和梯度出发，亲手训练一个解码器语言模型。</p>
<a class="cta" href="{start}">开始阅读 <span aria-hidden="true">→</span></a></header>
<div class="index-groups" id="catalog">{navigation(chapters, landing=True)}</div>
</article>"""
    (destination / "index.html").write_text(layout("教材目录", hero, chapters), encoding="utf-8")
    (static / "search-index.js").write_text(
        "window.BOOK_INDEX = " + json.dumps(index, ensure_ascii=False) + ";\n", encoding="utf-8")
    metrics = ROOT / "reports" / "numpy-quick.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    interpretation = ROOT / "reports" / "interpretability.json"
    attention = json.loads(interpretation.read_text()) if interpretation.exists() else {}
    (static / "experiment-data.js").write_text(
        "window.BOOK_RUN = " + json.dumps(rows) + ";\n"
        "window.BOOK_INTERPRETATION = " + json.dumps(attention) + ";\n", encoding="utf-8")
    print(f"构建完成：{len(chapters)} 页正文、{len(equations)} 个公式 → {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    build(parser.parse_args().out)

"""Generate the small original corpus under data/ with a held-out pair split."""

from pathlib import Path


SCENES = ("山间", "河边", "村口", "庭院", "街角", "桥头", "海边", "窗前")
CHANGES = ("春雨", "夏风", "秋雾", "冬雪", "白云", "青鸟", "晨光", "晚霞")


def record(scene, change):
    return (
        f"今早我走到{scene}，看见了{change}。我在纸上写下地点和天气，给朋友留作回信。"
        f"午后朋友来到{scene}，沿小路走了一圈，发现昨天的脚印还在。"
        f"我们讨论{change}是否会改变明天的计划，决定先把窗关好，再检查带来的地图。"
        f"傍晚离开{scene}时，我又想起{change}，便在本子的末尾写下两句话。"
        "第一句记录眼前的景象，第二句留给明天来此散步的人。"
    )


def make():
    root = Path(__file__).resolve().parents[1] / "data"
    root.mkdir(exist_ok=True)
    train, validation = [], []
    for i, scene in enumerate(SCENES):
        for j, change in enumerate(CHANGES):
            split = validation if i == j else train
            split.append(record(scene, change))
    for name, records in (("train", train), ("validation", validation)):
        (root / f"{name}.txt").write_text("\n".join(records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    make()

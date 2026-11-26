# 合成中文练习语料

本项目原创模板生成。使用 `python scripts/make_corpus.py` 可精确重建。每行一条记录，8 个场景与 8 个景物组合。对角的 8 种组合留作验证，其余 56 行用于训练。两组逐行互异，但共享模板、字符和大量短片段，只适合验证训练管线与有限的重组合能力。

生成器模板和生成的 train.txt、validation.txt 在适用法律范围内依 CC0 1.0 提供。许可原文见 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode)。无外部文本来源。

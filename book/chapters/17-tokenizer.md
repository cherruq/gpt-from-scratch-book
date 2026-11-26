# Tokenizer：从字节到可学习的 ID

## 学习目标

区分可见字符、Unicode 码点、UTF-8 字节和 Token。逐轮计算字节级 BPE 的相邻对计数与合并结果，用保存的顺序编码新文本，并核对边界标记和解码往返。

## 前置知识

[[02|第 2 章]] 的整数索引、[[09|第 9 章]] 的序列。

## 本章在 GPT 中的位置

分词器（tokenizer）确定模型的类别空间。训练目标中的每个整数都代表某个 Token；推理则把生成的整数解码回文本。

## 码点、字节和 Token 的边界

Unicode 为文本分配码点，UTF-8 将每个 Unicode 标量值编码为 1 至 4 个字节。下表按文本 <code>"A春🐍e\u0301"</code> 的码点顺序列出偏移。区间用左闭右开 <code>[起点,终点)</code>，相邻区间的终点和起点相同。

| 码点 | Unicode 编号 | UTF-8 十六进制 | 字节区间 |
| --- | --- | --- | --- |
| A | U+0041 | 41 | [0,1) |
| 春 | U+6625 | E6 98 A5 | [1,4) |
| 🐍 | U+1F40D | F0 9F 90 8D | [4,8) |
| e | U+0065 | 65 | [8,9) |
| 组合重音 | U+0301 | CC 81 | [9,11) |

Python 的 <code>len</code> 得到 5 个码点，编码后的 <code>len</code> 为 11 字节。最后两个码点通常显示成一个字形。预组合的 <code>é</code> 只有一个码点 U+00E9，字节是 C3 A9。本章的分词器保留输入的字节，不主动做 Unicode 规范化。ASCII 对应一个以 0 起始的字节。多字节序列的首字节包含长度前缀，后续字节以二进制 <code>10</code> 起始。[UTF-8 编码规则，RFC 3629 §3–4](https://www.rfc-editor.org/rfc/rfc3629.html#section-3)

以「春」为例，U+6625 的有效载荷按 4、6、6 位拆成 \(6,24,37\)。三字节格式用前缀 E0、80、80，加上载荷得到 E6、98、A5，十进制是 \([230,152,165]\)。没有合并时，三个字节分别对应三个 Token。合并后某个 Token 可以只包含码点的部分字节，也可以跨几个码点。Token 的切分依赖训练出的规则，词的边界没有参与本章实现的计算。

## 每轮训练先计数，再替换

字节对编码（byte pair encoding，BPE）在每轮训练中读取各条记录当前的 ID 序列。令第 \(r\) 条长度为 \(n_r\)，在每个能向右看一项的位置对当前的相邻对计数。

$$ pair-count
c(a,b)=\sum_r\sum_{j=0}^{n_r-2}\mathbf{1}\bigl[(x_{r,j},x_{r,j+1})=(a,b)\bigr].
$$

指示函数 \(\mathbf{1}[\cdot]\) 在括号内条件成立时为 1，其余为 0。只有零项或一项的记录没有可计数的位置。选择计数最大的对。平局时先比较左 ID，再比较右 ID，取数值较小的一对。新 ID 从 258 顺序分配。随后逐条从左到右扫描，遇到选中对就写一个新 ID 并前进两项，否则抄写当前项并前进一项。用替换后的记录重新计数，直到达到合并上限或没有相邻对。两条独立记录 <code>["a","b"]</code> 都只有一项，因此没有 <code>ab</code> 可以合并，即使预算足够。

| 单条记录 | 匹配的起点（从 0 计） | aa 计数 | 实际替换 | 完成首轮 |
| --- | --- | ---: | ---: | --- |
| aaa | 0、1 | 2 | 1 | [258,97] |
| aaaa | 0、1、2 | 3 | 2 | [258,258] |

例如 <code>aaa</code> 的两个匹配共享中间一项。扫描消耗了位置 0、1 后，位置 2 只剩单个 <code>a</code>。每成功替换一次，序列长度恰好减 1。上表的计数回答有多少个相邻窗口匹配；实际替换数回答序列减少多少项。教学版在每条记录内直接合并。GPT-2 的实现另含正则预切分与字节映射。[子词 BPE 原论文](https://arxiv.org/abs/1508.07909v5) · [GPT-2 encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)

## 编码按训练顺序执行

训练记录为 <code>["ab","ab","ba"]</code> 时，<code>ab</code> 有 2 次，<code>ba</code> 有 1 次。首轮保存 \((97,98)\to258\)，次轮保存 \((98,97)\to259\)。拿新文本 <code>baba</code> 检查顺序，初始 ID 是 \([98,97,98,97]\)。此时 <code>ba</code> 既最先出现，又有 2 次，而 <code>ab</code> 只有 1 次。编码仍先应用保存的 <code>ab</code> 规则，得到 \([98,258,97]\)，这一步消除了两个 <code>ba</code> 窗口。如果打乱应用顺序，在保持替换 ID 不变的情况下先把 <code>ba</code> 换成 259，就会得到 \([259,259]\)。两种切分都能展开成原字节，但会改变模型查表的 ID 和序列长度。存档时要同时保留每条规则的左右 ID 和规则次序。

## 解码保持字节内容

$$ merge-bytes
\operatorname{bytes}(m)=\operatorname{bytes}(a)\,\Vert\,\operatorname{bytes}(b).
$$

符号 \(\Vert\) 表示拼接，\(a,b\) 是已知 ID，\(m\) 是新 ID。把序列中相邻的 \([a,b]\) 换成 \(m\)，展开时该段仍是二者拼接，全序列的字节流保持不变。初始序列是 UTF-8 原字节；每次替换都维持这一性质，因此有限次合并后，按 ID 展开再整体解码，能还原输入文本。生成的 ID 也可以展开成非法 UTF-8。默认 <code>strict</code> 会抛出异常，选择 <code>errors="replace"</code> 会让无法解码的片段显示替换符。

固定保留的 256 是 BOS（beginning of sequence），257 是 EOS（end of sequence）。两项在 <code>encode(..., boundaries=True)</code> 的合并之后插入，都展开为空字节，模型仍各自分配一行嵌入。包含字面文本 <code>EOS</code> 的字符串会像普通文本一样编码。字节词表大小为 256，加上两个边界 ID 和实际学到的合并数，得到模型的词表大小 \(V\)。

## 手算「春春」的切分

训练记录只用「春」，编码输入是「春春」。把单条训练记录 \([230,152,165]\) 划成两个候选对 \((230,152)\) 与 \((152,165)\)，都只出现一次。按字典序选 \((152,165)\)，因为 \(152<230\)。训练记录变为 \([230,258]\)，只剩一个候选对，第二轮把它合并成 259。

| 合并上限 | 「春春」的正文 ID | 插入 BOS/EOS 后的长度 | 实际词表大小 |
| ---: | --- | ---: | ---: |
| 0 | [230,152,165,230,152,165] | 8 | 258 |
| 1 | [230,258,230,258] | 6 | 259 |
| 2 | [259,259] | 4 | 260 |

上限 1 时，第一个 <code>230</code> 占字节 \([0,1)\)，第一个 <code>258</code> 占 \([1,3)\)。二者单独做严格 UTF-8 解码都会失败。拼成 E6 98 A5 后才得到「春」。上限 2 时正文两个 259 的区间分别为 \([0,3)\) 和 \([3,6)\)，加上 BOS 和 EOS 得到 \([256,259,259,257]\)。这四个整数直接送入 [[18|第 18 章]] 的 Embedding 查表。

## 手算 abab

ASCII 的 a 为 97、b 为 98。初始 \([97,98,97,98]\)，ab 有两次，ba 有一次。第一条合并 \((97,98)\to258\)，结果 \([258,258]\)。第二条 \((258,258)\to259\)，结果 \([259]\)。解码时 259 展开为两份 258，最终得到 abab。目标词表含字节 256 项、两个特殊项与两条合并，共 260 项。

::demo bpe 查看训练好的两条合并如何编码 abab

静态读法：第 0 步 [97,98,97,98]；第 1 步 [258,258]；第 2 步 [259]。

## Python 实验

M3，运行 `python -m examples.language 17`，复用 `src/tokenizer/bpe.py::ByteBPE`。验证中文、代码缩进、换行和 emoji 往返。预期两条 toy 合并为 (97,98) 和 (258,258)，训练示例词表 280。

M3-v2 可更换短文本、训练记录和最大合并次数。每个 <code>--train</code> 参数是一条记录。默认训练「春」、编码「春春」并允许两次合并。输出 <code>code_points</code> 和 <code>token_spans</code> 可比对码点与 Token 的偏移。<code>training_trace</code> 与 <code>encoding_trace</code> 分别列出每轮实际规则、匹配数、替换数和结果。复算用整数精确比较。

~~~bash
python -m examples.tokenizer_embedding
python -m examples.tokenizer_embedding --train 春 --text 春春 --merges 1
python -m examples.tokenizer_embedding --train aaaa --text aaaa --merges 1
python -m examples.tokenizer_embedding --train ab --train ab --train ba --text 'baba春🐍' --merges 2
~~~

::source examples/tokenizer_embedding.py utf8_spans

::source examples/tokenizer_embedding.py trace_merges

::source src/tokenizer/bpe.py

## 常见错误

- 在测试集拟合合并规则，造成预处理泄漏。
- 将普通文本中的「EOS」字符串自动认作特殊标记。
- 只保存最后词表并丢失合并顺序，相同文本可能产生不同 ID 序列。

## 练习

1. **基础。** 未训练任何合并时，中文「春」需要多少 Token？
2. **推导。** 证明逐步合并不改变解码字节。
3. **工程。** 换一个含 emoji、制表符和尾部空格的文本，分别测零次与两次合并。逐项对照码点和 Token 区间，并确认解码字符串与输入相等。
4. **开放探索。** 词表加大对序列长度和 LM Head 计算有什么影响？

<details markdown="1">
<summary>参考答案</summary>

1. 3，各对应一个 UTF-8 字节。
2. 每次用字节拼接等价的新 ID 替换相邻两项；一步等价，有限次保持等价。
3. 在 Bash 中用 `sample=$'🐍\tA \r\n'`，分别传给 `--text "$sample" --train "$sample"`，让换行和制表符真正进入参数，并让合并在这段文本中发生。将 <code>--merges</code> 从 0 改到 2，对照 <code>token_spans</code> 的起止偏移与 <code>aligned_to_code_points</code>，比较 <code>decoded</code> 和 <code>text</code> 的完整字符串。
4. 合并可能减少时间轴，但 LM Head 参数与逐行 Softmax 代价随词表增大。报告每字符平均 Token 数和实际吞吐。
</details>

## 小结与下一章

分词器将连续文本变成可逆整数路径。[[18|第 18 章]] 讨论模型如何把这些 ID 变成浮点行并学习它们。

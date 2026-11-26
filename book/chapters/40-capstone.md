# 综合毕业项目：重建一个 GPT

## 学习目标

从空目录搭建可训练、可核验、可复现的 Decoder-only GPT，提交实验日志、推理样例和性能分析。

## 前置知识

[[15|第 15 章]] 的自动微分、[[17|第 17 章]] 的分词、[[28|第 28 章]] 的模型、[[30|第 30 章]] 的训练、[[32|第 32 章]] 的评估、[[35|第 35 章]] 的 KV Cache。

## 本章在 GPT 中的位置

项目验收覆盖从文本到概率，再到优化、生成及可重现实证。可先独立实现，再以本仓库为参考对拍。

## 从空目录开始的交付顺序

1. **约定。** 写下 Python 与依赖版本、数值类型、行向量右乘、词表与固定种子。用原创或许可清楚的语料，保留生成方式、记录边界和分割。
2. **算子。** 实现 Tensor 的加法、乘法、显式矩阵轴 matmul、归约、reshape、transpose、索引和指数对数。一开始以有限差分与重复索引累加验证。
3. **模型。** 先单头、再多头、再块。给出每步形状与参数来源。把解码器层接上可学习位置、词嵌入、最终 Norm 与共享输出。
4. **训练。** 样本用 \(T+1\) 项切成输入与标签。固定批先过拟合，再训练独立划分。保存参数、优化器、Token 映射、批采样随机流，再做续训对照。
5. **推理。** 实现 greedy、temperature、top-k、top-p、EOS 与窗口上限。先测无缓存，再加 prefill/decode。使用相同 prompt 和种子对照 logits 与序列。
6. **报告。** 汇总软硬件配置、参数与理论 FLOPs、实际指标和生成输出。分开解释模型的可用范围及明确未验证的配置。

## 最小验收矩阵

| 产物 | 可复核验收 |
| --- | --- |
| 模型与参数表 | 微型 GPT 544 参数，输入 [2,4] → logits [2,4,8] |
| 导数与共享 | 至少对 Q/K/V、归一化、重复索引、共享嵌入做测试 |
| 因果性 | 修改未来前，前缀 logits 保持一致 |
| 训练稳定 | 八项监督的固定批平均 NLL 明显下降；另测独立验证 |
| CLI 与续训 | 连续训练对比分段恢复，参数和 optimizer/RNG 一致 |
| 解码与性能 | 缓存与重算 logits 一致，报告缓存字节与估算方法 |
| 复现交付 | README、固定配置、语料许可、脚本、测试报告与解释边界 |

在报告里给出误差单位和阈值。可用

$$ overall-accept
\Delta_{\rm logit}=\max_{b,t,v}|z_{btv}^{\rm cache}-z_{btv}^{\rm full}|.
$$

手算例：若两行输出分别是 [1.00000000001, 0] 与 [1, 0]，最大差约 \(10^{-11}\)。若设 float64 缓存容差 \(10^{-12}\)，这一行检查不通过，即使 argmax 一样。形状 [2,4,8] 的稠密 logits 在 float64 下占 512 字节，缓存的 1024 字节还不包括模型参数；比较内存必须注明计数对象。

## Python 实验

M8，参考入口按顺序复算微型训练和缓存一致性，生成同批过拟合的真实序列。然后运行完整测试及无外部网络的站点检查。较大 CPU 配置见 [[34|第 34 章]]，全书代码映射见 [[41|附录]]。

~~~bash
python -m examples.modern 40
python -m pytest -q
python scripts/build.py
python scripts/check_book.py --full
~~~

::source examples/modern.py capstone

参考输出包括损失 2.090428 → 0.015276、贪心「春夏秋冬春夏秋冬」，以及缓存最大差约 \(1.39\cdot10^{-17}\)。用自己的配置时应提供自己的结果和阈值；容差应根据数值精度、参考计算和比较目的选择。

## 用统一输入验收自己的实现

[[32#context-floor]] 定义了末位必须读取前文的四行任务。参考运行使用词表 8、窗口 4、通道 16、4 个头、前馈宽度 48、2 个块，以及学习率 0.01 的 AdamW。这里的模型容量高于上面的 544 参数形状练习。每次固定同一个训练批更新 160 步。

读者可以把 <code>examples/capstone_check.py</code> 复制到自己的目录。传入候选工厂时，脚本只需要 NumPy 和 Python 标准库。工厂接收关键字 <code>seed, vocab, context, channels, heads, hidden, layers</code>，并返回具有三个方法的对象。

| 方法 | 输入与返回 |
| --- | --- |
| <code>forward(ids)</code> | 接收整数 [B,T]，返回 float64 的 [B,T,V] logits |
| <code>train_step(ids, targets)</code> | 接收等形整数数组，返回本次更新前全部目标的平均 NLL，并更新参数 |
| <code>generate(prompt, max_new_tokens, *, eos=None)</code> | 接收 ID 列表，按 greedy 返回含原提示的完整序列 |

~~~bash
python -m examples.capstone_check
python -m examples.capstone_check --seed 17 --token-seed 21

# 在独立目录里使用自己的模块名和工厂名
python capstone_check.py --candidate my_decoder:make_model
python capstone_check.py --candidate my_decoder:make_model --seed 17 --token-seed 21
~~~

入口按以下顺序核验。失败时先处理报错标出的阶段。

1. 检查前向输出的形状、类型、有限性，同种子构造两个实例并对照输出。
2. 分别前向整批和首行，替换未来项并独立计算各段前缀。训练前后都以绝对容差 \(10^{-10}\) 比较应当一致的 logits。
3. 从初始 logits 用 NumPy 独立计算全批 NLL，核对第一次更新的返回值，并确认更新确实改变了至少一个 logit。
4. 对比初末全批 NLL，要求末位 NLL 小于 0.2、四个答案全部正确。<code>--token-seed</code> 集中重排全部 ID，任务关系保持一致。
5. 用候选模型自己的前向独立重算四次输出，核对 [[31|第 31 章]] 的最后一行、窗口截断、EOS 和提示保留。

| 参考运行 | 初始全批 NLL | 最终全批 NLL | 末位 NLL | 末位输出 | 前缀最大误差 |
| --- | ---: | ---: | ---: | --- | ---: |
| 模型种子 42，原 ID | 2.139205 | 0.232363 | 0.001164 | [4,5,4,5] | \(3.55\times10^{-15}\) |
| 模型种子 17，ID 重排种子 21 | 2.117518 | 0.232530 | 0.001627 | [6,3,6,3] | \(1.78\times10^{-15}\) |

这是固定批次的验收。不同模型实现可以通过同一行为阈值，损失曲线允许不同。项目报告还要按 [[30|第 30 章]] 测量恢复后的下一次更新，按 [[35|第 35 章]] 对比缓存 logits，再用独立数据评估。

::source examples/capstone_check.py context_batch

::source examples/capstone_check.py prefix_check

::source examples/capstone_check.py verify_training

::source examples/capstone_check.py verify_greedy

::source examples/capstone_check.py reference_adapter

## 常见错误

- 先优化吞吐，后排查标签错位，难以分离问题来源。
- 交付只保存最佳 loss，丢失运行配置、数据划分和错误日志。
- 用现代组件扩展掩盖基线尚未完成的验收。

## 练习

1. **基础。** 写一条可运行的训练命令并列出它需要的所有输入文件。
2. **推导。** 从模型配置计算参数和缓存内存两个不同数字。
3. **工程。** 从空目录完成上述最小验收矩阵。用独立脚本跑原 ID 和重排 ID 两组，归档配置、JSON 结果和一次有意失败的日志。
4. **开放探索。** 选一个扩展并设计配对实验，例如 RoPE、GQA、量化或两种采样策略。

<details markdown="1">
<summary>参考答案</summary>

1. 见 [[30|第 30 章]]，输入包括安装环境、JSON 配置和其引用的两个数据文件。
2. 参数用 [[28#parameter-count]]，KV 用 [[35#kv-bytes]]。批与实际历史长度只进入后者。
3. 可以让模型只读取当前 ID 来触发上下文检查。归档失败阶段和对照数据，再单独记录连续训练与断点恢复、关闭与启用缓存两组实验的误差。
4. 只改变待测因素，固定数据、随机流、形状和预算，报告质量与资源两个维度。
</details>

## 小结与下一章

独立复现需要同时交付公式、可执行代码和尺度明确的证据。RAG（检索增强生成）、工具调用和 Agent 是应用层扩展，可以在这个预测器之外试验。[[41|附录]] 汇总阅读、符号和复核入口。

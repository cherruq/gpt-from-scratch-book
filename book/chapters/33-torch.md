# PyTorch 重实现与对齐

## 学习目标

用 PyTorch 的张量与自动微分重写同一模型，并用共享权重和输入对齐 logits、loss、梯度与一步 AdamW 更新。

## 前置知识

[[15|第 15 章]] 的自动微分、[[28|第 28 章]] 的参数表、[[32|第 32 章]] 的误差分析。

## 本章在 GPT 中的位置

NumPy 给出可解释的教学参考。PyTorch 用独立的张量算子、归一化、交叉熵和优化器验证相同目标，同时提供加速后端。

## 保持数学单位一致

实现直接以 \(XW+b\) 右乘，不需要在迁移时转置线性层权重。若改用 `nn.Linear`，其参数通常存为 \([C_{\rm out},C_{\rm in}]\)，迁移要转置。Norm 参数和方差计算 epsilon 相同，GELU 选择 tanh 近似，损失输入重排为 \([BT,V]\) 与 \([BT]\)，仍沿有效目标取均值。Torch 使用单个共享 Parameter 按两路线参与 \(\operatorname{LMHead}(Z)=ZE^\mathsf T\)。对齐批含重复 ID，帮助暴露散射漏项。

手算对齐用例：若输出行分数为 \([0,\log2]\)、目标为索引 1，则 loss \(=\log(3/2)\)，输出梯度 \([1/3,-1/3]\)。将同样一行重复两次并平均，每行梯度分别缩小一半；合并两条指向同一权重的路径时，对这条共享分数的总梯度不变。此原则适用于两框架。

## 比较顺序与误差

载入同一数组，固定 CPU float64，逐项对比：

1. 前向 \([B,T,V]\) 的最大绝对差。
2. loss 标量。
3. 反向后所有命名参数的最大元素误差。
4. 从零 AdamW 矩状态更新一次，权重衰减、学习率相同，关闭裁剪，比较更新后的整个参数集。

这样逐层发现差异的起点。若 logits 已偏，先查参数转置、mask 和 epsilon；若只有一步更新不同，核对矩步号、权重衰减与梯度裁剪。

| 指标 | 实测绝对误差（PyTorch 2.10 CPU） |
| --- | ---: |
| logits 最大 | \(1.39\cdot10^{-17}\) |
| loss | \(4.44\cdot10^{-16}\) |
| 参数梯度最大 | \(3.05\cdot10^{-16}\) |
| 单步后参数最大 | \(4.98\cdot10^{-16}\) |

绝对阈值统一 \(10^{-11}\) 便于覆盖接近零的项及不同 CPU 数值路径。默认前向显式计算分数、屏蔽与乘积。可选 `sdpa` 在已对齐后将内部注意力交给 PyTorch 的优化算子。由输入、设备及后端决定具体内核，不能从函数名判断是否使用 FlashAttention。[PyTorch 2.10 SDPA 文档](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 同一批次怎样测低精度的影响

先用本章的 float64 对拍确认公式、掩码和偏置路径。要检查混合精度，再在同一 PyTorch 实现中复制完全相同的 float32 初始权重、输入、标签和有效目标。一个副本用纯 float32，另一个只在前向与损失计算处进入 CPU 的 bfloat16 autocast。autocast 按算子选择运算类型。算负对数似然时需要稳定累加，训练参数和其梯度仍然保持 float32。[PyTorch 2.10 AMP 文档](https://docs.pytorch.org/docs/2.10/amp.html)

在 \(B=2,T=4,V=8,C=4\) 的基线中，第一行保留 1 个目标，第二行保留 3 个，两个副本都按这 4 个目标做均值。一次实际 CPU 运行记录如下。

| 比较点 | 纯 float32 | bfloat16 autocast |
| --- | --- | --- |
| LM Head logits | float32 | bfloat16 |
| 交叉熵 | float32 | float32 |
| 参数梯度与更新后权重 | float32 | float32 |
| 第一步更新前 NLL | 2.078781 | 2.078755 |

logits 最大绝对差约 \(4.14\cdot10^{-4}\)，梯度约 \(1.80\cdot10^{-3}\)，AdamW 一步后权重约 \(2.07\cdot10^{-3}\)。即使损失差小，个别小梯度的相对误差也会改变更新。这个测量只比较相同起点上的一小步，判断长期训练仍需比较曲线、数值异常和验证集。[[37|第 37 章]] 解释精度与计算预算如何分开评估。

## Python 实验

M7，安装 PyTorch 2.10 CPU 环境后运行 `python -m examples.torch_alignment`。源码是 `src/torch_gpt/model.py::TorchGPT` 与 `examples/torch_alignment.py::align`，前向未使用现成 Transformer 或 MultiheadAttention。

::source src/torch_gpt/model.py torch_forward

::source examples/torch_alignment.py align

运行 <code>python -m examples.torch_runtime</code> 复算 CPU 混合精度。加 <code>--gloo</code> 可运行 [[37|第 37 章]] 的两进程归约对照。默认 float32 训练 CLI 保持 [[34|第 34 章]] 的配置。

::source examples/torch_runtime.py amp

## 常见错误

- 两框架分别抽随机数，误把初始化的差异算作实现误差。
- 默认 GELU 路径不同还要求极小阈值。
- 只对比一个标量，漏掉输出末端或局部梯度的错位。

## 练习

1. **基础。** 为什么迁移到本章实现不用转置权重？
2. **推导。** 推导同一标量分数在两个位置相同且 loss 求均值时的总梯度。
3. **工程。** 改一处 epsilon，定位最早出现偏差的输出。
4. **开放探索。** 怎样为 float32、BF16 与融合算子选择容差？

<details markdown="1">
<summary>参考答案</summary>

1. 两边都显式用相同的行向量右乘约定。
2. 每行贡献缩放一半，两行加和回到单行梯度。
3. 先比对归一化输出，再看块和模型输出，恢复后用原测试验证。
4. 指定设备、数值范围与用途，使用绝对加相对容差，先记录误差分位数与第一处漂移。
</details>

## 小结与下一章

独立实现对齐了整个前向、反向和更新。[[34|第 34 章]] 将入口配置、日志与资源测量组织起来。

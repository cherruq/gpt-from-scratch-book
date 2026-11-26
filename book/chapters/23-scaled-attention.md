# Scaled Dot-Product Attention

## 学习目标

用 Q、K、V 手算完整注意力矩阵，解释缩放与 Softmax 的归约方向。

## 前置知识

[[10|第 10 章]] 的稳定 Softmax、[[22|第 22 章]] 的内容加权。

## 本章在 GPT 中的位置

缩放点积是多头注意力的单头内核，各头和各批样本使用同类运算。

## 前向与缩放假设

对 \(Q[T_q,D],K[T_k,D],V[T_k,D_v]\)，转置 K 的末两轴，沿通道 \(D\) 求和得 \(S[T_q,T_k]\)。按查询行归一化，再沿键位置归约得 \(O[T_q,D_v]\)：

$$ attention
S=QK^\mathsf T/\sqrt D,\qquad
A_{ij}=\frac{e^{S_{ij}}}{\sum_{j'=0}^{T_k-1}e^{S_{ij'}}},\qquad O=AV.
$$

若 \(q_d,k_d\) 独立、零均值、方差为 1，且各维乘积互不相关，则 \(\mathbb E(q_dk_d)=0\)，\(\operatorname{Var}(q_dk_d)=\mathbb E(q_d^2)\mathbb E(k_d^2)=1\)。未缩放求和的方差为 \(D\)，除 \(\sqrt D\) 后为 1。训练中分量未必独立同方差，推导只给出尺度设计的动机。分数尺度过大时 Softmax 容易集中到少数键，局部导数可能变小。[Transformer 原论文 §3.2.1](https://arxiv.org/abs/1706.03762v7)

## 手算两行

令 \(Q=K=[[1,0],[0,1]]\)，\(V=[[2,0],[0,4]]\)。第一行分数 \([1/\sqrt2,0]\)，第二行反过来。令 \(a=1/(1+e^{-1/\sqrt2})\approx0.669762\)。权重分别为 \([a,1-a]\) 与 \([1-a,a]\)：

$$ attention-example
O=\begin{bmatrix}2a & 4(1-a)\\2(1-a) & 4a\end{bmatrix}
\approx \begin{bmatrix}1.339523&1.320954\\0.660477&2.679046\end{bmatrix}.
$$

权重非负且行和为 1，每个输出落在两条值向量之间。后续输出投影会改变此几何关系。

::demo attention 逐步检查输入、分数、权重和加权和

静态读法：阶段 0 输入，1 缩放，2 行归一化，3 加权结果。具体数值见上面的手算。

## Python 实验

M4，运行 `python -m examples.transformer 23`。调用 `src/numpy_gpt/model.py::attention_weights` 与 `attention`，前面补 B、H 轴，打印时去掉。

::source examples/transformer.py scaled

### 用小例子定位轴错位

测试优先覆盖 \(T_q\ne T_k\) 的长方形分数矩阵。只用本章的方形对称输入时，列 Softmax 也可能给同样权重。若第二条查询改成 \([2,0]\)，两个查询都给键 0 更高权重，第二条更集中。正确的两行第一列权重约为 0.66976 与 0.80443，第二列各自补足到 1。若沿列归一化，第一列的和被迫等于 1，能暴露轴错误。

稳定归一化保留各行最大值 \([B,H,T_q,1]\)，广播到 Key 轴。用一个全局最大值在实数中也保持比值，但低分的整行浮点指数仍可能下溢。逐行偏移使每个有效查询至少一个指数为 1。添加掩码后需要先屏蔽，再取有效行最大值。

### 长方形算例：从点积算到值通道

取两个查询、三个键、每头 \(D=2\)。设 \(c=\sqrt2\log2\)，两个查询和键故意做成不同长度，这样可从形状发现错轴。

$$ rectangle-input
Q=\begin{bmatrix}c&0\\0&c\end{bmatrix}_{2\times2},\quad
K=\begin{bmatrix}0&2\\1&1\\2&0\end{bmatrix}_{3\times2},\quad
V=\begin{bmatrix}1&0\\0&2\\3&1\end{bmatrix}_{3\times2}.
$$

第一条查询和第三条键的点积为 \(2c\)，除以 \(\sqrt2\) 得 \(2\log2\)。其余五格沿同样的两个通道算出 \(S=(\log2)[[0,1,2],[2,1,0]]\)。每行最大值为 \(2\log2\)，减去后第一行指数为 \([1/4,1/2,1]\)，第二行顺序相反，行和都是 \(7/4\)。于是

$$ rectangle-out
A=\tfrac17\begin{bmatrix}1&2&4\\4&2&1\end{bmatrix}_{2\times3},\qquad
O=\tfrac17\begin{bmatrix}13&8\\7&5\end{bmatrix}_{2\times2}.
$$

第一行输出的第二通道用三个键位置的权重，计算是 \((1\cdot0+2\cdot2+4\cdot1)/7=8/7\)。算分数归约的是头通道 \(D\)，算输出归约的是键位置 \(T_k\)。本例第一列正确和为 \(5/7\)；误沿列做 Softmax 会强制每列和为 1。后续复用这组数做掩码与反向。

运行 <code>python -m examples.attention_labs 23</code>（M4-v2）。输出除以 \(\log2\) 的分数、行 Softmax 和输出。测试以手算分数和分数形式的权重验证浮点计算。

::source examples/attention_labs.py rectangular

## 常见错误

- 沿查询轴归一化。
- 使用 \(1/D\) 代替 \(1/\sqrt D\)。
- 行、列倒置。行发出查询，列是可以被读取的键。

## 练习

1. **基础。** 若某查询对所有键的分数都相同，输出是什么？
2. **推导。** 独立同方差且非单位方差时，求缩放分数方差。
3. **工程。** 用 NumPy 独立复算，并检查每行和。
4. **开放探索。** 固定数据，对比三种分母的权重熵。

<details markdown="1">
<summary>参考答案</summary>

1. 值的算术平均。
2. 单项方差为 \(\sigma_q^2\sigma_k^2\)，求和再除以 \(\sqrt D\) 后保持此值。
3. 实现稳定行 Softmax，并对 `weights.sum(-1)` 与 1 检查。
4. 比较 \(1,\sqrt D,D\)，固定可见掩码，跨头和样本汇总。
</details>

## 小结与下一章

注意力以矩阵乘法完成匹配与加权。[[24|第 24 章]] 明确每行可读取的键。

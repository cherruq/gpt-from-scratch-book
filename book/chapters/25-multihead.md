# Multi-Head Attention

## 学习目标

追踪拆头到合并的轴，计算参数与复杂度，推导注意力梯度并做差分验证。

## 前置知识

[[02|第 2 章]] 的 reshape 与 transpose、[[08|第 8 章]] 的 VJP、[[24|第 24 章]] 的掩码。

## 本章在 GPT 中的位置

多头注意力（multi-head attention，MHA）让同一位置并行查询多个关系，再混合各头结果。

## 前向与开销

\(X[B,T,C]\) 右乘三个 \([C,C]\) 投影，沿 \(C\) 求和，偏置 \([C]\) 广播。各结果 reshape 为 \([B,T,H,D]\)，再把头轴移前得到 \([B,H,T,D]\)，其中 \(D=C/H\)。各 \((b,h)\) 独立运行 [[23#attention]]，掩码 \([T,T]\) 跨 \(B,H\) 广播。输出逆转置、合并为 \([B,T,C]\)，再做一次 \([C,C]\) 变换。四个含偏置投影共 \(4(C^2+C)\) 参数。固定 \(C\) 增加头数时参数不变，每头表达维数减少。

两次注意力矩阵乘法耗时阶 \(O(BHT^2D)=O(BT^2C)\)，显式分数内存 \(O(BHT^2)\)，投影乘法耗时阶 \(O(BTC^2)\)。流式变体见 [[37|第 37 章]]。

## 反向推导

就一个批样本的一头而言，\(O=AV\) 的上游为 \(G_O[T,D]\)。由矩阵 VJP 得 \(G_A=G_OV^\mathsf T[T,T]\)、\(G_V=A^\mathsf T G_O[T,D]\)。Softmax 偏导 \(\partial a_i/\partial s_j=a_i(\delta_{ij}-a_j)\)，在每行收集项：

$$ mha-backward
G_S=A\odot\left(G_A-\operatorname{sum}_{\rm key}(G_A\odot A)\right),\qquad
G_Q=G_SK/\sqrt D,\quad G_K=G_S^\mathsf T Q/\sqrt D.
$$

求和保留长度 1 的键轴再广播。禁止项不向原分数传梯度。Q、K、V 的梯度都是 \([T,D]\)。合并各头后回传经过各自的投影，输入收到三支梯度之和。输出线性层使用 [[08#matmul-vjp]]。

## 手算拆头

输入 \([1,4,4]\) 的元素依次 0 到 15，位置 2 行为 [8,9,10,11]。头 0 取前两项，头 1 取 [10,11]。在 \([1,2,4,2]\) 中查看 \((0,1,2)\) 得到后两项。只 reshape 会让位置与头的对应关系错位。

::demo heads 选一个位置，核查拆头通道

静态读法：头 1 对位置 2 取原通道 2、3，数值 10、11。

## Python 实验

M4，运行 `python -m examples.transformer 25`，检查 `split_heads`、`merge_heads` 与输入梯度。测试用 float64 中心差分扰动两个头中的每个 Q、K、V 坐标，绝对容差 \(2\cdot10^{-9}\)、相对容差 \(2\cdot10^{-7}\)。

::source src/numpy_gpt/model.py head_axes

::source examples/transformer.py multihead

### 手算长方形矩阵的反向

用 [[23#rectangle-out]] 的 \(A,O\)，设 \(G_O=[[1,-1],[2,1]]\)，定义标量探针 \(\langle G_O,O\rangle\)。对第一个查询，\(G_A\) 是上游 \([1,-1]\) 分别点乘三个值，得 \([1,-2,2]\)。第二个查询得 \([2,2,7]\)。第一行的 \(G_A\odot A\) 沿键轴求和为 \((1-4+8)/7=5/7\)，所以该行第二项分数梯度是 \((2/7)(-2-5/7)=-38/49\)。各项依次算出

$$ hand-attention-vjp
G_S=\tfrac1{49}\begin{bmatrix}2&-38&36\\-20&-10&30\end{bmatrix},
\quad
G_V=\tfrac17\begin{bmatrix}9&3\\6&0\\6&-3\end{bmatrix}.
$$

\(G_S\) 行和为零，与整行分数平移不改变概率相符。第一条查询的通道 0 梯度是 \((0-38+72)/(49\sqrt2)\)，第二通道为其相反数。\(G_Q\) 两行分别为 \([34,-34]/(49\sqrt2)\)、\([50,-50]/(49\sqrt2)\)，\(G_K=(\log2)G_S^\mathsf T\) 形状为 \([3,2]\)。改变 \(K_{jd}\) 会同时影响所有可见查询，因而 \(G_K\) 沿查询轴归约。

### 从单头梯度回到四个投影

记拼回的头输出为 \(M[B,T,C]\)，且 \(Y=MW_O+b_O\)。由 [[08#matmul-vjp]] 得 \(G_M=G_YW_O^\mathsf T[B,T,C]\)，沿展平的批与时间轴得到 \(G_{W_O}=M_{\mathrm{flat}}^\mathsf T G_{Y,\mathrm{flat}}[C,C]\)。按前向通道顺序拆 \(G_M\)，才得到正确的头内上游。反过来，对 \(\overline Q,\overline K,\overline V\) 各做逆置换再合并，分别应用 [[08#matmul-vjp]]。三个投影共用 \(X\)，故 \(G_X\) 是三项输入梯度之和；四个偏置各沿 \(B,T\) 归约。

运行 <code>python -m examples.attention_labs 25</code>（M4-v2）。先复算长方形反向，再对两个批次、三个位置、两个不同头的因果注意力比较手写 NumPy 与张量引擎。实验省略偏置，四个矩阵梯度最大绝对差约 \(2.2\cdot10^{-16}\)。查询投影下标 \((1,2)\) 的解析梯度和独立中心差分均约 \(-0.12471628\)。

::source examples/attention_labs.py manual_attention_vjp

::source examples/attention_labs.py projected_mha

## 常见错误

- 对整个张量反转全部轴，只需交换 K 的末两轴。
- 合并时直接 reshape，破坏头与通道对应关系。
- 差分检查时同时修改掩码。

## 练习

1. **基础。** \(C=128,H=4\) 时头宽是多少？
2. **推导。** 从 Softmax Jacobian 展开行 VJP。
3. **工程。** 验证拆头和合并互逆。
4. **开放探索。** 固定 \(C\) 增加 \(H\) 为何显式权重内存仍增加？

<details markdown="1">
<summary>参考答案</summary>

1. 32。
2. 以 \(G_{A,i}\) 乘偏导，再沿 \(i\) 求和并提取 \(a_j\)。
3. 用不重复值检查多个 B、T、H 组合，避免常数掩盖布局错误。
4. 每头仍有一张 \(T\times T\) 的权重表，只是每项匹配使用更短的向量。
</details>

## 小结与下一章

MHA 在各头汇总历史并混合通道。[[26|第 26 章]] 使用 FFN 重组汇总后的状态。

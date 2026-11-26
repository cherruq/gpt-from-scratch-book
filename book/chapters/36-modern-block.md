# 现代架构：旋转、分组与稀疏路由

## 学习目标

把 RoPE、RMSNorm、SwiGLU、MQA/GQA 和 MoE 映射到基线，解释最小实现的范围。

## 前置知识

[[16|第 16 章]] 的归一化、[[21|第 21 章]] 的位置、[[25|第 25 章]] 的 MHA、[[35|第 35 章]] 的缓存。

## 本章在 GPT 中的位置

基线使用学习式绝对位置、LayerNorm、GELU FFN 与标准 MHA。现代变体仍可使用因果下一项预测，但改写块内表示与存储。

## RoPE 与 RMSNorm

给每对通道 \((a,b)\) 按位置 \(t\) 旋转 \(\theta=t\omega\)，得到 \((a\cos\theta-b\sin\theta,a\sin\theta+b\cos\theta)\)。Q/K 的 \(D\) 按两项配对，角度表 \([T,D/2]\) 跨 \(B,H\) 广播。按列向量定义旋转，归约点积时：

$$ rope-relative
(R(m)q)^\mathsf T(R(n)k)=q^\mathsf T R(n-m)k.
$$

转置等于逆，反向旋转用逆角度。取 \(q=k=[1,0]\)，频率 1，\(m=0,n=1\)，得分为 \(\cos1\approx0.5403\)。两位置同加 6 结果相同。相对位移一致只保证旋转因子一致，内容本身仍参与得分。[RoFormer](https://arxiv.org/abs/2104.09864v5)

RMSNorm 用 \((\operatorname{mean}_C(x^2)+\epsilon)^{-1/2}x\odot\gamma\)，参数 \([C]\) 跨前缀轴广播。取 [1,3]、增益 1，暂忽略 epsilon，输出为 \([1/\sqrt5,3/\sqrt5]\)，保留非零均值。反向差别见 [[16|第 16 章]]。[RMSNorm](https://arxiv.org/abs/1910.07467)

## SwiGLU 和 GQA

SwiGLU 以两组 \([C,F]\) 投影产生激活门与载荷，相乘后由 \([F,C]\) 投影回 \(C\)。门输入为 0 时 SiLU 为 0，此通道的乘积为零。中间负输入也可能翻转载荷符号。参数计数见 [[26|第 26 章]]。[GLU Variants](https://arxiv.org/abs/2002.05202)

MQA 令多组 Q 共用一组 K/V。分组查询注意力（grouped-query attention，GQA）令 \(H_q\) 个 Q 均分到 \(H_{\rm kv}\) 个 KV 组，要求整除且头维相同。取 4 个 Q 与 2 个 KV 头，头 0、1 共用 KV 0，头 2、3 用 KV 1。缓存相对于 MHA 减半，但每个查询头的匹配仍需计算。最小实现展开索引，实际内核可直接消费分组布局。[GQA](https://arxiv.org/abs/2305.13245v3)

### 从投影列数推到缓存字节

保持 \(C=16,H_q=4,D=4\)。\(W_q[C,H_qD]\) 和 \(W_o[H_qD,C]\) 保持不变。键和值的权重各为 \([C,H_{\rm kv}D]\)，沿 \(C\) 归约。以 GQA 的 \(H_{\rm kv}=2\) 为例，\(W_k\) 只有 8 列。头 0、1 各自与同一个键头算分数，因此同一个键向量在前向被读两次，在反向会收到两个查询头的梯度。每头的分数矩阵仍有 \(T_qT_k\) 项。

下面只比较一个块的四组投影权重数量，忽略偏置。缓存则按全模型 2 层、批大小 1、历史 128 项和元素大小 2 字节计算。

| 架构 | KV 头数 | 单块投影权重 | 全模型 KV 字节 |
| --- | ---: | ---: | ---: |
| MHA | 4 | 1024 | 16 384 |
| GQA | 2 | 768 | 8192 |
| MQA | 1 | 640 | 4096 |

通用权重计数是 \(CD(2H_q+2H_{\rm kv})\)，缓存按 [[35#kv-bytes]] 计算。若头维、历史长度、精度和批量相同，缓存随 KV 头数线性变化。把 KV 临时展开到查询头数的实现仍可保留压缩缓存，但展开区域本身按查询头数消耗内存。

## MoE 与负载均衡

混合专家（mixture of experts，MoE）对 \(X[B,T,C]\) 右乘路由器 \([C,E]\)，得 \([B,T,E]\)。沿专家轴选最高 \(k\) 个，对选中分数做 Softmax，并加权相应的专家输出。硬选择集合在局部不变化时视为常量。top-1 权重恒等于 1，任务梯度不会经权重连续更新选择；负载均衡可通过完整分布给路由器信号：

$$ router
L_{\rm aux}=E\sum_{e=1}^E f_e\,\overline p_e.
$$

\(f_e\) 是所有被选中槽位中的份额，\(\overline p_e\) 是该专家完整 Softmax 概率沿 \([B,T]\) 的均值。两专家一个样本，分数 [2,0]、top-1，则 \(f=[1,0]\)、\(\overline p_0=0.880797\)，辅助损失 1.761594。均匀份额时为 1。微实现为对照数学而先计算全部专家输出；稀疏系统还要调度、限制容量、移交超额 Token。[Switch Transformers](https://arxiv.org/abs/2101.03961v3)

再看三个 Token 的路由梯度。路由器令前两个 Token 的分数为 \([\log3,0]\)，最后一个为 \([0,\log3]\)。top-1 的离散份额是 \([2/3,1/3]\)，完整分布均值是 \([7/12,5/12]\)，辅助损失为 \(19/18\)。记第 \(i\) 个 Token 的全概率为 \(p_{i,e}\)、分数为 \(s_{i,j}\)，固定离散份额后，用 Softmax 的 VJP 得

$$ router-gradient
\frac{\partial L_{\rm aux}}{\partial s_{i,j}}
=\frac{E}{N}p_{i,j}\left(f_j-\sum_e p_{i,e}f_e\right),
$$

其中 \(N=3,E=2\)。每条分数行梯度都是 \([1/24,-1/24]\)。若前两行输入为 \([1,0]\)，第三行为 \([0,1]\)，乘回输入转置后，路由器权重的两行分别为 \([1/12,-1/12]\) 和 \([1/24,-1/24]\)。下降步会减小过载专家 0 的相对分数。这里观察的是尚未跨过选中集合边界的局部更新。对 task loss 单独反向，本例的路由参数梯度为零，专家参数仍可从已选 Token 收到梯度。

MLA（multi-head latent attention）通过低秩潜在表示与可吸收投影减少缓存，另行处理旋转的位置分量。它属于进阶阅读范围，本项目实现到 GQA 与 MoE。[DeepSeek-V2](https://arxiv.org/abs/2405.04434v5)

## Python 实验

M8，运行 <code>python -m examples.modern 36</code>。Pre-RMS、RoPE、GQA 与 SwiGLU 的组合输出 [1,2,4] 且输入梯度有限；MoE 样本输出 [2,0]，辅助值 1.761594。测试检查 RoPE 相对位移、逆旋转梯度及 GQA 展开等价性。

::source src/modern/architecture.py

<code>python -m examples.modern_objectives</code> 复算上表和三个 Token 的梯度。<code>routing</code> 分别记录只反向任务输出与只反向辅助目标得到的路由器梯度。

::source examples/modern_objectives.py routing_budget

## 常见错误

- 旋转 V，或遗漏缓存中的绝对位置偏移。
- 用 GQA 缓存公式描述已展开的中间数组的实际内存。
- 把专家总参数量当成每项计算量，或把密集教学实现当成稀疏调度。

## 练习

1. **基础。** Q 头 8、KV 头 2，每组几个查询头？
2. **推导。** 解释逆角度的 VJP。
3. **工程。** 检查 top-k 等于专家数时的路由和负载。
4. **开放探索。** MHA 与 GQA 的质量、吞吐、缓存可能怎样变化？

<details markdown="1">
<summary>参考答案</summary>

1. 四个。
2. 旋转是线性正交算子，VJP 由转置完成，即逆旋转。
3. 掩码全真，等价于普通 Softmax 混合，每个专家槽位份额相同。
4. 固定数据与计算预算，分开报告验证 NLL、首项与单项延迟、峰值缓存和后端。
</details>

## 小结与下一章

架构替换改变参数布局、激活与缓存。[[37|第 37 章]] 用 FLOPs、带宽和数值实验分析代价。

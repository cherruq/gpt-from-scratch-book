# Transformer FFN

## 学习目标

推导逐位置 FFN，计算 GELU 与 SwiGLU 参数量，说明它们与注意力的职责差异。

## 前置知识

[[06|第 6 章]] 的激活、[[13|第 13 章]] 的 MLP、[[25|第 25 章]] 的多头输出。

## 本章在 GPT 中的位置

注意力把可见前缀信息汇入当前位置，FFN 用跨位置共享的权重加工这条表示。上下文改变输入，从而影响 FFN 激活。

## 扩维、激活、收缩

\(X[B,T,C]\) 右乘 \(W_u[C,F]\) 沿 \(C\) 归约并广播 \(b_u[F]\)，得到 \([B,T,F]\)。激活逐元素作用，\(W_d[F,C]\) 沿 \(F\) 汇总回 \([B,T,C]\)，再广播偏置 \(b_d[C]\)。

$$ ffn
\operatorname{FFN}(X)=\operatorname{GELU}(XW_u+b_u)W_d+b_d.
$$

参数量 \(2CF+F+C\)。注意力提供跨位置的信息流；FFN 处理每行通道。消融实验可能发现某些维度与特定模式相关，但层间交互影响结果，不能直接把整个 FFN 当成确定的词典。

SwiGLU 采用 \([\operatorname{SiLU}(XW_g)\odot(XW_u)]W_d\)。前两支都是 \([B,T,F_g]\)，最后沿 \(F_g\) 归约。省偏置时共 \(3CF_g\)，设 \(F_g\approx2F/3\) 可接近普通 FFN 的权重数。门控让一条支路连续调节另一条支路的通道贡献。

## 手算一个位置

\(x=[1,-1]\) 乘 \([[1,0,-1],[0,1,0]]\) 得 \(z=[1,-1,-1]\)，GELU 后约为 \([0.841192,-0.158808,-0.158808]\)。下投影 \([[1,0],[0,1],[1,1]]\) 得 \([0.682384,-0.317616]\)。取输出两项之和为目标，反向得到 \(G_x\approx[1.248892,-0.082964]\)。

## Python 实验

M4，运行 `python -m examples.transformer 26`，复用 `src/numpy_gpt/model.py::gelu`。手算省偏置，基线保留偏置。

::source examples/transformer.py ffn

### 推导实际实现的 GELU 梯度

标准 GELU 为 \(z\Phi(z)\)，\(\Phi\) 是标准正态累积分布，\(\varphi\) 是其密度。对精确表达求导得到 \(\Phi(z)+z\varphi(z)\)。基线采用 [[06|第 6 章]] 的 tanh 近似，反向也应来自同一个近似。令 \(a=\sqrt{2/\pi}\)、\(b=0.044715\)，写 \(u=a(z+bz^3)\) 和 \(t=\tanh u\)。用乘积法则，且 \(\mathrm dt/\mathrm du=1-t^2\)，得

$$ gelu-tanh-vjp
\widetilde{\operatorname{GELU}}'(z)
=\tfrac12(1+t)+\tfrac z2(1-t^2)a(1+3bz^2).
$$

将 \(z=-1,0,1\) 代入，斜率约为 \(-0.08296408,0.5,1.08296408\)。近似 GELU 在 \(-1\) 附近可出现负斜率。回到本章的 FFN 行，取 \(G_y=[1,1]\)。下投影的 VJP 给激活层 \(G_a=[1,1,2]\)。对 \(z=[1,-1,-1]\) 逐项乘斜率，得 \([1.08296408,-0.08296408,-0.16592817]\)。上投影第一行是 \([1,0,-1]\)，故 \(G_{x,0}\approx1.08296408-(-0.16592817)=1.24889225\)。第二行只读中间项，得约 \(-0.08296408\)。FFN 激活及其 VJP 对 \([B,T,F]\) 的每个分量各自计算。

SwiGLU 再多一个乘法节点。若 \(H=\operatorname{SiLU}(A)\odot B\)，则 \(\overline A=G_H\odot B\odot\operatorname{SiLU}'(A)\)、\(\overline B=G_H\odot\operatorname{SiLU}(A)\)。使用 \(\operatorname{SiLU}'(a)=\sigma(a)+a\sigma(a)(1-\sigma(a))\)。手算 \(a=0,b=4,G_H=1\)，输出 0，\(\overline A=2,\overline B=0\)。门在当前点关掉一条路线，门自身仍可收到梯度。

运行 <code>python -m examples.attention_labs 26</code>（M4-v2），比较三个点的解析斜率、自动微分和中心差分。实测最大差分误差约 \(9.5\cdot10^{-12}\)。

::source examples/attention_labs.py gelu_prime

## 常见错误

- 沿时间轴扩维，混入未授权的跨位置路径。
- 将 SwiGLU 的逐元素门控写为矩阵乘法。
- 比较不同宽度时只报告参数，忽略激活内存与计算。

## 练习

1. **基础。** \(C=4,F=16\) 的 GELU FFN 参数量是多少？
2. **推导。** 无偏置 SwiGLU 与普通 FFN 参数近似相等时的宽度比例。
3. **工程。** 增加另一位置，修改它并检查原位置输出。
4. **开放探索。** 怎样避免将激活相关性误称为知识存储？

<details markdown="1">
<summary>参考答案</summary>

1. 148。
2. \(3CF_g=2CF\)，得 \(F_g=2F/3\)。
3. 沿时间轴拼第二行，改变其数值，因计算独立第一行应保持。
4. 预设反例和因果干预，对照随机消融，并检查效果在不同前缀中是否稳定。
</details>

## 小结与下一章

FFN 逐位置加工通道，注意力跨位置汇总。[[27|第 27 章]] 用两条残差支路将它们组成块。

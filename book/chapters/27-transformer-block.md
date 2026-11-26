# Transformer Block

## 学习目标

实现并验证完整 Pre-Norm 块，比较 Post-Norm，追踪残差中的直接梯度。

## 前置知识

[[16|第 16 章]] 的 LayerNorm、[[25|第 25 章]] 的 MHA、[[26|第 26 章]] 的 FFN。

## 本章在 GPT 中的位置

解码器重复堆叠多个相同结构的块。每块参数独立，输入与输出都保留 \([B,T,C]\) 残差形状。

## 两条残差支路

基线先归一化再走变换：

$$ prenorm
U=X+\operatorname{MHA}(\operatorname{LN}_1(X)),\qquad
Y=U+\operatorname{FFN}(\operatorname{LN}_2(U)).
$$

两次层归一化都沿通道 \(C\) 进行，增益、偏置 \([C]\) 跨时间和批广播；两个加法操作都是同形 \([B,T,C]\)。Post-Norm 把归一化放在和之后，例如 \(U=\operatorname{LN}(X+\operatorname{MHA}(X))\)。此时直接残差也经过归一化的 Jacobian。原始 Transformer 使用 Post-Norm 的组织方式，[Transformer 原论文](https://arxiv.org/abs/1706.03762v7)。

把其中一个子层写成 \(y=x+f(x)\)，向量展平后对上游 \(G_y\) 有

$$ residual-backward
G_x=G_y+G_yJ_f.
$$

第一项来自恒等映射，第二项包含子层的每个投影及 LayerNorm。两个残差分支连续堆叠有多条梯度路径。该式没有保证所有方向上的梯度都保持大小，因为两项可能相抵。

## 手算残差与完整块

为便于手算先取标量 \(f(x)=2x\)，输入 2，输出 6。如果上游是 1，则直路贡献 1，支路贡献 2，总梯度 3。完整块中，若将注意力的输出投影和 FFN 下投影连同偏置都清零，两支值都为零，因此任意输入保留原样。梯度从输入角度只剩恒等路径，但输出投影的参数本身仍可能有非零梯度。此区别有助于解释某些零初始化策略。

::demo residual 调整标量残差支路，查看梯度相加与抵消

静态读法：a=2 得输出 6、梯度 3；a=0 得输出 2、梯度 1；a=−1 时两路径梯度抵消。

## Python 实验

M5，运行 `python -m examples.transformer 27`。先输出随机块与输入梯度，再清零两个末端投影，验证输出精确等于原 `arange(8)/8`、输入梯度全 1。下面是模型实际调用的块，缓存支路在 [[35|第 35 章]] 展开。

::source src/numpy_gpt/model.py transformer_block

::source examples/transformer.py block

### LayerNorm 的边界与完整块导数

依 [[16#layernorm-grad]]，设 \(g=G_y\odot\gamma\)，\(r=(\operatorname{mean}(u^2)+\epsilon)^{-1/2}\)。即使去均值后 \(u=0\)，在 \(\epsilon>0\) 时输入梯度仍为 \(r(g-\operatorname{mean}(g))\)。以两通道 \([2,2]\)、增益 \([1,1]\)、偏置零、\(\epsilon=1/4\) 为例。输出全零，但取上游 \([1,-1]\) 后，输入梯度为 \([2,-2]\)，增益梯度为 0，偏置梯度为 \([1,-1]\)。这里扩大 \(\epsilon\) 便于手算，基线用 \(10^{-5}\)。

改输入为 \([1,3]\) 后，去均值为 \([-1,1]\)，方差为 1。取同一上游，代入得 \(\overline x=(2/(5\sqrt5))[1,-1]\)。当 \(\epsilon=0\) 且方差非零，按正数缩放不改变标准化，该方向的导数为零；正 \(\epsilon\) 会破坏精确的缩放不变性，平移不变性仍成立。若方差也是零，\(\epsilon=0\) 的公式无定义。

若某步的输出为 \(y=x+F(\operatorname{LN}(x))\)，经过内部子层与 Norm 后取得 \(G_{\rm branch}\)，输入梯度为 \(G_y+G_{\rm branch}\)。将第二次残差的全部梯度传回第一次残差。原输入经注意力成为别的位置的 K 或 V 时，可能从这些位置收回额外梯度，由 QKV 的 VJP 还原。

运行 <code>python -m examples.attention_labs 27</code>（M5-v2）。使用非均匀上游，对真实块的一个输入元素做中心差分，解析值与差分均约 0.19708119。原实验将两条支路清零以检查恒等路径，新实验的两条支路都参与前向。

::source examples/attention_labs.py norm_boundary

## 常见错误

- 把两个 Norm 共用参数，而实验和参数公式假定分别学习。
- 将第二次残差加回最初的 \(X\)，丢失 \(U\) 的直接路径。
- 只检查形状，遗漏加残差与归一化的次序。

## 练习

1. **基础。** 输入 [2,4,4] 的块输出什么形状？
2. **推导。** 两个标量支路 \(f(x)=2x,g(u)=3u\)，求完整残差的导数。
3. **工程。** 只清零一个末端投影，检查哪一支消失。
4. **开放探索。** 比较 Pre-Norm 和 Post-Norm，如何减少其他变化干扰？

<details markdown="1">
<summary>参考答案</summary>

1. [2,4,4]。
2. \((1+2)(1+3)=12\)。
3. 清零注意力末端只能得到 \(U=X\)；FFN 保留。反之 \(Y=U\)。
4. 使用同种子、数据与参数尺寸，考虑两种结构的最终 Norm 选择，比较多层梯度尺度和损失。
</details>

## 小结与下一章

块负责在同一形状循环更新状态。[[28|第 28 章]] 把块、嵌入和输出头装成完整 GPT，逐项计数。

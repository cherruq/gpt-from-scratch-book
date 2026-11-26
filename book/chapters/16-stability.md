# 训练稳定性与泛化

## 学习目标

分析初始尺度、梯度连乘、残差和归一化，推导 LayerNorm 与 RMSNorm 的反向，区分训练模式、推理模式及验证。

## 前置知识

[[09|第 9 章]] 的方差，[[15|第 15 章]] 的反向图。

## 本章在 GPT 中的位置

本书 GPT 的每个子层先做 LayerNorm，再走残差支路。最后还有一层 Norm。RMSNorm 和 Dropout 作为可独立检查的扩展。

## 为什么尺度会失控

若反向沿多层不断乘局部导数，典型因子持续小于 1 时梯度会快速衰减，持续大于 1 时可能膨胀。宽层随机初始化要考虑 fan-in。如果每个输入通道近似独立、零均值同方差，权重独立标准差取 \(1/\sqrt C\)，则输出单分量方差近似输入量级。真实层间相关性会违反这些假设。残差 \(x+F(x)\) 给反向提供恒等直路，但加法项也可能相互抵消，不能保证越深越稳定。

## 归一化及其反向

每个 \([B,T,C]\) 的通道行独立，定义 \(\mu=\operatorname{mean}_C(x)\)、\(u=x-\mu\)、\(r=(\operatorname{mean}_C(u^2)+\epsilon)^{-1/2}\)，归一化后 \(n=ur\)。增益 \(\gamma[C]\) 与偏置 \(\beta[C]\) 跨批和时间广播，\(y=n\odot\gamma+\beta\)。设上游 \(G_y\)，记 \(g=G_y\odot\gamma\)。

先看 \(n\) 对 \(u\)。扰动 \(\mathrm dr=-r^3\operatorname{mean}(u\,\mathrm du)\)，因此 \(\mathrm dn=r\,\mathrm du+u\,\mathrm dr\)。把 \(\langle g,\mathrm dn\rangle\) 中 \(\mathrm du\) 的系数收集得到 \(r[g-n\operatorname{mean}(gn)]\)。再过中心化算子，其反向减去通道均值；因为 \(n\) 均值为零，得到

$$ layernorm-grad
\overline x=r[g-\operatorname{mean}_C(g)-n\operatorname{mean}_C(gn)].
$$

增益梯度为 \(\sum_{b,t}G_y\odot n\)，偏置梯度为 \(\sum_{b,t}G_y\)，均为 \([C]\)。RMSNorm 直接取 \(u=x\)，通常省略偏置，因此 \(\bar x\) 少去减 \(g\) 均值的项。两者都在平方平均后加 \(\epsilon\) 保证常数或全零输入的分母非零。[Layer Normalization 原论文](https://arxiv.org/abs/1607.06450)

### 将梯度拆到均值、方差和增益

为避免混淆，把一行的方差记为 \(v=C^{-1}\sum_cu_c^2\)。通道行和为零的原因是 \(\sum_c(x_c-\mu)=0\)。反向可按前向相反的顺序登记各节点收到的贡献：

| 节点 | 形状 | 反向与归约 |
| --- | --- | --- |
| \(y=n\odot\gamma+\beta\) | \([B,T,C]\) | 传给 \(n\) 的是 \(g=G_y\odot\gamma\)。增益、偏置分别沿 \(B,T\) 求和 |
| \(n=ur\) | \([B,T,C]\) | 直接传给 \(u\) 的是 \(rg\)，传给 \(r\) 的是沿 \(C\) 求和的 \(\sum_c g_cu_c\) |
| \(r=(v+\epsilon)^{-1/2}\) | \([B,T,1]\) | \(\overline v=-\overline r\,r^3/2\) |
| \(v=\operatorname{mean}_C(u^2)\) | \([B,T,1]\) | 向 \(u\) 广播 \(\overline v(2u/C)\)，与直接路径相加 |
| \(u=x-\operatorname{mean}_C(x)\) | \([B,T,C]\) | 从相加后的 \(\overline u\) 减去它沿 \(C\) 的均值 |

将表中方差支路代回，\(u\) 收到的总量是 \(rg-u r^3 C^{-1}\sum_cg_cu_c\)。令 \(n=ur\)，第二项可写成 \(r n\,\operatorname{mean}_C(gn)\)。然后执行最后一行的中心化反向就得到 [[16#layernorm-grad]]。中心化本身等同于对行右乘 \(I-\mathbf1\mathbf1^\mathsf T/C\)，这个矩阵对称，所以它的 VJP 和前向相同。写程序时用广播减均值即可。

### 用不变量定位归约轴

在同一行的各通道同时加常量 \(a\)，\(\mu\) 也加 \(a\)，故 \(u\)、\(v\) 和 \(y\) 完全不变。可导时相应方向导数是 \(\sum_c\overline x_c=0\)。这应逐行成立。在批轴上求总和为零还不够，几行错误也可能抵消。正数缩放则不同。正 \(\epsilon\) 使 \(\operatorname{mean}_C(n^2)=v/(v+\epsilon)\)，当 \(v\) 很小时，归一化输出的方差可以明显小于 1。

若 \(C=1\)，单行减均值后就是零，\(y=\beta\)，输入和增益梯度均为零。若 \(C>1\) 且输入行恰好是常数，\(n\) 仍然为零，但只扰动一个通道会打破常数行，\(\overline x=r(g-\operatorname{mean}_C(g))\) 可以非零。若取很小的 \(\epsilon\)，这个点上的 \(r=1/\sqrt\epsilon\) 较大，验证时需关注差分步长，而不是把有限扰动误当成精确导数。RMSNorm 去掉 \(u=x-\mu\) 的中心化步骤，所以不应用逐行梯度零和这条判据。

## 手算归一化

用 \(x=[1,3],\gamma=[1,1],\beta=[0,0],\epsilon=1/4\)。中心值 \(u=[-1,1]\)，\(r=2/\sqrt5\)，输出 \([-0.894427,0.894427]\)。取上游 \([1,2]\)，手算 \(\operatorname{mean}(g)=1.5\)、\(\operatorname{mean}(gn)=1/\sqrt5\)，代入得输入梯度 \([-0.089443,0.089443]\)，两项之和为 0，与整体平移不改变输出一致。

训练模式若启用 Dropout，每项独立取以 \(1-p\) 为概率的保留标记，保留项乘 \(1/(1-p)\)，所以随机输出的期望等于输入。推理直接传递输入。LayerNorm 在两种模式计算方式一致。固定种子和独立验证集可以比较不同训练阶段；训练损失继续下降而验证损失上升时，模型可能对训练样本过拟合。

## Python 实验

M2，运行 `python -m examples.networks 16`。手推公式 `norm_vjp` 与张量版 `layer_norm` 的自动梯度对齐，测试也用中心差分检查了 RMSNorm 的输入梯度。

::source src/math_basics/normalization.py norms

### 换一组增益、偏置和上游

仍以一行两通道手算，取 \(x=[0,2]\)，\(\epsilon=3\) 仅用于把 \(r\) 化成 \(1/2\)。此时 \(u=[-1,1]\)、\(v=1\)、\(n=[-1/2,1/2]\)。令 \(\gamma=[2,-1]\)、\(\beta=[1,1/2]\)，两项输出均为 0。取上游 \(G_y=[1,3]\)，有 \(g=[2,-3]\)，而 \(\operatorname{mean}(g)=-1/2,\operatorname{mean}(gn)=-5/4\)。

按表跟踪方差支路也能复得同一答案。\(\overline r=2(-1)+(-3)(1)=-5\)，故 \(\overline v=(-5)(-(1/2)^3/2)=5/16\)。它经 \(2u/C=[-1,1]\) 向 \(u\) 贡献 \([-5/16,5/16]\)，与直接路线 \(rg=[1,-3/2]\) 加和为 \([11/16,-19/16]\)。减去该向量的均值 \(-1/4\)，得到

$$ norm-hand-dx
\overline x=[15/16,-15/16],\qquad
\overline\gamma=[-1/2,3/2],\qquad
\overline\beta=[1,3].
$$

这里假设只有一条输入行，参数梯度还未跨其他行累加。若放进批次，增益和偏置的梯度沿其共享的 \(B,T\) 都要加起来。数值实验在 \([2,3,4]\) 张量中放入一条常数行与随机上游，对 \(x,\gamma,\beta\) 的每个元素分别用纯 NumPy 前向做中心差分。它也让真实 <code>layer_norm</code> 反向一遍，核对手写公式。默认 \(\epsilon=0.01\) 时，三组梯度的差分最大绝对误差低于 \(9.1\cdot10^{-10}\)，与自动微分的误差不超过 \(4.5\cdot10^{-16}\)。

M2-v3 的工程练习可替换批次、位置、通道数和种子。运行下面两次再把通道数改成 1，观察输入梯度的验证结果。给出的练习参数使用 \(10^{-8}\) 作为差分绝对容差。若继续大幅减小 \(\epsilon\)，对于常数行应同时缩小差分步长并重新判断舍入误差。

~~~bash
python -m examples.norm_lab
python -m examples.norm_lab --batch 1 --time 4 --channels 3 --epsilon 0.05 --seed 19
~~~

::source examples/norm_lab.py same_loss

::source examples/norm_lab.py norm_check

## 常见错误

- 沿批次或时间平均 LayerNorm，使位置之间产生额外依赖。
- 实现训练 Dropout 时遗漏保留缩放，导致切换模式时均值变化。
- 用训练批选择超参数并把同一结果当成独立验证。

## 练习

1. **基础。** 常数向量经过 LayerNorm 的理论输出是什么？
2. **推导。** 证明 \(\bar x\) 沿通道和为零。
3. **工程。** 给随机梯度增加有限差分检查，并改变 epsilon。
4. **开放探索。** 要识别过拟合及数据泄漏，应分开观察哪些曲线？

<details markdown="1">
<summary>参考答案</summary>

1. 中心化后为全零，输出 \(\beta\)。
2. \(g-\operatorname{mean}(g)\) 的和为零，第三项含零均值 \(n\)，和也是零。
3. 对标量目标 \(\langle G_y,y(x)\rangle\) 差分即可。让样本含非零方差，再加入接近常量的边界样本。
4. 观察训练和独立验证平均 NLL、训练覆盖率及按记录分组的误差，检查近重复记录是否跨集合。
</details>

## 小结与下一章

归一化与残差改善数值条件，验证集提供独立证据。[[17|第 17 章]] 开始把真实文本转换成可学习的整数序列。

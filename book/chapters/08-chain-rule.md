# 链式法则与矩阵微分

## 学习目标

沿计算图传播梯度，推导矩阵乘法与广播的向量–Jacobian 乘积，理解共享参数的加和路径。

## 前置知识

[[05|第 5 章]] 的转置与乘法、[[07|第 7 章]] 的导数。

## 本章在 GPT 中的位置

每个仿射投影、位置偏置和输出权重共享都使用本章的反向规则。完整计算图在 [[15|第 15 章]] 自动构建。

## 局部线性近似怎样复合

设 \(u=f(x)\)，\(y=g(u)\)。对小扰动 \(\mathrm dx\)，有 \(\mathrm du\approx f'(x)\mathrm dx\)，\(\mathrm dy\approx g'(u)\mathrm du\)，得到 \(y'=g'(f(x))f'(x)\)。若向量到向量映射把输入、输出分别展平为 \(n,m\) 项，Jacobian \(J[m,n]\) 的第 \((i,j)\) 项就是输出 \(i\) 对输入 \(j\) 的偏导。对标量损失，上游梯度写作行 \(g[1,m]\)，反向只需 \(gJ[1,n]\)。称为向量–Jacobian 乘积（VJP）。像 \([B,T,C]\) 这样的大张量不需要显式构造 \(J\)。

### 从方向扰动到向量–Jacobian 乘积

设 \(F:\mathbb R^n\to\mathbb R^m\)，行向量扰动为 \(\delta x[1,n]\)，Jacobian 按输出行、输入列排列，则 \((\delta y)^\mathsf T=J_F(\delta x)^\mathsf T\) 到一阶成立。再接可微的 \(H:\mathbb R^m\to\mathbb R^k\) 时，最终扰动的首项是 \(J_HJ_F(\delta x)^\mathsf T\)。因为 \(\|\delta y\|=O(\|\delta x\|)\)，外层的高阶余项仍是 \(o(\|\delta x\|)\)。这解释了链式法则为何需要局部线性近似。

若损失传到 \(y\) 的上游是行向量 \(g\)，把 \(g(\delta y)^\mathsf T=gJ_F(\delta x)^\mathsf T\) 中乘在输入扰动前的系数读出来，就得到输入梯度 \(gJ_F\)。例如 \(F(a,b)=(2a+b,4a-3b)\)，其 Jacobian 两行是 \([2,1]\) 与 \([4,-3]\)。若 \(g=[1,2]\)，则两个输入依次得到 \(1\cdot2+2\cdot4=10\) 与 \(1\cdot1+2\cdot(-3)=-5\)。输出维 \(m\) 是这一乘法的求和轴。

## 对矩阵乘法求导

设 \(X[N,C],W[C,M],b[M]\)，\(Y=XW+b\) 形状为 \([N,M]\)，偏置跨 \(N\) 行广播。让 \(G[N,M]\) 是 \(\partial\mathcal L/\partial Y\)。一阶扰动由 \(\mathrm dY=(\mathrm dX)W+X(\mathrm dW)+\mathrm db\) 给出，两输入扰动相乘是二阶项。代入 \(\mathrm d\mathcal L=\sum_{ij}G_{ij}\mathrm dY_{ij}\)，按各扰动项收集系数：

$$ matmul-vjp
\begin{aligned}
\overline X_{ik}&=\sum_jG_{ij}W_{kj}, & \overline X&=GW^\mathsf T,\\
\overline W_{kj}&=\sum_iX_{ik}G_{ij}, & \overline W&=X^\mathsf T G,\\
\overline b_j&=\sum_iG_{ij}.
\end{aligned}
$$

横线表示相对于标量损失的梯度。GPT 的 \(i\) 合并批次与位置，因此共享权重的梯度跨两轴相加。若相同参数还有输入查表等另一条路径，整个损失扰动等于各路径扰动的和，梯度也加起来。广播反向会对被扩展的轴求和，再恢复原有形状。目标函数如果取了均值，缩放已经包含在 \(G\) 里。

## 手算一个反向传播

设 \(X=[[1,2],[3,4]],W=[[2,-1],[0,3]],G=[[1,0],[-1,2]]\)。前向得到 \(Y=[[2,5],[6,9]]\)。对 \(W_{00}\) 的梯度为 \(1\cdot1+3\cdot(-1)=-2\)。对 \(X_{10}\) 的梯度为 \((-1)\cdot 2+2\cdot(-1)=-4\)。在两行都加同一偏置时 \(\overline b=[0,2]\)，第一通道的两个路径抵消。

## Python 实验

M0，运行 `python -m examples.math_basics 8`。预期 \(\overline X=[[2,0],[-4,6]]\)、\(\overline W=[[-2,6],[-2,8]]\)。同一规则落在 `src/autograd/tensor.py::Tensor.__matmul__`，推广到批广播。

::source examples/math_basics.py vjp

### 批次广播与两条参数路径

把手算例的两条输入放进独立的批次，每个批次只有一个位置。此时 \(X[2,1,2]\)、输出与上游 \(G[2,1,2]\)；投影矩阵 \(W[2,2]\) 和偏置 \(b[2]\) 都跨批次用同一份。令 \(b=[1,-1]\)，两条输出变成 \([3,4]\) 和 \([7,8]\)。每批分别得到矩阵梯度 \(X_b^\mathsf T G_b[2,2]\)，两批沿批轴求和，恰好等于先把 \(X,G\) 展平成 \([BT,2]\) 再用 [[08#matmul-vjp]]。输入梯度保留各自的批轴和位置轴；偏置同时沿 \(B,T\) 求和。

再给标量目标增加一个直接依赖 \(W\) 的二次项。记 \(\langle G,Y\rangle\) 为两个同形张量逐元素相乘后归约全部轴，则

$$ shared-affine
L=\langle G,Y\rangle+\tfrac14\sum_{ij}W_{ij}^2,\qquad
\overline W=X_{\mathrm{flat}}^\mathsf T G_{\mathrm{flat}}+\tfrac12 W.
$$

原数据路径给出 \([[-2,6],[-2,8]]\)，直接路径给出 \([[1,-0.5],[0,1.5]]\)，最后 \(\overline W=[[-1,5.5],[-2,9.5]]\)。\(\overline b=[0,2]\) 不变，输入梯度仍分别为 \([2,0]\) 与 \([-4,6]\)。这里的二次项专门展示同一变量的多个使用点；第 11 章的 AdamW 衰减发生在优化器更新，与这一损失构造分别计算。

运行 <code>python -m examples.gradient_labs 8</code>（M0-v2）。用张量引擎计算图上的两个路径，再用纯 NumPy 标量目标逐个扰动 \(W\) 的四项。该例用 \(h=10^{-5}\)，最大误差约 \(1.2\cdot10^{-10}\)，验收界为 \(10^{-8}\)。

::source examples/gradient_labs.py shared_affine

## 常见错误

- 使用 `W.T @ G` 更新输入，在维数偶然相同时可能静默得到错误值。
- 将广播梯度取平均，造成随批量或长度改变的额外缩放。
- 共享参数先后收到两次梯度时，用赋值覆盖第一项。

## 练习

1. **基础。** 给出 \(\overline X,\overline W,\overline b\) 的形状。
2. **推导。** 从 \(\mathrm d\mathcal L\) 展开 \(\overline W_{kj}\)。
3. **工程。** 用标量函数 \(\langle G,XW\rangle\) 差分检查 \(W_{00}\)。
4. **开放探索。** 如何验证添加一条共享路径后梯度确实相加？

<details markdown="1">
<summary>参考答案</summary>

1. \([N,C],[C,M],[M]\)。
2. 代入 \(\sum_{ij}G_{ij}\sum_k X_{ik}\mathrm dW_{kj}\)，交换有限求和，提取固定 \(k,j\) 的系数。
3. 中心差分预期 −2；改变一项权重而其他项固定。
4. 先对两支分别反传，再对两支的标量损失总和反传。三次均清空梯度，比较总梯度与两支梯度之和。
</details>

## 小结与下一章

链式法则把局部线性近似传回输入。为了确定梯度应降低的目标，[[09|第 9 章]] 首先从概率拆解完整序列。

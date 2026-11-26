# 附录：符号、公式与代码索引

## 数学符号

本书采用行向量右乘 \(XW\)。若参考文献写列向量 \(W' x^\mathsf T\)，则同一线性映射的参数关系为 \(W'=W^\mathsf T\)。矩阵批维在最前面，最后两轴做乘法，偏置通常沿前缀轴广播。

| 记号 | 含义 | 首次展开 |
| --- | --- | --- |
| \(B\) | 批中序列数 | [[02|张量]] |
| \(T,T_{\max}\) | 当前时间长度和最大上下文 | [[21|位置]] |
| \(V\) | 词表大小，在注意力上下文也用字母 V 表示值矩阵 | [[17|分词]]、[[22|注意力]] |
| \(C\) | 残差通道数 | [[03|向量]] |
| \(H,H_{\rm kv},D\) | 查询头数、KV 头数、每头通道 \(D=C/H\) | [[25|多头]]、[[36|GQA]] |
| \(F,L\) | FFN 隐藏宽度和块数 | [[26|FFN]]、[[28|GPT]] |
| \(x,y,z\) | 输入、目标或结果、归一化前分数，随公式定义 | [[01|总览]] |
| \(E,P\) | Token 与位置嵌入表 | [[18|嵌入]]、[[21|位置]] |
| \(Q,K,V; S,A\) | 查询、键、值；分数与归一化权重 | [[23|注意力]] |
| \(\bar x,G_x\) | 标量损失对 \(x\) 的梯度，与 \(x\) 同形 | [[08|VJP]] |
| \(\odot,\Vert\) | 按元素乘、拼接 | [[14|反向传播]]、[[17|分词]] |
| \(\eta,\beta_1,\beta_2,\epsilon\) | 学习率、矩系数、数值稳定项 | [[11|优化]] |
| \(\theta,\tau\) | 模型参数的集合、采样温度 | [[30|训练]]、[[31|生成]] |

书中微型设置 B=2、T=4、V=8、C=4、H=2、D=2、F=16、L=2，使用「春夏秋冬风雨晴雪」及 PCG64 种子 42。手算常取其子例并标明，用少数项写出完整归约。较大实测配置是 C=128、H=4、L=4、T=128，词表按训练集生成；结果见 [[34|项目章]]。

## 张量形状速查

| 阶段 | 输入或输出 | 归约或广播 |
| --- | --- | --- |
| Token 及标签 | [B,T] 整数 | 标签相对输入右移一个 |
| 嵌入相加 | [B,T,C] + [T,C] | 位置表沿 B 广播 |
| Norm | [B,T,C] | 沿 C 求均值或均方，增益与偏置沿 B、T 广播 |
| 线性投影 | [B,T,C] @ [C,M] → [B,T,M] | 沿 C 求和，加 [M] |
| 拆头 | [B,T,C] → [B,T,H,D] → [B,H,T,D] | reshape 后转置 |
| 分数 | [B,H,Tq,D] @ [B,H,D,Tk] → [B,H,Tq,Tk] | 沿 D，mask [Tq,Tk] 广播 |
| 权重 | [B,H,Tq,Tk] | 每个查询在 Tk 上归一化 |
| 上下文值 | [B,H,Tq,Tk] @ [B,H,Tk,D] → [B,H,Tq,D] | 沿 Tk 求和 |
| 合头与 FFN | [B,H,T,D] → [B,T,C] → [B,T,F] → [B,T,C] | 前者逆置换，后者逐位置 |
| LM Head | [B,T,C] @ [C,V] → [B,T,V] | 沿 C 求和 |
| 损失 | [B,T,V] 与 [B,T] → 标量 | 词表归一化，批与时间取有效均值 |

## 梯度速查

| 操作 | 反向 |
| --- | --- |
| \(Y=XW+b\) | \(G_X=G_YW^\mathsf T,\ G_W=X^\mathsf T G_Y,\ G_b=\sum_i G_{Y,i,:}\) |
| \(L=\operatorname{mean}(u^2)\) | 逐项 \(2u/N\) |
| gather | scatter-add，重复行沿命中位置相加 |
| \(p=\operatorname{softmax}(z)\) | \(G_z=p\odot(G_p-\sum_j G_{p,j}p_j)\)，归约轴与正向相同 |
| 平均交叉熵 | \(G_z=(p-u)/N\)，\(u\) 为 one-hot |
| LayerNorm | [[16#layernorm-grad]]，包含通道均值梯度 |
| Attention | [[25#mha-backward]]，禁止分数梯度为 0 |
| 残差 | 直路贡献 \(G_y\)，再加子层的 VJP |
| 拼接后按切片输入 | 每个输入收到对应输出区间 |

## Python / NumPy / PyTorch 速查

| 操作 | NumPy | PyTorch |
| --- | --- | --- |
| 构造数组或张量 | <code>np.asarray(data, dtype=np.float64)</code> | <code>torch.as_tensor(data, dtype=torch.float64)</code> |
| 堆叠数组 | <code>np.stack(sequence, axis=0)</code> | <code>torch.stack(sequence, dim=0)</code> |
| 按末轴平均并保形 | <code>x.mean(axis=-1, keepdims=True)</code> | <code>x.mean(dim=-1, keepdim=True)</code> |
| 交换 Q/K 的矩阵轴 | <code>k.swapaxes(-1, -2)</code> | <code>k.transpose(-1, -2)</code> |
| 将所有索引加到原表 | <code>np.add.at(table, ids, updates)</code> | <code>table.index_add_(0, ids, updates)</code> |
| 随机源 | <code>np.random.default_rng(42)</code> | <code>torch.Generator().manual_seed(42)</code> |
| 比较结果 | <code>np.testing.assert_allclose(a,b)</code> | <code>torch.testing.assert_close(a,b)</code> |

从仓库根目录用 <code>python -m package.module</code> 启动示例，确保模块导入路径一致。函数片段引用同文件中的定义；各章的命令是可直接运行的入口。[NumPy 2.3 文档](https://numpy.org/doc/2.3/) · [PyTorch 2.10 文档](https://docs.pytorch.org/docs/2.10/)

## 术语与索引

| 查找 | 英文 | 章节 |
| --- | --- | --- |
| 广播、步长、布局 | broadcasting, stride, layout | [[02|2]] |
| 基、范数、正交 | basis, norm, orthogonality | [[03|3]]、[[04|4]] |
| 向量–Jacobian 乘积 | vector–Jacobian product (VJP) | [[08|8]] |
| 对数和指数、困惑度 | logsumexp, perplexity | [[10|10]] |
| 动量、自适应矩估计 | momentum, adaptive moment estimation | [[11|11]] |
| 自动微分、计算图 | autodiff, computational graph | [[15|15]] |
| 层归一化、随机失活 | layer normalization, dropout | [[16|16]] |
| 字节对编码、词表 | BPE, vocabulary | [[17|17]] |
| 散射加、参数绑定 | scatter-add, weight tying | [[18|18]] |
| 位置表示、旋转 | positional encoding, RoPE | [[21|21]]、[[36|36]] |
| 因果遮罩 | causal mask | [[24|24]] |
| 前馈网络、解码器块 | FFN, decoder block | [[26|26]]、[[27|27]] |
| 教师强制、检查点 | teacher forcing, checkpoint | [[29|29]]、[[30|30]] |
| 贪心、核采样 | greedy, nucleus sampling | [[31|31]] |
| 预填充、解码、缓存 | prefill, decode, KV cache | [[35|35]] |
| 分组查询、混合专家 | GQA, MoE | [[36|36]] |
| 在线归约、推测解码 | online reduction, speculative decoding | [[37|37]] |
| 偏好优化、强化学习 | preference optimization, reinforcement learning | [[38|38]] |
| 探针、消融 | probe, ablation | [[39|39]] |

## 论文阅读路线

- 从构建块开始，阅读 Transformer 的 §3.2、§3.3、§3.5，手算两行分数，再对照 GPT-2 如何组织解码器。[Attention Is All You Need v7](https://arxiv.org/abs/1706.03762v7) · [GPT-2 原始实现](https://github.com/openai/gpt-2)
- 从公式对照优化器的矩状态和解耦衰减，再阅读字节/子词编码的边界及词向量负采样。[Adam v9](https://arxiv.org/abs/1412.6980v9) · [AdamW v3](https://arxiv.org/abs/1711.05101v3) · [子词 BPE v5](https://arxiv.org/abs/1508.07909v5) · [负采样](https://arxiv.org/abs/1310.4546v1)
- 读相对位置旋转与 KV 投影压缩，再检查 FlashAttention 的块统计。[RoFormer v5](https://arxiv.org/abs/2104.09864v5) · [GQA v3](https://arxiv.org/abs/2305.13245v3) · [DeepSeek-V2 v5](https://arxiv.org/abs/2405.04434v5) · [FlashAttention v2](https://arxiv.org/abs/2205.14135v2)
- 从示范监督、偏好对到奖励更新，逐一检查参考策略和旧策略的角色。[InstructGPT](https://arxiv.org/abs/2203.02155) · [DPO v3](https://arxiv.org/abs/2305.18290v3) · [PPO v2](https://arxiv.org/abs/1707.06347v2) · [DeepSeekMath v3](https://arxiv.org/abs/2402.03300v3)

## 故障排查

| 症状 | 检查 |
| --- | --- |
| 导入模块失败 | 在仓库根目录通过 <code>python -m</code> 运行；确认选中装过 requirements 的解释器 |
| 没有公式 | 运行 <code>npm ci</code> 后重新构建；浏览器检查 KaTeX CSS 与字体相对路径 |
| Python 安装 Torch 失败 | 使用有对应 wheel 的 Python 与 CPU 索引；基础 NumPy 实验独立可运行 |
| logits 非有限 | 检查全遮罩行、输入 ID、Norm epsilon 与数据类型 |
| loss 接近均匀基准 | 先确认标签移位、命中和模型参数更新，再做固定 batch 过拟合 |
| 修改未来影响过去 | 检查 mask 的查询、键轴和所有跨时间归约 |
| 续训偏离 | 比对语料指纹、学习率、批大小、矩状态、步号与 RNG |
| KV 正常但长生成发散 | 检查位置偏移和达到上限后是否重建窗口 |
| 生成含替换符 | 字符模型可能采到 UNK，字节模型可能生成非法 UTF-8 |

## 全书源文件与版本入口

每个里程碑有带源码 SHA-256 清单的可解压快照。下载后在解压目录按其中的 README 安装 Python 依赖并运行入口。快照可脱离教材仓库运行。

[M0-v1](downloads/M0-v1.zip) · [M1-v1](downloads/M1-v1.zip) · [M2-v1](downloads/M2-v1.zip) · [M3-v1](downloads/M3-v1.zip) · [M4-v1](downloads/M4-v1.zip) · [M5-v1](downloads/M5-v1.zip) · [M6-v1](downloads/M6-v1.zip) · [M7-v1](downloads/M7-v1.zip) · [M8-v1](downloads/M8-v1.zip)

[M0-v2 导数与矩阵 VJP](downloads/M0-v2.zip) · [M2-v2 自动微分](downloads/M2-v2.zip) · [M4-v2 注意力与 FFN](downloads/M4-v2.zip) · [M5-v2 Transformer 块与权重共享](downloads/M5-v2.zip) · [M6-v2 数据与训练](downloads/M6-v2.zip) · [M8-v2 完整合集](downloads/M8-v2.zip) · [M8-v3 探针对照](downloads/M8-v3.zip) · [M8-v4 长上下文与探针对照](downloads/M8-v4.zip) · [M8-v5 毕业项目验收](downloads/M8-v5.zip)

[M3-v2 UTF-8、BPE 与嵌入梯度](downloads/M3-v2.zip) · [M3-v3 共现、bigram 与位置表示](downloads/M3-v3.zip)

[M0-v3 有效目标的损失](downloads/M0-v3.zip) · [M2-v3 LayerNorm 梯度验收](downloads/M2-v3.zip)

[M1-v2 从输出梯度到参数更新](downloads/M1-v2.zip) · [M4-v3 查询、键与值的对照](downloads/M4-v3.zip)

[M7-v2 CPU 混合精度与归约](downloads/M7-v2.zip) · [M8-v6 路由、效率与策略目标](downloads/M8-v6.zip)

基线保留以下里程碑的独立运行入口。每组在现版本测试时持续复跑。数字参数逐章选择例子，源文件段落与 HTML 由构建器保持一致。

| 里程碑 | 章节 | 入口及关键源码 |
| --- | --- | --- |
| M0 | [[01|1]]–[[11|11]] | <code>python -m examples.preview</code>；<code>python -m examples.math_basics 2</code>（可换 2–11） |
| M1 | [[12|12]]–[[14|14]] | <code>python -m examples.networks 13</code>；<code>src/math_basics/mlp.py</code> |
| M2 | [[15|15]]–[[16|16]] | <code>python -m src.autograd.scalar</code>；<code>python -m examples.networks 15</code>；<code>src/autograd/tensor.py</code> |
| M3 | [[17|17]]–[[21|21]] | <code>python -m examples.language 17</code>（可换 17–21）；<code>python -m examples.tokenizer_embedding</code> 复算 [[17|17]]–[[18|18]]；<code>python -m examples.language_pipeline</code> 复算 [[19|19]]–[[21|21]]；<code>src/tokenizer/</code>、<code>src/math_basics/language.py</code> |
| M4 | [[22|22]]–[[26|26]] | <code>python -m examples.transformer 23</code>（可换 22–26）；<code>attention_weights</code>、<code>split_heads</code>、<code>gelu</code> |
| M5 | [[27|27]]–[[28|28]] | <code>python -m examples.transformer 28</code>；<code>transformer_block</code> 与 <code>GPT</code> |
| M6 | [[29|29]]–[[32|32]] | <code>python -m src.numpy_gpt.train fit --config configs/quick.json</code>；<code>examples.training</code> |
| M7 | [[33|33]]–[[34|34]] | <code>python -m examples.torch_alignment</code>；<code>python -m src.torch_gpt.train fit --config configs/larger.json --sdpa</code> |
| M8 | [[35|35]]–[[40|40]] | <code>python -m examples.modern 35</code>（可换 35–40）；<code>src/modern/</code> 与基线缓存；39 需按该章训练 NumPy checkpoint |

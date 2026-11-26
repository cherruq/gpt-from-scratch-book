# 《从零实现 GPT：数学、神经网络与大语言模型》教材建设任务书

## 1. 任务与目标
你是本项目的教材作者、数学教师、软件工程师和技术审校者。请在当前 Git 仓库中建设一本完整的中文 HTML 教材，而非仅输出目录、示例章节或项目脚手架。核心目标是让具有软件开发经验、但未系统学习机器学习的读者，从基本数学出发，最终独立实现、训练、验证并运行一个 Decoder-only GPT。不要假定读者已掌握微积分或线性代数。教材应达到一本技术专著的深度，建议 40 章、约 15–25 万中文字，篇幅以教学完整性为准，不以凑字数为目标。

最终读者应能解释每个参数的来源、形状、前向计算、梯度、训练目标和推理行为；能够从随机初始化开始训练小模型，并理解其与现代大模型的区别。不要把调用现成 Transformer API 当作从零实现。

## 2. 教学设计（强制）
采用目标驱动、螺旋式学习：先展示 GPT 全貌，再拆解实际计算，按依赖关系逐步学习，最后组合。数学章节必须优先采用后续 GPT 的真实运算作为例子，避免与目标无关的纯理论堆积。可以先展示尚未完全解释的公式，但标注后续解释章节，不能要求读者提前掌握。

贯穿全书的微型示例采用固定词表、固定随机种子、短序列和小矩阵。例如 B=2、T=4、V=8、C=4、H=2、D=2、FFN=16、L=2；这些是教学配置，不代表真实模型。逐步扩大到可训练配置，例如 C=128、H=4、L=4、T=128。所有例子明确区分实际计算结果与概念示意。建立统一符号表，采用行向量批处理约定 XW；说明与列向量 Wx 写法的关系。每个矩阵公式标明形状、轴含义、广播和求和方向。

每章至少包括：学习目标；前置知识和链接；本章在 GPT 中的位置；直观解释；严格定义与推导；一个可手算的数值例子；Python/NumPy 实现；常见错误；练习（基础、推导、工程、开放探索）；可折叠参考答案；本章小结及下一章衔接。适合交互的内容提供滑块、矩阵高亮、逐步执行或图形演示；不为装饰而制作动画。每章代码应可独立复现，并注明与主工程对应的文件、函数和版本。

## 3. 全书目录与必讲内容

### 第一篇：先看到完整目标
1. GPT 到底是什么：文本到 Token、概率、采样、自回归；展示最小 GPT 的端到端运行，区分训练与推理、模型与应用系统；给出后续依赖地图。
2. 计算机中的张量：标量、向量、矩阵、高维数组、shape、dtype、stride、索引、广播、批处理；用 [B,T,C] 和 [B,T,V] 作为例子；NumPy 基础与可复现实验环境。

### 第二篇：GPT 所需数学
3. 向量与向量空间：坐标、基、线性组合、维度、子空间、范数；Token 向量示例；不要把每个维度强行解释成单一语义。
4. 点积、距离与相似度：几何与代数推导、余弦相似度、投影、正交；用 Q/K 匹配分数及 Embedding 检索作为实例；交互显示角度与点积变化。
5. 矩阵与线性变换：矩阵乘法的行列解释、基变换、转置、复合变换、秩；用 XWq、XWk、XWv 和输出投影示范；手算完整小矩阵。
6. 函数与非线性：复合函数、指数、对数、ReLU、Sigmoid、GELU、SiLU；证明纯线性层堆叠仍是线性变换；对应 GPT FFN。
7. 导数与偏导：极限的直观意义、变化率、偏导、梯度、方向导数；从单个权重对预测误差的影响入手；数值差分。
8. 链式法则与矩阵微分：计算图、Jacobian、向量-Jacobian 乘积；推导 Y=XW 的 dX、dW，广播梯度及共享参数梯度累加；无需显式构造巨大 Jacobian。
9. 概率论：条件概率、联合概率、链式分解、期望、方差、分类分布；从序列概率 P(x1,...,xn) 推导自回归语言模型。
10. Softmax 与信息论：Logits、稳定 Softmax、logsumexp、熵、交叉熵、KL、最大似然、困惑度；推导 softmax-cross-entropy 梯度 p-y；以真实小词表演示。
11. 优化算法：梯度下降、SGD、mini-batch、Momentum、Adam、AdamW、学习率、梯度裁剪；用 GPT 的输出权重更新举例；展示不收敛的原因。

### 第三篇：神经网络如何学习
12. 单个神经元与线性模型：从随机权重到拟合；手写训练循环，观察 loss 和参数变化。
13. 多层感知机：XOR、非线性表达能力、隐藏层、参数计数、初始化；实现 Linear、激活函数、MLP。
14. 反向传播：逐节点推导，矩阵版本、批处理、参数共享；手写两层网络梯度并进行有限差分验证。
15. 自动微分引擎：DAG、拓扑排序、反向累积、标量 micrograd 风格实现；进一步实现支持 NumPy 张量的最小 autograd，覆盖 matmul、sum、reshape、transpose、broadcast、gather、exp、log 等必要算子。
16. 训练稳定性与泛化：初始化、梯度消失/爆炸、残差、LayerNorm、RMSNorm、Dropout、过拟合、验证集；推导归一化及其梯度；说明训练与推理模式。

### 第四篇：语言的数值表示
17. Tokenizer：字符、字节、Unicode、UTF-8、BPE、特殊 Token、encode/decode；从零实现 byte-level BPE，测试中文和代码往返；明确 Token 不等于词。
18. One-hot 与 Embedding：查表等价于 one-hot 矩阵乘法；随机初始化、gather、scatter-add 梯度、重复 Token 梯度累加；输出权重共享。
19. 从上下文学习语义：共现矩阵、CBOW、Skip-gram、负采样；实现小型 Word2Vec 实验；说明它是教学桥梁而非 GPT 必需组件；讨论静态与上下文表示。
20. 最小语言模型：unigram、bigram、Embedding+Linear；从计数模型到可训练概率模型；训练、采样、困惑度；明确能力边界。
21. 位置表示：为何无位置信息的 Attention 缺少顺序表达；学习式位置向量、正弦编码；RoPE 的二维旋转直觉及相对位置性质预告。

### 第五篇：Attention 与 Transformer
22. 从加权平均到 Attention：检索、加权求和、内容相关权重；从固定权重到可学习 Q/K/V，解释为什么需要三种投影，而非声称唯一设计。
23. Scaled Dot-Product Attention：完整推导 QK^T/sqrt(d)、Softmax、AV；方差缩放的假设；逐元素手算、交互矩阵和 NumPy 实现。
24. Causal Attention：掩码、未来信息泄漏、padding 与 causal mask 区别；训练并行与生成串行；验证修改未来 Token 不改变过去输出。
25. Multi-Head Attention：拆头、转置、合并、输出投影；[B,H,T,D] 形状跟踪；参数量与复杂度；手写前向和梯度。
26. Transformer FFN：逐位置计算、扩维、GELU、SwiGLU、参数量；解释与 Attention 的分工及其局限，不能将 MLP 简化成确定的知识字典。
27. Transformer Block：Pre-Norm、Post-Norm、残差、Attention、FFN；实现并验证完整 block；展示残差梯度路径。
28. 完整 GPT：Embedding、位置编码、堆叠 block、最终 Norm、LM Head、权重共享；完整参数计数、逐层张量形状和前向/反向传播。

### 第六篇：从零训练并运行 GPT（全书核心）
29. 数据管线：合法公开语料、清洗、分词、训练/验证划分、滑动窗口、batch、输入与标签错位；防止数据泄漏；提供本地小语料。
30. NumPy GPT 训练：使用前面自建 autograd 或手写梯度，禁止使用 PyTorch/JAX 自动微分；实现 loss、反向、优化器、checkpoint、恢复训练；先过拟合一个 batch，再训练小语料。
31. 文本生成：greedy、temperature、top-k、top-p、随机种子、EOS、上下文截断；实现采样，展示概率变化；说明低温与正确性并非等价。
32. 评估与调试：loss 曲线、perplexity、梯度检查、因果性检查、参数更新检查、数值稳定性、生成样例；区分记忆、泛化与幻觉；记录真实实验结果。
33. PyTorch 重实现：使用 Tensor/autograd 和基础算子重写相同 GPT，不直接调用 Transformer/MultiheadAttention；与 NumPy 共享权重和输入，对齐 logits、loss、梯度和单步更新；再展示工程优化。
34. 完整训练项目：CLI、配置、日志、断点恢复、生成接口、CPU 默认运行、可选 GPU；提供最小可复现实验与较大实验；明确硬件、时间和资源预算。

### 第七篇：从教学 GPT 到现代 LLM
35. 推理引擎：Prefill、Decode、KV Cache、位置偏移、缓存与无缓存结果一致性；实现 KV Cache，分析内存和计算复杂度，区分吞吐与延迟。
36. 现代架构：RoPE、RMSNorm、SwiGLU、MQA/GQA、MoE Router/Top-k/负载均衡；给出与基线 GPT 的逐项差异及最小实现；MLA 作为进阶专题，明确实现范围。
37. 训练与推理效率：参数量、FLOPs、显存、带宽、混合精度、量化、FlashAttention、分布式并行、Speculative Decoding；核心公式、适用条件和测量方法；不要求自写 CUDA 内核。
38. 后训练与推理模型：SFT、偏好优化、DPO、RLHF、RLVR、PPO/GRPO 的目标和差异；用小型可验证任务展示奖励与策略更新；区分训练时计算和推理时计算，不声称复现闭源模型。
39. 知识与可解释性：Embedding/隐藏状态、探针、消融、注意力可视化的局限、记忆与泛化；以自建 GPT 做可重复实验；区分已知结论、假设和开放问题。
40. 综合毕业项目：从空目录独立构建 GPT；设计、实现、训练、评估、推理、性能分析和技术报告；提供验收清单、参考实现和扩展挑战。RAG、工具调用、Agent 仅作为系统层扩展，不喧宾夺主。

附录：数学符号表、张量形状速查、梯度公式、Python/NumPy/PyTorch 速查、术语中英对照、论文阅读路线、故障排查、索引、全书代码映射。

## 4. 贯穿全书的代码演进
采用同一套符号、接口和测试数据，按里程碑逐步演进：M0 张量与数学实验；M1 MLP 手动梯度；M2 张量 autograd；M3 tokenizer/embedding/bigram；M4 单头和多头 attention；M5 transformer block；M6 NumPy GPT 完整训练；M7 PyTorch 对齐与训练；M8 KV Cache 和现代架构扩展。每个里程碑保留可运行快照或明确版本标签，不让后续重构破坏前面章节。核心 GPT 基线使用学习式位置编码、Pre-LayerNorm、GELU、标准 MHA，现代变体独立实现。禁止把尚未讲解的复杂库作为核心实现捷径。标准库、NumPy、测试框架可用；PyTorch 仅在相应篇章使用及作为独立参考验证。

## 5. HTML 产品要求
- 输出完整静态网站，中文为主，英文术语首次出现时附原文。建议 Markdown 源文件 + Python 构建脚本 + HTML/CSS/JS，最终 dist/index.html 可作为入口。具体框架由 Codex 在 ADR 中说明；优先低依赖、可离线阅读、无需后端。
- 支持章节目录、上一篇/下一篇、章内目录、全局搜索、公式编号、交叉引用、代码复制、明暗主题、移动端适配、打印样式、学习进度（本地存储即可）。不要依赖外部 CDN 才能显示公式或核心内容；本地打包 KaTeX/MathJax 等依赖并保留许可。
- 公式采用 LaTeX，代码语法高亮，SVG/Canvas 用于计算图、矩阵、Attention、梯度下降等真实教学演示。交互组件提供重置、初始值、说明、键盘可用性及静态替代说明。避免生成无实际功能的按钮。
- HTML 代码片段与实际源码保持单一来源或自动校验；提供完整可复制运行命令、依赖版本和预期结果。浏览器中的小型 JS 实验可以直接运行，Python 实验提供本地运行方式，不伪装成已执行的代码。
- 推荐交互实验：矩阵变换、点积角度、链式法则计算图、梯度下降、Softmax 温度、Embedding 查表及梯度累加、BPE 合并、QKV 逐步运算、因果掩码、多头 reshape、残差路径、训练 loss、采样、KV Cache 内存估算。交互数值须与参考实现一致。

## 6. 仓库结构（建议）
```text
README.md
AGENTS.md
PLAN.md
PROGRESS.md
DECISIONS.md
book/                 # 章节源文件、术语表、引用、习题
src/                  # 逐步演进的参考实现
  math_basics/
  autograd/
  tokenizer/
  numpy_gpt/
  torch_gpt/
  modern/
examples/             # 章节实验与运行入口
assets/               # 图形、交互组件、本地依赖
data/                # 小语料、来源及许可证
tests/                # 数学、模型、网站测试
scripts/              # 构建、校验、实验脚本
dist/                 # 生成的静态 HTML
```
README 提供从安装到构建、训练、阅读的完整命令。AGENTS.md 固化本任务书的约束；PLAN.md 维护章节依赖和里程碑；PROGRESS.md 记录每章状态、实验结果、未完成事项和下一步；DECISIONS.md 记录技术选择及原因。

## 7. 验证与验收
- 数学：有限差分与解析梯度对齐（float64，避开不可导点）；明确绝对/相对误差阈值及理由。所有数值示例由脚本复算。
- 模型：检查形状、参数量、Softmax 行和、掩码、未来不泄漏、重复索引梯度累加、共享权重梯度、数值稳定性；对齐 NumPy 与 PyTorch 的前向、loss、梯度及优化一步结果。
- 训练：固定随机种子，在小数据上验证 loss 下降、单 batch 可过拟合、checkpoint 恢复和生成可运行；验证集指标真实记录，不虚构训练结果或承诺小模型具备通用智能。
- 推理：缓存与非缓存 logits 在容差内一致；采样边界、EOS、上下文限制有测试。
- 网站：所有 40 章及附录实际存在，无 TODO 占位、断链、缺图、未渲染公式；桌面/移动端抽查；交互和搜索可用；离线检查；所有章节代码运行或明确标注资源受限未运行。
- 内容：逐章审校数学、代码和术语；避免循环依赖、未解释符号、误导性类比；引用原始论文和官方文档，记录版本和来源；不复制受版权保护教材的大段文字。
- 建议统一命令（按实际实现调整并写入 README）：python -m pytest；python scripts/build.py；python scripts/check_book.py；python -m src.numpy_gpt.train --config ...。提供 CPU 快速测试与可选长实验，不默认启动昂贵训练或下载巨量数据。

## 8. Codex 执行策略
1. 先检查当前仓库，保留已有用户文件，建立章节依赖图、技术选型、统一符号和贯穿示例，创建计划与进度文件。
2. 先完成第一章和一个端到端最小参考模型及网站构建链，验证技术方案；参考模型可提前存在，但教材必须按教学依赖逐步解释。
3. 按篇章、里程碑分批实现；每批同时完成正文、公式、例子、源码、测试和 HTML，而非先生成全部空页面。
4. 每批执行测试和构建，修复错误后更新 PROGRESS.md；若允许 Git 提交，按里程碑提交，不自动推送或覆盖用户改动。
5. 若会话上下文或运行预算不足，将准确进度、失败命令、未完成章节和下一步写入 PROGRESS.md。后续会话读取本任务书、AGENTS.md、PLAN.md、PROGRESS.md 后继续，避免重复生成已完成内容。不要将分批策略当作停止在大纲的理由。
6. 不需要为普通可逆的实现细节反复询问用户；自行选择合理默认并记录。涉及高成本计算、外部服务、不可逆操作、许可不清的数据或重大范围变更时先征求确认。
7. 最终交付完整 HTML 网站、全部源文件、可运行模型、测试报告、构建和阅读说明。列出实际未验证项目、限制及后续可选扩展；只有全部必需章节和验收条件完成才标记项目完成。

## 9. 推荐原始资料
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- GPT-2: https://github.com/openai/gpt-2
- Word2Vec: https://arxiv.org/abs/1301.3781
- Adam: https://arxiv.org/abs/1412.6980
- AdamW: https://arxiv.org/abs/1711.05101
- RoPE: https://arxiv.org/abs/2104.09864
- FlashAttention: https://arxiv.org/abs/2205.14135
- micrograd: https://github.com/karpathy/micrograd
- Neural Networks: Zero to Hero: https://github.com/karpathy/nn-zero-to-hero
- LLMs from Scratch: https://github.com/rasbt/LLMs-from-scratch
- NumPy: https://numpy.org/doc/ ; PyTorch: https://pytorch.org/docs/
以上为参考入口，写作时核实论文、版本、公式和许可证。教材必须原创组织、推导和实现，不是参考资料的拼接。

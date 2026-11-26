# 从零实现 GPT

面向有软件开发经验的读者的中文静态教材。主线从下一 Token 预测出发，经过数学、手写梯度、分词和解码器块，完成小模型训练、生成和验证。

## 离线阅读

在浏览器直接打开 [dist/index.html](dist/index.html)。本地公式、导航、搜索、学习进度、明暗主题与交互无需后端。附录提供全部章节和 M0–M8 代码快照。源码在 [book/chapters/](book/chapters/)，运行数据与源码摘录在构建时嵌入。

完整交付包由 <code>python scripts/package_book.py</code> 生成到 <code>release/</code>。解压后可直接打开 <code>dist/index.html</code>。包内还有源码、语料、配置、测试、验证报告和逐文件 SHA-256 清单。重建或修改实验时按下文安装运行依赖。

## 安装与构建

需要 Python 3.11 或更新版本、Node.js 与 npm。已验证 Python 3.14.4、NumPy 2.3.5、Node 24.18.0。以下命令均在仓库根目录执行。

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm ci
python scripts/build.py
python scripts/check_book.py --full
~~~

## NumPy 实验

~~~bash
python -m examples.preview
OPENBLAS_NUM_THREADS=1 python -m src.numpy_gpt.train fit --config configs/quick.json --checkpoint runs/quick.npz
python -m src.numpy_gpt.train generate --checkpoint runs/quick.npz --prompt 今早我 --top-k 8 --cache
python -m examples.modern 39 --checkpoint runs/quick.npz
~~~

固定批 80 步平均损失约 2.090428 → 0.015276。语料训练 120 步后验证 NLL 约 1.53047。语料划分及许可见 [data/README.md](data/README.md)。使用续训时提供和原训练一致的配置、原检查点路径与目标总步数。

~~~bash
OPENBLAS_NUM_THREADS=1 python -m src.numpy_gpt.train fit --config configs/quick.json --checkpoint runs/quick.npz --steps 150 --resume
~~~

### 第二轮手算与训练诊断

~~~bash
python -m examples.gradient_labs 7
python -m examples.attention_labs 25
python -m examples.training_labs 29
OPENBLAS_NUM_THREADS=1 python -m examples.training_labs 30
~~~

三个入口覆盖第 7、8、15 章，第 23–28 章以及第 29–30 章。对应章节可替换命令后的章号。训练诊断包括全局裁剪、错误标签对照、检查点恢复与验证窗口分组，临时检查点由脚本自动清理。附录提供 M0-v2 至 M8-v2 对应实验档案。

### 探针对照与长序列推理

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.probe_controls
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.kv_benchmark
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.kv_benchmark --long
~~~

探针入口按快速配置复训，比较随机标签、同种子未训练模型和两种新模板，临时检查点随进程清理。KV 基准默认运行短序列冒烟测试。长序列参数每种模式测 256 次 decode，保留分配峰值、缓存负载与原始计时样本。实测记录见 [探针对照](reports/probe-controls.json) 和 [长上下文](reports/kv-long-context.json)。附录的 M8-v4 快照包含两个入口。

### 损失、归一化与独立实现验收

~~~bash
python -m examples.loss_lab --batch 2 --time 3 --vocab 4 --seed 17
python -m examples.norm_lab --batch 1 --time 2 --channels 3 --epsilon 0.001 --seed 17
python -m examples.capstone_check
python -m examples.capstone_check --seed 17 --token-seed 21
~~~

前两个入口可替换张量尺寸和种子，分别核对有效目标损失与 LayerNorm 三组梯度。毕业项目入口检查因果性、固定批训练和贪心生成。把 [验收脚本](examples/capstone_check.py) 放进单独目录，可以用 <code>python capstone_check.py --candidate my_decoder:make_model</code> 测试自己的模型。工厂接口见 [第 40 章](dist/40.html)，参考实测见 [capstone-context.json](reports/capstone-context.json)。

### 从 UTF-8 到 Embedding

~~~bash
python -m examples.tokenizer_embedding
python -m examples.tokenizer_embedding --text 春春 --train 春 --merges 1
python -m examples.tokenizer_embedding --text 'baba春🐍' --train ab --train ab --train ba --merges 2
~~~

第 17–18 章共用这个入口。每个 <code>--train</code> 参数代表一条记录，<code>--merges</code> 指定合并上限。输出串起码点与 Token 的字节区间、训练和编码的逐轮变化、查表结果及重复 ID 的梯度。附录的 M3-v2 快照可独立运行。复算见 [tokenizer-embedding.json](reports/tokenizer-embedding.json)。

### 从共现和 bigram 到位置表示

~~~bash
python -m examples.language_pipeline
python -m examples.language_pipeline --radius 2
python -m examples.language_pipeline --train ab --train ba --eval aa --eval '🐍' --merges 0 --radius 2 --alpha 0.1 --seed 7
~~~

第 19–21 章共用相同的分词和记录边界。实验依次给出共现、负采样与梯度差分、留出预测概率，以及真实 GPT 输入的同 Token 跨位置差值。默认用「春春」和「春」训练，用「春春春」和空文本留出评估。附录的 M3-v3 快照可独立运行。复算见 [language-pipeline.json](reports/language-pipeline.json)。

### 输出梯度与 Q/K/V 对照

~~~bash
python -m examples.training_bridge
python -m examples.training_bridge --rate 0.3
python -m examples.retrieval_lab
python -m examples.retrieval_lab --logit-gap 0
~~~

第 11–14 章从交叉熵和输出权重到手写两行 MLP 的整步更新。第 22 章分别改变查询、键、值，比较权重和载荷。附录提供 M1-v2 和 M4-v3。

## PyTorch 实验

先安装适合环境的 PyTorch 2.10。已实测的 Linux CPU 包可以按以下方式安装。

~~~bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
python -m examples.torch_alignment
python -m examples.torch_runtime
python -m examples.torch_runtime --gloo
python -m src.torch_gpt.train fit --config configs/larger.json --checkpoint runs/larger.pt --sdpa
python -m src.torch_gpt.train generate --checkpoint runs/larger.pt --prompt 今早我 --top-k 8
~~~

CPU 较大配置约 82.6 万参数，在 i7-10700 单线程实测约 9 秒，1000 余个验证目标。运行时实验比较一次 bfloat16 autocast 更新。<code>--gloo</code> 需要支持 Gloo 的 PyTorch，用两个本机 CPU 进程验证梯度归约。GPU 训练可用兼容的发行包和 <code>--device cuda</code>，本次未实测。较大运行及训练、评估边界见 [测试报告](reports/validation.md)。

### 现代架构与策略目标

~~~bash
python -m examples.modern_objectives
~~~

第 36–38 章复算 GQA 的投影和 KV 字节数、路由辅助梯度、在线注意力、推测采样补样和 PPO/GRPO 的小型目标对照。M7-v2 和 M8-v6 各有独立入口。

## 验证

~~~bash
python -m pytest -q
python scripts/build.py
python scripts/check_book.py --full
python scripts/check_examples.py --torch
BOOK_CHROME=/usr/bin/google-chrome node tests/browser.cjs
python scripts/package_book.py
~~~

未安装 PyTorch 时，pytest 跳过该组，复跑章节入口时去掉 <code>--torch</code>。浏览器用已安装的 Chromium/Chrome，通过 <code>BOOK_CHROME</code> 指定可执行文件，也可用 Playwright 安装的 Chromium。测试将网站作为本地文件打开，禁用外网，覆盖桌面和移动视口。章节与快照复跑使用独立临时目录，结束后清理。

## 维护与交付证据

- [任务书](TEXTBOOK_SPEC.md)
- [维护约定](AGENTS.md)、[建设计划](PLAN.md)、[当前进度](PROGRESS.md)、[技术决策](DECISIONS.md)
- [测试报告](reports/validation.md)、[逐章审校](reports/chapter-audit.md)、[各章与快照复跑输出](reports/examples.json)、[浏览器记录](reports/browser.json)

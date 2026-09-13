# LLM 学习总路线（第一性原理优化版）

> 硬件：Mac M4 / 16GB / MPS。时间预算：每天 1~2 小时（每周约 10 小时，每 4 周一个阶段 ≈ 40 小时）。
> 本文档是总纲；每日任务见各阶段计划文档。所有阶段共用同一套机制：**每天"看→练→验收"，每周日缓冲/复盘，每阶段末关书重做**。

## 一、终点定义：什么叫"学会了 LLM"

学习路线必须先有可验证的终点，否则永远学不完。本路线的终点能力（约 4 个月后）：

1. **能手写**：从空文件写出一个可训练的 GPT（字符级，莎士比亚语料），并讲清每一层（embedding、attention、残差、LayerNorm、采样）在做什么、为什么。
2. **能复现**：在本地或云端跑通"预训练 → SFT 微调"全流程的小模型（MiniMind / nanoGPT 规模），知道 loss 曲线每一步意味着什么。
3. **能应用**：用 Hugging Face 生态完成一次真实微调（LoRA）和一个 RAG 小应用。

原路线（网上通用版）的问题是没有终点、没有验收，资料堆砌。本路线把一切都挂在上面三个终点上。

## 二、第一性原理取舍：为什么砍掉原路线的一半内容

原则：**只保留终点能力的因果链上的内容；数学按需学（just-in-time），不单独设阶段；重复资料只留一套。**

| 原路线内容 | 处理 | 理由 |
|---|---|---|
| 3Blue1Brown 数学全套（线代 16 集 + 微积分 12 集 + 概率） | **砍**，改为按需看 | 数学是工具不是门槛。阶段一训练中遇到链式法则/矩阵乘法时，看对应那几集即可，单独花 2-4 周学数学会严重推迟正反馈 |
| 吴恩达 ML Specialization、Fast.ai、Goodfellow《深度学习》 | **砍** | 与 d2l + Zero to Hero 内容高度重复，同一件事学三遍是最常见的半途而废原因 |
| Word2Vec、RNN/LSTM | **降级为选读** | 现代 LLM 关键路径上没有它们；makemore 第 5 讲（WaveNet）已覆盖序列建模的核心直觉 |
| BERT / T5 架构对比 | **压缩为 1 天阅读** | GPT 路线之外的知识，知道差异即可，不配练习 |
| nanochat 全流程复现 | **降级为云端可选** | 需 8×H100（约 $100），16GB Mac 物理上跑不了；作为阶段五的压轴可选项 |
| LangChain / LlamaIndex / 向量数据库全家桶 | **收敛** | 应用层只学 HF + 轻量方案（见阶段四），框架会过时，原理不会 |
| Karpathy Zero to Hero | **保留并升格为主线** | 唯一同时覆盖终点 1 和终点 2 的资料，难度递增设计极好 |
| MiniMind | **保留** | 中文 + 低资源全流程（预训练/SFT/DPO），正好适配 M4 16GB |
| 论文《Attention Is All You Need》 | **保留** | 但放在手写 GPT **之后**读——先会写再读论文，顺序反了会读不动 |

**阶段间依赖关系**（不能跳级的硬依赖）：

```
阶段一 PyTorch 核心 ──→ 阶段二 语言模型从零构建 ──→ 阶段三 Transformer/GPT ──→ 阶段四 全流程+应用
 (张量/autograd/训练循环)   (micrograd→makemore)        (手写GPT+分词器+论文)      (MiniMind/LoRA/RAG)
```

## 三、硬件现实（约束条件，不接受幻想）

| 任务 | M4 16GB 可行性 |
|---|---|
| nanoGPT 字符级莎士比亚（~1M 参数） | ✅ MPS 可跑，分钟级 |
| MiniMind 26M 预训练（小语料） | ✅ 可行，需调小 batch |
| GPT-2 124M 微调（LoRA） | ⚠️ 勉强，小 batch + 短序列 |
| GPT-2 从零预训练 | ❌ 需云端 GPU（AutoDL / Colab） |
| nanochat 完整流程 | ❌ 官方配置 8×H100，仅云端可选 |

**规则：本地跑不通的实验不硬跑，标记为"云端实验"，攒到阶段五统一处理。**

## 四、阶段总览

| 阶段 | 周 | 目标 | 阶段验收 | 计划文档 |
|---|---|---|---|---|
| 一：PyTorch 核心 | 1-4 | 张量/autograd/训练循环/CNN | 独立写 MNIST 全流程；CIFAR-10 ≥70% | `docs/前4周每日计划.md` |
| 二：语言模型从零构建 | 5-8 | micrograd → makemore 五讲 | 关视频重写 makemore MLP 并讲清反向传播 | `docs/阶段二每日计划.md` |
| 三：Transformer 与 GPT | 9-12 | 手写 GPT、分词器、读原始论文 | 从零写出可训练的字符级 GPT；跑通 nanoGPT | `docs/阶段三每日计划.md` |
| 四：全流程与应用 | 13-16 | MiniMind 中文全流程、LoRA、RAG | 完成一次真实微调 + 一个 RAG 小应用 | 见下方周计划 |
| 五：前沿深耕（长期可选） | 17+ | nanochat 云端复现、DPO/RLHF、部署 | — | 见下方说明 |

### 阶段四：周计划（第 13-16 周，里程碑粒度）

- **第 13 周**：读 `minimind/README.md` 与 `model/` 源码（对比你手写的 GPT）；`pip install -r minimind/requirements.txt`；跑通 MiniMind 预训练小样例（`scripts/` 里的 train_pretrain，调小 batch_size）。验收：loss 正常下降，能说出与你手写 GPT 的三处差异。
- **第 14 周**：MiniMind SFT 微调 + 用 `eval_llm.py` 对话；选读 DPO 脚本。验收：微调前后同一 prompt 的输出对比，写进 `notes/`。
- **第 15 周**：HF 生态实战——`transformers` 加载一个开源小模型（如 Qwen 0.5B），用 PEFT 做 LoRA 微调。验收：跑通一次 LoRA 训练并保存 adapter。
- **第 16 周**：RAG 小应用——本地文档向量化（sentence-transformers + FAISS）+ 检索结果喂给模型。验收：对自己的 `notes/` 目录做一次问答，结果合理。

### 阶段五：长期可选（不设时间表）

- **nanochat 云端复现**（压轴）：`nanochat/` 已在本地，先读 `README.md`；需要时租 8×H100（约 $100）跑完整流程：Rust 分词器 → FineWeb 预训练 → SFT → 可选 GRPO → WebUI。
- **方向深耕**：对齐（DPO/RLHF，可读 minimind 的 DPO 脚本入门）、推理部署（Ollama 本地跑开源模型、vLLM）、多模态。
- **信息源**：arXiv、Hugging Face 社区、Karpathy 的新项目。

## 五、环境与资料清单

| 环境/资料 | 位置 | 用于 |
|---|---|---|
| `pytorch_env`（conda） | 已装好 | 阶段一 |
| `llm_env`（conda） | 已装好（torch + transformers + datasets + tiktoken + jupyter） | 阶段二起，激活：`conda activate llm_env` |
| d2l-zh | `d2l-zh/` | 阶段一主教材，后续按需查 |
| thorough-pytorch | `thorough-pytorch/` | 阶段一辅助 |
| Zero to Hero 代码 | `nn-zero-to-hero/lectures/` | 阶段二、三主线（视频在 YouTube/B 站搜 "Karpathy Zero to Hero"） |
| minbpe | `minbpe/` | 阶段三分词器练习 |
| nanoGPT | `nanoGPT/` | 阶段三实战 |
| MiniMind | `minimind/` | 阶段四主线 |
| nanochat | `nanochat/` | 阶段五可选（云端） |
| 数据集 | `data/` | MNIST / CIFAR-10 |
| 实验记录 | `notes/` | 每阶段的 loss 曲线、对比实验、总结都存这里 |

## 六、通用规则（继承自阶段一，全程适用）

1. 代码必须手敲；视频可以 1.25-1.5 倍速，但代码必须自己跟着写。
2. 每天看与练约各占一半；看视频超过当天一半时间就说明今天没练够。
3. 遇到报错先读 traceback 15 分钟再搜。
4. 每周日缓冲；每阶段最后一天"关书重做"——做不出就回炉，不带病进入下一阶段。

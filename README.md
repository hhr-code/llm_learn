# llm_learn

Mac (Apple M4 / 16GB) 上从零学习 LLM 的工作目录。总目标与路线见 `docs/LLM学习总路线.md`。

## 环境

```bash
# 阶段一（PyTorch 基础，第 1-4 周）
conda activate pytorch_env

# 阶段二起（Zero to Hero / nanoGPT / MiniMind，第 5 周+）
conda activate llm_env

cd ~/code/llm_learn
```

- `pytorch_env`：Python 3.11 + PyTorch（MPS 已验证可用）+ torchvision / jupyter / matplotlib / pandas
- `llm_env`：Python 3.11 + PyTorch 2.14（MPS 已验证）+ transformers 5.17 / datasets / tiktoken / tokenizers / accelerate / jupyter
  - 重建方式：`nohup ./install_llm_env.sh &`（走清华镜像直连，不走本地代理）
- 若遇到 MPS 算子报错：`export PYTORCH_ENABLE_MPS_FALLBACK=1`

## 目录结构

| 路径 | 内容 |
|------|------|
| `docs/LLM学习总路线.md` | **总路线（先读这个）**：终点定义、取舍理由、阶段总览 |
| `docs/前4周每日计划.md` | 阶段一每日任务清单（PyTorch 核心，第 1-4 周） |
| `docs/阶段二每日计划.md` | 阶段二每日任务清单（Zero to Hero 1-6 讲，第 5-8 周） |
| `docs/阶段三每日计划.md` | 阶段三每日任务清单（手写 GPT + 分词器 + 论文，第 9-12 周） |
| `code/` | 阶段一练习代码（已验证可运行） |
| `data/` | MNIST、CIFAR-10 数据集（已下载，418MB） |
| `d2l-zh/` | 动手学深度学习中文版（阶段一主教材，在线版 zh.d2l.ai） |
| `thorough-pytorch/` | PyTorch Handbook（阶段一辅助） |
| `nn-zero-to-hero/` | Karpathy Zero to Hero 配套代码（阶段二、三主线） |
| `minbpe/` | Karpathy 分词器练习仓库（阶段三） |
| `nanoGPT/` | Karpathy nanoGPT（阶段三实战） |
| `minimind/` | MiniMind 中文小模型全流程（阶段四主线） |
| `nanochat/` | Karpathy nanochat（阶段五可选，需云端 GPU） |
| `notes/` | 每阶段的实验记录、loss 曲线、总结（自己积累） |

## 学习主线

PyTorch 核心（阶段一）→ 语言模型从零构建（阶段二）→ 手写 Transformer/GPT（阶段三）→ 中文全流程 + LoRA + RAG（阶段四）→ nanochat 云端复现等前沿深耕（阶段五，可选）。

原则：**先直觉后严格、先应用后原理、以战代学**——看视频建立直觉，手写代码形成理解，关书重做作为验收。

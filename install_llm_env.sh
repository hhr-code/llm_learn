#!/bin/bash
# llm_env 依赖安装脚本（nohup 后台运行，日志写到同目录 llm_env_install.log）
# 不走本地代理（127.0.0.1:3247 太慢），改用清华 PyPI 镜像直连
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

PIP=~/miniforge3/envs/llm_env/bin/pip
PY=~/miniforge3/envs/llm_env/bin/python

$PIP install --timeout 60 --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch torchvision transformers datasets tiktoken tokenizers accelerate tqdm matplotlib jupyter
$PY -c "import torch, transformers, tiktoken, datasets; print('VERIFY_OK torch', torch.__version__, '| MPS:', torch.backends.mps.is_available(), '| transformers', transformers.__version__)"

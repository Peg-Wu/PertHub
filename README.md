<u>**本仓库 (PertHub) 是我的手撕代码，主要复现一些常见的药物扰动模型，使用 HuggingFace 生态**</u>

## Installation

### Environment Setup

### Step 1: Set up a python environment

We recommend creating a virtual Python environment with [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/):

- Required version: `python >= 3.11`

```bash
conda create -n perthub python=3.11
conda activate perthub
```

### Step 2: Install pytorch

Install `PyTorch` based on your system configuration. Refer to [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

For the exact command, for example:

- You may choose any version to install, but make sure the PyTorch version is not too old.
- We recommend `torch ≥ 2.6`.

```bash
# Installation Example: torch v2.7.1
# CUDA 11.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install deepspeed (optional)

Install `DeepSpeed` based on your system configuration. Refer to [DeepSpeed installation instructions](https://www.deepspeed.ai/tutorials/advanced-install/).

For the exact command, for example:

```bash
pip install deepspeed
```

### Step 4: Install perthub and dependencies

To install `perthub`, run:

```bash
git clone https://github.com/Peg-Wu/PertHub.git
cd PertHub
pip install [-e] .
```
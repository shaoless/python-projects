# Mini-GPT 训练项目

在本地电脑上从零理解 AI 大模型训练。包含两个完整流程：

1. **从零训练 Mini-GPT** — 纯 PyTorch 实现，25M 参数 Transformer
2. **LoRA 微调** — 工业界标准做法，微调 Qwen2.5-0.5B / SmolLM2

## 硬件要求

- **最低**：8GB 显存 GPU（RTX 2060/3060/4060）+ 16GB 内存
- **推荐**：RTX 4060 8GB（本项目开发用机）
- 纯 CPU 也能跑（慢几十倍），设置 `CUDA_VISIBLE_DEVICES=""`

## 快速开始

### 0. 配置国内镜像（推荐）

PyTorch 2.5GB 直连下载很慢，用国内镜像快 10 倍以上。

**pip 镜像（选一个）：**

```bash
# 清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 或者临时使用（不修改全局配置）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**PyTorch 专用（清华 tuna 镜像，~2GB，国内下载快很多）：**

```bash
# 方法 1：用清华镜像
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 方法 2：如果方法 1 还是慢，下 whl 文件手动安装
# 先去 https://download.pytorch.org/whl/cu124 找到对应文件，用迅雷/IDM 下载，然后：
pip install torch-2.6.0+cu124-cp312-cp312-win_amd64.whl
```

**HuggingFace 镜像（下载模型和数据集用）：**

```bash
# 设置环境变量（临时生效）
set HF_ENDPOINT=https://hf-mirror.com

# 或者加到系统环境变量永久生效
# 之后 python data/download.py 和 train_lora.py 都会走镜像
```

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载训练数据

```bash
python data/download.py tinystories    # TinyStories（推荐，小模型友好）
python data/download.py wikitext       # WikiText-2（更正式）
python data/download.py alpaca         # Alpaca 指令数据（给 LoRA 用）
```

### 3. 从零训练 Mini-GPT

```bash
# 快速测试 100 步（~2 分钟）
python scripts/train_gpt.py --max_steps 100 --eval_interval 50

# 完整训练 10000 步（RTX 4060 约 30-60 分钟）
python scripts/train_gpt.py --max_steps 10000 --batch_size 8 --grad_accum 4
```

### 4. 生成文本

```bash
python scripts/generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"
```

### 5. LoRA 微调

```bash
python scripts/train_lora.py --model HuggingFaceTB/SmolLM2-360M --max_steps 500
```

## 完整训练

```bash
# 从零训练 10000 步（RTX 4060 约 30-60 分钟）
python scripts/train_gpt.py --max_steps 10000 --batch_size 8 --grad_accum 4

# 查看训练曲线
tensorboard --logdir logs
```

## 项目结构

```
├── src/
│   ├── config.py      # 模型和训练配置
│   ├── model.py       # Mini-GPT 模型（纯 PyTorch）
│   ├── tokenizer.py   # BPE 分词器训练/加载
│   ├── dataset.py     # 数据集封装
│   └── trainer.py     # 训练循环（AMP + 梯度累积 + checkpoint）
├── scripts/
│   ├── train_gpt.py   # 从零训练入口
│   ├── train_lora.py  # LoRA 微调入口
│   └── generate.py    # 文本生成
├── data/
│   └── download.py    # 下载数据集
└── checkpoints/       # 模型保存目录
```

## 关键技术

| 技术 | 说明 |
|------|------|
| **混合精度 (AMP)** | bfloat16 训练，加速 ~2x，显存减半 |
| **梯度累积** | 4 步累积 → 模拟 4 倍 batch size |
| **余弦学习率** | Warmup 500 步 → Cosine decay |
| **QLoRA** | 4-bit 量化基座模型，只训练 adapters（~1% 参数） |
| **Flash Attention** | 如果 PyTorch 版本支持，自动启用 |

## Windows 注意

- `bitsandbytes` 在 Windows 上可能不工作，QLoRA（4-bit）会失败
- 解决：`train_lora.py` 不传 `--use_4bit`，用全精度 LoRA，约需 3-4GB 显存

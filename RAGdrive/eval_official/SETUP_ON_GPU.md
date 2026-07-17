# 在 GPU 机器上建 ragdrive 环境 + 跑完整评测

> 本机（Intel 核显，无 NVIDIA GPU）不能跑步骤②的 VLM 推理，改到有 NVIDIA GPU（≥24GB，如 3090/4090）的机器上。
> 把整个 `D:\lhy\RAG4drive` 拷过去（含已下载的 data/ 数据）。

## 1. 建环境

```bash
conda create -n ragdrive python=3.10 -y
conda activate ragdrive

# 深度学习（GPU 版 torch —— 按 GPU 机器的 CUDA 版本选，下面是 CUDA 12.1 示例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece

# 检索
pip install FlagEmbedding hnswlib rank_bm25 qwen-vl-utils pandas pyarrow

# 评估（官方口径打分需要）
pip install nltk rouge-score pycocoevalcap openai
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

验证 GPU 可用：
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 2. 环境变量（Linux GPU 机器）

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export HF_ENDPOINT=https://hf-mirror.com      # 国内镜像，可选
export OPENAI_API_KEY=sk-...                   # 步骤④完整分需要
# export OPENAI_BASE_URL=...                    # 本地/兼容端点，可选
```

## 3. 跑完整评测（4 步）

```bash
cd RAGdrive/eval_official

# ① 准备评测集（CPU，秒级）
python prepare_eval_set.py \
    --train ../../data/json/v1_1_train_nus.json \
    --n-scenes 5 \
    --challenge-dir ../../../DriveLM/challenge \
    --out-dir .

# ② RAG + VLM 生成预测（GPU，耗时主要在这）
cd ../data_tools
python generate.py \
    --val-json ../eval_official/my_test_eval.json \
    --output   ../eval_official/preds.jsonl \
    --n 0 --top-k 5 --multi-camera 6

# ③ 转官方格式（CPU）
python convert_preds_to_official.py \
    --preds     ../eval_official/preds.jsonl \
    --test-eval ../eval_official/my_test_eval.json \
    --out       ../eval_official/preds_official.json

# ④ 官方打分（CPU + OpenAI API）
cd ../eval_official
python eval_drivelm_official.py \
    --pred preds_official.json \
    --test my_test_eval.json
# 没配 key 先加 --no-chatgpt 跑通其余三项
```

## 4. 先决条件（步骤② generate.py 还需要）

generate.py 依赖这些**已构建的索引和模型**（见 RAGdrive/README.md）：
- `RAGdrive/index/hnsw_bge.bin`（1.6GB）、`chunks_meta.parquet` —— 若没有，先跑 `build_index.py`
- BGE embedding / reranker 模型、Qwen2.5-VL-7B 模型权重

> 如果这些在原机器上已经建好，一起拷到 GPU 机器；否则在 GPU 机器上按 RAGdrive/README.md 的 "运行命令" 章节先构建。

## 5. 规模调整

`--n-scenes` 控制评测集大小：5=快验证(195题)，50=更稳，0=全部696 scene（VLM 推理会很久）。
两处都要一致：prepare 的 `--n-scenes` 决定题目，generate 的 `--n 0` 表示跑完 my_test_eval 里的全部题。
```

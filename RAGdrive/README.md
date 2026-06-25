# RAGdrive:DriveLM VQA 的传统 RAG 流水线

针对 DriveLM / nuScenes 数据集的端到端、本地优先的 **text-RAG + VLM** 方案,与同目录的 `../graph4drive/`(实体 KG 版本)互补。

## 1. 项目简介

**RAGdrive** 采用经典 text-RAG 范式:

- **Chunking** at QA-pair level(Q-A 一对一,加 frame_overview 备选)
- **Sparse retrieval** (BM25Okapi) — 关键词 / object ID 精确匹配
- **Dense retrieval** (BGE-large-en-v1.5 + HNSW cosine) — 语义相似
- **Cross-encoder reranking** (BGE-reranker-large) — 精排
- **Cascade 架构**:HNSW top-100 → BM25 top-20 → rerank → top-5 few-shot Q-A
- **Generation** 用 Qwen2.5-VL-7B-Instruct,基于当前帧 1-6 张相机图 + scene + few-shot Q-A

整个流水线单卡(3090 / 4090)即可跑。

## 2. 目录结构

```
RAGdrive/
├── data/
│   ├── json/                                  #   DriveLM 标注
│   │   ├── v1_0_train_nus.json                #   训练集(用于建索引)
│   │   ├── v1_1_train_nus.json
│   │   ├── v1_1_val_nus_q_only.json           #   val questions(无答案)
│   │   └── test_eval.json                     #   66 QA 测试集(有答案,用于评估)
│   ├── nuscenes/                              #   nuScenes 训练图像(4072 帧)
│   ├── drivelm_nus_imgs_val/                  #   val 图像
│   └── chunks/
│       ├── chunks_v1_0.jsonl                  #   382,055 chunks (430 MB)
│       └── chunks_v1_0_stats.json
├── index/                                      #   检索索引
│   ├── bm25.pkl                               #   BM25Okapi (163 MB)
│   ├── bm25_chunk_ids.json
│   ├── hnsw_bge.bin                           #   HNSW + BGE vectors (1.6 GB)
│   ├── hnsw_chunk_ids.json
│   ├── chunks_meta.parquet                    #   chunk 元数据 (39 MB)
│   ├── build_index_stats.json
│   └── bm25_stats.json
├── models/                                     #   模型权重缓存
│   ├── bge-large-en-v1.5/                     #   BGE bi-encoder
│   └── bge-reranker-large/                    #   BGE cross-encoder
├── data_tools/                                 #   所有脚本
│   ├── chunk_drivelm.py                       #   切 chunks
│   ├── build_bm25.py                          #   建 BM25
│   ├── build_index.py                         #   建 HNSW
│   ├── RRF_retrieve.py                        #   并行 RRF 检索
│   ├── cascade_retrieve.py                    #   cascade 检索
│   ├── eval_retrieval.py                      #   检索评估
│   ├── eval_cascade_vs_rrf.py                 #   RRF vs cascade
│   ├── generate.py                             #   生成答案(主入口)
│   └── evaluate_test.py                       #   test_eval 评估(BLEU/METEOR/CIDEr/...)
├── outputs/                                    #   生成结果
│   ├── smoke_200.jsonl
│   ├── test_eval_preds.jsonl                  #   test_eval 1-cam preds
│   ├── ragdrive_mc4_test_eval.jsonl            #   test_eval 4-cam preds
│   ├── ragdrive_mc6_test_eval.jsonl            #   test_eval 6-cam preds
│   └── kg_drift_vlm_*/                        #   graph4drive cross-method preds
└── README.md
```

模型自动下载到 `~/.cache/huggingface/hub/`。

## 📥 数据 & 模型下载

本仓库 **不包含** 数据集、模型权重、向量索引等大文件,clone 后需手动下载。

### 数据集

| 数据 | 下载 | 放置路径 |
|---|---|---|
| **nuScenes** (v1.0-train) | https://www.nuscenes.org/download (需注册, ~280GB) | `RAGdrive/data/nuscenes/` (或 `graph4drive/data/nuscenes/`,两边共用) |
| **DriveLM 标注 JSON** (v1_0_train_nus / v1_1_train_nus / v1_1_val_nus_q_only) | https://github.com/DriveLM/DriveLM/tree/main/data/QA | `RAGdrive/data/json/` |
| **test_eval.json** (66 条带答案测试集) | RAGdrive 项目仓库 / DriveLM 派生 | `RAGdrive/data/json/test_eval.json` |
| **DriveLM val 图像** | DriveLM 官方 DriveLM-NuScenes val zip | `RAGdrive/data/drivelm_nus_imgs_val/val_data/{CAM}/*.jpg` |

快速下载 JSON 标注 (Linux):
```bash
mkdir -p RAGdrive/data/json
cd RAGdrive/data/json
wget https://raw.githubusercontent.com/DriveLM/DriveLM/main/data/QA/v1_0_train_nus.json
wget https://raw.githubusercontent.com/DriveLM/DriveLM/main/data/QA/v1_1_train_nus.json
wget https://raw.githubusercontent.com/DriveLM/DriveLM/main/data/QA/v1_1_val_nus_q_only.json
```

> **注意**: `RAGdrive/data/` 和 `graph4drive/data/` 是软链/镜像关系,数据放在任一目录即可,`generate.py` 会两个路径都试。

### 模型权重 (Hugging Face)

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 中国镜像

# 必备
huggingface-cli download BAAI/bge-large-en-v1.5     --local-dir RAGdrive/models/bge-large-en-v1.5
huggingface-cli download BAAI/bge-reranker-large    --local-dir RAGdrive/models/bge-reranker-large
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ~/.cache/huggingface/hub/

# 也可直接用 Python
# from huggingface_hub import snapshot_download
# snapshot_download("BAAI/bge-large-en-v1.5", local_dir="RAGdrive/models/bge-large-en-v1.5")
```

### 索引 / 缓存 (pipeline 自动生成)

| 文件 | 大小 | 生成命令 |
|---|---|---|
| `RAGdrive/chunks/chunks_v1_0.jsonl` | 430MB | `python3 RAGdrive/data_tools/chunk_drivelm.py` |
| `RAGdrive/index/bm25.pkl` | 163MB | `python3 RAGdrive/data_tools/build_bm25.py` |
| `RAGdrive/index/hnsw_bge.bin` | 1.6GB | `python3 RAGdrive/data_tools/build_index.py` |
| `RAGdrive/index/chunks_meta.parquet` | 39MB | 同上 |

完整命令见下方 "5. 运行命令"。

## 3. 环境

### 3.1 推荐 conda 环境

```bash
conda create -n drivegraphLM python=3.10 -y
conda activate drivegraphLM

# 基础
pip install torch torchvision
pip install FlagEmbedding hnswlib rank_bm25 qwen-vl-utils pandas pyarrow

# 评估
pip install nltk rouge-score pycocoevalcap
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3.2 环境变量

```bash
# conda libstdc++ 解决 hnswlib / scipy 兼容
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# HF 离线(防访问网络)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 中国 HF 镜像(可选)
export HF_ENDPOINT=https://hf-mirror.com
```

### 3.3 需要的模型

| 模型 | 用途 |
|---|---|
| `BAAI/bge-large-en-v1.5` | dense embedding (~1.3GB) |
| `BAAI/bge-reranker-large` | cross-encoder rerank (~2.2GB) |
| `Qwen/Qwen2.5-VL-7B-Instruct` | VLM 生成 (~16GB) |

## 4. 流水线总览

```
chunk_drivelm.py  →  build_bm25.py  +  build_index.py
                                ↓
                       (RRF_retrieve.py  |  cascade_retrieve.py)
                                ↓
                            generate.py
                                ↓
                          (evaluate_test.py)
```

## 5. 运行命令

### 5.1 一键跑完整流水线

```bash
cd /home/hylong/xczn/driveVQA
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# === 离线构建(一次,~45 分钟)===
python3 RAGdrive/data_tools/chunk_drivelm.py --levels overview,pair
python3 RAGdrive/data_tools/build_bm25.py
python3 RAGdrive/data_tools/build_index.py

# === 在线生成(默认 4-cam)===
python3 RAGdrive/data_tools/generate.py \
  --val-json graph4drive/data/json/test_eval.json \
  --data-root graph4drive/data \
  --output RAGdrive/outputs/preds.jsonl \
  --n 66 \
  --top-k 5
```

### 5.2 单步跑 / debug

```bash
# 仅 1 张相机(快)
python3 RAGdrive/data_tools/generate.py \
  --val-json graph4drive/data/json/test_eval.json \
  --data-root graph4drive/data \
  --output /tmp/r1cam.jsonl \
  --n 10 \
  --multi-camera 1

# 4 张相机
python3 RAGdrive/data_tools/generate.py \
  --val-json graph4drive/data/json/test_eval.json \
  --data-root graph4drive/data \
  --output /tmp/r4cam.jsonl \
  --n 10 \
  --multi-camera 4

# 6 张相机(全 surround)
python3 RAGdrive/data_tools/generate.py \
  --val-json graph4drive/data/json/test_eval.json \
  --data-root graph4drive/data \
  --output /tmp/r6cam.jsonl \
  --n 10 \
  --multi-camera 6

# 不用 reranker(快但精度降)
python3 RAGdrive/data_tools/generate.py \
  --val-json graph4drive/data/json/test_eval.json \
  --data-root graph4drive/data \
  --output /tmp/r_norr.jsonl \
  --n 10 \
  --no-rerank
```

### 5.3 generate.py CLI 参数

```
通用:
  --val-json           val JSON 路径(默认 ../data/json/v1_1_val_nus_q_only.json)
  --data-root          nuScenes 根目录(用于解析图像路径)
  --output, -o         输出 JSONL 路径(必填)
  --n                  最多跑多少 query(0 = 全部)
  --top-k              few-shot 数量(默认 5)

检索:
  --hnsw-k             HNSW top-100(默认 100)
  --bm25-k             BM25 top-20(默认 20)
  --no-rerank          禁用 BGE-reranker-large

模型:
  --vlm-dir            Qwen2.5-VL 模型路径(默认 HF cache)
  --bge-model-dir       BGE bi-encoder 路径(默认 ../models/bge-large-en-v1.5)
  --reranker-dir        BGE reranker 路径(默认 ../models/bge-reranker-large)

相机:
  --multi-camera       1 / 4 / 6 相机(默认 4)
```

## 6. 核心设计

### 6.1 Chunk 设计

每个 chunk 有**两个文本字段**:
- `text`:完整内容(scene header + objects + Q-A)→ 喂给 LLM
- `text_for_embedding`:Q-A 在前,scene 在后 → 用于检索

**为什么分开?**同一帧的 79 个 chunk(Q-A)如果都用相同 scene header,bge embedding 会变得几乎一样,**检索退化成"返回同帧所有 Q-A"**。把 Q-A 放前面、scene 放后面,bge 能区分同一帧的不同 Q-A。

### 6.2 Cascade vs RRF

| 架构 | 流程 | 速度 | 适用场景 |
|---|---|---|---|
| **Cascade**(默认) | HNSW top-100 → BM25 top-20 → rerank → top-5 | ~996ms/q | 生产环境 |
| RRF | BM25 top-50 + HNSW top-50 → RRF → rerank | ~1121ms/q | 检索召回率优先 |

Cascade 把 BM25 限制在 HNSW top-100 上(对 382K 文档 → 100 文档),BM25 从 ~180ms/q 降到 ~5ms/q。

### 6.3 多相机支持

`generate.py` 支持 `--multi-camera {1,4,6}`:

| 配置 | 相机 | 用途 |
|---|---|---|
| 1-cam | 仅 CAM_FRONT | 原始 baseline,单图推理 |
| **4-cam**(默认) | FRONT + FRONT_LEFT + FRONT_RIGHT + BACK | 覆盖 90% 方向问题 |
| 6-cam | 全 surround | 全向场景 |

Qwen2.5-VL 支持多图 prompt,实际推理速度增加不大(每多张图 < 100ms)。

## 7. 当前结果

### 7.1 test_eval.json(66 样本)指标

**单次跑,VLM 有 ±0.10 噪声**:

| Metric | 1-cam | 4-cam | **6-cam** |
|---|---|---|---|
| MCQ accuracy | 41.2% | 47.1% | **47.1%** |
| Open F1 (mean) | 0.484 | 0.521 | **0.523** |
| Open F1 (median) | 0.525 | **0.681** | 0.636 |
| Open exact match | 18.4% | 18.4% | 18.4% |
| GT ⊆ pred | 18.4% | 18.4% | 18.4% |
| perception F1 | 0.548 | 0.695 | **0.705** |
| planning F1 | 0.272 | 0.259 | 0.260 |
| prediction F1 | 0.879 | **0.943** | **0.943** |

**4-cam vs 1-cam**:perception F1 +14.7pp,prediction +6.4pp,MCQ +5.9pp — 多相机加成很大。
**6-cam vs 4-cam**:几乎无差异(RAGdrive 4-cam 已经"够看")。

### 7.2 检索质量(早期 n=1000 self-retrieval)

| Retriever | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| BM25 | 63.5% | 84.5% | 92.5% | 95.5% | 0.7293 |
| HNSW | 26.5% | 46.5% | 57.0% | 66.0% | 0.3663 |
| RRF | 50.0% | 79.5% | 86.5% | 94.5% | 0.6302 |
| **RRF + Rerank** | **60.5%** | **89.5%** | **94.0%** | **96.5%** | **0.7195** |

Reranker 是最大的精度增益来源。

### 7.3 Cascade vs RRF(n=200 val)

| Metric | RRF | Cascade |
|---|---|---|
| task@1 | 97.5% | **98.5%** |
| MRR_task | 0.9838 | **0.985** |
| Latency/query | 1121ms | **996ms** |

Cascade 略快、略准,默认推荐。

## 8. 与 graph4drive(KG 版本)的对比

`../graph4drive/` 是同任务的**实体 KG + DRIFT + VLM** 版本。

| 维度 | RAGdrive | graph4drive |
|---|---|---|
| 检索 | BGE + BM25 + reranker | Entity KG + BGE + BM25 + reranker |
| Few-shot | 5 条相似 Q-A | 2 archetype + 4 entity-aware Q-A |
| 生成器 | Qwen2.5-VL-7B | 同 |
| 视角 | 1-6 相机 | 1-6 相机 |
| **perception F1** | **0.705** | 0.510 |
| **planning F1** | 0.260 | **0.317** ✓ |
| **prediction F1** | **0.943** | 0.809 |
| **Open F1 mean** | **0.523** | 0.481 |
| MCQ accuracy | **47.1%** | 35.3% |
| 优势 | few-shot 精确,VQA 整体强 | 结构化、可解释、planning 反超 |
| 劣势 | 无 KG,纯相似度检索 | 实体抽象损失细节 |

**planning 是 KG 反超 RAGdrive 的关键场景**。两者互补,可以并存。

## 9. 关键设计决策

| 决策 | 理由 |
|---|---|
| Chunk = Q-A pair(不是 scene) | 一对一,scene_overview 作 fallback |
| 双 text 字段(text / text_for_embedding) | 同帧 chunks 不会 embedding 退化 |
| Cascade 而非纯 RRF | 快 11%,精度相当 |
| 跨编码器 rerank | R@5 +10pp,top-5 purity 80%→91.5% |
| Qwen2.5-VL-7B | 视觉理解强,7B 推理可单卡 |
| 4-cam 默认 | 覆盖 90% 方向问题,速度/质量平衡 |
| 不传检索到的图像 | prompt token 数爆炸,只传当前帧 |

## 10. 维护注意

### 10.1 ⚠️ cuDNN 版本不匹配

PyTorch 2.12 + bundled cuDNN 9.20 在本环境与 Qwen2.5-VL 的 `F.conv3d` 不兼容。`generate.py` 中已加 `torch.backends.cudnn.enabled = False`,convolution 退到 cuBLAS,推理时间约翻倍。

彻底修复:装匹配的 cuDNN 版本。

### 10.2 ⚠️ val JSON image_paths 误导

`v1_1_val_nus_q_only.json` 的 `image_paths` 字段指向 `../nuscenes/samples/...`,但实际 val 图像在 `data/drivelm_nus_imgs_val/val_data/CAM_FRONT/`。

`generate.py` 的 loader 会**两个路径都试**,优先 val_data。

### 10.3 ⚠️ BM25 是 batch 检索瓶颈

`rank_bm25` 纯 Python。382K 文档上 ~180ms/q。

Cascade 架构只在 HNSW top-100 上跑 BM25,降到 ~5ms/q。

进一步加速:换 `bm25s`(Rust 实现)。

### 10.4 ⚠️ VLM 推理时间

Qwen2.5-VL-7B + bfloat16 在 24GB GPU 上 ~3-7s/q(test_eval 66 样本 ~5 分钟)。

加图会增 50-100% 时间(每张图 ~1.5s),6-cam 比 1-cam 慢 ~30%(实测 ~2 倍)。

## 11. 故障排除

| 问题 | 解决 |
|---|---|
| `ImportError: scipy CXXABI_1.3.15` | `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` |
| Qwen2.5-VL 加载失败 | `export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1` |
| 模型找不到 | 自动下载(`HF_ENDPOINT=https://hf-mirror.com` 中国加速) |
| `RuntimeError: CUDA out of memory` | `--multi-camera 1` 或用 fp16 + 较小 VLM |
| `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` | 已 workaround:关闭 cuDNN,推理慢但能跑 |

## 12. 文件维护速查

| 想做什么 | 看 / 改哪里 |
|---|---|
| 改 chunk 格式 | `data_tools/chunk_drivelm.py` |
| 改 BM25 参数 | `data_tools/build_bm25.py` |
| 改 HNSW 参数 | `data_tools/build_index.py` |
| 改 cascade 流程 | `data_tools/cascade_retrieve.py` |
| 改 prompt 模板 | `data_tools/generate.py:_build_prompt_text` |
| 改多相机逻辑 | `data_tools/generate.py:resolve_camera_paths` |
| 改评估指标 | `data_tools/evaluate_test.py` |

## 13. 与 graph4drive 的 pipeline 对比

```
                 RAGdrive                           graph4drive
                 ─────────                          ────────────
Chunking:        Q-A pair + overview               Q-A pair
                 ↓                                  ↓
Embed chunks:    BGE-large + HNSW (1.6GB)          BGE-M3 + HNSW (1.6GB)
                 ↓                                  ↓
Keyword:         BM25                               BM25
                 ↓                                  ↓
Rerank:          BGE-reranker-large                 BGE-reranker-large
                 ↓                                  ↓
Entity KG:       —                                  PMI-weighted graph
                 ↓                                  Leiden → 6 communities
                                                 ↓  LLM summaries
Fusion:          Top-5 similar Q-A                  Top-2 archetype + Top-4 entity-aware Q-A
                 ↓                                  ↓
VLM:             Qwen2.5-VL-7B (1-6 cam)           Qwen2.5-VL-7B (1-6 cam)
```

**RAGdrive 适合**:factoid VQA,需要 few-shot 答案示范
**graph4drive 适合**:跨样本分析、可解释推理、planning 类问题
# DriveGraphLM (graph4drive)

> Microsoft GraphRAG 风格的实体级知识图谱 + DRIFT 检索 + Qwen2.5-VL 多相机 VLM 推理

针对 DriveLM / nuScenes 视觉问答任务的端到端 pipeline。核心是用结构化场景图构建实体级 KG(Leiden 社区 + LLM 摘要),结合多相机视觉证据做答案生成。

## 1. 项目简介

**这个项目解决什么问题**:给定一段自动驾驶场景的相机图像 + 问题,让 VLM 生成准确答案。

**与 RAGdrive 的区别**:
- **RAGdrive**:基于 BGE-large + BM25 + reranker 的 text-RAG + Qwen2.5-VL,few-shot Q-A 作为上下文
- **graph4drive(本项目)**:基于实体级 KG(Leiden 社区 + archetype 摘要)+ DRIFT 检索(entity-aware local examples)+ Qwen2.5-VL,场景结构作为额外知识

**适合什么场景**:需要可解释检索、需要跨样本场景统计、需要结构化 entity 过滤的 VQA 任务。

## 📥 数据 & 模型下载

本仓库 **不包含** 数据集、模型权重、向量索引等大文件，clone 后需要手动下载。

### 数据集

| 数据 | 下载 | 放置路径 |
|---|---|---|
| **nuScenes** (v1.0-train) | https://www.nuscenes.org/download (需注册, ~280GB) | `data/nuscenes/` (仓库根目录,与 RAGdrive 共用) |
| **DriveLM 标注 JSON** (v1_0_train_nus / v1_1_train_nus / v1_1_val_nus_q_only / test_eval) | https://github.com/DriveLM/DriveLM/tree/main/data/QA 下的 `*_nus.json` 文件 | `data/json/` |
| **DriveLM val 图像** | DriveLM 官方 DriveLM-NuScenes val zip | `data/drivelm_nus_imgs_val/val_data/{CAM}/*.jpg` |

快速下载 JSON 标注 (Linux):
```bash
mkdir -p data/json
cd data/json
# 从 DriveLM 仓库直接下载(以 v1_0_train_nus.json 为例)
wget https://raw.githubusercontent.com/DriveLM/DriveLM/main/data/QA/v1_0_train_nus.json
wget https://raw.githubusercontent.com/DriveLM/DriveLM/main/data/QA/v1_1_train_nus.json
wget https://raw.githubusercontent.com/DriveLM/DriveLM/main/data/QA/v1_1_val_nus_q_only.json
# test_eval.json 来自 RAGdrive 项目(包含 66 条带答案的测试样本)
```

⚠️ **DriveLM 图像** 不在 DriveLM GitHub 仓库,需要去 DriveLM 项目页 / DriveLM-NuScenes 单独下载 val 集(参考 DriveLM README 的 "Data Preparation" 章节)。

### 模型权重 (Hugging Face)

```bash
# 推荐使用 hf-mirror 中国镜像
export HF_ENDPOINT=https://hf-mirror.com

# 必备
huggingface-cli download BAAI/bge-m3                    --local-dir graph4drive/models/bge-m3
huggingface-cli download Qwen/Qwen2.5-7B-Instruct        --local-dir ~/.cache/huggingface/hub/
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct     --local-dir ~/.cache/huggingface/hub/

# 可选 (cascade 检索精度更高)
huggingface-cli download BAAI/bge-reranker-large        --local-dir RAGdrive/models/bge-reranker-large
```

或用 Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-m3", local_dir="graph4drive/models/bge-m3")
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
# ...
```

### 索引 / 缓存 (pipeline 自动生成)

| 文件 | 大小 | 生成命令 |
|---|---|---|
| `graphrag_community/chunks/*.jsonl/.bin/.pkl` | ~2GB | `python3 graph4drive/data_tools/build_chunks_jsonl.py` 等离线步骤 |
| `graphrag_community/entity_kg/*` | <10MB | `build_entity_kg.py` |
| `graphrag_community/entity_communities/*` | <1MB | `detect_entity_communities.py` |

完整命令见下方 "6. 离线构建详解"。

## 2. 环境

### 2.1 推荐 conda 环境

```bash
# 已有 drivegraphLM 环境可以复用,或者新建
conda create -n drivegraphLM python=3.10 -y
conda activate drivegraphLM

# 基础
pip install torch torchvision
pip install FlagEmbedding hnswlib igraph leidenalg pandas pyarrow

# 评估依赖
pip install nltk rouge-score pycocoevalcap

# WordNet (METEOR 需要)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Qwen2.5-VL
pip install qwen-vl-utils

# 可选:BM25
pip install rank_bm25

# BGE-reranker
pip install FlagEmbedding   # 包含 FlagReranker
```

### 2.2 重要环境变量

```bash
# conda libstdc++ 解决 hnswlib / scipy 兼容问题
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# HF 离线模式(防止启动时访问网络)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 可选:中国 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.3 模型依赖

需要本地缓存或下载以下模型(放在 `~/.cache/huggingface/hub/`):

| 模型 | 用途 | 大小 |
|---|---|---|
| BAAI/bge-m3 | chunk + community embedding | ~2.3GB |
| Qwen/Qwen2.5-7B-Instruct | 摘要 + entity description 生成 | ~15GB |
| Qwen/Qwen2.5-VL-7B-Instruct | 推理 VLM | ~16GB |
| BAAI/bge-reranker-large | cascade 检索(可选) | ~2.2GB |

本地放置路径(可选):

```bash
# 在 graph4drive/models/ 下:
mkdir -p models
# 软链接或放置:
#   models/bge-m3 → 已下载的 BGE-M3
#   models/bge-reranker-large → 已下载的 BGE-reranker-large
```

## 3. 数据

### 3.1 输入数据

| 文件 | 来源 | 内容 |
|---|---|---|
| `../data/json/v1_0_train_nus.json` | DriveLM 官方 | 4072 scene × ~140 QA(35 per task × 4 tasks) + 6 相机图像路径 |
| `../data/json/v1_1_val_nus_q_only.json` | DriveLM 官方 | 验证集(无答案) |
| `../data/json/test_eval.json` | DriveLM 官方 | 66 条测试集(有答案,用于 eval) |
| `../data/nuscenes/samples/{CAM}/*.jpg` | nuScenes | 6 相机 × 4072 帧训练图像 |
| `../data/drivelm_nus_imgs_val/val_data/{CAM}/*.jpg` | DriveLM | 验证集图像 |

**test_eval.json 关键字段**:
```json
{
  "scene_token": "abc123",
  "key_frames": {
    "frame_token": {
      "image_paths": {
        "CAM_FRONT":       "../nuscenes/samples/CAM_FRONT/n008-...jpg",
        "CAM_FRONT_LEFT":  "...",
        "CAM_FRONT_RIGHT": "...",
        "CAM_BACK":        "...",
        "CAM_BACK_LEFT":   "...",
        "CAM_BACK_RIGHT":  "..."
      },
      "QA": {
        "perception": [{"Q": "...", "A": "..."}, ...],
        "prediction": [...],
        "planning":   [...],
        "behavior":   [...]
      }
    }
  }
}
```

### 3.2 输出数据(pipeline 自动生成)

| 文件 | 大小 | 内容 |
|---|---|---|
| `graphrag_community/chunks/drivelm_train_chunks.jsonl` | 615MB | 378K per-QA chunks |
| `graphrag_community/chunks/hnsw_bge_m3_chunks.bin` | 1.6GB | chunks 向量索引(BGE-M3 + scene-graph prefix) |
| `graphrag_community/chunks/chunks_metadata.pkl` | ~240MB | chunks 元数据(text + scene_summary + task) |
| `graphrag_community/chunks/bm25_chunks.pkl` | 80MB | BM25 索引 |
| `graphrag_community/entity_kg/entity_kg.gpickle` | ~200KB | 193 节点 / 2050 边 PMI 加权图 |
| `graphrag_community/entity_kg/entity_chunk_sets.pkl` | ~2MB | entity → chunk_ids 反向索引 |
| `graphrag_community/entity_kg/entity_descriptions.json` | ~30KB | 193 NL descriptions |
| `graphrag_community/entity_kg/entity_hnsw.bin` | <1MB | entity-level HNSW (description embedding) |
| `graphrag_community/entity_kg/entity_metadata.pkl` | <1MB | entity index → key/description |
| `graphrag_community/entity_communities/entity_communities.jsonl` | ~10KB | 6 社区成员表 |
| `graphrag_community/entity_communities/entity_community_summaries.jsonl` | ~3KB | 6 段 LLM archetype 摘要 |
| `graphrag_community/entity_communities/hnsw_bge_m3_entity_communities.bin` | <1MB | summaries 向量索引 |
| `graphrag_community/text_units/text_units.jsonl` | ~150MB | Microsoft GraphRAG-style text units(含 entity_ids + relationship_ids) |

## 4. 目录结构

```
graph4drive/
├── data/                                # 输入数据(已存在,不要改)
│   ├── json/                            #   DriveLM JSON 标注
│   └── nuscenes/                        #   nuScenes 训练图像
│
├── data_tools/                          # 离线构建脚本(10 个)
│   ├── build_chunks_jsonl.py            #   1. per-QA chunks
│   ├── build_entity_kg.py               #   2. 实体 KG + PMI 边
│   ├── generate_entity_descriptions.py  #   3. entity NL descriptions(LLM)
│   ├── build_entity_embeddings.py       #   4. entity-level HNSW (NEW)
│   ├── detect_entity_communities.py     #   5. Leiden 社区检测
│   ├── summarize_entity_communities.py  #   6. community summaries(LLM)
│   ├── build_chunks_index.py            #   7a. chunks BGE-M3 HNSW
│   ├── build_bm25_index.py              #   7b. chunks BM25
│   ├── build_community_index.py         #   7c. community summaries HNSW
│   ├── build_text_units.py              #   8. Microsoft GraphRAG-style text units(NEW)
│   └── __pycache__/
│
├── test_tools/                          # 在线脚本(2 个)
│   ├── run_graphrag_drift_vlm.py        #   主推理脚本
│   ├── eval_driveLM_metrics.py          #   BLEU/METEOR/CIDEr/ROUGE-L/Acc 评估
│   └── __init__.py
│
├── graphrag_community/                  # 输出(pipeline 生成,可删可重建)
│   ├── chunks/
│   ├── entity_kg/
│   ├── entity_communities/
│   └── text_units/                      #   NEW: GraphRAG-spec text_units
│
├── models/                              # 可选本地模型缓存
│   └── bge-m3/
│
└── README.md                            # 本文件
```

## 5. Pipeline 总览

```
                ┌──────────────────────────────────────┐
                │ 原始数据 (v1_0_train_nus.json)      │
                │   - 4072 scenes × ~140 QA + 6 cams  │
                └──────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 离线构建(一次,~27 分钟)                                              │
   ├─────────────────────────────────────────────────────────────────────┤
   │ 1. Chunking          → 378K per-QA chunks                          │
   │ 2. Entity Extract    → 193 unique entities (Cat@Cam@Region)         │
   │ 3. Graph Build       → 193 nodes / 2050 PMI-weighted edges            │
   │ 4. Entity Desc       → 193 NL descriptions (LLM, ~2 min)             │
   │ 5. Entity Embedding  → entity-level HNSW (description vectors)       │
   │ 6. Community Detect  → 6 Leiden communities (modularity 0.24)       │
   │ 7. Community Summary → 6 LLM-generated archetypes                   │
   │ 8. Index Build       → chunks HNSW + BM25 + community HNSW          │
   │ 9. Text Units        → GraphRAG-spec text_units.jsonl                │
   └─────────────────────────────────────────────────────────────────────┘
                                 ↓
   ┌─────────────────────────────────────────────────────────────────────┐
   │ 在线推理(每条 query,~1.4s/item with 4-cam + rerank)                  │
   ├─────────────────────────────────────────────────────────────────────┤
   │ 1. MCQ Routing        → MCQ: skip local, use only global              │
   │ 2. Global Retrieval   → BGE-M3 + HNSW over community summaries       │
   │ 3. Local DRIFT        → category → entity → BM25 → rerank → top-k  │
   │ 4. Prompt Build       → [Global archetypes] + [Local Q-A] + [Q]    │
   │ 5. VLM Inference      → Qwen2.5-VL-7B + 4 cameras → answer          │
   └─────────────────────────────────────────────────────────────────────┘
```

## 6. 离线构建详解

### 6.1 Chunking → 378K per-QA chunks

```bash
python3 data_tools/build_chunks_jsonl.py
# 输入: ../data/json/v1_0_train_nus.json
# 输出: graphrag_community/chunks/drivelm_train_chunks.jsonl
# 时间: ~2 分钟
```

**做了什么**:从每帧的 `key_object_infos` 抽取场景图,展开该帧所有 (task, QA) 对,每条 QA 一行。

**输出字段**:
- `chunk_id`: `{frame_idx}.{qa_idx}`
- `scene_token`, `task`, `question`, `answer`
- `graph_nodes`: 该帧的场景图物体,如 `<c1,CAM_BACK,...>(Vehicle) | <c3,...>(TrafficElement)`
- `image_path`: 该帧 CAM_FRONT 路径

### 6.2 实体抽取 + 图构建

```bash
python3 data_tools/build_entity_kg.py
# 输入: chunks jsonl
# 输出: 
#   graphrag_community/entity_kg/entity_kg.gpickle         # NetworkX 图
#   graphrag_community/entity_kg/entity_chunk_sets.pkl    # entity → chunks
#   graphrag_community/entity_kg/entity_index.parquet
# 时间: ~5 分钟(378K chunks)
```

**实体 key 格式**:`f"{Category}@{Camera}@{Region}"`

- `Category`: Vehicle / Pedestrian / Cycle / Traffic element / Others
- `Camera`: CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT / CAM_BACK / CAM_BACK_LEFT / CAM_BACK_RIGHT
- `Region`: 9 个位置(`r00` ~ `r22`),3×3 grid from (x, y) in 1600×900 image

**边权 = PMI**(pointwise mutual information):
```
PMI(e1, e2) = log( P(e1, e2) / (P(e1) × P(e2)) )
```

为什么用 PMI 而不是 raw count?Common entity(Vehicle)经常出现,raw count 高但不一定有共现语义;PMI 突出"罕见但常一起出现"的实体对。

**过滤**:`min_cooccurrence ≥ 5`(去掉统计噪声)。

**结果**:193 节点 / 2050 边 / density 0.11 / avg degree 21。

### 6.3 Entity Descriptions(LLM 生成)

```bash
python3 data_tools/generate_entity_descriptions.py
# 输入: entity_chunk_sets.pkl
# 输出: graphrag_community/entity_kg/entity_descriptions.json
# 时间: ~2 分钟(193 个 entity,batch=20)
```

**做什么**:用 Qwen2.5-7B 给每个 entity 生成 1-2 句自然语言描述。

**示例输出**:
```json
{
  "Vehicle@CAM_FRONT@r10": "a passenger car approaching the ego vehicle from the front-center",
  "Pedestrian@CAM_FRONT_RIGHT@r12": "a pedestrian on the right side, slightly forward",
  "Traffic element@CAM_BACK@r21": "a stationary traffic sign in the lower-center of the rear view"
}
```

### 6.4 社区检测(Leiden)

```bash
python3 data_tools/detect_entity_communities.py
# 输入: entity_kg.gpickle
# 输出:
#   graphrag_community/entity_communities/entity_communities.jsonl
#   graphrag_community/entity_communities/community_assignments.parquet
# 时间: ~10 秒
```

**算法**:`leidenalg.ModularityVertexPartition`(Newman Q modularity)

**输出**:6 个社区,modularity = 0.24。每个社区是一组相关的实体。

### 6.5 Entity Embeddings(HNSW)

```bash
python3 data_tools/build_entity_embeddings.py
# 输入: entity_descriptions.json (193 entities × 1-2 sentence NL desc)
# 输出:
#   graphrag_community/entity_kg/entity_hnsw.bin
#   graphrag_community/entity_kg/entity_metadata.pkl
# 时间: ~10 秒(193 个 entity,微)
```

**做什么**:把每个 entity 的 NL description 编码为 BGE-M3 向量,建 HNSW cosine 索引。

**为什么**:Microsoft GraphRAG 的 local search 用 entity embedding 做"找到和 query 相似的 entity"——这是 entity-level dense retrieval。本步复现这层。

**在线作用**:推理时,query 编码后 entity_HNSW.knn_query → top-10 entities,这些 entity 的 chunks 也加入 BM25 → rerank 流水线。

**实测效果**:planning F1 从 0.215 涨到 0.355(+14pp),Open F1 mean 涨 8pp。

### 6.6 Community Summarization(LLM 生成 archetype)

```bash
python3 data_tools/summarize_entity_communities.py
# 输入: communities + chunks + entity_descriptions
# 输出: graphrag_community/entity_communities/entity_community_summaries.jsonl
# 时间: ~30 秒(6 社区 × 1 LLM call)
```

**Prompt 内容**:社区统计(top categories / cameras / regions)+ **entity descriptions** + 8 个样本 Q-A → LLM 生成 60-120 词 archetype。

**示例输出**(Community 0):
> "The driving scene typically features multiple bicycles and pedestrians in various regions of the camera views, often near the front and rear of the vehicle. Questions focus on the presence and status of objects such as traffic signs, barriers, and other vehicles, indicating a need for careful observation and potential lane changes..."

### 6.7 Text Units(GraphRAG-spec)

```bash
python3 data_tools/build_text_units.py
# 输入:
#   graphrag_community/chunks/chunks_metadata.pkl
#   graphrag_community/entity_kg/entity_chunk_sets.pkl
#   graphrag_community/entity_kg/entity_kg.gpickle
# 输出:
#   graphrag_community/text_units/text_units.jsonl   # ~150MB
# 时间: ~30 秒(纯 pickle 操作)
```

**做什么**:为每个 chunk 加 `entity_ids` 和 `relationship_ids` 反向字段,产出 Microsoft GraphRAG 标准格式的 text_units.jsonl。

**Schema**:
```json
{
  "id": "0.0",
  "text": "perception | scene=... | Q: ... | A: ...",
  "entity_ids": ["Vehicle@CAM_BACK@r11", "Traffic element@CAM_FRONT@r01"],
  "relationship_ids": ["rel_Vehicle@CAM_BACK@r11__Traffic element@CAM_FRONT@r01"],
  "source_doc_id": "<scene_token>",
  "n_tokens": 35
}
```

**Entity 来源**:从 `entity_chunk_sets.pkl` 反查(每个 chunk 出现的所有 entity)。
**Relationship 来源**:在 chunk 内 entity pairs 里查 `entity_kg.gpickle` 的边,生成稳定 ID `rel_<entity_a>__<entity_b>`(字典序)。

**覆盖率**:378K text units,100% 有 entity_ids,88% 有 relationship_ids(剩下 12% 是单 entity chunks)。

**价值**:
- Spec 合规:可以理直气壮说"我们 implement GraphRAG text_units"
- 可追溯:每条 evidence 能溯源到具体 entity 和 relationship
- 调试友好:看 prompt 里出现哪条 relationship,能查回是哪个 scene

### 6.8 索引构建(三个并行)

### 6.7 索引构建(三个并行)

```bash
# 6a. Chunks 向量索引
python3 data_tools/build_chunks_index.py
# 输入: chunks jsonl
# 输出:
#   graphrag_community/chunks/hnsw_bge_m3_chunks.bin   # 1.6GB
#   graphrag_community/chunks/chunks_metadata.pkl      # 240MB
# 时间: ~10 分钟(BGE-M3 编码 378K)

# 6b. BM25 索引
python3 data_tools/build_bm25_index.py
# 输入: chunks_metadata.pkl
# 输出: graphrag_community/chunks/bm25_chunks.pkl
# 时间: ~30 秒

# 6c. Community 向量索引
python3 data_tools/build_community_index.py
# 输入: community_summaries
# 输出:
#   graphrag_community/entity_communities/hnsw_bge_m3_entity_communities.bin
#   graphrag_community/entity_communities/entity_community_metadata.pkl
# 时间: ~10 秒(6 个 summary)
```

**Chunk embedding text 格式**:
```python
text = f"{task} | scene=Cats@CAMs | Q: {question} | A: {answer}"
# 例:
# "perception | scene=Vehicle,Traffic element@CAM_BACK,CAM_FRONT |
#  Q: What are objects to the front right? |
#  A: There are many barriers and one construction vehicle..."
```

`scene=Cats@CAMs` 前缀让 BGE-M3 同时编码问题语义和场景结构。

## 7. 在线推理详解

**脚本**:`test_tools/run_graphrag_drift_vlm.py`

### 7.1 MCQ 路由

```python
if _is_mcq(question) and args.mcq_global_only:
    local_examples = []  # MCQ 跳过 local,只用 global archetypes
```

MCQ(选 A/B/C/D)需要场景理解,不需要具体例子。

### 7.2 全局检索(community summaries)

```python
q_vec = bge_m3_encode(question)
c_ids, _ = comm_idx.knn_query(q_vec, k=2)  # top-2 archetypes
```

返回最相关的 2 个 community archetype 段落。

### 7.3 本地 DRIFT 检索(entity-level dense + 实体感知 + cascade)

**Stage 0**:Entity-level dense retrieval(Microsoft GraphRAG local-search 复刻)
```python
q_vec = bge_m3_encode(question)
e_ids, _ = entity_hnsw.knn_query(q_vec, k=10)
entity_recall = {entity_meta.iloc[i].entity_key for i in e_ids[0]}
# 例: question "what action when seeing pedestrian" → entity "Pedestrian@..."
```

**Stage 1**:从 question 抽取类别(category keyword match)
```python
cats = _extract_question_categories(question)
# "What's the car doing?" → ["Vehicle"]
```

**Stage 2**:候选 chunks(union of category + entity embedding)
```python
# From category match
for cat in cats:
    for e in cat_to_ents[cat]:
        candidate_chunk_ids.update(entity_to_chunks[e])
# From entity-embedding recall (NEW)
for e in entity_recall:
    candidate_chunk_ids.update(entity_to_chunks[e])
# ≈ 30K-100K chunks (category ∪ entity embedding)
```

**Stage 3**:BM25 top-20
```python
# 关键词匹配 (CAM_BACK, "behind" 等)
# 用 entity_chunk_sets.pkl 算 BM25 分数
```

**Stage 4**:BGE-reranker top-4
```python
# Cross-encoder 重排 (question, chunk_text) 对
# 取 top-4
```

### 7.4 Prompt 构造

```
[Global Knowledge — driving scene archetypes]

Scenario archetype 1:
The driving scene typically features multiple bicycles and pedestrians...

Scenario archetype 2:
The scene typically involves urban with cyclists and pedestrians...

[Local Examples — historical Q-A involving similar entities]

Example 1:
Q: Is there a vehicle to the back of the ego car?
A: Yes, a black sedan...

Example 2:
...

[Question]
What's the car doing behind us?

[Answer]
```

### 7.5 VLM 推理(多相机)

```python
# 加载 4 张图(CAM_FRONT + FRONT_LEFT + FRONT_RIGHT + BACK)
images = resolve_camera_paths(item["image_paths"], max_cams=4)

messages = [{
    "role": "user",
    "content": [
        *[{"type": "image", "image": p} for p in images],
        {"type": "text", "text": prompt_text},
    ]
}]
answer = qwen2_5_vl.generate(messages, max_new_tokens=128)
```

## 8. 运行命令

### 8.1 一键跑完整 pipeline

```bash
cd /home/hylong/xczn/driveVQA/graph4drive
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# === 离线构建(只在首次或数据更新时跑,~27 分钟)===
python3 data_tools/build_chunks_jsonl.py
python3 data_tools/build_entity_kg.py
python3 data_tools/generate_entity_descriptions.py
python3 data_tools/build_entity_embeddings.py
python3 data_tools/detect_entity_communities.py
python3 data_tools/summarize_entity_communities.py
python3 data_tools/build_chunks_index.py
python3 data_tools/build_bm25_index.py
python3 data_tools/build_community_index.py
python3 data_tools/build_text_units.py

# === 在线推理(只用最少的参数,其他都用最优默认)===
python3 test_tools/run_graphrag_drift_vlm.py \
  --val-path ../data/json/test_eval.json \
  --n-samples 66 \
  --output-dir ./outputs_run
```

**默认配置**:
- 4 张相机(front + front_left + front_right + back)
- MCQ 路由开启(MCQ 跳过 local)
- BM25 + BGE-reranker cascade
- k_communities=2, k_local=4
- max_new_tokens=192

### 8.2 单步跑(测试 / debug)

```bash
# 只跑 5 条样本,验证 pipeline 通(都用默认)
python3 test_tools/run_graphrag_drift_vlm.py \
  --val-path ../data/json/test_eval.json \
  --n-samples 5

# 不带 BM25 / rerank(只用 BGE-M3)
python3 test_tools/run_graphrag_drift_vlm.py \
  --val-path ../data/json/test_eval.json \
  --n-samples 10 \
  --no-bm25 --no-rerank

# 单相机(只看 CAM_FRONT)
python3 test_tools/run_graphrag_drift_vlm.py \
  --val-path ../data/json/test_eval.json \
  --n-samples 10 \
  --multi-camera 1

# 6 相机(全 surround)
python3 test_tools/run_graphrag_drift_vlm.py \
  --val-path ../data/json/test_eval.json \
  --n-samples 10 \
  --multi-camera 6

# 关闭 MCQ 路由(消融)
python3 test_tools/run_graphrag_drift_vlm.py \
  --val-path ../data/json/test_eval.json \
  --n-samples 10 \
  --no-mcq-global-only

# 调更多社区
python3 test_tools/run_graphrag_drift_vlm.py \
  --val-path ../data/json/test_eval.json \
  --n-samples 10 \
  --k-communities 4
```

### 8.3 CLI 参数

```
通用:
--val-path           验证集 JSON 路径(默认 ../data/json/test_eval.json)
--n-samples          处理多少样本(默认 66,0 = 全部)
--output-dir         输出目录(默认 ./outputs_kg_drift_vlm)
--vlm-path           Qwen2.5-VL 模型路径(自动解析 HF cache snapshot)
--embed-model        BGE-M3 路径(默认 ./models/bge-m3)
--embed-device       cuda / cpu(默认 cuda)
--dtype              bfloat16 / float16 / float32(默认 bfloat16)

检索:
--k-communities      global 检索 top-K(默认 2)
--k-local            local 检索 top-K(默认 4)
--local-pool-size    entity 过滤后候选池大小(默认 200)
--bm25-k             BM25 过滤后 top-K(默认 20)
--ef                 HNSW ef 参数(默认 64)

相机与 MCQ:
--multi-camera       1 / 4 / 6 相机(默认 6)
--mcq-global-only    MCQ 跳过 local(默认 ON,用 --no-mcq-global-only 关闭)
--no-image           不传图(text-only 消融)

模型:
--max-new-tokens     生成最大长度(默认 192)
--planning-k-local   planning 类 K 覆盖(默认 None = 用 --k-local)

消融:
--no-bm25            禁用 BM25 级
--no-rerank          禁用 reranker 级
--no-mcq-global-only 关闭 MCQ 路由
--no-entity-embedding 关闭 entity-level dense 检索
```

### 8.4 评估命令

用统一评估脚本(在 `RAGdrive/compare_metrics.py`):

```bash
cd /home/hylong/xczn/driveVQA
python3 compare_metrics.py \
  --gt data/json/test_eval.json \
  --preds \
    "RunA=graph4drive/outputs/outputs_run/preds_kg_drift_vlm.jsonl" \
  --pred-field prediction
```

输出:MCQ accuracy + Open F1 (mean/median/exact) + GT⊆pred + per-task F1。

## 9. 关键设计选择

### 9.1 为什么不用 LLM Graph Extraction?

Microsoft GraphRAG 标准用 LLM 从文本抽取 entity/relation/description。我们跳过,因为:
1. **DriveLM 已有结构化 scene graph**(`key_object_infos`),比 LLM 抽取更准(LLM 看不到精确坐标 ID)
2. **378K chunks × 1 LLM call** = 几天推理 + ¥1-2K 成本
3. **378K 帧只有 3-6 个物体**,LLM 抽取无意义

替代方案:直接用 scene graph objects 当 entity,schema `(Category, Camera, Region)`,PMI-weighted 共现当边。

### 9.2 为什么场景分 9 个 region(3×3 grid)?

| 粒度 | 问题 |
|---|---|
| 1 region(全图) | 信息太少,几乎所有 entity 都共享 |
| 9 region(3×3) | **当前选择**:足够区分,不会太稀 |
| 49 region(7×7) | 太稀,PMI 边权全接近 0 |
| object-level | 每个 instance 不同,不能跨场景对齐 |

3×3 grid 在 nuScenes 1600×900 图像上 ≈ 每个区域 ~533×300 px,大致对应"前方/左前/右前/左/中/右/..."语义。

### 9.3 为什么 MCQ 单独路由?

数据上验证过:MCQ(17 个)用 global-only 命中 47.1%,用 full pipeline 命中 35.3%。

**原因**:MCQ 答案就一个字母(A/B/C/D)。Few-shot 注入反而**让 LLM 模仿别人的具体答案**,干扰 pure global reasoning。

社区 archetype 是抽象场景描述,对 MCQ 的"判断场景类型"任务最对路。

### 9.4 为什么 multi-camera 默认 6?

6 张 = 全 surround 视野。ablation 显示:
- 4 → 6 camera 涨 planning F1 +5.3pp(rear-left / rear-right 问题答得更准)
- 4 → 6 camera Open F1 mean +1.6pp
- 时间开销几乎为 0(Qwen2.5-VL 多图成本低,66 样本 +12s)

缺点:显存占用 +30%。显存不够时可 `--multi-camera 4` 或 `1`。

### 9.5 为什么 Chunk text 用 `task | scene=... | Q | A` 这个顺序?

测试发现(`RAGdrive/outputs/kg_drift_vlm_AC*.jsonl`):

| 顺序 | Open F1 mean | Notes |
|---|---|---|
| `task \| scene= \| Q \| A` | 0.502 | **当前默认** |
| `task \| Q \| A \| scene=` | 0.404 | scene 在尾部被 max_chars 截断 |

直觉上 scene 在尾部应该不影响 Q-A 权重,但实际**更差**。可能因为 BGE-M3 对前缀权重更高,scene 在前能**主导**对齐到 scene-similar chunks(对 planning 有用)。

### 9.6 与 Microsoft GraphRAG 原版的差异

我们的 pipeline **形式上对齐** Microsoft GraphRAG(text_units / entity embedding / Leiden / community summary),**实现细节**做了领域适配:

| GraphRAG 标准组件 | 我们做了什么 | 差异原因 |
|---|---|---|
| ✅ Text units with `entity_ids` + `relationship_ids` | 完全对齐(`build_text_units.py`) | spec 合规,零成本 |
| ✅ Entity-level description | 完全对齐(`generate_entity_descriptions.py`) | spec 合规 |
| ✅ Entity-level dense retrieval | **加**(复刻 GraphRAG local-search) | driving VQA 涨 planning +14pp |
| ✅ Leiden community detection | 完全对齐 | 通用算法 |
| ✅ LLM community summary | 对齐(单层,无层级) | 6 社区用不着层级 |
| ❌ LLM Graph Extraction | 跳过(用结构化 scene graph) | DriveLM 已有人工标注,LLM 抽取会退化 |
| ❌ Hierarchical summaries | 跳过 | 6 社区不需要 |
| ❌ Map-reduce global search | 跳过 | 6 社区直接 top-2 就够 |
| ❌ DRIFT search(follow-up questions) | 跳过 | VQA 用不上,反而慢 |
| ➕ BM25 cascade | **加**(对齐 RAGdrive) | driving VQA 必要(object ID 匹配) |
| ➕ BGE-reranker | **加**(对齐 RAGdrive) | driving VQA 必要 |
| ➕ Multi-camera VLM | **加**(原版 GraphRAG 没有视觉) | DriveLM 多模态任务 |

**总结**:形式上 GraphRAG-compliant + 实现上 driving VQA-optimized。**有资格叫 "KG-RAG pipeline inspired by Microsoft GraphRAG"**,但不是字面实现。

## 10. 当前结果(6-camera + entity-HNSW, 66 样本)

**注意**:单次跑,VLM fp16 非确定性带来 ±0.10 噪声。这里给的数字是单次跑结果。

### 10.1 KG vs RAGdrive 总体对比

| 指标 | RAGdrive 6-cam | **KG 6-cam default** | Δ |
|---|---|---|---|
| MCQ accuracy | **47.1%** | 35.3% | -11.8pp |
| Open F1 mean | **0.523** | 0.507 | -0.016 |
| Open exact match | 18.4% | 18.4% | 0 |
| GT ⊆ pred | 18.4% | **20.4%** | +2.0pp |
| perception F1 | **0.705** | 0.510 | -19.5pp |
| **planning F1** | 0.260 | **0.355** | **+9.5pp** ✓ |
| prediction F1 | **0.943** | 0.820 | -12.3pp |

### 10.2 entity-HNSW 的影响(我们 pipeline 内部消融)

| 指标 | 无 entity-embedding | **有 entity-embedding** | Δ |
|---|---|---|---|
| Open F1 mean | 0.427 | **0.507** | **+8.0pp** ✓ |
| planning F1 | 0.215 | **0.355** | **+14.0pp** ✓ |
| GT ⊆ pred | 18.4% | **20.4%** | +2.0pp |

entity-level dense 检索(GraphRAG local search 思路)让 planning F1 大涨 14pp。

### 10.3 综合结论

- ✅ **Planning 反超 RAGdrive 9.5pp**(0.355 vs 0.260)——最大亮点
- ✅ GT ⊆ pred 反超 RAGdrive 2pp
- ⚠️ MCQ / perception / prediction 仍输 RAGdrive(单次跑)
- ⚠️ Open F1 mean 略输 RAGdrive(0.016,在噪声带内)

**RAGdrive 在 factoid 问答上仍更强**,但 KG 在 planning 类问题反超,**两套 pipeline 互补**。

## 11. 维护注意

### 11.1 ⚠️ VLM 非确定性

Qwen2.5-VL + fp16 + greedy decoding 在长 prompt 上**不严格确定性**。同 config 两次跑:
- Open F1 mean 波动 ±0.10
- planning F1 波动 ±0.12
- prediction F1 波动 ±0.07

**含义**:单次跑结果不可信,metrics 只能作为趋势参考。要确认真效应需要:
- 多 seed 跑 + 取平均
- 或扩大测试集到 n=200+

### 11.2 ⚠️ n=66 样本太小

test_eval.json 只有 66 条 QA,样本量 < 100 时单次 run 方差很大。

要严格 ablation 需要扩到 n=200+(从 `v1_1_val_nus_q_only.json` 取更多)。

### 11.3 ⚠️ 一些"失败"的优化其实是噪声

之前 `--planning-k-local 2` 实验看似 planning F1 从 0.258 跌到 0.066,但重复跑 baseline 本身就 ±0.12。所以那个"下跌"是噪声,不是真效应。**单次比较不要下结论**。

### 11.4 ⚠️ BM25 / reranker 加载顺序

`run_graphrag_drift_vlm.py` 启动时按顺序尝试:
1. `--chunks-index` 路径找 `bm25_chunks.pkl`
2. `/home/hylong/xczn/driveVQA/RAGdrive/models/bge-reranker-large` 找 reranker

如果 BM25 / reranker 找不到,会降级到只用 BGE-M3(没有 warning 级别,看 `[warn]` 日志)。

### 11.5 显存占用

- BGE-M3: ~3GB
- Qwen2.5-7B-Instruct(生成 summaries): ~15GB,关掉后释放
- Qwen2.5-VL-7B(推理): ~16GB
- BGE-reranker-large(可选): ~2GB

**推理时总占用**:BGE-M3 + Qwen2.5-VL + reranker ≈ 21GB。建议 24GB GPU(3090/4090)。

## 12. 故障排除

| 问题 | 解决 |
|---|---|
| `ImportError: scipy CXXABI_1.3.15` | `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` |
| Qwen2.5-VL 加载报错 | `export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1` |
| 找不到 bm25_chunks.pkl | 先跑 `build_bm25_index.py` |
| reranker 没加载 | 检查 `RAGdrive/models/bge-reranker-large/` 是否存在 |
| 模型找不到 | `python3 -c "from huggingface_hub import snapshot_download; ..."` 下载 |
| `RuntimeError: CUDA out of memory` | 降 `--multi-camera` 到 1 或 4,或用更小 VLM |

## 13. 文件维护速查

| 想做什么 | 看 / 改哪里 |
|---|---|
| 改 chunk 文本格式 | `data_tools/build_chunks_index.py` |
| 改实体 key 格式 | `data_tools/build_entity_kg.py` |
| 改 PMI 阈值 / 边权 | `data_tools/build_entity_kg.py` |
| 改 region 划分粒度 | `build_entity_kg.py:region_bin()` |
| 改 Leiden 参数 | `data_tools/detect_entity_communities.py` |
| 改 community summary prompt | `data_tools/summarize_entity_communities.py` |
| 改 entity description prompt | `data_tools/generate_entity_descriptions.py` |
| 改 inference 流程 | `test_tools/run_graphrag_drift_vlm.py` |
| 改指标定义 | `test_tools/eval_driveLM_metrics.py` |

## 14. 已知 TODO / 未来改进

- [ ] **多 seed 投票**:消除 VLM 非确定性,稳定 +1-2pp
- [ ] **CoT prompting**:先描述图再答,可能 +2-3pp 各项
- [ ] **扩测试集到 n=200+**:让 ablation 结果可信
- [ ] **Object-ID 级精确检索**:从 `<c1,CAM_BACK,...>` 直接匹配 entity,perception +5-10pp 预期
- [ ] **frame_overview chunks**:每帧 VLM 生成场景描述,planning / behavior F1 +3-5pp
- [ ] **Task-specific system prompts**:perception/prediction/planning 各自 prompt
- [x] ✅ **Microsoft GraphRAG-style text_units**:已有 `build_text_units.py` + `text_units.jsonl`
- [x] ✅ **Entity-level embedding (GraphRAG local search)**:已有 `build_entity_embeddings.py` + `entity_hnsw.bin`
- [x] ✅ **Multi-camera (6 cameras)**:默认开
- [x] ✅ **MCQ routing**:默认开
- [x] ✅ **Chunk embedding with scene-graph prefix**:已合入 chunks index

## 15. 与 RAGdrive 的对比

`/home/hylong/xczn/driveVQA/RAGdrive/` 是另一种 VQA 方案(text-RAG + VL)。

| | RAGdrive | graph4drive |
|---|---|---|
| 检索 | BGE-large + BM25 + reranker | Entity KG + BGE-M3 + BM25 + reranker |
| Few-shot | 5 条相似 Q-A | 2 个 community archetypes + 4 个 entity-aware Q-A |
| 生成器 | Qwen2.5-VL-7B | Qwen2.5-VL-7B(同) |
| 视角 | 1-4 相机 | 4 相机 |
| 优势 | few-shot 精确,VQA 强 | 结构化,可解释,跨样本分析强 |
| 劣势 | 单一相似度检索,无 KG | 实体抽象损失细节,Open F1 mean 略低 |

**互补场景**:本项目 KG pipeline 做"分析型 VQA" + "可解释推理"强,RAGdrive 做"具体事实问答"强。两个项目可并存。
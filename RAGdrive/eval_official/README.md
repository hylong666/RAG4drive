# 用 DriveLM 官方标准评测 RAGdrive

这个目录把 RAGdrive 的预测接到 **DriveLM 官方评测口径**（accuracy / ChatGPT-judge / language / match，加权 final score）。

## 完整流程（4 步）

```
v1_1_train_nus.json
   │  ① prepare_eval_set.py  (抽样 + 官方 extract_data + convert_data)
   ▼
my_test_eval.json  ── 带 GT 答案 + tag 的评测集
   │  ② RAGdrive generate.py  (你的 RAG+VLM 跑答案，需 GPU)
   ▼
preds.jsonl
   │  ③ convert_preds_to_official.py  (按官方 id 规则重编号)
   ▼
preds_official.json
   │  ④ eval_drivelm_official.py  (官方口径打分)
   ▼
accuracy / chatgpt / match / language / final score
```

## 各步命令

### ① 准备评测集（已验证，无需 GPU）

```bash
cd RAGdrive/eval_official
python prepare_eval_set.py \
    --train ../../data/json/v1_1_train_nus.json \
    --n-scenes 5 \
    --challenge-dir ../../../DriveLM/challenge \
    --out-dir .
```

产出 `my_test_eval.json`（5 scene → 195 题）。改 `--n-scenes` 调整规模（0 = 全部 696）。

### ② 用 RAGdrive 生成预测（需要 GPU + drivegraphLM 环境）

关键：让 generate.py 读**带答案的评测集**当题库，题目文本才与评测集对齐。

```bash
conda activate drivegraphLM
cd RAGdrive/data_tools
python generate.py \
    --val-json ../eval_official/my_test_eval.json \
    --output   ../eval_official/preds.jsonl \
    --n 0 \
    --top-k 5 \
    --multi-camera 6
```

> `--n 0` = 跑全部题。generate.py 只读 `Q` 字段，不会看到答案，无泄漏。

### ③ 转成官方格式（无需 GPU）

```bash
cd RAGdrive/data_tools
python convert_preds_to_official.py \
    --preds     ../eval_official/preds.jsonl \
    --test-eval ../eval_official/my_test_eval.json \
    --out       ../eval_official/preds_official.json
```

会打印 `matched / missing`。理想是 `missing 0`。

### ④ 官方打分（需要 OpenAI key 才有完整 final score）

```bash
export OPENAI_API_KEY=sk-...          # 你的 key
# export OPENAI_BASE_URL=...          # 可选：本地/兼容端点
cd RAGdrive/eval_official
python eval_drivelm_official.py \
    --pred preds_official.json \
    --test my_test_eval.json
```

没配 key 时加 `--no-chatgpt` 先跑通其余三项。

## 与官方 evaluation.py 的差异

只有两处环境适配，**评分逻辑和权重完全一致**：

1. **language 后端**：用 `pycocoevalcap`（RAGdrive 已依赖）替代难装的 `language_evaluation`（省去 Java 依赖）。BLEU/ROUGE/CIDEr 的 key 顺序和 0-100 缩放保持一致，final score 聚合不变。
2. **API key**：从环境变量 `OPENAI_API_KEY` 读，不硬编码。

## 评分口径（与官方相同）

```
final = 0.4*chatgpt + 0.2*language + 0.2*match + 0.2*accuracy
tag: 0=accuracy(选择题/精确匹配)  1=chatgpt  2=language(BLEU/ROUGE/CIDEr)  3=match(坐标F1)+chatgpt
```

**图门控**：评测时第一题建图，后续题目只有当问题里的物体坐标在图中（`eval_graph` 通过）才计分——这正是论文图结构在评测里的体现。脚本会打印实际计分题数 `[graph-gated] scored N`。

## 你还需要做的两件事

1. **建 drivegraphLM 环境**（当前机器只有 base，缺 numpy/torch/pycocoevalcap）：见 `RAGdrive/README.md` 第 120 行起的 conda 命令。
2. **准备 OpenAI key**（第 ④ 步完整 final score 需要）。
```

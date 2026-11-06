# Generative Retriever RL

以下单机历史 recipe 为八卡（每卡至少 80GB）、`2e-6` constant LR，
总计 1,000 个 rollout，每个 rollout 对应一次 optimizer step、处理 1,024 query × 64 sample。
存档 run `261015-104046-gr-lr2e6`
以每 query 16 sample 完成训练，最终 checkpoint 为 `iter_0001000`。

## 环境

Miles 使用仓库内 `.venv` 和 `uv.lock` 管理完整 Python 环境。
Miles、Megatron-LM 和 SGLang 均从锁定源码安装：

```bash
cd /home/aiscuser/miles
SGLANG_BUILD_RUST_EXTS=none uv sync --extra generative-retriever
uv run python -m examples.experimental.generative_retriever.runtime.prepare_uv_cuda
uv lock --check
uv sync --extra generative-retriever --check
uv run ruff check examples/experimental/generative_retriever
uv run pyright examples/experimental/generative_retriever/{eval,index,model,reward,runtime,services} examples/experimental/generative_retriever/{run_eval,run_grpo}.py
```

Megatron-LM 锁定 commit `fd15ee20a4f03b03680529baf4b8f7eeed64df1d`，
SGLang 锁定 commit `7f79d11d948aa37b49821475e075369bb0e801f3`；
CUDA 环境使用 Torch 2.13.0+cu130、Transformers 5.12.1。

MI300X 双机（ROCm 10.0.0、Torch `2.11.0+rocm10.0.0`、AITER `acf8fdf93`）使用仓库内的
[prepare_rocm_env.py](../../examples/experimental/generative_retriever/runtime/prepare_rocm_env.py)
与锁定依赖清单
[requirements-rocm.freeze.txt](../../examples/experimental/generative_retriever/runtime/requirements-rocm.freeze.txt)
一键构建 `.venv`、生成 `.venv/activate-rocm.sh` 并同步到所有节点：

```bash
cd /home/aiscuser/miles
python3 -m examples.experimental.generative_retriever.runtime.prepare_rocm_env \
  --hosts node-0,node-1 --slm-root /home/aiscuser/SLMGR_dev/SLMv3
```

## 实验定义

给定 query，训练 GR 生成 doc ID，使其对应的 keyword 高相关，
并提供冻结原始 GR baseline 的合格 keyword 集合之外的增量。

- GR 输出为四段 RQ semantic doc ID。
- Baseline 使用原始 Stage5 checkpoint，训练期间不更新。
- 训练默认学习率为 `2e-6`，使用 constant schedule。
- **Group Size 对齐协议**：
  - **训练阶段**：Rollout（constrained `beam_sample` 无放回采样）与冻结 Baseline（确定性 constrained top-$K$ beam）的 group size **永远保持一致**（默认 `DEFAULT_TRAIN_GROUP_SIZE = 512`）。
  - **离线评测阶段**：Policy 与冻结 Baseline 的确定性 constrained top-$K$ beam group size **永远保持一致**（默认 `DEFAULT_EVAL_GROUP_SIZE = 64`，支持最高 `512`）。
- 使用 SLMGR_dev 的 constrained decoder（`infer_mode=2`）与统一列式 ID 映射。
- Baseline 比较集合按 PM（≥ 0.58）与 SM（≥ 0.82）分别构建 $B_{\text{PM}}(q)$ 和
  $B_{\text{SM}}(q)$；两侧 unique 分别按候选文本是否属于对应集合判定。

### SLMv3 外部源码 patch

`SLMv3/model.py` 所需兼容补丁保存在 Miles 的
[slmv3-transformers-cache.patch](../../examples/experimental/generative_retriever/runtime/patches/slmv3-transformers-cache.patch)，
基于 SLMGR_dev commit `afb5096219af0b64a72feb09b009726fcb513dea`。

在未修改的 SLMGR_dev checkout 上应用（路径按实际 checkout 设置）：

```bash
MILES_ROOT=/path/to/miles
SLMGR_ROOT=/path/to/SLMGR_dev
PATCH="$MILES_ROOT/examples/experimental/generative_retriever/runtime/patches/slmv3-transformers-cache.patch"
git -C "$SLMGR_ROOT" apply --check "$PATCH"
git -C "$SLMGR_ROOT" apply "$PATCH"
```

Miles 的 `--slm-root` 仍指向该 checkout 下的 `SLMv3` 目录。

### SGLang embedding 取消补丁

训练完成目标 query 数后会中止多余的推理请求。锁定的 SGLang commit
`7f79d11d948aa37b49821475e075369bb0e801f3` 对取消的 embedding
请求返回无 `embedding` 字段的结果，但 OpenAI 层仍按成功结果读取该字段，
产生 `KeyError` 和误导性的内部 500。
[sglang-embedding-abort.patch](../../examples/experimental/generative_retriever/runtime/patches/sglang-embedding-abort.patch)
仅在返回结果的 `finish_reason` 是 `abort` 时给出明确的 503，不改变正常的
embedding 输出或模型行为。每台训练节点上的锁定 SGLang checkout 均须应用；
launcher 会在启动前验证，不会自动修改外部源码：

```bash
MILES_ROOT=/home/aiscuser/miles
SGLANG_ROOT="$MILES_ROOT/.venv/src/sglang"
PATCH="$MILES_ROOT/examples/experimental/generative_retriever/runtime/patches/sglang-embedding-abort.patch"
git -C "$SGLANG_ROOT" apply --check "$PATCH"
git -C "$SGLANG_ROOT" apply "$PATCH"
```

此补丁仅影响后续新启动的 SGLang 服务，已在运行的旧服务需先退出。

## 模型与数据

Cosmos 路径以运行时环境变量 `$cosmos_data_path` 为前缀。

| 用途 | 路径 |
| --- | --- |
| Relevance model | `$cosmos_data_path/local/User/yugao/share/qualitymodel/GCQC/GV2.1/ForArcher/Large/` |
| Relevance source snapshot | `$cosmos_data_path/local/User/yugao/share/qualitymodel/GCQC/GV2.1/Student/GPT5_0806/PF_E3.5-2/Multi_Cup/code/` |
| 原始 GR checkpoint | `$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553/` |
| Qwen encoder | `$cosmos_data_path/local/Users/tianchiyang/LLM/Qwen/Qwen3-4B/` |
| NA + INTL 统一列式索引（SLMGRv3Part25） | `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/slmgrv3_part25_na_intl_20261028_20261029/` |
| NA SStream 源表（2026-10-28） | `$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/SLMGR/FlightTable/SLMGRv3Part25/Keyword_NATAPart.2026-10-28.ss` |
| INTL SStream 源表（2026-10-29） | `$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/SLMGR/FlightTable/SLMGRv3Part25/Keyword_INTLTAPart.2026-10-29.ss` |
| 当前训练/推理 ID-text 映射（Global index 1 TAListing） | `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/globalindex1_latest_ta_listing_20261019/asiindex_mapping.ta_listing.tsv` |
| 旧 run / 存档评测使用的 NA 映射 | `$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/UniASI/PublishTables/asiv3index5selmarketna/RawData/asiindex_mapping20261005022223.tsv` |
| 训练数据 | `$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/TA_PA_UnifiedData_202604to202605_Weekly/TAKeyword/train_15M.queryfeat.2026-04-01/Part0.json` |
| TAKeyword 验证集 | `$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/TA_PA_UnifiedData_202411to202511/TAKeyword/valid_100K.queryfeat/Part0` |
| Stage5 RQVAE 重建的 SID-keyword 索引 | `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/demand_10m_stage5_rqvae_20261019/sid_keyword.tsv` |

映射第一列为四段 positional ID，第二列为 keyword 文本。一个 ID 可对应多个文本。
映射中的 ID 等于 decoder token ID 减 2；local RQ code 各位置落在 `[0,255]`。
新索引使用 `TAKeyword/demand_10M.queryfeat/Part0` 的全部 10,000,000 条记录和
冻结 Stage5 的文档编码器/RQVAE；对完全相同的 SID-keyword 对去重后有
9,201,813 对、5,439,670 个 SID。生成与 baseline constrained 推理/文本查询
均已验证；该 TAKeyword 索引与当前 Global TAListing 映射分别保存。
Global index 1 的 `asiindex_mapping.latest.tsv` 按第 9 列
`SALLTAListing` 过滤，保留原 9 列中的 238,836,331 / 267,009,981 行；
子集文件的行数与 SHA-256 已验证，是当前训练、baseline constrained 解码、
reward 映射和后续离线评测共同使用的 ID-text 映射。

### 统一 `(market, docid, keyword)` 列式索引与动态映射协议

由 [build_sstream_index.py](../../examples/experimental/generative_retriever/index/build_sstream_index.py)
直接解析 `SLMGRv3Part25` 下同周期对齐的
`Keyword_NATAPart.2026-10-28.ss`（`353,163,790` 行，2 列 `docid, Keyword`）与
`Keyword_INTLTAPart.2026-10-29.ss`（`132,601,263` 行，3 列 `Market, docid, Keyword`），
合计 `485,765,053` 原始行：

1. **Keyword 原文精确匹配与 Market 并集**：
   - 对 `keyword` 按原始字节严格 exact match 去重（不做 `strip()` 或大小写折叠），
     共得到 `265,127,714` 个唯一 `keyword`（仅 NA `175,014,980` 个、仅 INTL
     `86,594,168` 个、NA 与 INTL 共有 `3,518,566` 个）。
   - 因为 keyword 编码器（`TA-Keyword`）输入仅含 `keyword` 文本、不含 `market`，
     后续更新 `docid2keyword` 时同一 `keyword` 映射到唯一 `docid`；因此将可投放
     market 集合定义为 `keyword` 自身的静态属性（汇总该 `keyword` 在 NA 与 INTL
     所有出现行中的 market 并集），避免同一 `keyword` 因不同召回 `docid` 而只允许部分 market。
2. **静态词表与动态映射解耦存储**：
   - **静态词表 `keywords.parquet`**（`6.83 GB`，训练期间保持不变）：
     `(keyword_id: uint32, keyword: large_binary, in_na: bool, markets: large_string)`，
     其中 `keyword_id` 为连续整数 `0 .. N_kw - 1`，`markets` 为去重排序后的逗号分隔
     market 集合（如 `"NA,DE-AT,DE-CH,DE-DE"`）。
   - **动态映射表 `docid_keywords.parquet` 与 CSR 数组**（`2.63 GB` Parquet +
     `docid_unique.u32.npy`、`docid_offsets.i64.npy`、`docid_kw_ids.u32.npy`）：
     将四段 positional ID `(c0, c1, c2, c3)` 无损打包为单个 `uint32`：
     `docid_u32 = (c0 << 24) | ((c1 - 256) << 16) | ((c2 - 512) << 8) | (c3 - 768)`，
     按 `(docid_u32, keyword_id)` 排序去重后保存全部 `483,649,422` 对映射
     （覆盖 `88,527,153` 个唯一 `docid`）。
   - 训练期若 `docid2keyword` 发生变化，直接在 `mapping_service` 内存中更新
     `docid_u32 <-> keyword_id` 的 CSR 数组，无需从磁盘重读；需要持久化快照时只需写出
     `docid_keywords.parquet` 与 CSR `.npy` 文件，静态 `keywords.parquet` 通过符号链接复用。

```bash
uv run --env-file .env python -m examples.experimental.generative_retriever.index.build_sstream_index \
  --na-ss "$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/SLMGR/FlightTable/SLMGRv3Part25/Keyword_NATAPart.2026-10-28.ss" \
  --intl-ss "$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/SLMGR/FlightTable/SLMGRv3Part25/Keyword_INTLTAPart.2026-10-29.ss" \
  --output "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/slmgrv3_part25_na_intl_20261028_20261029"
```

## Reward

| 评分头 | 输出索引 | 阈值 |
| --- | --- | --- |
| PM | 1 | 0.58 |
| SM | 2 | 0.82 |

Baseline 比较集合（训练期 $K = 512$，评测期默认 $K = 64$，始终与同阶段被测模型一致）：

```text
I_K(q) = frozen_original_GR_constrained_topK_doc_ids(q)
B_PM(q) = {k | k 对应 I_K(q) 中的某个 ID，s_PM(q,k) >= 0.58}
B_SM(q) = {k | k 对应 I_K(q) 中的某个 ID，s_SM(q,k) >= 0.82}
U_PM(q,k) = 1[k not in B_PM(q)]
U_SM(q,k) = 1[k not in B_SM(q)]
```

### 统一 relevance、分头 unique 权重与同 Query 词频折减

```text
P = 1[s_PM >= 0.58]
S = 1[s_SM >= 0.82]
W_PM = clip(s_PM / 0.58, 0, 1)
W_SM = clip(s_SM / 0.82, 0, 1)
Q = (W_PM + W_SM) / 2
c_q(k) = max(1, count_of_rollout_docids_under_q_containing_k)
pm_unique_bonus = (UNIQUE_WEIGHT * 0.5 / c_q(k)) * U_PM * P * Q
sm_unique_bonus = (UNIQUE_WEIGHT * 0.5 / c_q(k)) * U_SM * S * Q
unique_bonus = pm_unique_bonus + sm_unique_bonus
```

- `UNIQUE_WEIGHT = 2.0`：PM / SM unique bonus 分别在该头过阈值且 keyword 不属于对应
  baseline 集合时生效，乘跨头连续质量因子 `Q`，并按该词在同一 query 的 rollout group
  中出现的 doc ID 频次 $c_q(k)$ 做 $1/c_q(k)$ 折减（抑制多个 doc ID 映射到同一热门词的重复刷分）。
- 单头 bonus 最大为 `1.0`，总 `unique_bonus` 最大为 `2.0`，两种 reward mode 的单 keyword 范围均为 `[0,4]`。
- 实现在 [relevance_reward.py](../../examples/experimental/generative_retriever/reward/relevance_reward.py)。

### `discrete_relevance_unique`

```text
discrete_base = P + S
R_discrete = discrete_base + unique_bonus
```

- 默认训练模式；`--reward-mode` 可切换。

### `continuous_relevance_unique`

```text
continuous_base = W_PM + W_SM
R_continuous = continuous_base + unique_bonus
```

- 阈值以下的 relevance 提供连续 shaping signal；训练中作为 shadow metric 汇报。

### Doc ID reward、4 层前缀树 Credit Assignment 与耦合 `doc2k` 更新

- **Doc ID Reward（带 `DOCID_REWARD_FLOOR = 25` 下界）**：
  对同一个 doc ID $d$ 对应的去重 keyword 集合 $K(d)$ 取带分母下界的平均：
  `R_docid(q, d) = sum_{k in K(d)} R_keyword(q, k) / max(|K(d)|, 25)`。
- **4 层 RQ 前缀树 Credit Assignment**（[rollout_metrics.py](../../examples/experimental/generative_retriever/reward/rollout_metrics.py)）：
  在同一 query 的 rollout group 内计算标准化 leaf advantage $A_{\text{leaf}}(d)$ 后，
  按 4 层 RQ 前缀树自底向上聚合子树 `top-k` 均值（`TREE_CREDIT_TOP_K = (8, 4, 2)`，
  分别对应 depth 3 / 2 / 1），并以每层权重 `TREE_CREDIT_LEVEL_WEIGHT = 0.25` 分配到 4 个
  decoder token 位置。
- **耦合 `doc2k` 在线动态映射更新**：
  每个 rollout step 结束时，各 reward worker 通过 `/harvest_step_delta` 汇总本步的动态映射增量并提交至 `mapping_service`（`/update`）：
  1. **Tail-to-Head 注入**：将尾部（rank > `DOC2K_HEAD_RANK_CUTOFF = 64`）命中的 unique good keywords 注入头部 doc ID（每 query 最多 `DOC2K_MAX_ADDS_PER_QUERY = 8` 个，单 doc ID 词数上限 `DOCID_MAX_KEYWORDS = 150`）；
  2. **Dead Keyword 裁剪**：对单步至少被 `DOC2K_PRUNE_MIN_QUERIES = 5` 个不同 query 召回且词数超过 `DOCID_REWARD_FLOOR = 25` 的 doc ID，裁剪本步 0 次过阈值的死词（保留至少 25 个词）。
  同时，Baseline `/prepare` 始终查询初始静态映射（`use_initial=True`），保证 $B(q)$ 参照系固定不变。

GRPO launcher 以 `--reward-mode` 指定训练模式，HTTP reward 请求用 `reward_mode`；
默认 `discrete_relevance_unique`。W&B 汇报 PM/SM 原始分数、pass、threshold
weight、`relevance_quality`、`unique_keyword_rate`、`unique_eligible_rate`、
`unique_quality`、`unique_bonus` 和两种模式的 base / total。

## 官方生成与评分链路

```text
query
  → SLMGR_dev 原始 GR + constrained beam search
  → doc ID
  → 固定官方映射中的 keyword 文本
  → 官方 relevance model
  → 归一化 PM/SM + unique bonus
```

Baseline 与候选均对 ID 映射出的不同 keyword 逐项评分；训练按 `B_PM(q)`、`B_SM(q)`
分别判断两侧 unique。Baseline 由 [model_loading.py](../../examples/experimental/generative_retriever/model/model_loading.py)
加载原始 checkpoint，按当前映射的 ID 前缀约束执行 top-64 beam search。

Relevance 使用原源码快照中的 `MultiCupGraphenePipeline`：

- 从 relevance model 的 `SOURCE_CHECKPOINT.txt` 和 `$cosmos_data_path`
  解析 `Multi_Cup/code/`。
- 源码快照 commit 为 `a57cb597de0891c1e0a2612d12366188b96aa0cc`。
- 文本输入前缀为 `Search query: `、`Document: `；两侧 max length 为 24。
- Query、keyword tower 各输出 `[B,2,128]`，cross scorer 输出 `[B,4]`。
HTTP reward hook 对每个 sample 返回一个 doc ID 的平均 keyword reward。

### Rollout 配置

- [constrained_decode.py](../../examples/experimental/generative_retriever/model/constrained_decode.py)
  使用当前映射的 ID 前缀约束。训练阶段 Baseline 为确定性 top-512 beam，训练 rollout 为
  constrained 512-beam 无放回采样（`decoding_mode="beam_sample"`，每 query 返回 512 个
  互不重复的有效 doc ID 及 4 个 decoder token logprob）。
- Rollout 经 SGLang router 的 `/v1/embeddings` 返回 token 和 logprob；
  同一 query group 只分词一次并共享生成请求，并行准备 reward baseline。Prepare 请求
  每 10ms 合并，最多 512 个 query；learner 使用同一前缀约束在 GPU 上向量化构造 response mask 并计算 logprob。
- 单机 Rollout engine 使用两卡 DP=2，`sglang-server-concurrency=8192`，
  `mem-fraction-static` 按并发与 group size 自适应推导（约 `0.15`）。
  Launcher 检查 mapping、baseline 和 reward 服务的 `/health`，
  复用配置一致的常驻服务。

### 训练规模（单机 8 卡）

- `n_samples_per_prompt=512`（Rollout 与 Baseline 统一使用该 group size）。
- 每轮处理 `2048000 / 512 = 4000` 个 query，即 `rollout_batch_size=4000`。
- Prompt dataset 为 2026-04-01 的 `Part0.json`，包含 15,000,000 条记录；
  在 run 目录通过 `.jsonl` 符号链接接入，按固定 seed shuffle。
- `3,750` 个 rollout 各消费 `4,000` 条记录，恰好覆盖全部 `15,000,000` 条记录（1.0 epoch）；
  源数据中的 query 文本可重复。
- 已完成的 batch 清理多余推理请求时，取消响应不会使训练中断；
  SGLang 对被取消的 embedding 请求返回明确的 503。
  在 batch 尚未完成时，现有 rollout 仍会记录失败 group 并另取 query
  补足；这不等同于对原始数据精确遍历，须单独处理，不能宣称已修复。
- `global_batch_size=2048000`，每次 rollout 恰好形成一个 training step。
- Learner 使用动态 token batching（`--use-dynamic-batch-size`，`max_tokens_per_gpu=131072`，
  保持每个 query 的 512 个连续 sample 不被打散以最大化 prompt 去重效率）；
  冻结 Qwen encoder 使用 backbone 的 last hidden state。
- 每 20 个 training step 保存 checkpoint，最终 step 也保存；包括 model、
  optimizer、LR scheduler、RNG 及 rollout 数据位置。
- 默认执行 `train_async.py`，Train N 与 Rollout N+1 重叠，
  每步同步权重，并使用 rollout logprob 作为 behavior-policy denominator。
  FSDP 跳过额外的 actor 无梯度 logprob 前向；反向训练的前向仍计算
  policy loss 和梯度。此模式不再输出该额外前向生成的 `rollout/log_probs`
  与 `rollout/entropy` 指标。
- 输出根目录为 `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/`。
  `--resume` 使用原 `--run-id` 的最新完整 checkpoint 恢复；
  `--num-rollout` 指总 rollout 数。

```bash
uv run --env-file .env python -m examples.experimental.generative_retriever.run_grpo \
  --run-id <run_id>

uv run --env-file .env python -m examples.experimental.generative_retriever.run_grpo \
  --run-id <existing_run_id> --resume
```

### 64-sample 单步容量

从旧 run 的 `iter_0001000` 和旧映射分别运行一次完整的 1,024 query × 64 sample
（`global_batch_size=65536`），baseline 同时使用 top-64；两次均完成
`iter_0001001`。每卡已有约 12 GiB 常驻显存。仅改变 learner 的
`max_tokens_per_gpu`：

| token 上限 | microbatch 数 | 训练期间 GPU 0 / 1 峰值显存（含常驻，MiB） | learner `train` | `log_probs` | `actor_train` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 33,024 | 33 | 28,461 / 27,441 | 597.9 s | 284.4 s | 283.3 s |
| 65,536 | 17 | 36,079 / 35,061 | 605.9 s | 282.8 s | 291.5 s |

65,536 能跑通，但该单步未缩短 learner 耗时，GPU 0 / 1 训练峰值分别增加
7,618 / 7,620 MiB；当前保留 33,024。逐卡采样、完整日志和状态保存在
`$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/261015-104046-gr-lr2e6/eval/gr64_microbatch_benchmark_20261019/{33024,65536}/`。

### 训练日志

- W&B 图表按指标选择 `train/step`、`rollout/step` 或 `eval/step` 作为横轴。
  全局 `Step` 计数每次日志写入。
- 每个 query group 汇报 reward std、range、unique reward count、
  top-1/top-2 gap 和 zero-std group 比例；GRPO 同时汇报 advantage
  absolute mean/max。

### GPU 分配（单机 8 卡）

当前单机 recipe 使用 8 卡，角色如下（全 8 卡 `GPU 0-7` 并行承载常驻 Relevance Reward Workers，端口 `30490..30497`）：

| 物理 GPU | 角色 | 数量 |
| --- | --- | --- |
| 0-1 | learner / train（与 reward worker 0-1 共卡） | 2 |
| 2-3 | train rollout inference（与 reward worker 2-3 共卡） | 2 |
| 4-5 | frozen baseline constrained beam（与 reward worker 4-5 共卡） | 2 |
| 0-7 | relevance reward workers（全 8 卡，端口 `30490..30497`） | 8 |

Launcher 设置各服务的 GPU visibility；离线评测与训练使用不同的
生成和评分端口。

### 常驻 ID-text 服务

[mapping_service.py](../../examples/experimental/generative_retriever/services/mapping_service.py)
加载上述统一列式索引目录（`keywords.parquet` + `docid_keywords.parquet` / CSR `.npy`），启动后常驻内存提供服务：

```bash
# 加载统一列式索引目录（约 30s 启动）
uv run python -m examples.experimental.generative_retriever.services.mapping_service \
  --mapping "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/slmgrv3_part25_na_intl_20261028_20261029" \
  --host 127.0.0.1 --port 30488
```

- `POST /lookup` 返回 ID 对应的去重 keyword 列表；支持 `use_initial=True` 查询只读初始静态映射（供 Baseline `/prepare` 使用），并支持可选 `market` 参数按离散 `keyword_id` 过滤。
- `POST /update_delta` 与 `POST /update` 在内存中基于离散 `keyword_id: uint32`（`_overlay_kw_ids`）增量更新动态映射（单 doc ID 最多保留 `DOCID_MAX_KEYWORDS = 150` 个词），训练期间无需从磁盘重新读取；`POST /save` 将 `_overlay_kw_ids` 与初始 CSR 合并写出为新的 `docid_keywords.parquet` 与 `.u32.npy` 快照。
- 客户端 [lookup_service](../../examples/experimental/generative_retriever/index/id_text_mapping.py)
  按最多 4,096 个不同 ID 分批请求；服务地址 `http://127.0.0.1:30488`。

### 常驻 baseline 服务

[baseline_service.py](../../examples/experimental/generative_retriever/services/baseline_service.py)
在 GPU 4-5 各加载一份冻结原始 GR，使用初始静态映射前缀树执行确定性 constrained top-$K$ beam（训练期默认 `--docids-per-query 512`，与 rollout `n_samples_per_prompt` 一致）。

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run python -m \
  examples.experimental.generative_retriever.services.baseline_service \
  --checkpoint /path/to/stage5 \
  --qwen-model /path/to/qwen \
  --slm-root /path/to/SLMv3 \
  --id-mapping "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/slmgrv3_part25_na_intl_20261028_20261029" \
  --docids-per-query 512 \
  --port 30487
```

- `POST /generate` 每次最多 512 个 query，每个 query 返回 `docids_per_query` 个不同有效 ID
  及四段 token logprob；完整 query group 缓存在 8,192 条 LRU 中。

### 常驻 reward 服务与二级缓存（Per-Node 8 卡显存直连共享：L1 Score Cache + L2 $q$/$k$ GPU Embedding Tables）

[reward_service.py](../../examples/experimental/generative_retriever/services/reward_service.py)
是单节点多 GPU 显存直连服务（`--devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7`，端口 `30490` / 多机 `31583`）。
单机在单个进程内统一调度本机全部 8 张 GPU（`cuda:0..7`），多机在每台服务节点上运行 1 个管理本机 8 张 GPU 的进程，由 [reward_router_service.py](../../examples/experimental/generative_retriever/services/reward_router_service.py) 按 `q-hash` 将同一 query 的 `/prepare` 和 `/score` 路由到同一节点，并采用 **节点级二级缓存 + 进程内 8 卡 `k-hash` 显存共享架构**：

1. **一级缓存（L1 Score Cache，`--score-cache-size`）**：
   - 节点内全局共享一份 `(q, k) -> (pm_score, sm_score)` 缓存（容量 `8,388,608` 对）。
   - 同一个 query 在 `/prepare`（Baseline 512 个 doc ID 打分）与 `/score`（Rollout 512 个 doc ID 打分）之间重叠的 `(q, k)` 对、同一 query 跨多个 `/score` 批次的重复词、以及数据集中的重复高频 query 直接在 L1 命中返回，零 GPU 计算。
2. **二级缓存（L2 Per-Node 8 卡共享 GPU 显存表，`--query-cache-size` + `--keyword-cache-size`）**：
   - 仅当 `(q, k)` 在 L1 未命中时触发，全部常驻在各 GPU HBM 显存预分配的 `bfloat16` `[capacity, 2, 128]` 张量表（[GpuEmbeddingTable](../../examples/experimental/generative_retriever/reward/relevance_model.py)）中，进程内多线程直连调度 `cuda:0..7`，零跨进程 HTTP RPC、零主机 `mmap`：
     - **`query_features`（`--query-cache-size`）**：按 `xxhash64(q) % 8` 分片常驻在对应 GPU 显存中（每条仅 `512 Bytes`），Baseline `/prepare` 编码一次 $q$ 后，后续 Rollout `/score` 直接在显存中 $O(1)$ 复用 `q_emb`；
     - **`keyword_features`（`--keyword-cache-size`）**：按 `xxhash64(k) % 8`（`k-hash`）在节点内 8 张 GPU 显存间全局分片（每卡 `8,388,608` 条仅占 `4.0 GiB` HBM，单节点 8 卡合并不重复常驻 **`6,710 万`条** 唯一 keyword 向量）。同一节点内任何 keyword 最多只编码一次，8 张 GPU 通过进程内线程池并行执行查表/编码与 `cross_model(q_emb, k_emb)` 打分，多机之间节点级独立闭环、零跨机网络通信。

### 两机 16 GPU 与多机 `2n × 8` GPU 异步 GRPO 配置

[run_grpo_multinode_async.py](../../examples/experimental/generative_retriever/run_grpo_multinode_async.py)
是 Git 仓库内的多机入口，底层执行 **`train_async.py`**（Train N 与 Rollout N+1 重叠）。
`--hosts` 按「服务/rollout 节点、learner 节点」依次成对排列；每个节点 8 卡，总计 `2n` 台：

- 每对节点（16 卡）分配 8 learner（节点 `2k+1` 的 GPU 0–7）、4 rollout（节点 `2k` 的 GPU 4–7）、4 frozen baseline（节点 `2k` 的 GPU 0–3）和 8 reward worker（节点 `2k` 的 GPU 0–7）。
- 每步处理 **`8000n` query × `512` sample**（两机 16 卡即 `8,000` query/step，`global_batch_size = 4,096,000n`）。
- 完整遍历 `15,000,000` 条数据的 optimizer step 数为 **`1875/n`**；当 `1875` 不能被 `n` 整除时拒绝启动，不隐式复用或丢弃数据。
支持 `--num-rollout` 指定不超过该上限的短验收步骤；正式训练省略此参数。
此精确遍历前提是每轮无请求或 group 失败；否则现有 rollout 会
跳过失败 group 并补取 query，不能将运行结果视为精确遍历。
旧 run `261021-114423-121` 在早期 rollout 已跳过失败 group，即使其
`iter_0000040` checkpoint 完整，也不能用它证明精确遍历 1500 万条。

先在首个服务节点检查各节点共享的 Cosmos 路径、ROCm 环境及免密 SSH。
所有节点必须部署同一 Miles commit，且运行工作树干净；入口启动前检查
commit 和工作树状态，另行验证外部 SGLang 补丁、SLMv3 模型源码与
Stage5/Qwen 索引。不一致就拒绝启动，不会覆盖其他节点的工作树。
入口优先检测首个服务节点的 `31480` 常驻 mapping；仅当 `/health` 的 `api_version == 2`、
版本名称和映射路径与当前统一列式索引一致时才复用，
并且不会在本次任务退出时停止该服务。没有常驻服务时才在 `31580` 启动任务专属
mapping；有服务但映射不符则明确报错，不使用错误数据。
冻结 baseline 同样在每个服务节点的 `31581` 验证健康状态和完整配置
（Stage5、Qwen、SLMv3、当前映射、4 GPU、constrained top-512）；一致则复用，
端口被不匹配服务占用时拒绝启动。首次启动的 baseline 在本次训练结束后仍常驻
服务节点 GPU 0–3，供下一次相同配置的 run 复用，不由训练任务清理。确认没有
训练任务依赖它之后，可根据 `31581` 的监听 PID 单独停止；不要按进程名批量清理。
AzureML FUSE 挂载的符号链接不会自动出现在其他节点；入口会在各节点就地创建
训练数据的 `.jsonl` 别名及 SGLang bundle 的模型/Tokenizer 链接，并验证可读，
不会复制或替换原始数据、权重。
W&B 必须提供 `WANDB_API_KEY`；入口与单机 launcher
使用相同的 `miles-run_grpo` project 和 run-id group，密钥不写入 run manifest。
`.env` 只用于导入运行环境，不传给服务进程；建议仅允许文件所有者读取：

Ray 的自动 `uv run` 环境继承在 worker 的打包目录中会再次寻找 `.env`；
该文件不随源码打包。多机入口在本进程内关闭此自动重放，使用节点上已有
`.venv`。W&B 密钥只随训练 driver 的进程环境和 SSH 标准输入到达各节点的
Ray 进程，不写入共享文件、命令行或 Ray 会记录的 runtime-env 配置；不需要
将 `.env` 或密钥文件复制到其他节点。

```bash
cd /home/aiscuser/miles
chmod 600 .env
source .venv/activate-rocm.sh

# 不启动服务或 GPU 任务：核对固定输入、1500 万行、训练参数与 W&B 开关。
uv run --no-sync --env-file .env python -m \
  examples.experimental.generative_retriever.run_grpo_multinode_async \
  --hosts node-0,node-1 --run-id <run_id> --dry-run

# 两机 16 卡：8000 query/step × 1875 step = 1500 万条。
uv run --no-sync --env-file .env python -m \
  examples.experimental.generative_retriever.run_grpo_multinode_async \
  --hosts node-0,node-1 --run-id <run_id>

# 同一配置的断点续跑，--num-rollout 始终表示总步骤数。
uv run --no-sync --env-file .env python -m \
  examples.experimental.generative_retriever.run_grpo_multinode_async \
  --hosts node-0,node-1 --run-id <run_id> --resume

# 六机 48 卡（n=3）举例：24000 query/step × 625 step = 1500 万条。
uv run --no-sync --env-file .env python -m \
  examples.experimental.generative_retriever.run_grpo_multinode_async \
  --hosts node-0,node-1,node-2,node-3,node-4,node-5 --run-id <another_run_id>
```

全局 reward score 请求并发为 `8n`、prepare 并发为 `n`，每个 worker
的转发并发仍为 1。入口只在端口未占用时建立本次运行所需的 reward/router 服务
及 Ray 集群，不终止别的任务；run manifest、服务/训练日志、checkpoint
位于 `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/<run_id>/`。
成功条件为 `latest_checkpointed_iteration.txt` 等于目标 step 数。
新多机入口需先以 `--num-rollout 2` 单独 run-id 验收 W&B、GPU 归属和 checkpoint，
再执行全量训练；短跑并不能证明全量运行已完成。

## 离线 good-count eval 与训练资源隔离

### 固定评测集

- 数据源：`$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/`
  `TA_PA_UnifiedData_202411to202511/TAKeyword/valid_100K.queryfeat/Part0`。
- 从非空 `query` 字段中的 99,812 个 distinct 文本按 RecordIO 源顺序与固定 seed
  `20261019` 做 reservoir 抽样，抽取 **10,000** 个 query 作为固定评测集
  （`queries.jsonl` SHA-256 为 `d36822a89c715a73281af87118bfde972516f3183f99ef6646a82585cf1ce732`；
  早期单机存档评测曾使用同一 seed 抽取的 1,000-query 子集，SHA-256 为
  `ebfb9943bfb58ab0e354dafbfd00fc17729c500d4810231e90d680697185dc1a`）。
- Frozen baseline 与 policy checkpoint 均使用同一映射、同一 Qwen3-4B 编码器与确定性 constrained
  top-$K$ beam（默认 `--docids-per-query 64`，最高支持 `512`，且同一次评测中 Policy 与 Baseline 的 `docids_per_query` 必须严格一致）；映射展开后按 keyword 原文精确匹配、
  大小写敏感去重。
- 评测目录结构为
  `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/<run_id>/eval/valid_100k_10k_top64_step<step>/`，
  包含 `queries.jsonl`、`evaluation.json`、`baseline.jsonl`、`policy-head-step<step>.pt`、
  `policy.jsonl` 与 `summary.json`。

### 评测公式与指标协议

仓库内提供一键式八卡离线评测入口
[run_eval.py](../../examples/experimental/generative_retriever/run_eval.py)
与底层评测模块
[good_count_eval.py](../../examples/experimental/generative_retriever/eval/good_count_eval.py)，
复用 [good_count_service.py](../../examples/experimental/generative_retriever/services/good_count_service.py)
和训练 reward 相同的 [RelevanceScorer](../../examples/experimental/generative_retriever/reward/relevance_model.py)。

给定 query $q$，记 baseline 与 policy 的 top-64 ID 映射去重后的 keyword 集合分别为
$K_B(q)$ 与 $K_P(q)$，对任意集合 $K \in \{K_B(q), K_P(q)\}$ 及口径
$m \in \{\text{PM}, \text{SM}, \text{PM}\cup\text{SM}\}$ 定义过阈值合格子集：

```text
G_PM(K, q) = {k in K | s_PM(q,k) >= 0.58}
G_SM(K, q) = {k in K | s_SM(q,k) >= 0.82}
G_PM_or_SM(K, q) = G_PM(K, q) U G_SM(K, q)

B_m(q) = G_m(K_B(q), q)
P_m(q) = G_m(K_P(q), q)
U_m(q) = P_m(q) - B_m(q)
good_count_m(q) = |U_m(q)|
```

对每个口径 $m \in \{\text{PM}, \text{SM}, \text{PM}\cup\text{SM}\}$ 分别统计两类指标：

1. **Overall Generation Quality（`overall_quality`）**（分别对 $S \in \{B, P\}$ 计算）：
   - `good_density`：$\mathbb{E}_q[|S_m(q)|]$
   - `query_good_rate`：$\mathbb{P}_q(|S_m(q)| > 0)$
   - `keyword_good_rate`：$\mathbb{E}_q[|S_m(q)| / |K_S(q)|]$
2. **Policy Unique Gain（`unique_gain`）**（相对于对应 baseline 合格集合 $B_m(q)$ 的增量）：
   - `unique_good_density`：$\mathbb{E}_q[|U_m(q)|]$（即 $\text{mean}(\text{good\_count}_m)$）
   - `unique_query_good_rate`：$\mathbb{P}_q(|U_m(q)| > 0)$
   - `unique_keyword_good_rate`：$\mathbb{E}_q[|U_m(q)| / |K_P(q)|]$

### 评测接口

评分服务每次同时返回两侧过阈值集合及逐 `(query, keyword)` 的 PM/SM 原始分数。
Baseline 和 policy JSONL 各保存一次该 query 下去重 keyword 的分数、
mapping 版本和 relevance 模型路径；policy 同时保存 PM/SM 的 good count。
评测输出保存两头的原始分数与各自 good count；`--score-head` 选择顶层
`good_count` 分位统计的主口径。`summary.json`（可通过 `--summary-output` 直接写出）
同时汇报：

- 顶层 `good_count` 统计：total、zero fraction、mean、median、p10、p90、p95、p99、max，
  以及 `pm_good_density`、`pm_good_rate`、`sm_good_density`、`sm_good_rate`、
  `baseline_keywords_per_query`、`policy_keywords_per_query`；
- `overall_quality` 与 `unique_gain`（含 `pm`、`sm`、`pm_or_sm` 三种口径）。

Mapping 查询按最多 4,096 个不同 ID 分批；baseline 服务缓存 query 生成结果。

- [model_loading.py](../../examples/experimental/generative_retriever/model/model_loading.py)
  中的 `export_policy_head_checkpoint` 仅读取 FSDP DCP checkpoint（`checkpoints/iter_XXXXXXX/model`）
  中训练更新的 44 个解码头张量（`docEnc2decoder.`、`decoder.`、`lm_head.`）所在分片前缀，
  快速导出 `policy-head-step<step>.pt` 供 `baseline_service.py --policy-checkpoint` 加载。
- 离线 generator 使用 `--docids-per-query 64`（默认在 GPU 0–3 上并行），
  评分服务默认在 GPU 4–7 上以 `--batch-size 256` 运行，评测客户端默认
  `--batch-size 64`；当前批评分与下一批生成流水线并行，且支持从已完成 query 断点续跑。
- 也可通过 [good_count_router_service.py](../../examples/experimental/generative_retriever/services/good_count_router_service.py)
  将同一 query 固定路由到多个隔离单 GPU 进程。离线评测与八卡训练分时运行。

```bash
# 一键运行单机八卡 10,000-query top-64 离线评测（未指定 --step 时自动读取 latest_checkpointed_iteration.txt）
uv run --env-file .env python -m examples.experimental.generative_retriever.run_eval \
  --run-id <run_id> --step <step>

# 或在已启动本地服务（30488 / 30587 / 30588 / 30689）时单独调用底层评测 CLI：
EVAL_ROOT="$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/<run_id>/eval/<eval_id>"

uv run python -m examples.experimental.generative_retriever.eval.good_count_eval \
  --mode baseline --docids-per-query 64 --batch-size 64 \
  --manifest "$EVAL_ROOT/queries.jsonl" --output "$EVAL_ROOT/baseline.jsonl" \
  --generate-url http://127.0.0.1:30587/generate \
  --score-url http://127.0.0.1:30689/score

uv run python -m examples.experimental.generative_retriever.eval.good_count_eval \
  --mode policy --docids-per-query 64 --batch-size 64 --score-head pm \
  --manifest "$EVAL_ROOT/queries.jsonl" \
  --baseline-results "$EVAL_ROOT/baseline.jsonl" \
  --output "$EVAL_ROOT/policy.jsonl" \
  --summary-output "$EVAL_ROOT/summary.json" \
  --generate-url http://127.0.0.1:30588/generate \
  --score-url http://127.0.0.1:30689/score
```

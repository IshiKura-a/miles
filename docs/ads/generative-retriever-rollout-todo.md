# Generative Retriever RL

更新：2026-10-15。当前实现的两卡 learner 更新与 checkpoint 重载已验证；
默认学习率 `2e-6` 的完整在线训练尚未重新运行。

## 环境

Miles 使用仓库内 `.venv` 和 `uv.lock` 管理完整 Python 环境。Miles、
Megatron-LM 和 SGLang 均从锁定源码安装，不使用同名 PyPI wheel：

```bash
python3 -m pip install uv
cd /home/aiscuser/miles
SGLANG_BUILD_RUST_EXTS=none uv sync --extra generative-retriever
uv run python -m examples.experimental.generative_retriever.prepare_uv_cuda
uv lock --check
uv sync --extra generative-retriever --check
uv run ruff check examples/experimental/generative_retriever
uv run pyright examples/experimental/generative_retriever/{baseline_service,constrained_decode,good_count_eval,good_count_router_service,good_count_service,id_text_mapping,relevance_model,relevance_reward,reward,reward_router_service,reward_service,rollout_metrics,run_grpo,sglang_encode_generate,transformers_infer}.py
```

当前锁定 Megatron-LM commit 为
`fd15ee20a4f03b03680529baf4b8f7eeed64df1d`，SGLang commit 为
`7f79d11d948aa37b49821475e075369bb0e801f3`。验证时使用
Torch 2.13.0+cu130、Transformers 5.12.1，CUDA 可用且识别 8 张 GPU。

## 实验定义

给定 query，训练 GR 生成 doc ID，使其对应的 keyword 高相关，
并提供冻结原始 GR baseline 的合格 keyword 集合之外的增量。

- GR 输出为四段 RQ semantic doc ID。
- Baseline 使用原始 Stage5 checkpoint，训练期间不更新。
- 训练默认学习率为 `2e-6`，使用 constant schedule。
- Baseline 和 train rollout 的每 query doc ID 数量必须相同；当前正式值为 16。
- 使用 **SLMGR_dev 的 constrained decoder**：`infer_mode=2`，有效 ID 来自固定版本的官方映射。
- Baseline 比较集合由这 16 个 ID 对应的全部 keyword 中，PM ≥ 0.58
  且 SM ≥ 0.82 的 keyword 去重构成。
- Unique 按候选 keyword 是否属于该比较集合判定。
- SLMGR_dev 除获准的 `SLMv3/model.py` 外只读。

### SLMv3 外部源码 patch

获准的 `SLMv3/model.py` 改动保存在 Miles 的
[slmv3-transformers-cache.patch](../../examples/experimental/generative_retriever/patches/slmv3-transformers-cache.patch)，
基于 SLMGR_dev commit `afb5096219af0b64a72feb09b009726fcb513dea`。
该 patch 兼容新旧 decoder cache、增量生成输入及 beam cache 重排，
不包含 checkpoint 或其他 SLMGR_dev 文件。

在未修改的 SLMGR_dev checkout 上应用（路径按实际 checkout 设置）：

```bash
MILES_ROOT=/path/to/miles
SLMGR_ROOT=/path/to/SLMGR_dev
PATCH="$MILES_ROOT/examples/experimental/generative_retriever/patches/slmv3-transformers-cache.patch"
git -C "$SLMGR_ROOT" apply --check "$PATCH"
git -C "$SLMGR_ROOT" apply "$PATCH"
```

已有同样改动的 checkout 不要重复应用；可用
`git -C "$SLMGR_ROOT" apply --reverse --check "$PATCH"` 验证已应用状态。
Miles 的 `--slm-root` 仍指向该 checkout 下的 `SLMv3` 目录。

## 模型与数据

Cosmos 路径以运行时环境变量 `$cosmos_data_path` 为前缀。

| 用途 | 路径 |
| --- | --- |
| Relevance model | `$cosmos_data_path/local/User/yugao/share/qualitymodel/GCQC/GV2.1/ForArcher/Large/` |
| Relevance source snapshot | `$cosmos_data_path/local/User/yugao/share/qualitymodel/GCQC/GV2.1/Student/GPT5_0806/PF_E3.5-2/Multi_Cup/code/` |
| 原始 GR checkpoint | `$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553/` |
| Qwen encoder | `$cosmos_data_path/local/Users/tianchiyang/LLM/Qwen/Qwen3-4B/` |
| 固定官方 ID-text 映射 | `$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/UniASI/PublishTables/asiv3index5selmarketna/RawData/asiindex_mapping20261005022223.tsv` |
| Weekly 训练数据根目录 | `$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/TA_PA_UnifiedData_202604to202605_Weekly/` |
| 已指定候选数据 | `$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/TA_PA_UnifiedData_202604to202605_Weekly/TAKeyword/train_15M.queryfeat.2026-04-01/Part0` |
| TAKeyword 验证集 | `$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/TA_PA_UnifiedData_202411to202511/TAKeyword/valid_100K.queryfeat/Part0` |
| TAListing 验证集 | `$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/TA_PA_UnifiedData_202411to202511/TAListing/valid_100K.queryfeat/Part0` |
| 既有训练操作参考 | `~/SLMGR_dev/SLMv3/scripts/202511_train_qw3_stage1_4x256.sh` |

映射第一列为四段 positional ID，第二列为 keyword 文本。一个 ID 可对应多个文本。
映射中的 ID 等于 decoder token ID 减 2；local RQ code 各位置落在 `[0,255]`。

## 已批准 reward

| 评分头 | 输出索引 | 固定阈值 | 本阶段使用 |
| --- | --- | --- | --- |
| EEM | 0 | 0.99 | 否 |
| PM | 1 | 0.58 | 是 |
| SM | 2 | 0.82 | 是 |

Baseline 比较集合：

```text
I_16(q) = frozen_original_GR_constrained_top16_doc_ids(q)
B(q) = {k | k 对应 I_16(q) 中的某个 ID，
             s_PM(q,k) >= 0.58 且 s_SM(q,k) >= 0.82}
U(q,k) = 1[k not in B(q)]
```

### 统一 relevance 和 unique 权重

```text
P = 1[s_PM >= 0.58]
S = 1[s_SM >= 0.82]
W_PM = clip(s_PM / 0.58, 0, 1)
W_SM = clip(s_SM / 0.82, 0, 1)
Q = (W_PM + W_SM) / 2
G = max(P, S)
unique_bonus = U * G * Q
```

- Baseline 集合筛选要求 PM、SM 同时满足各自阈值。
- 每个 head 的权重在低于 threshold 时按 `score / threshold` 线性增长，
  达到 threshold 后饱和为 1。
- 至少一个 head 达到 threshold 时才启用 unique bonus；两个 head 都未达到时，
  baseline 外 keyword 也不获得 novelty 奖励。
- Unique bonus 最大为 1，两个 reward mode 的最终范围均为 `[0,3]`。
- 实现在 [relevance_reward.py](../../examples/experimental/generative_retriever/relevance_reward.py)。
- 组合思路参考 [Retrieval-GRPO §3.2.1](https://arxiv.org/html/2511.13885v2)。

### `discrete_relevance_unique`

```text
discrete_base = P + S
R_discrete = discrete_base + unique_bonus
```

- 两头均未达到 threshold：0。
- 单头达到且非 unique：1。
- 单头达到且 unique：`1 + Q`，范围 `[1.5,2)`。
- 双头达到且非 unique：2。
- 双头达到且 unique：3。
- 当前默认使用该模式；`--reward-mode` 可在正式实验中切换训练 reward。

### `continuous_relevance_unique`

```text
continuous_base = W_PM + W_SM
R_continuous = continuous_base + unique_bonus
```

- Threshold 以下的 relevance 作为连续 shaping signal。
- Unique bonus 仍受 `G` gate 约束，不奖励两个 head 都不合格的 novelty。
- 训练时同时计算并汇报该值作为 shadow metric，不参与 optimizer。

### Doc ID reward

两种模式均对同一个 doc ID 对应的去重 keyword reward 取算术平均：
`R_docid = mean(R_keyword)`。GRPO 使用各 doc ID reward 计算组内 advantage。

选择入口：

- Baseline 明细脚本：`--reward-mode continuous_relevance_unique|discrete_relevance_unique`，
  同时输出 `keyword_rewards.jsonl` 和 `docid_rewards.jsonl`。
- GRPO launcher：`--reward-mode`，传入训练参数 `--generative-retriever-reward-mode`。
- HTTP reward 请求通过 `reward_mode` 字段指定模式。
- 训练默认值为 `discrete_relevance_unique`，continuous/discrete 均为受支持的训练超参数。
- W&B 汇报 PM/SM 原始分数、pass、threshold weight、`relevance_quality`、
  `unique_keyword_rate`、`unique_eligible_rate`、`unique_quality`、
  `unique_bonus`、discrete/continuous base 和 total；`train_total`
  等于所选训练模式的 `reward`。

## 官方生成与评分链路

```text
query
  → SLMGR_dev 原始 GR + constrained beam search
  → doc ID
  → 固定官方映射中的 keyword 文本
  → 官方 relevance model
  → 归一化 PM/SM + unique bonus
```

Baseline 的 16 个 ID 展开后，逐 keyword 打分，双阈值筛选并去重得到 `B(q)`。
训练候选同样逐 keyword 打分，按 `B(q)` 判定 unique，再聚合为 doc ID reward。
同一 keyword 即使出现在不同 doc ID 下，也按同一个 baseline 集合元素比较。

Constrained baseline 调用：

- [model_loading.py](../../examples/experimental/generative_retriever/model_loading.py)
  构建 SLMv3 模型并加载原始 checkpoint。
- 官方 `utils.get_restrict_decode_vocab(args, docid_list, id_bias=2)` 构建有效 ID 前缀树。
- `args.infer_mode=2`，约束函数为 `constraint_restrict_decode_vocab`。
- [SLMv3Retriever](../../examples/experimental/generative_retriever/transformers_infer.py)
  调用模型 `generate`，`num_beams=num_return_sequences=16`。

Relevance 使用原源码快照中的 `MultiCupGraphenePipeline`：

- 从 relevance model 的 `SOURCE_CHECKPOINT.txt` 和 `$cosmos_data_path`
  自动解析上述 `Multi_Cup/code/`，无需手工指定；最终解析路径写入 baseline manifest。
- 源码快照 commit 为 `a57cb597de0891c1e0a2612d12366188b96aa0cc`。
- 文本输入前缀为 `Search query: `、`Document: `；两侧 max length 为 24。
- Query、keyword tower 各输出 `[B,2,128]`，cross scorer 输出 `[B,4]`。
- Checkpoint 记录的 GR Transformers 版本为 4.57.3；当前统一 uv 环境使用
  Transformers 5.12.1，并应用批准的 SLMv3 cache patch。真实 16-beam
  baseline 和 relevance reward 均已在该统一环境运行通过。

Miles 已有 SGLang rollout、FSDP policy、GRPO launcher 和 HTTP reward hook。
HTTP reward hook 对每个 sample 获取一个 doc ID 的平均 keyword reward。

### Rollout 配置

- Baseline 使用 [constrained_decode.py](../../examples/experimental/generative_retriever/constrained_decode.py)
  加载 SLMGR_dev 官方前缀约束，并执行 constrained beam search。
- 训练 rollout 使用同一个官方前缀约束执行随机采样，不使用 beam search。
- Baseline 和训练 rollout 都是每个 query 返回 16 个 doc ID；每个 ID 为 4 个
  decoder token，且完整路径必须存在于固定官方映射的前缀树中。
- Baseline beam 返回不同 ID；训练随机采样允许重复，汇报 `distinct_docid_ratio`。
- `/encode` 返回每条 sample 的 4 个 decoder token 和 4 个 logprob。
  同一训练 group 共用一次请求，各 sample 使用不同随机采样结果。
- 每个 query group 在提交共享 `/encode` 请求时，同时提交 reward baseline
  prepare；prepare client 用 10ms 窗口合并最多 512 个 query。Group completion
  同时等待 rollout 与 prepare，耗时为两条链路的最大值而非串行和。
- Miles custom rollout 通过 router 标准 `/v1/embeddings` API 请求 packed token/logprob。
- Learner 对每个响应 prefix 使用同一官方约束计算 logprob。
- GR 服务配置为两卡、DP=2、`mem-fraction-static=0.7`。
- Launcher 先检查 mapping、baseline 和 reward `/health`，配置匹配时复用
  现有常驻服务；仅在服务不存在时启动独立进程并等待 ready 后提交训练。
  健康服务跨训练进程常驻；启动失败时清理本次创建的进程。
- 常驻 mapping/baseline/reward 服务日志写入
  `~/.local/state/miles/generative_retriever/`；
  `<output_dir>/<run_id>/` 保存 prompt alias、launcher 日志和 checkpoint。

### 训练规模

- `n_samples_per_prompt=16`。
- 每轮处理 `16384 / 16 = 1024` 个 query，即 `rollout_batch_size=1024`。
- Prompt dataset 读取完整 15,000,000 条 queryfeat JSONL，确定性 shuffle。
  源文件通过 run 目录内的 `.jsonl` 符号链接接入 Miles。
- 当前 recipe 固定 1,000 steps，共消费 1,024,000 个不重复 dataset row，约
  0.068 epoch；源数据自身可包含相同 query 文本。
  Launcher 校验固定 steps 所需的数据量；重复数据使用显式 epoch recipe。
- `global_batch_size=16384`，每次 rollout 恰好形成一个 training step。
- Learner 使用动态 token batching，每卡每 microbatch 上限 33,024 tokens；
  microbatch 数按实际 prompt 长度计算并在两个 FSDP rank 间取最大值。
- 冻结 Qwen encoder 调用 `Qwen3Model` backbone 获取 last hidden state，
  不计算词表 logits，不保留逐层 hidden states 或 KV cache。
- Checkpoint 默认每 20 个完成的 training step 保存一次，最后一个 step 强制保存；
  路径为 `<output_dir>/<run_id>/checkpoints/iter_<step>/`，包含 model、
  optimizer、LR scheduler、RNG 和下一 rollout ID。
- Rollout/reward/cache 指标以对应的 training global step 汇报。
- 正式在线汇报需要由运行环境或 AzureML secret 注入 `WANDB_API_KEY`；
- 输出根目录为
  `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/`。
- 默认执行 `train_async.py`，Train N 与 Rollout N+1 重叠，
  `update_weights_interval=1`，使用 rollout logprob 作为 behavior-policy denominator；
  `--no-one-step-async` 切换为同步训练。

```bash
uv run python -m examples.experimental.generative_retriever.run_grpo \
  --run-id <run_id> --lr 2e-6
```

### 训练日志

- W&B 指标按各自的 train/rollout/eval step 绑定，支持 1–8 层 metric 路径。
- 每个 query group 汇报 reward std、range、unique reward count、
  top-1/top-2 gap 的 mean、median、p10、p90、min、max，以及 zero-std group 比例。
- GRPO 且 `kl_coef=0` 时，按 group sample std 归一化语义汇报 advantage
  absolute mean/max 和每组 absolute mean 分布。

### GPU 分配

当前单机 recipe 使用 8 卡，角色如下：

| 物理 GPU | 角色 | 数量 |
| --- | --- | --- |
| 0-1 | learner / train | 2 |
| 2-3 | train rollout inference | 2 |
| 4-5 | frozen baseline constrained beam | 2 |
| 6-7 | relevance reward | 2 |

Launcher 固定设置四个角色的 GPU visibility：Ray 只看到 0-3，
baseline 进程只看到 4-5，reward 进程只看到 6-7。

### 常驻 ID-text 服务

[mapping_service.py](../../examples/experimental/generative_retriever/mapping_service.py)
启动时加载一次固定版本官方映射，后续查询读取内存。

```bash
uv run python -m examples.experimental.generative_retriever.mapping_service \
  --mapping "$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/UniASI/PublishTables/asiv3index5selmarketna/RawData/asiindex_mapping20261005022223.tsv" \
  --host 127.0.0.1 --port 30488
```

- `GET /health`：返回映射版本、ID 数和记录数。
- `POST /lookup`：请求 `{"docids":[[244,394,617,1009]]}`，
  返回按输入顺序排列的 keyword list、ID 和 `ok` / `not_found` 状态。
- 同一 ID 内完全相同的 keyword 去重。
- 客户端入口为 `id_text_mapping.lookup_service`，去重后自动拆成最多 4,096 IDs 的请求。
- 当前服务地址为 `http://127.0.0.1:30488`，驻留内存约 25 GiB。

### 常驻 baseline 服务

[baseline_service.py](../../examples/experimental/generative_retriever/baseline_service.py)
在 GPU 4-5 各加载一份冻结原始 GR，使用官方固定映射前缀树执行 16-beam。

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run python -m \
  examples.experimental.generative_retriever.baseline_service \
  --checkpoint /path/to/stage5 \
  --qwen-model /path/to/qwen \
  --slm-root /path/to/SLMv3 \
  --id-mapping /path/to/mapping.tsv \
  --port 30487
```

- `GET /health`：返回两个 worker、内部 device、固定输出数量和 cache 指标。
- `POST /generate`：输入最多 512 个 query，每个 query 返回 16 个不同有效 ID
  及四段 token logprob。
- Query 按 worker 交错分桶并行生成，结果恢复原输入顺序。
- 相同 query 的完整 16-ID group 使用容量 8,192 的 LRU cache。

### 常驻 reward 服务

[reward_service.py](../../examples/experimental/generative_retriever/reward_service.py)
是单 GPU worker。GPU 6 和 7 各运行一个隔离进程，分别缓存 baseline set、
query/keyword feature 和 query-keyword 分数；[reward_router_service.py](../../examples/experimental/generative_retriever/reward_router_service.py)
在 CPU 上按 query 做 rendezvous hashing，并恢复输入 sample 顺序。
Baseline cache miss 时 worker 调用常驻 baseline 服务生成同样数量的 ID，
经 mapping 服务展开并评分后在线构建 `B(q)`。

```bash
CUDA_VISIBLE_DEVICES=6 uv run python -m \
  examples.experimental.generative_retriever.reward_service \
  --baseline-url http://127.0.0.1:30487/generate \
  --mapping-url http://127.0.0.1:30488 \
  --mapping-version asiindex_mapping20261005022223.tsv \
  --relevance-model "$cosmos_data_path/local/User/yugao/share/qualitymodel/GCQC/GV2.1/ForArcher/Large/" \
  --port 30490

CUDA_VISIBLE_DEVICES=7 uv run python -m \
  examples.experimental.generative_retriever.reward_service \
  <同一模型和服务参数> --port 30491

uv run python -m examples.experimental.generative_retriever.reward_router_service \
  --worker-urls http://127.0.0.1:30490/score,http://127.0.0.1:30491/score \
  --port 30489
```

- Worker `GET /health`：单 worker、映射版本、已准备 query、配置和各 cache 指标。
- Router `GET /health`：全部 worker 的 readiness、映射一致性和配置。
- `POST /score`：`samples` 中提供 `prompt` 和四个 `docid_tokens`，
  `reward_mode` 选择连续或离散模式。
- `POST /prepare`：输入最多 512 个 query，router 按同一 rendezvous affinity
  分配到 reward worker；worker 并发调用 baseline service，并提前完成 mapping、
  baseline relevance、baseline set、两侧 feature 和 query-keyword score cache。
- Rollout reward hook 在同一 executor 内用 10ms 窗口合并并发 group，每个
  HTTP request 最多 512 samples；同一时刻只保留一个 router request in-flight，
  由 router 按 query 拆到两个 worker 并行评分，结果恢复到原 group 和 sample 顺序。
- Worker 批量准备 baseline，按最多 4,096 pairs 分块评分，分别去重编码 query
  和 keyword。Baseline、query feature、keyword feature 各缓存 8,192 条，
  query-keyword score 缓存 131,072 条。
- Worker、router 和 rollout reward hook 复用 HTTP 连接；reward request 的
  total timeout 默认为 300 秒。依赖服务失败或超时使对应 rollout group 失败。
- W&B 单独汇报 prepare latency、每 worker routed query、cache delta 和 gauge；
  同一个合并 prepare request 的指标只记录一次。
- Cache 指标包括 capacity、size、hits、misses、evictions；W&B 汇报请求内
  hit/miss/eviction delta 和 capacity/size gauge。
- 返回 `rewards`、逐 ID/keyword 的 `details`、request cache delta 和 cache gauge。
- 同一 query 按 rendezvous hashing 稳定路由到同一个 worker；
  router 汇报每 worker routed sample 累计值。
- Launcher 自动使用 `http://127.0.0.1:30489/score`，无需单独配置 URL。

## 离线 good-count eval 与训练资源隔离

### 固定评测集与已有结果

- 数据源：`$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/`
  `TA_PA_UnifiedData_202411to202511/TAKeyword/demand_1M.queryfeat/Part0`。
  该数据的 `query` 字段为空；**使用 `keyword` 原文作为评测 query**，
  不是从搜索 query 验证集抽样，也不对 keyword 添加模板或改写。
- 从全部 distinct、非空 keyword 文本中做 reservoir sampling，固定 seed
  `20261015`，本次样本数为 **1,000**。同一个 manifest 用于 baseline 和 policy。
- 已评测 checkpoint：run `261014-112122-gr-full`，`iter_0000200`。
  `/home/aiscuser/models/gr-policy-step200.pt` 的全部 44 个 policy-head tensor
  已与该 distributed checkpoint 逐一比对一致。
- 本次 manifest 与结果目录：
  `~/.local/state/miles/generative_retriever/eval/261014-112122-gr-full-step200-1k-20261015/`。
  `queries.jsonl` 保存 query、源 row index 和 market；`evaluation.json` 保存
  数据来源、字段、seed、抽样计数及 manifest SHA-256；
  `baseline.jsonl` 和 `policy.jsonl` 保存逐 query 结果。

### Step 200 的 1K 评测结果（2026-10-15）

Baseline 和 policy 均完成 1,000 个 query 的 constrained top-16，逐 query
校验 ID 去重和 `good_keywords = policy_SM_keywords - baseline_SM_keywords`。

| 指标 | 结果 |
| --- | --- |
| Good count total / mean | 19 / 0.019 |
| Good count 为 0 的 query | 997 / 1,000（99.7%） |
| Median、p10、p90、p95、p99 | 均为 0 |
| Good count max | 17 |
| Top-16 doc ID 集合发生变化的 query | 3 / 1,000 |
| Baseline / policy SM keyword 平均数量 | 64.225 / 64.233 |
| 新增 / 丢失 / 净增 SM keyword 总数 | 19 / 11 / 8 |
| Baseline / policy 评测耗时 | 123.4 秒 / 24.5 秒 |

Policy 阶段复用同一 scorer 的 baseline 评分缓存，耗时不能解读为模型自身
推理速度提升。3 个产生新增 SM keyword 的 query 为：
`best hotels new york`（17）、`download gps maps app`（1）、
`grand mondial login`（1）。该 step 200 与 baseline 的匹配权重数值相同，
以上结果不作为训练收益；权重比较详见评测目录的 `checkpoint_diff.json`。

完整统计保存在上述目录的 `summary.json`；原有常驻服务未替换，评测专用的
policy 和 scoring 服务已在完成后关闭。

### 评测接口

[good_count_eval.py](../../examples/experimental/generative_retriever/good_count_eval.py)
提供 `baseline` / `policy` 两种离线模式，复用
[good_count_service.py](../../examples/experimental/generative_retriever/good_count_service.py)
和训练 reward 相同的 [RelevanceScorer](../../examples/experimental/generative_retriever/relevance_model.py)。
训练 baseline 要同时通过 PM/SM，good-count eval 只看 SM：

```text
B_SM(q) = {k in baseline keywords | SM(q,k) >= 0.82}
P_SM(q) = {k in policy keywords | SM(q,k) >= 0.82}
good_count(q) = |P_SM(q) - B_SM(q)|
```

汇报 total、zero fraction、mean、median、p10、p90、p95、p99、max；
baseline 结果按同一 manifest 和模型配置跨 policy checkpoint 复用。

Mapping 查询统一使用 HTTP connection pool，按 doc ID 去重并自动拆成最多
4,096 IDs 的请求；baseline 服务对同一请求中的重复 query 只执行一次生成。
Policy checkpoint 的全部 head keys 与模型一致。

- 每个 query 生成 16 个 positional doc IDs，展开后的全部 distinct keywords 参与评分。
- 当前批评分与下一批生成重叠；每个服务最多一个 request in-flight，按 manifest
  顺序写入完成的批次；服务错误使评测失败。
- 运行前拒绝空/重复 query manifest、缺失 baseline 和不匹配的续跑输出；
  已完成 query 不重复生成。输出文件按 manifest、checkpoint 和 mode 隔离。
- 离线 worker 可用 `--devices` 指定独立 GPU，或用
  [good_count_router_service.py](../../examples/experimental/generative_retriever/good_count_router_service.py)
  将 query 固定路由到单 GPU 进程。HTTP connections 和 tower/pair caches 都可复用。
- 离线 eval 在训练暂停后运行，或使用独立 GPU 资源。

```bash
uv run python -m examples.experimental.generative_retriever.good_count_eval \
  --mode baseline --manifest /path/to/queries.jsonl \
  --output /path/to/baseline.jsonl \
  --generate-url http://127.0.0.1:30587/generate \
  --score-url http://127.0.0.1:30589/score

uv run python -m examples.experimental.generative_retriever.good_count_eval \
  --mode policy --manifest /path/to/queries.jsonl \
  --baseline-results /path/to/baseline.jsonl --output /path/to/policy.jsonl \
  --generate-url http://127.0.0.1:30588/generate \
  --score-url http://127.0.0.1:30589/score
```

## 待完成

- 默认 `2e-6` 的完整在线训练与新 checkpoint 评测。
- 训练内定期 good-count eval；当前入口仅用于离线评测。
- Relevance kernel 可用性及真实负载性能验证。

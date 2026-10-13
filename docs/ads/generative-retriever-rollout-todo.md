# Generative Retriever RL

更新：2026-10-13。当前阶段：最小端到端已通过，待正式规模和在线 W&B 验收。

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
uv run pyright examples/experimental/generative_retriever/{baseline_service,constrained_decode,relevance_model,relevance_reward,reward,reward_router_service,reward_service,rollout_metrics,run_grpo,sglang_encode_generate,transformers_infer}.py
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
- Baseline 和 train rollout 的每 query doc ID 数量必须相同；当前正式值为 16。
- 使用 **SLMGR_dev 的 constrained decoder**：`infer_mode=2`，有效 ID 来自固定版本的官方映射。
- Baseline 比较集合由这 16 个 ID 对应的全部 keyword 中，PM ≥ 0.58
  且 SM ≥ 0.82 的 keyword 去重构成。
- Unique 按候选 keyword 是否属于该比较集合判定。
- 先完成少量验证，再扩大规模。
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

- 官方 `utils.get_tokenizer_and_model` 加载原始 checkpoint。
- 官方 `utils.get_restrict_decode_vocab(args, docid_list, id_bias=2)` 构建有效 ID 前缀树。
- `args.infer_mode=2`，约束函数为 `constraint_restrict_decode_vocab`。
- 官方 `ASITrainer.prediction_step` 调用生成，`num_beams=num_return_sequences=16`。

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
- `/encode` 返回每条 sample 的 4 个 decoder token 和 4 个 logprob。
  同一训练 group 共用一次请求，各 sample 使用不同随机采样结果。
- Miles custom rollout 通过 router 标准 `/v1/embeddings` API 请求 packed
  token/logprob；不读取 router 中脱敏的 worker URL。
- Learner 对每个响应 prefix 使用同一官方约束计算 logprob。
- GR 服务配置为两卡、DP=2、`mem-fraction-static=0.7`。
- Launcher 先检查 mapping、baseline 和 reward `/health`，配置匹配时复用
  现有常驻服务；仅在服务不存在时启动独立进程并等待 ready 后提交训练。
  训练完成或失败均不停止健康服务，避免重复加载模型和 25GB mapping。
  只有本次新建但未通过 health 的失败进程会按明确 PID 清理。

### 训练规模

- `n_samples_per_prompt=16`。
- 每轮处理 `8192 / 16 = 512` 个 query，即 `rollout_batch_size=512`。
- `global_batch_size=8192`，每次 rollout 恰好形成一个 training step。
- Checkpoint 默认每 20 个完成的 training step 保存一次，最后一个 step 强制保存；
  路径为 `<output_dir>/<run_id>/checkpoints/iter_<step>/`，包含 model、
  optimizer、LR scheduler、RNG 和下一 rollout ID。
- GR rollout 固定按 training global step 汇报；该行为由 rollout 注册时设置，
  不是可调实验参数。Rollout/reward/cache 指标的 `rollout/step` 必须等于
  对应的 training global step。
- 正式在线汇报需要由运行环境或 AzureML secret 注入 `WANDB_API_KEY`；
  禁止写入仓库、workspace 或 launcher 参数。
- 输出根目录为
  `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/`。

### GPU 分配

单机必须恰好使用 8 卡，固定角色如下，禁止角色复用、空卡或把服务临时挪到 CPU：

| 物理 GPU | 角色 | 数量 |
| --- | --- | --- |
| 0-1 | learner / train | 2 |
| 2-3 | train rollout inference | 2 |
| 4-5 | frozen baseline constrained beam | 2 |
| 6-7 | relevance reward | 2 |

Launcher 固定设置四个角色的 GPU visibility：Ray 只看到 0-3，
baseline 进程只看到 4-5，reward 进程只看到 6-7。服务停止时只终止
launcher 自己创建并持有 PID 的失败进程；禁止按名称清理或终止任何已有 GPU 进程。

### Baseline / reward cache

- Baseline query 结果和每个 reward worker 的 query-keyword 分数分别使用 LRU cache。
- 每个 cache 实例的容量固定为 8,192；总容量随 reward worker 数量横向扩展。
- Reward HTTP 响应和 `/health` 暴露 capacity、size、hits、misses、evictions。
- W&B 对每个 request 汇报 hit/miss/eviction delta，对 capacity/size 汇报 gauge；
  不对累计计数快照做 sample 平均。
- 服务跨训练进程常驻并复用，模型和映射只加载一次，不允许每个 step 或每次
  launcher 重载。

### 常驻 ID-text 服务

[mapping_service.py](../../examples/experimental/generative_retriever/mapping_service.py)
启动时加载一次固定版本官方映射，后续查询读取内存。

```bash
python -m examples.experimental.generative_retriever.mapping_service \
  --mapping "$cosmos_data_path/shares/bingads.data.selection/data/User/tianchiyang/OnlineServing/UniASI/PublishTables/asiv3index5selmarketna/RawData/asiindex_mapping20261005022223.tsv" \
  --host 127.0.0.1 --port 30488
```

- `GET /health`：返回映射版本、ID 数和记录数。
- `POST /lookup`：请求 `{"docids":[[244,394,617,1009]]}`，
  返回按输入顺序排列的 keyword list、ID 和 `ok` / `not_found` 状态。
- 同一 ID 内完全相同的 keyword 去重。
- 客户端入口为 `id_text_mapping.lookup_service`，每批最多 4,096 个 ID。
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
query feature 和 query-keyword 分数；[reward_router_service.py](../../examples/experimental/generative_retriever/reward_router_service.py)
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
- 返回 `rewards`、逐 ID/keyword 的 `details`、request cache delta 和 cache gauge。
- 同一 query 稳定落到同一个 worker；增加 worker 时 rendezvous hashing 只迁移
  部分 query。每个 worker 的 LRU 容量为 8,192。
- Router 汇报每 worker routed sample 累计值；只有观测到热点 query 造成队列
  饱和后才启用显式 secondary-worker overflow，并单独计量重复 cache，避免
  无依据的动态负载策略破坏 query affinity。
- Launcher 自动使用 `http://127.0.0.1:30489/score`，无需单独配置 URL。

## 已验证

- 固定映射含 **43,844,117 个不同有效 ID**。
- uv lock/sync、Ruff 和本次变更文件 Pyright 均通过。
- GPU 4-5 的真实 baseline 服务 health 返回两个 worker；真实 query 生成
  16 个不同 ID、四段 logprob 全部有限，首次延迟约 1.16 秒。
- 这 16 个 ID 全部存在于固定官方映射，共展开 204 个 keyword；
  第二次 baseline 请求命中 LRU，服务端延迟约 32 微秒。
- Standalone SGLang rollout health 和真实 `/v1/embeddings` 通过；单 query
  返回 16 个不同 constrained ID、128 个 packed value，全部 logprob 有限。
- 两个隔离 reward worker 和 query-affinity router 的真实 16-sample group
  通过；discrete/continuous reward、threshold weight、unique gate 和 doc ID
  keyword 平均均与公式一致。
- 1-query × 16-sample 最小 GRPO 端到端成功；16 个 sample 全部进入训练，
  rollout 通过 router `/v1/embeddings`，reward 分项和 cache 指标进入 W&B。
- 最小 learner step 0 的 loss 为 0.0017508、gradient norm 为 4.15625，
  train-rollout logprob absolute difference 为 0，forward/backward 和 optimizer
  完成；训练权重同步到 rollout 成功，weight version 从 1 更新到 2。

## 待完成

- 用 512 个 query 完成一次 8,192 global batch 的端到端验证。
- 设置 `WANDB_API_KEY` 后验证在线 W&B、reward/cache 指标和 training global step。

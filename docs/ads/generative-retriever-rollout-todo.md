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
uv run python -m examples.experimental.generative_retriever.prepare_uv_cuda
uv lock --check
uv sync --extra generative-retriever --check
uv run ruff check examples/experimental/generative_retriever
uv run pyright examples/experimental/generative_retriever/{baseline_service,constrained_decode,data_source,fsdp_policy,good_count_eval,good_count_router_service,good_count_service,id_text_mapping,model_loading,relevance_model,relevance_reward,reward,reward_router_service,reward_service,rollout_metrics,run_grpo,sglang_encode_generate,transformers_infer}.py
```

Megatron-LM 锁定 commit `fd15ee20a4f03b03680529baf4b8f7eeed64df1d`，
SGLang 锁定 commit `7f79d11d948aa37b49821475e075369bb0e801f3`；
CUDA 环境使用 Torch 2.13.0+cu130、Transformers 5.12.1。

MI300X 双机（ROCm 10.0.0、Torch `2.11.0+rocm10.0.0`、AITER `acf8fdf93`）使用仓库内的
[prepare_rocm_env.py](../../examples/experimental/generative_retriever/prepare_rocm_env.py)
与锁定依赖清单
[requirements-rocm.freeze.txt](../../examples/experimental/generative_retriever/requirements-rocm.freeze.txt)
一键构建 `.venv`、生成 `.venv/activate-rocm.sh` 并同步到所有节点：

```bash
cd /home/aiscuser/miles
python3 -m examples.experimental.generative_retriever.prepare_rocm_env \
  --hosts node-0,node-1 --slm-root /home/aiscuser/SLMGR_dev/SLMv3
```

## 实验定义

给定 query，训练 GR 生成 doc ID，使其对应的 keyword 高相关，
并提供冻结原始 GR baseline 的合格 keyword 集合之外的增量。

- GR 输出为四段 RQ semantic doc ID。
- Baseline 使用原始 Stage5 checkpoint，训练期间不更新。
- 训练默认学习率为 `2e-6`，使用 constant schedule。
- 训练 rollout 每 query 使用 constrained stochastic beam search 无放回采样 64 个不同 doc ID；
  冻结 baseline 和离线评测每 query 各使用 constrained top-64 beam。
- 使用 SLMGR_dev 的 constrained decoder（`infer_mode=2`）与固定版本官方 ID 映射。
- Baseline 比较集合按 PM（≥ 0.58）与 SM（≥ 0.82）分别构建 $B_{\text{PM}}(q)$ 和
  $B_{\text{SM}}(q)$；两侧 unique 分别按候选文本是否属于对应集合判定。

### SLMv3 外部源码 patch

`SLMv3/model.py` 所需兼容补丁保存在 Miles 的
[slmv3-transformers-cache.patch](../../examples/experimental/generative_retriever/patches/slmv3-transformers-cache.patch)，
基于 SLMGR_dev commit `afb5096219af0b64a72feb09b009726fcb513dea`。

在未修改的 SLMGR_dev checkout 上应用（路径按实际 checkout 设置）：

```bash
MILES_ROOT=/path/to/miles
SLMGR_ROOT=/path/to/SLMGR_dev
PATCH="$MILES_ROOT/examples/experimental/generative_retriever/patches/slmv3-transformers-cache.patch"
git -C "$SLMGR_ROOT" apply --check "$PATCH"
git -C "$SLMGR_ROOT" apply "$PATCH"
```

Miles 的 `--slm-root` 仍指向该 checkout 下的 `SLMv3` 目录。

### SGLang embedding 取消补丁

训练完成目标 query 数后会中止多余的推理请求。锁定的 SGLang commit
`7f79d11d948aa37b49821475e075369bb0e801f3` 对取消的 embedding
请求返回无 `embedding` 字段的结果，但 OpenAI 层仍按成功结果读取该字段，
产生 `KeyError` 和误导性的内部 500。
[sglang-embedding-abort.patch](../../examples/experimental/generative_retriever/patches/sglang-embedding-abort.patch)
仅在返回结果的 `finish_reason` 是 `abort` 时给出明确的 503，不改变正常的
embedding 输出或模型行为。每台训练节点上的锁定 SGLang checkout 均须应用；
launcher 会在启动前验证，不会自动修改外部源码：

```bash
MILES_ROOT=/home/aiscuser/miles
SGLANG_ROOT="$MILES_ROOT/.venv/src/sglang"
PATCH="$MILES_ROOT/examples/experimental/generative_retriever/patches/sglang-embedding-abort.patch"
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

## Reward

| 评分头 | 输出索引 | 阈值 |
| --- | --- | --- |
| PM | 1 | 0.58 |
| SM | 2 | 0.82 |

Baseline 比较集合：

```text
I_64(q) = frozen_original_GR_constrained_top64_doc_ids(q)
B_PM(q) = {k | k 对应 I_64(q) 中的某个 ID，s_PM(q,k) >= 0.58}
B_SM(q) = {k | k 对应 I_64(q) 中的某个 ID，s_SM(q,k) >= 0.82}
U_PM(q,k) = 1[k not in B_PM(q)]
U_SM(q,k) = 1[k not in B_SM(q)]
```

### 统一 relevance 和分头 unique 权重

```text
P = 1[s_PM >= 0.58]
S = 1[s_SM >= 0.82]
W_PM = clip(s_PM / 0.58, 0, 1)
W_SM = clip(s_SM / 0.82, 0, 1)
Q = (W_PM + W_SM) / 2
pm_unique_bonus = 0.5 * U_PM * P * Q
sm_unique_bonus = 0.5 * U_SM * S * Q
unique_bonus = pm_unique_bonus + sm_unique_bonus
```

- PM / SM unique bonus 分别在该头过阈值且 keyword 不属于对应 baseline 集合时生效，
  并乘跨头连续质量因子 `Q`（对单过 SM 的新词提供把 PM 推向 `0.58` 的连续梯度）；
  单头 bonus 最大为 `0.5`，总 `unique_bonus` 最大为 `1`，两种 reward mode 的范围均为 `[0,3]`。
- 实现在 [relevance_reward.py](../../examples/experimental/generative_retriever/relevance_reward.py)。

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

### Doc ID reward

两种模式均对同一个 doc ID 对应的去重 keyword reward 取算术平均：
`R_docid = mean(R_keyword)`。GRPO 使用各 doc ID reward 计算组内 advantage。

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
分别判断两侧 unique。Baseline 由 [model_loading.py](../../examples/experimental/generative_retriever/model_loading.py)
加载原始 checkpoint，按当前映射的 ID 前缀约束执行 top-64 beam search。

Relevance 使用原源码快照中的 `MultiCupGraphenePipeline`：

- 从 relevance model 的 `SOURCE_CHECKPOINT.txt` 和 `$cosmos_data_path`
  解析 `Multi_Cup/code/`。
- 源码快照 commit 为 `a57cb597de0891c1e0a2612d12366188b96aa0cc`。
- 文本输入前缀为 `Search query: `、`Document: `；两侧 max length 为 24。
- Query、keyword tower 各输出 `[B,2,128]`，cross scorer 输出 `[B,4]`。
HTTP reward hook 对每个 sample 返回一个 doc ID 的平均 keyword reward。

### Rollout 配置

- [constrained_decode.py](../../examples/experimental/generative_retriever/constrained_decode.py)
  使用当前映射的 ID 前缀约束。Baseline 为确定性 top-64 beam，训练 rollout 为每 query
  64-beam constrained stochastic beam search（Kool et al., 2019 Gumbel-Top-k 无放回采样，
  返回 64 个互不重复的有效 doc ID 及未缩放的真实策略 token logprob）。
- Rollout 经 SGLang router 的 `/v1/embeddings` 返回 token 和 logprob；
  同一 query group 只分词一次并共享生成请求，并行准备 reward baseline。Prepare 请求
  每 10ms 合并，最多 512 个 query；learner 使用同一前缀约束计算 logprob。
- Rollout engine 使用两卡 DP=2、`mem-fraction-static=0.7`；
  `sglang-server-concurrency=4096` 可同时容纳 64 个完整的 64-sample query group。
  Launcher 检查 mapping、baseline 和 reward 服务的 `/health`，
  复用配置一致的常驻服务。

### 训练规模

- `n_samples_per_prompt=64`。
- 每轮处理 `65536 / 64 = 1024` 个 query，即 `rollout_batch_size=1024`。
- Prompt dataset 为 2026-04-01 的 `Part0.json`，包含 15,000,000 条记录；
  在 run 目录通过 `.jsonl` 符号链接接入，按固定 seed shuffle。
- 1,000 个 rollout 各消费 1,024 条记录，合计 1,024,000 条记录
  （约 0.068 epoch）；源数据中的 query 文本可重复。
- 已完成的 batch 清理多余推理请求时，取消响应不会使训练中断；
  SGLang 对被取消的 embedding 请求返回明确的 503。
  在 batch 尚未完成时，现有 rollout 仍会记录失败 group 并另取 query
  补足；这不等同于对原始数据精确遍历，须单独处理，不能宣称已修复。
- `global_batch_size=65536`，每次 rollout 恰好形成一个 training step。
- Learner 使用动态 token batching，每卡每 microbatch 上限 33,024 tokens；
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

### GPU 分配

当前单机 recipe 使用 8 卡，角色如下：

| 物理 GPU | 角色 | 数量 |
| --- | --- | --- |
| 0-1 | learner / train | 2 |
| 2-3 | train rollout inference | 2 |
| 4-5 | frozen baseline constrained beam | 2 |
| 6-7 | relevance reward | 2 |

Launcher 设置各服务的 GPU visibility；离线评测与训练使用不同的
生成和评分端口。

### 常驻 ID-text 服务

[mapping_service.py](../../examples/experimental/generative_retriever/mapping_service.py)
启动时加载一次当前 Global TAListing 映射，后续查询读取内存。

```bash
uv run python -m examples.experimental.generative_retriever.mapping_service \
  --mapping "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/globalindex1_latest_ta_listing_20261019/asiindex_mapping.ta_listing.tsv" \
  --host 127.0.0.1 --port 30488
```

- `POST /lookup` 返回 ID 对应的 keyword 列表；同一 ID 内相同文本去重。
- 客户端 [lookup_service](../../examples/experimental/generative_retriever/id_text_mapping.py)
  按最多 4,096 个不同 ID 分批请求；服务地址 `http://127.0.0.1:30488`。

### 常驻 baseline 服务

[baseline_service.py](../../examples/experimental/generative_retriever/baseline_service.py)
在 GPU 4-5 各加载一份冻结原始 GR，使用当前映射前缀树执行 64-beam。

```bash
CUDA_VISIBLE_DEVICES=4,5 uv run python -m \
  examples.experimental.generative_retriever.baseline_service \
  --checkpoint /path/to/stage5 \
  --qwen-model /path/to/qwen \
  --slm-root /path/to/SLMv3 \
  --id-mapping "$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/index/globalindex1_latest_ta_listing_20261019/asiindex_mapping.ta_listing.tsv" \
  --port 30487
```

- `POST /generate` 每次最多 512 个 query，每个 query 返回 64 个不同有效 ID
  及四段 token logprob；完整 query group 缓存在 8,192 条 LRU 中。

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
  --mapping-version asiindex_mapping.ta_listing.tsv \
  --relevance-model "$cosmos_data_path/local/User/yugao/share/qualitymodel/GCQC/GV2.1/ForArcher/Large/" \
  --port 30490

CUDA_VISIBLE_DEVICES=7 uv run python -m \
  examples.experimental.generative_retriever.reward_service \
  <同一模型和服务参数> --port 30491

uv run python -m examples.experimental.generative_retriever.reward_router_service \
  --worker-urls http://127.0.0.1:30490/score,http://127.0.0.1:30491/score \
  --port 30489
```

- `POST /prepare` 接收最多 512 个 query，准备 baseline 集合与 relevance 缓存；
  `POST /score` 每次接收最多 2,048 个 sample，包含 `prompt`、四个
  `docid_tokens` 和 `reward_mode`，
  返回 reward、逐 keyword 明细及 cache 指标。
- Router 按 query 做 rendezvous hashing；worker 使用 2,048 条 keyword 的
  tower batch，最多 8,192 个 query-keyword pair 一块；cross scorer
  每批 256 对。Query/keyword tower batch 必须为 4 的倍数，末批补齐到 4 的倍数，
  以保持评分不随请求大小变化。
  Baseline 与两侧 feature 各缓存 8,192 条，pair score
  缓存 131,072 条。
- Rollout hook 在 10ms 窗口合并最多 2,048 个 sample，训练 HTTP 超时默认 1,200 秒；
  日志分别记录 prepare 和 score 请求的数量与平均服务时延。
  训练 reward URL 为 `http://127.0.0.1:30489/score`。
  Worker 日志的 `reward_baseline_stage`、`reward_score_stage` 和
  `graphene_score_stage` 分别记录 baseline 生成、映射、tower/cross 评分、
  请求锁等待与聚合耗时；router 日志记录转发耗时。不输出 query/keyword 原文，
  用于区分模型计算和 HTTP/CPU 排队瓶颈。

### 两机 16 GPU reward 配置

两机 16×MI300X 使用
8 learner、4 rollout、4 frozen baseline、8 reward worker（其中 4 个与
rollout 同卡）。先前的两机吞吐测试每步使用 4,096 query × 64 sample；以下
正式全量训练为精确遍历 15,000,000 条数据，改用 4,000 query × 64 sample。
Reward 配置为
每个 `/score` 请求最多 2,048 sample、客户端并发 8、router 每 worker
转发并发 1、prepare 并发 1；baseline 模型和 reward 定义不变。
首轮完整 rollout 中，最后一个 `/prepare` 到最后一个 `/score` 的间隔
从客户端并发 1 时约 58–59 秒降至并发 8 时的 2.2 秒。

### 多机异步 GRPO：2n × 8 GPU

[run_grpo_multinode_async.py](../../examples/experimental/generative_retriever/run_grpo_multinode_async.py)
是 Git 仓库内的多机入口，底层执行 **`train_async.py`**（Train N 与 Rollout N+1
重叠），而非 `train.py`。不依赖 `local/` 中的实验脚本或复制的模型/数据：
Stage5、Qwen、relevance、训练集和当前 TAListing 映射均取上表固定原始路径；
reward 定义、64-beam stochastic beam rollout、`2e-6` constant LR 和每步权重同步保持不变。

`--hosts` 按「服务/rollout 节点、learner 节点」依次成对排列；每个节点 8 卡，
总计 `2n` 台。每对分配 8 learner、4 rollout、4 frozen baseline 和
8 reward worker；rollout 与对应服务节点的 4 个 reward worker 共用 GPU 4–7。
learner/rollout 的 Ray bundle 分别固定到对应节点 IP，避免 Ray 按 IP 排序
误分配训练卡；角色 IP 由入口从 `--hosts` 推导并在内部传递，无需额外
CLI 参数。每步 `4000n` query × 64 sample，`global_batch_size=256000n`，
learner token budget 每卡 `524288`。完整遍历的 optimizer step 数为
`3750/n`；当 `3750` 不能被 `n` 整除时拒绝启动，不隐式复用或丢弃数据。
支持 `--num-rollout` 指定不超过该上限的短验收步骤；正式训练省略此参数。
此精确遍历前提是每轮无请求或 group 失败；否则现有 rollout 会
跳过失败 group 并补取 query，不能将运行结果视为精确遍历。
旧 run `261021-114423-121` 在早期 rollout 已跳过失败 group，即使其
`iter_0000040` checkpoint 完整，也不能用它证明精确遍历 1500 万条。

先在首个服务节点检查各节点共享的 Cosmos 路径、ROCm 环境及免密 SSH。
所有节点必须部署同一 Miles commit，且运行工作树干净；入口启动前检查
commit 和工作树状态，另行验证外部 SGLang 补丁、SLMv3 模型源码与
Stage5/Qwen 索引。不一致就拒绝启动，不会覆盖其他节点的工作树。
入口优先检测首个服务节点的 `31480` 常驻 mapping；仅当 `/health` 的版本、
238,836,331 行、文件大小和底层文件的逐字节内容均与上表固定映射一致时才复用，
并且不会在本次任务退出时停止该服务。没有常驻服务时才在 `31580` 启动任务专属
mapping；有服务但映射不符则明确报错，不使用错误数据。
冻结 baseline 同样在每个服务节点的 `31581` 验证健康状态和完整配置
（Stage5、Qwen、SLMv3、当前映射、4 GPU、constrained top-64）；一致则复用，
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

# 两机 16 卡：4000 query/step × 3750 step = 1500 万条。
uv run --no-sync --env-file .env python -m \
  examples.experimental.generative_retriever.run_grpo_multinode_async \
  --hosts node-0,node-1 --run-id <run_id>

# 同一配置的断点续跑，--num-rollout 始终表示总步骤数。
uv run --no-sync --env-file .env python -m \
  examples.experimental.generative_retriever.run_grpo_multinode_async \
  --hosts node-0,node-1 --run-id <run_id> --resume

# 四机 32 卡举例：8000 query/step × 1875 step = 1500 万条。
uv run --no-sync --env-file .env python -m \
  examples.experimental.generative_retriever.run_grpo_multinode_async \
  --hosts node-0,node-1,node-2,node-3 --run-id <another_run_id>
```

全局 reward score 请求并发为 `8n`、prepare 并发为 `n`，每个 worker
的转发并发仍为 1。入口只在端口未占用时建立本次运行所需的 reward/router 服务
及 Ray 集群，不终止别的任务；run manifest、服务/训练日志、checkpoint
位于 `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/<run_id>/`。
成功条件为 `latest_checkpointed_iteration.txt` 等于目标 step 数。
新多机入口需先以 `--num-rollout 2` 单独 run-id 验收 W&B、GPU 归属和 checkpoint，
再执行全量训练；短跑并不能证明全量运行已完成。

## 离线 good-count eval 与训练资源隔离

### 固定评测集与最终 checkpoint

- 数据源：`$cosmos_data_path/local/Users/tianchiyang/Data/MZLog/`
  `TA_PA_UnifiedData_202411to202511/TAKeyword/valid_100K.queryfeat/Part0`。
  从非空 `query` 字段中的 99,812 个 distinct 文本用固定 seed `20261019`
  抽取 1,000 个 query。
- Frozen baseline 和 run `261015-104046-gr-lr2e6` 的最终 `iter_0001000`
  的存档评测使用上表旧版 NA 映射及同一 Qwen 模型、constrained 64-beam / top-64。
- 结果持久化在训练 run 的
  `$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/261015-104046-gr-lr2e6/eval/valid_100k_1k_top64_20261019/`。
  `queries.jsonl` 保存固定样本，`evaluation.json` 保存来源和 manifest SHA-256，
  `baseline.jsonl`、`policy.jsonl` 保存逐 query SM 结果，`pm/results.jsonl`
  保存 PM 结果；两头汇总分别为 `summary.json`、`pm/summary.json`。

Baseline 和 policy 均返回每 query 64 个不同 ID；映射后按 keyword 原文精确匹配、
大小写敏感去重，平均分别为 298.785 和 314.400 个不同 keyword / query。

| 评分头 | Baseline 合格 keyword / query | Policy 合格 keyword / query | Good density | Good rate |
| --- | ---: | ---: | ---: | ---: |
| SM ≥ 0.82 | 183.812 | 192.518 | 40.324 | 88.5% |
| PM ≥ 0.58 | 63.406 | 69.743 | 15.361 | 77.4% |

Good density 为 `mean(good_count)`；good rate 为 `good_count > 0` 的 query 占比。

### 评测接口

[good_count_eval.py](../../examples/experimental/generative_retriever/good_count_eval.py)
提供 `baseline` / `policy` 两种离线模式，复用
[good_count_service.py](../../examples/experimental/generative_retriever/good_count_service.py)
和训练 reward 相同的 [RelevanceScorer](../../examples/experimental/generative_retriever/relevance_model.py)。
训练 baseline 要同时通过 PM/SM；离线 good-count eval 对 PM 和 SM 分别计算：

```text
B_SM(q) = {k in baseline keywords | SM(q,k) >= 0.82}
P_SM(q) = {k in policy keywords | SM(q,k) >= 0.82}
B_PM(q) = {k in baseline keywords | PM(q,k) >= 0.58}
P_PM(q) = {k in policy keywords | PM(q,k) >= 0.58}
good_count_SM(q) = |P_SM(q) - B_SM(q)|
good_count_PM(q) = |P_PM(q) - B_PM(q)|
```

评分服务每次同时返回两侧过阈值集合及逐 `(query, keyword)` 的 PM/SM 原始分数。
Baseline 和 policy JSONL 各保存一次该 query 下去重 keyword 的分数、
mapping 版本和 relevance 模型路径；policy 同时保存 PM/SM 的 good count。
评测输出保存两头的原始分数与各自 good count；`--score-head` 只选择
`good_count` 和 CLI 汇总的主口径，默认 `sm`。分数可用于离线阈值重算。

汇报 total、zero fraction、mean、median、p10、p90、p95、p99、max；
baseline 结果按同一 manifest 和模型配置跨 policy checkpoint 复用。

Mapping 查询按最多 4,096 个不同 ID 分批；baseline 服务缓存 query 生成结果。

- 离线 generator 使用 `--docids-per-query 64`，评测 CLI 同样指定 64；
  训练 reward 的 baseline 也是 top-64。映射得到的不同 keyword 均参与评分。
- 评测按 manifest 顺序写入结果，可从已完成 query 续跑；当前批评分与下一批
  生成并行。
- 离线 worker 可用 `--devices` 指定 GPU，或用
  [good_count_router_service.py](../../examples/experimental/generative_retriever/good_count_router_service.py)
  将同一 query 固定路由到单 GPU 进程。离线评测与八卡训练分时运行。

```bash
EVAL_ROOT="$cosmos_data_path/local/User/zihao/online_rl/gr_offline_syn/<run_id>/eval/<eval_id>"

uv run python -m examples.experimental.generative_retriever.good_count_eval \
  --mode baseline --docids-per-query 64 --batch-size 8 \
  --manifest "$EVAL_ROOT/queries.jsonl" --output "$EVAL_ROOT/baseline.jsonl" \
  --generate-url http://127.0.0.1:30587/generate \
  --score-url http://127.0.0.1:30689/score

uv run python -m examples.experimental.generative_retriever.good_count_eval \
  --mode policy --docids-per-query 64 --batch-size 8 \
  --manifest "$EVAL_ROOT/queries.jsonl" \
  --baseline-results "$EVAL_ROOT/baseline.jsonl" --output "$EVAL_ROOT/policy.jsonl" \
  --generate-url http://127.0.0.1:30588/generate \
  --score-url http://127.0.0.1:30689/score
```

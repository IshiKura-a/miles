# Generative Retriever RL

更新：2026-10-10。当前阶段：少量推理和评分验证。

## 实验定义

给定 query，训练 GR 生成 doc ID，使其对应的 keyword 高相关，
并提供冻结原始 GR baseline 的合格 keyword 集合之外的增量。

- GR 输出为四段 RQ semantic doc ID。
- Baseline 使用原始 Stage5 checkpoint，固定生成 500 个不同 doc ID，训练期间不更新。
- 使用 **SLMGR_dev 的 constrained decoder**：`infer_mode=2`，有效 ID 来自固定版本的官方映射。
- Baseline 比较集合由这 500 个 ID 对应的全部 keyword 中，PM ≥ 0.58
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
| 原始 GR checkpoint | `$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553/` |
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
I_500(q) = frozen_original_GR_constrained_top500_doc_ids(q)
B(q) = {k | k 对应 I_500(q) 中的某个 ID，
             s_PM(q,k) >= 0.58 且 s_SM(q,k) >= 0.82}
U(q,k) = 1[k not in B(q)]
```

### `continuous_relevance_unique`

```text
N_PM = max(0, (s_PM - 0.58) / 0.42)
N_SM = max(0, (s_SM - 0.82) / 0.18)
r = N_PM + N_SM
R = r + lambda * r * U
lambda = 1
```

- Baseline 集合筛选要求 PM、SM 同时满足各自阈值。
- 候选 keyword 的连续 relevance reward 将两头分别归一化后相加。
- `r` 范围为 `[0,2]`；最终 `R` 范围为 `[0,4]`。
- 两头均未超过阈值时 reward 为 0。
- Unique bonus 作用于合并 relevance。
- 实现在 [relevance_reward.py](../../examples/experimental/generative_retriever/relevance_reward.py)。
- 组合思路参考 [Retrieval-GRPO §3.2.1](https://arxiv.org/html/2511.13885v2)。

### `discrete_relevance_unique`

| Keyword 条件 | Reward |
| --- | --- |
| PM、SM 均未达到阈值 | 0 |
| PM、SM 仅一个达到阈值 | 1 |
| 两头均达到阈值，keyword 属于 `B(q)` | 2 |
| 两头均达到阈值，keyword 在 `B(q)` 外 | 3 |

### Doc ID reward

两种模式均对同一个 doc ID 对应的去重 keyword reward 取算术平均：
`R_docid = mean(R_keyword)`。GRPO 使用各 doc ID reward 计算组内 advantage。

选择入口：

- Baseline 明细脚本：`--reward-mode continuous_relevance_unique|discrete_relevance_unique`，
  同时输出 `keyword_rewards.jsonl` 和 `docid_rewards.jsonl`。
- GRPO launcher：`--reward-mode`，传入训练参数 `--generative-retriever-reward-mode`。
- HTTP reward 请求通过 `reward_mode` 字段指定模式。
- 默认值为 `continuous_relevance_unique`。

## 官方生成与评分链路

```text
query
  → SLMGR_dev 原始 GR + constrained beam search
  → doc ID
  → 固定官方映射中的 keyword 文本
  → 官方 relevance model
  → 归一化 PM/SM + unique bonus
```

Baseline 的 500 个 ID 展开后，逐 keyword 打分，双阈值筛选并去重得到 `B(q)`。
训练候选同样逐 keyword 打分，按 `B(q)` 判定 unique，再聚合为 doc ID reward。
同一 keyword 即使出现在不同 doc ID 下，也按同一个 baseline 集合元素比较。

Constrained baseline 调用：

- 官方 `utils.get_tokenizer_and_model` 加载原始 checkpoint。
- 官方 `utils.get_restrict_decode_vocab(args, docid_list, id_bias=2)` 构建有效 ID 前缀树。
- `args.infer_mode=2`，约束函数为 `constraint_restrict_decode_vocab`。
- 官方 `ASITrainer.prediction_step` 调用生成，`num_beams=num_return_sequences=500`。

Relevance 使用原源码快照中的 `MultiCupGraphenePipeline`：

- 文本输入前缀为 `Search query: `、`Document: `；两侧 max length 为 24。
- Query、keyword tower 各输出 `[B,2,128]`，cross scorer 输出 `[B,4]`。
- 已验证的分进程运行环境：GR 使用 Transformers 4.57.3，relevance 使用 5.12.1。

Miles 已有 SGLang rollout、FSDP policy、GRPO launcher 和 HTTP reward hook。
HTTP reward hook 对每个 sample 获取一个 doc ID 的平均 keyword reward。

### SGLang constrained rollout

- 使用 [constrained_decode.py](../../examples/experimental/generative_retriever/constrained_decode.py)
  加载 SLMGR_dev 的官方前缀约束，并调用其生成模型的 constrained beam search。
- Bundle 参数：`--slm-root`、`--id-mapping`、`--num-beams`、`--num-return-sequences`。
  Beam 默认 500，返回数量由 `n_samples_per_prompt` 决定。
- `/encode` 返回每条 beam 的 4 个 decoder token 和 4 个 logprob，按 beam 顺序展开。
  同一训练 group 共用一次请求，各 sample 使用不同 beam rank。
- Learner 对每个响应 prefix 使用同一官方约束计算 logprob。
- GR 服务配置为两卡、DP=2、`mem-fraction-static=0.7`。

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

### 常驻 reward 服务

[reward_service.py](../../examples/experimental/generative_retriever/reward_service.py)
加载完成的 baseline manifest 和官方 relevance pipeline，缓存 query-keyword 分数。
调用常驻映射服务取得 keyword，按请求模式评分并对每个 doc ID 取平均。

```bash
python -m examples.experimental.generative_retriever.reward_service \
  --baseline-dir /path/to/filtered-baseline \
  --mapping-url http://127.0.0.1:30488 --host 127.0.0.1 --port 30489
```

- `GET /health`：映射版本及已准备 baseline query 数。
- `POST /score`：`samples` 中提供 `prompt` 和四个 `docid_tokens`，
  `reward_mode` 选择连续或离散模式。
- 返回 `rewards` 与逐 ID、逐 keyword 的分项 `details`。
- 当前地址为 `http://127.0.0.1:30489/score`，已准备首条验证 query 的 baseline。
- 新 query 需要先构建 baseline；缺失 baseline 返回 HTTP 409。
- Launcher 使用 `--reward-url` 和 `--reward-mode` 接入该服务。

## 已验证

- 使用 SLMGR_dev 官方数据处理与模型入口完成少量推理。
- 固定映射含 **43,844,117 个不同有效 ID**。
- 官方 constrained decoder 对 **4 个 query 各生成 500 个不同 ID，全部属于固定映射**。
- 官方 relevance model 可对映射文本输出有限 PM/SM 分数；归一化 reward 边界检查通过。
- Miles 的 MT5 tokenizer 别名兼容修复可加载未修改的官方 utils。
- 真实 SGLang `/encode` 在两卡 DP=2、70% 显存配置下返回 500 个不同 ID、
  2,000 个有限 token logprob；500 个 ID 全部经常驻服务查到文本。
- 相同小模型权重下，SGLang 使用的生成 adapter 与官方模型的 500-beam
  token/logprob 一致；learner logprob 对齐、梯度有限，group 返回不同 beam。
- 常驻服务对 500 个 constrained ID 的三次 HTTP 查询耗时约 19、9、7 ms；
  该 query 共返回 5,779 个 ID-keyword 对。
- 真实 reward 服务对 500 个 baseline ID 的连续、离散两种输出均与保存的
  keyword 平均分一致；缓存请求耗时约 101 ms / 80 ms。
- 现有 rollout HTTP hook 对两种模式各 8 个实际 sample 调用通过。
- 11 个实际 SGLang 生成的 baseline 外 ID 共对应 111 个 keyword；
  两种模式的逐 keyword 集合判定及 ID 均值检查通过。

真实 SGLang 的 top-500 与已保存的官方 Torch prefill 结果重合 489 个 ID；
两者使用不同 prefill 实现和运行环境。上述相同权重数值对照为小模型检查。

Constrained 输出保存于本次会话的 `files/official-gr-constrained-reward-inputs.json`，
记录了解码模式、约束函数、映射版本和逐 query 的 ID 成员检查。

### Baseline keyword set

[build_baseline_rewards.py](../../examples/experimental/generative_retriever/build_baseline_rewards.py)
读取官方 constrained 输出，通过常驻映射服务展开并去重 keyword，
调用官方 relevance pipeline，保存双阈值 baseline set 和逐 keyword reward。

首条验证 query：`moana necklace heart of te fiti child`。

| 项目 | 数量 |
| --- | --- |
| 原始 GR constrained ID | 500 |
| 成功映射 ID | 500 |
| ID-keyword 对 | 5,779 |
| 去重后评分 keyword | 4,003 |
| PM ≥ 0.58 | 750 |
| SM ≥ 0.82 | 2,217 |
| 双阈值 baseline keyword set | 689 |

输出位于本次会话 `files/filtered-baseline-one-query/`：
`baseline_keywords.json`、`keyword_rewards.jsonl`、`summary.json` 及完成 manifest。
Baseline set 成员关系和全部逐 keyword reward 已按批准公式验证。
当前 unique 定义下，单头过阈值且 baseline set 外的 keyword 获得对应
连续 relevance 和 unique bonus。

## 待完成

- 扩展少量 query 的 baseline keyword set 验证。
- 完成少量训练端到端验证，包括权重同步后的 constrained 输出与 reward。
- 完成少量 GRPO 更新验证，再决定扩大数据与训练规模。

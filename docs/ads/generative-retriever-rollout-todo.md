# Generative Retriever Rollout TODO

状态：已实现可供 Miles 使用的 custom generate；当前推荐使用 GR 的
Transformers 进程内 rollout，不依赖 SGLang。

## 已完成

- [x] 新增可通过 `--custom-generate-function-path` 加载的
      `examples.experimental.generative_retriever.generate.generate`。
- [x] 复用 Miles 的 `compute_prompt_ids_from_sample`、
      `compute_request_payload` 和 `update_sample_from_response`。
- [x] 将 SGLang 返回的输出 token 和 rollout log probability 写入 `Sample`。
- [x] 支持固定 `docid_size`，并在达到长度后标记 sample 完成。
- [x] 将本轮 docid token 序列写入
      `sample.metadata["generative_retriever"]`。
- [x] 要求 SGLang 返回每个 docid token 的 rollout log probability；缺失时显式失败，
      避免生成无法参与 policy loss 的 success-shaped sample。
- [x] 对固定长度 docid 正确处理 partial rollout，并将不足长度的响应标记为
      `TRUNCATED`。
- [x] 增加不依赖 SGLang/vLLM 的 Transformers 进程内 rollout：
      `gr_transformers_rollout.generate`。
- [x] 使用 GR checkpoint
      `SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553` 和
      `models/Qwen3-4B` 完成真实 CPU smoke test。
- [x] 对 `uniqueid=1` 的 docid vocabulary 实现按位置的 constrained decoding：
      第 `i` 位限制到 `[2 + i * code_dim, 2 + (i + 1) * code_dim)`。
- [x] CPU smoke test 已验证 4 个 docid token 和 4 个 token-level logprob。

## 后续验证

- [x] 用 `test query` 完成：
      query → docid token → token logprob。
- [x] 检查 `docid_size`、EOS、stop token 和 constrained decoding 的行为；
      EOS 不允许出现在固定长度 docid 的 codebook 区间之外。
- [x] 检查 rollout 输出覆盖每个 docid token 的 logprob。
- [ ] 完成 GPU smoke test，并确认 batch/concurrency 下的显存占用和吞吐。
- [ ] 实现 docid token 到索引中文档的解码器。
- [ ] 接入检索 reward（至少 Recall@1、MRR 或 NDCG）。
- [ ] 为生成失败、非法 docid 和索引 miss 定义明确的 reward/status 行为。
- [ ] 评估是否需要为 docid vocabulary 添加 constrained decoding。

## 训练侧待办

- [ ] 验证默认 Miles learner 能对该 encoder-decoder 模型计算 policy logprob。
- [ ] 验证 PPO/GRPO 使用的是 docid token 的 rollout logprob，而不是文本重编码
      后的 logprob。
- [ ] 决定 retrieval reward 使用 GRPO group advantage 还是 PPO value baseline。
- [ ] 将 InfoNCE、RQ-VAE quantization 等 auxiliary loss 与 RL policy loss
      分开记录并配置权重。
- [ ] 如果默认 model provider/loss 不兼容，分别实现
      `--custom-model-provider-path` 和 `--custom-loss-function-path`。

## 运行入口

进程内 Transformers rollout（推荐）：

```bash
export GR_CHECKPOINT="$cosmos_data_path/local/Users/tianchiyang/Checkpoints/ASI_RQ/SLMv3-202511-4x256-stage5/epoch1-part24/checkpoint-30553"
export GR_SLM_ROOT="$PWD/SLMGR_dev/SLMv3"
export GR_DOC_ENCODER_CHECKPOINT="/path/to/Qwen3-4B"
export PYTHONPATH="$PWD:$PWD/miles:$PYTHONPATH"

--custom-generate-function-path gr_transformers_rollout.generate
```

该实现直接在 rollout worker 中加载 GR checkpoint，并执行
`query -> Qwen encoder -> decoder`，每个 sample 使用固定长度 docid。
`GR_DOC_ENCODER_CHECKPOINT` 必须指向训练时使用的 Qwen encoder checkpoint，例如：

```bash
export GR_DOC_ENCODER_CHECKPOINT="$PWD/models/Qwen3-4B"
```

当前 smoke test 使用的 query 是 `test query`，生成的 token 是：

```text
[170, 347, 679, 882]
```

在 `uniqueid=1`, `code_dim=256` 下对应的 RQ code 是：

```text
[168, 89, 165, 112]
```

这组 code 目前还不能映射到具体文档，因为 checkpoint 及其上两级目录内没有
`doc2docid.txt`、`code2doc.json` 或 `docid2doclist` 映射文件。

SGLang 版本（仅用于后续性能实验）：

```bash
--custom-generate-function-path \
examples.experimental.generative_retriever.generate.generate \
--generative-retriever-docid-size <DOCID_SIZE>
```

SGLang 版本仅保留作后续性能实验；当前推荐入口是
`gr_transformers_rollout.generate`。SGLang 版本只负责通过
`/generate` 取得 docid，不负责把 docid 映射回索引文档，也不提供 retrieval
reward。

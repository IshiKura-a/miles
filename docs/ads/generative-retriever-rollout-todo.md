# Generative Retriever RL

状态：已完成 SLMv3 的 Transformers 推理、SGLang rollout、FSDP learner、
GRPO launcher 和 learner-to-SGLang 权重同步闭环。

## 已完成

- [x] 新增基于 SGLang `/encode` 的 custom generate。
- [x] 将 SGLang 返回的 docid token 和 rollout log probability 写入 `Sample`。
- [x] 支持固定 4-token positional docid。
- [x] 将本轮 docid token 序列写入
      `sample.metadata["generative_retriever"]`。
- [x] 严格校验每个 docid token 的 rollout log probability。
- [x] 使用 SGLang native Qwen3 prefill 和自定义 T5/RQ head，通过 `/encode`
      返回 4 个 positional docid token 及其 rollout logprob。
- [x] 冻结 Qwen3、RQ-VAE/glue 和旧 T5 query encoder，仅训练
      `docEnc2decoder`、T5 decoder、docid embedding 与 LM head。
- [x] learner 每步仅向 SGLang 同步 Stage5 policy head。

## 后续验证

- [x] 用 `test query` 完成：
      query → docid token → token logprob。
- [x] 检查 `docid_size`、EOS、stop token 和 constrained decoding 的行为；
      EOS 不允许出现在固定长度 docid 的 codebook 区间之外。
- [x] 检查 rollout 输出覆盖每个 docid token 的 logprob。
- [x] 完成 4 卡 FSDP learner + 4 卡 SGLang rollout 的一步 GRPO GPU smoke。
- [ ] 实现 docid token 到索引中文档的解码器。
- [x] 提供检索 reward 的 HTTP 接口；待接入实际 docid-to-document 索引和评分服务。
- [ ] 为生成失败、非法 docid 和索引 miss 定义明确的 reward/status 行为。
- [x] 对四个位置分别限制到 `[2,258)`、`[258,514)`、`[514,770)`、
      `[770,1026)`。

## 训练侧待办

- [x] 验证 Miles learner 能对该 encoder-decoder policy 计算 policy logprob。
- [x] 验证 GRPO 使用的是 docid token 的 rollout logprob，而不是文本重编码
      后的 logprob。
- [x] 使用 GRPO group advantage。
- [x] RL 仅使用 policy loss；InfoNCE 与 RQ-VAE auxiliary loss 不参与 Stage5 RL。
- [x] 通过 FSDP custom model factory 接入 GR policy，复用 Miles 现有 GRPO loss。

## 运行入口

Transformers checkpoint inference：

```bash
python -m examples.experimental.generative_retriever.transformers_infer \
  --checkpoint /path/to/checkpoint \
  --qwen-model /path/to/Qwen3-4B \
  --slm-root /path/to/SLMv3 \
  --transformers-path /path/to/transformers-4.57.3 \
  --query "cheap accommodation oporto with global travelers"
```

所有模型和源码路径均为输入。`--transformers-path` 可省略，但当前运行环境的
Transformers 版本必须与 checkpoint `config.json` 中记录的版本一致。输出包含：

- `decoder_token_ids`：1026 维 decoder vocabulary 中的原始 token；
- `positional_docid`：训练评估使用的 `decoder_token_ids - 2`；
- `rq_codes`：每层归一到 `[0, 255]` 的 RQ code；
- `token_logprobs`：每个生成 token 的归一化 log probability。

SGLang rollout 使用：

```bash
--custom-generate-function-path \
  examples.experimental.generative_retriever.sglang_encode_generate.generate \
--generative-retriever-docid-size 4
```

该 hook 从 Miles router 查询健康 worker，再直接调用 worker 的 `/encode`；
当前 SGLang router 不代理 embedding endpoint。

完整 GRPO 入口：

```bash
MILES_HOST_IP=<node-ip> python \
  examples/experimental/generative_retriever/run_grpo.py \
  --checkpoint /path/to/slmv3-checkpoint \
  --qwen-model /path/to/Qwen3-4B \
  --slm-root /path/to/SLMv3 \
  --prompt-data /path/to/prompts.jsonl \
  --reward-url http://reward-host:port/score
```

默认单节点拓扑为 8 卡：4 卡 FSDP learner、4 卡单 GPU SGLang engines。
所有模型、数据及服务路径均通过参数输入。

Reward hook：

```bash
--custom-rm-path \
  examples.experimental.generative_retriever.reward.reward \
--generative-retriever-reward-url http://<reward-host>:<port>/score \
--generative-retriever-reward-timeout 30
```

Reward endpoint 的请求和响应分别为：

```json
{"samples":[{"prompt":"query","response":"","label":"optional ground truth","docid_tokens":[170,347,679,882],"metadata":{"generative_retriever":{"docid_tokens":[170,347,679,882]}}}]}
```

```json
{"rewards":[1.0]}
```

每个 reward 必须是有限数，且数目必须与请求样本严格相等；HTTP 失败、无效 docid
或索引 miss 必须由 reward 服务显式返回对应的 reward 或错误，训练侧不会静默降级。
实际 reward 服务可基于其 docid-to-document 映射计算 Recall@1、MRR 或 NDCG。

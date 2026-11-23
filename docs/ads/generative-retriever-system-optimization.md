# Generative Retriever RL: Lossless System Acceleration & Colocated Architecture

本文档独立归档生成式检索强化学习在 MI300X 单机（8 卡）与多机（32 卡 / 64 卡）上的无损吞吐加速、HBM3 词向量缓存与自产自销架构验收数据（所有优化均保持数学输出与梯度逐位等价）。

## 文档分类导航（`docs/ads/`）

| 类别 | 文档 | 核心内容范围 |
| :--- | :--- | :--- |
| **运行与评测协议** | [generative-retriever-rollout-todo.md](./generative-retriever-rollout-todo.md) | 环境、列式索引、现行奖励公式、多机启动、10k 离线评测与 `SLMv3` 权重导出协议 |
| **静态索引与理论上界** | [generative-retriever-index-and-oracle.md](./generative-retriever-index-and-oracle.md) | `part25` 桶容量/市场分布、5 类桶型储量、10k-Query `Oracle_R / Oracle_Metric` 上界与 4 层树掉出诊断 |
| **系统加速与显存工程** | [generative-retriever-system-optimization.md](./generative-retriever-system-optimization.md) | `S0`–`S3` 单机无损优化与 `S4-Coloc` 多机自产自销架构（`26.25×` 端到端提速与 HBM3 遥测） |
| **RL 算法消融与收敛动力学** | [generative-retriever-ablation-study.md](./generative-retriever-ablation-study.md) | 15M 全量五代演进、`~150` 步策略熵跌破 $\ln 64$（Mode Collapse）实证，及早期正交机制消融归档 |

---

## 1. 训练系统无损加速与显存优化消融实验（`S0`–`S3`，数学等价性验证 + 端到端实测）

针对 64 卡主实验单步耗时达 `21.5 ~ 27.0` 分钟（8 卡单机 `~9.3` 分钟）且每张 MI300X（192 GB）仍有约 `45 ~ 50 GB` 显存闲置的工程瓶颈，定位并实现了三项 **数学输出 100% 等价（零超参改动、零梯度/奖励漂移）** 的系统优化，在 `online_rl_abl` 4 台 8 卡 MI300X 节点上同步运行阶梯消融（统一采用 `R3_Z3` 算法设定，每步 `4,000` queries $\times$ `512` beams = `2,048,000` 样本）：

1. **`OPT1`（`GR_PRECOMPILED_KW_FEATURES=1`）：全表 2.65 亿 Keyword 静态特征预编译 + `/dev/shm` 直寻址 + 整数 `kid` 打分链路**
   - 使用 32 张 MI300X 并行将 `part25` 索引全部 `265,127,714` 个唯一 Keyword 预编译为 `[265127714, 2, 128]` `bfloat16` 静态表（`135,745,389,568` Bytes = `126.4 GiB`，抽样 900 条与分片边界 140 条核验 `cross_score_max_diff = 0.0`），并已同步至 Cosmos 索引同级目录 `slmgrv3_part25_na_intl_20261028_20261029/keyword_features_gv21_large.{bf16.bin,meta.json}`。
   - 训练时单节点共享一份 `/dev/shm` 映射表，省去每卡 `16 GB` `GpuEmbeddingTable`、省去加载 `keyword_engine` 模型、省去 0% 命中率的 `scorer.scores` LRU 字典及 UTF-8 字符串解码/哈希，直接按 `uint32 kid` `O(1)` 取特征进 `cross_model`。CPU/GPU 对齐测试验证奖励、26 项指标与 `doc2k` 投票误差 `< 1e-6`。
2. **`OPT2`（`GR_FAST_DOC2K_SYNC=1`）：微批增量二进制追加写盘 + 跨 Rank 分片向量化 `doc2k` 投票归约**
   - 将 `_score_batch_in_process` 每微批全量重序列化 `pickle.dumps`（`$O(M^2)$`）改为定长二进制增量追加（`< 0.3s` / 25 微批）；
   - 将 FSDP `rank 0` 单核 Python 大字典合并与 `sorted()` 改为按 `u32 % world_size` `all_to_all_single` 分片 + GPU `torch.argsort / scatter_add` 向量化归约（8 卡 $\times$ 12 步验证 `additions / removals` 100% 逐元素一致）。在模拟主实验 **Step 90 规模（`2,450 万`历史累积键）** 的 8 卡实测中，全局同步耗时从 **`41.70s` 降至 `0.81s`（`51.5×` 提速；64 卡上由 `207.1s` 降至 `< 1s`，彻底消除随步数线性变慢与 `rank 0` 内存膨胀隐患）**。
3. **`OPT3`（`GR_FAST_TRAIN_IPC=1`）：紧凑 Group Token 传输 + T5 分组 Cross-Attention + `5×` 大训练微批**
   - 同组 512 个束仅 `beam 0` 携带完整 Prompt，其余 511 个束仅携带 `[pad_id, c0, c1, c2, c3]`（5 个 int），使 Ray IPC 序列与 FSDP `thd` 打包长度缩小 **`6.72×`**；
   - 将 6 层 T5 Decoder 的 Cross-Attention 改为 `[G, H, 512*4, D_k] @ [G, H, D_k, L_enc]` 分组矩阵乘，消除 `512×` 冗余 Encoder K/V 投影与激活显存；
   - 关闭冗余的 6 层小 T5 梯度检查点，将 `sglang_mem_fraction_static` 收紧至 `0.105`，并将 FSDP `micro_batch_size` 从 `5,120`（10 queries）放大 **`5×` 至 `25,600`（50 queries，仅 10 次 FSDP All-Gather/步）**。全参数相对梯度误差实测仅 **`1.08e-7`**。

#### 表 5H：8 卡单机（`4,000` queries/步）稳态（Steps 2–4 均值）端到端提速与显存对比

| 实验组 | 启用优化项 | **单步总耗时 `step_time`** (提速比) | **`rollout_time`** (生成+打分) | **单条 Query 打分延迟** | **`actor_train_time`** (FSDP 前反向) | **常驻显存 / 峰值显存** (单卡 192 GB) | Step 4 累积 `doc2k` 更新桶数 (`overlay`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S0`：Baseline** | 原版基线 (`micro_bs=5120`) | `558.79s` / `9.31 min` (`1.00×`) | `381.48s` | `691.1 ms` | `123.41s` | `90.2 GB` / `145.9 GB` | `666` |
| **`S1`：`+ OPT1`** | 预编译 Keyword `/dev/shm` 表 | `218.47s` / `3.64 min` (**`2.56×`**) | **`42.09s` (`9.06×`!)** | **`29.1 ms` (`23.7×`!)** | `122.73s` | `78.1 GB` / `134.7 GB` | `650` |
| **`S2`：`+ OPT1 + OPT2`** | `+ 分片向量化 doc2k 同步` | `224.16s` / `3.74 min` (**`2.49×`**) *(注)* | `40.25s` (`9.48×`) | `42.3 ms` (`16.3×`) | `123.86s` | `79.2 GB` / `134.6 GB` | `641` |
| **`S3`：`+ OPT1 + OPT2 + OPT3`** | **`+ 紧凑 IPC + 分组 T5 + micro_bs=25600`** | **`189.69s` / `3.16 min` (`2.95×`! 最快单步 `183.3s`)** | `55.96s` (`6.82×`) | `48.7 ms` (`14.2×`) | **`80.52s` (`1.53×`!)** | **`71.6 GB` / `140.8 GB` (峰值反降 `5.1 GB`!)** | `646` |

*(注：前 4 步全局历史键数尚少，`OPT2` 的收益主要体现在长程多卡规模——在 2,450 万历史键规模实测下将同步耗时从 `41.7s` [8 卡] / `207.1s` [64 卡主实验 Step 90] 压至 `0.81s`；折算到 **64 卡 Update Index 主实验**，`OPT1+OPT2+OPT3` 组合可将单步总耗时从当前的 **`1,622s` (`27.0 min`) 压降至约 `350s` (`~5.8 min`，约 `4.6×` 提速)**。)*

---

---

## 2. 多机纯自产自销架构与高显存无损加速验收报告（`S4-Coloc`，4 机 32 卡 MI300X 实测 vs. 8 机 64 卡主实验遥测）

针对单机测试未覆盖的多机星型通信瓶颈（`node-0` 单点 `RolloutExecutor` 跨机发送 `64,000` 次 HTTP、单进程聚合/切分 `3,277 万` 样本耗时 `209s` 且内存飙至 `77 GB`）以及 GPU 显存闲置（仅用 `49%~59%`）、FSDP 训练算力受限于每样本细粒度 kernel 启动等问题，在隔离分支 `ablation/speed-opt`（提交 `15fb39d5`，不含离线 Baseline）上实现并完成了 **4 机 32 卡（`pairs=2`，每卡负载 `1,000 Queries/step = 512,000 Samples/step` 与 8 机 64 卡完全等比例）** 完整 5 步联训验收（W&B Run: `j5h1zj7w`）：

1. **本机本卡纯“自产自销”去中心化 Rollout（`GR_LOCAL_COLOC_ROLLOUT=1` + `gr_compact_shard`）**：
   - 每台机器部署 1 个节点亲和 `_NodeColocRolloutWorker`，本机直接批量向本机 8 个 SGLang 引擎发送 `125-Query` 批请求（每卡每步仅 `8` 次本机回环调用，**0 跨机 HTTP**）；
   - 将每个 DP 分片压缩为 **5 个连续零拷贝 `np.ndarray / torch.Tensor` 缓冲区（`0` 个 Python 列表对象）** 直接写入本机 `/dev/shm` Ray Plasma，由同机同卡的 FSDP Actor 直接零网络读取（**0 跨机样本传输**，消除 `142s` GIL 序列化与 `209s` 单点等待）。
2. **HBM3 常驻高频预编译词向量缓存 + 全 GPU 向量化打分与规约（`HbmKeywordCache` + `score_batch_gpu_vectorized`）**：
   - 按自适应显存水位将高频预编译 Keyword 特征（加动态 FIFO 环缓）常驻每张 GPU 的 HBM3 显存，未命中冷词走 Pinned `/dev/shm` DMA（`max_diff = 0.0`）；
   - 将 `score_batch_with_precompiled_kids` 的百万次三层 Python 循环全量搬上 GPU（`float64` 精度，实测 `raw_max_diff = 0.0`、`metrics26_max_diff = 0.0`、`doc2k` 投票字典 `100%` 逐项一致），同时为 Rollout 树搜索补齐 Grouped Cross-Attention（`128-Query` chunk）。
3. **FSDP 大微批与向量化损失前反向（`mbs = 25,600` + Vectorized `get_batch / get_log_probs`）**：
   - 消除 `get_batch` 与 `get_log_probs_and_entropy` 中每微批 `2.56 万` 次逐样本 `F.pad / torch.arange / SliceBackward` 小 kernel 启动，改为单次 `index_select` / 分段前缀和（前反向逐位等价），将多机 `actor_train_time` 从 `260.8s` 压至 **`26.2s`**。

#### 表 5P：多机等单卡负载（`1,000 Queries = 512,000 Samples / GPU / step`）端到端性能与硬件遥测对比

| 核心性能 / 硬件指标（同单卡负载口径） | 64 卡 Fix 主实验 (`tod6o6tg`) | 64 卡 Update 主实验 (`wi49s1wh`) | **32 卡 4 机 `S4-Coloc + R4_D4` 稳态实测 (`j5h1zj7w`, Steps 1~4 均值)** | **相对 Update 主实验加速 / 改善** |
| :--- | ---: | ---: | :---: | :---: |
| **`perf/step_time`（单步端到端总耗时）** | `1,365.3s` (`22.8 min`) | `1,685.9s` (`28.1 min`) | **`64.22s` (`1.07 min`)**<br>*(各步: `62.28s, 63.47s, 64.96s, 66.17s`)* | **`26.25×` 端到端提速！**<br>*(远超 `<= 3.0 min` 验收线)* |
| **`perf/rollout_time`（编码 + 树搜索 + 打分 + 本地入库）** | `926.7s` | `1,026.9s` | **`16.43s`** | **`62.5×` 提速** |
| **单条 Query 打分延迟 (`score/latency_seconds`)** | `0.624s` | `0.674s` | **`0.0085s`** (`8.5 ms`) | **`79.3×` 提速** |
| **Rollout 有效吞吐 (`tokens_per_gpu_per_sec`)** | `2,213` | `1,995` | **`125,020`** | **`62.7×` 吞吐提升** |
| **跨机分发与后处理干等 (`train_wait - rollout - upd_w`)** | `135.5s` | `209.0s` | **`2.42s`** | **`86.4×` 削减（0 跨机样本传输）** |
| **Learner 数据上卡解包 (`train_time - actor_train_time`)** | `46.6s` | `48.3s` | **`4.08s`** *(含 4 机 Gloo 日志聚合)* | **`11.8×` 提速** |
| **`perf/actor_train_time`（多机 FSDP 前反向）** | `241.0s` (`24.9 TF`) | `260.8s` (`23.1 TF`) | **`26.32s`** (`71.5 TF` 名义 / 含紧凑序列实际 **`10×`**) | **`9.91×` 提速** |
| **`perf/update_weights_time`（含动态 `doc2k` 全局同步）** | `15.6s` *(无 `doc2k`)* | `140.9s ~ 198.9s` | **`14.86s`** *(含动态 `doc2k` GPU 归并)* | **`10.9× ~ 13.4×` 提速（追平无 `doc2k`！）** |
| **GPU 显存容量利用率 (`memoryAllocated` / `191.5 GiB`)** | 均值 `49%` / 峰值 `68.5%` | 均值 `59%` / 峰值 `78.6%` | **均值 `70.7%` / 训练峰值 `91.3%` (`174.8 GiB`)**<br>*(含 `12.3%` 保活进程；纯任务峰值 `~79.0%`)* | **显存打满甜点区，全程 0 OOM** |
| **GPU 满载功耗占比 (`powerPercent` / `750W`)** | 均值 `60.6%` / p95 `100%` | 均值 `63.3%` / p50 `43.9%` | **计算期（`42.7s / 64.2s`）持续 `86% ~ 92%` (`650~698W`)** | 消除碎 kernel 半休眠，剩余 `14.9s` 为框架权重同步 |
| **`node-0` 驱动进程内存峰值 (`proc.memory.rssMB`)** | `76.3 GB` | `77.0 GB` | **`13.5 GB`** | **主节点内存暴降 `-82.5%`** |

---

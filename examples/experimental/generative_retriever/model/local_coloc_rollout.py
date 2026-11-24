"""Node-local self-produce-self-consume colocated rollout for Generative Retriever."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import logging
import socket
import time
from typing import Any, cast
from urllib.parse import urlparse

import aiohttp
import httpx
import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import torch
from transformers import AutoTokenizer

from examples.experimental.generative_retriever.index.data_source import GRRolloutDataSource
from examples.experimental.generative_retriever.reward.in_process_protocol import (
    IN_PROCESS_METRIC_KEYS,
    expected_in_process_embedding_width,
    pack_group_tokens_flat,
    unpack_in_process_embedding,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    QUERY_MAX_TOKENS,
    RQ_CODE_NUM,
)
from examples.experimental.generative_retriever.reward.rollout_metrics import (
    add_distribution_metrics,
    build_rollout_advantages_3d,
    summarize_rollout_prob_concentration,
)
from examples.experimental.generative_retriever.runtime.gr_config import AntiCollapseMode, GRConfig
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainOutput,
)
from miles.utils.ray_utils import Box
from miles.utils.types import COMPACT_TOKEN_SHARD_KEY

logger = logging.getLogger(__name__)
_TRANSPORT_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 0.5
_SOCK_READ_TIMEOUT_SECONDS = 300.0


def ip_sort_key(host_or_ip: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in host_or_ip.split("."))
    except ValueError:
        try:
            resolved = socket.gethostbyname(host_or_ip)
            return tuple(int(part) for part in resolved.split("."))
        except (socket.gaierror, ValueError):
            return tuple(ord(ch) for ch in host_or_ip)


def batch_encode_prompts(
    tokenizer: Any,
    chat_pair: tuple[str, str],
    queries: list[str],
) -> list[list[int]]:
    if not queries:
        return []
    raw_ids = tokenizer(queries, add_special_tokens=False)["input_ids"]
    trunc_ids = [ids[:QUERY_MAX_TOKENS] for ids in raw_ids]
    trunc_texts = tokenizer.batch_decode(trunc_ids, skip_special_tokens=False)
    p0, p1 = chat_pair
    rendered = [f"{p0}{txt}{p1}" for txt in trunc_texts]
    return [list(ids) for ids in tokenizer(rendered, add_special_tokens=False)["input_ids"]]


def build_local_gpu_shard(
    *,
    prompt_ids_list: list[list[int]],
    out_mat: np.ndarray,
    base_sample_idx: int,
    group_size: int,
    pad_tok_id: int,
    unique_weight: float,
    rewards_norm: bool,
    grpo_std_norm: bool,
    anti_collapse_mode: AntiCollapseMode = "none",
) -> tuple[Box, dict[str, Any]]:
    """Convert raw SGLang embeddings for one local GPU into a zero-copy compact DP shard + summary."""
    num_q = len(prompt_ids_list)
    n_samples = num_q * group_size
    tok_i32, lp_f32, u32_mat, raw_f64, metrics_2d = unpack_in_process_embedding(
        out_mat,
        group_size,
        unique_weight,
    )

    s_u32 = np.sort(u32_mat, axis=1)
    distinct_sum = float(np.sum(1.0 + np.sum(s_u32[:, 1:] != s_u32[:, :-1], axis=1).astype(np.float64)))
    over_quota_sum, top16_mass_sum, samp_mass_sum = summarize_rollout_prob_concentration(lp_f32)

    norm_3d, cred_f64 = build_rollout_advantages_3d(
        u32_mat,
        raw_f64,
        lp_f32,
        anti_collapse_mode=anti_collapse_mode,
        rewards_norm=rewards_norm,
        grpo_std_norm=grpo_std_norm,
    )
    rewards_tensor = norm_3d.reshape(n_samples, RQ_CODE_NUM).contiguous()

    stds_arr = np.std(cred_f64, axis=1, ddof=1) if group_size > 1 else np.zeros(num_q, dtype=np.float64)
    ordered_cred = np.sort(cred_f64, axis=1)
    ranges_arr = ordered_cred[:, -1] - ordered_cred[:, 0]
    top_gaps_arr = ordered_cred[:, -1] - ordered_cred[:, -2] if group_size > 1 else np.zeros(num_q, dtype=np.float64)
    unique_cnt_arr = 1.0 + np.sum(ordered_cred[:, 1:] != ordered_cred[:, :-1], axis=1).astype(np.float64)
    abs_1d = norm_3d[:, :, 0].detach().abs().cpu().numpy().astype(np.float64)

    flat_tokens_i32, token_lengths_i32 = pack_group_tokens_flat(
        prompt_ids_list,
        tok_i32,
        pad_tok_id=pad_tok_id,
    )
    raw_zero_mask = np.all(raw_f64 == raw_f64[:, :1], axis=1)

    shard: dict[str, Any] = {
        COMPACT_TOKEN_SHARD_KEY: True,
        "flat_tokens_i32": flat_tokens_i32,
        "token_lengths_i32": token_lengths_i32,
        "rollout_log_probs_2d": np.ascontiguousarray(lp_f32.reshape(n_samples, RQ_CODE_NUM), dtype=np.float32),
        "rewards": rewards_tensor,
        "raw_reward": torch.from_numpy(np.ascontiguousarray(raw_f64.reshape(n_samples), dtype=np.float32)),
        "base_sample_idx": int(base_sample_idx),
        "response_length": RQ_CODE_NUM,
    }
    summary = {
        "num_q": num_q,
        "n_samples": n_samples,
        "stds": stds_arr,
        "ranges": ranges_arr,
        "unique_counts": unique_cnt_arr,
        "top_gaps": top_gaps_arr,
        "zero_std_groups": int(np.sum(ranges_arr == 0.0)),
        "adv_abs_group_means": np.mean(abs_1d, axis=1),
        "adv_abs_sum": float(np.sum(abs_1d)),
        "adv_abs_max": float(np.max(abs_1d)) if abs_1d.size > 0 else 0.0,
        "distinct_sum": distinct_sum,
        "over_quota_sum": over_quota_sum,
        "top16_mass_sum": top16_mass_sum,
        "samp_mass_sum": samp_mass_sum,
        "raw_sum": float(np.sum(raw_f64)),
        "all_zero_cnt": int(np.sum(raw_zero_mask & (np.round(raw_f64[:, 0], 1) == 0.0))),
        "all_one_cnt": int(np.sum(raw_zero_mask & (np.round(raw_f64[:, 0], 1) == 1.0))),
        "metrics_sum": np.sum(metrics_2d, axis=0),
        "metrics_last": metrics_2d[-1].copy() if num_q > 0 else np.zeros(len(IN_PROCESS_METRIC_KEYS), dtype=np.float64),
    }
    return Box(ray.put(shard)), summary


def aggregate_cluster_rollout_metrics(
    summaries: list[dict[str, Any]],
    *,
    rollout_time: float,
    rollout_gpus: int,
    unique_weight: float,
) -> dict[str, Any]:
    total_q = sum(int(s["num_q"]) for s in summaries)
    total_samples = sum(int(s["n_samples"]) for s in summaries)
    if total_q <= 0 or total_samples <= 0:
        return {}
    stds = np.concatenate([s["stds"] for s in summaries])
    ranges = np.concatenate([s["ranges"] for s in summaries])
    unique_counts = np.concatenate([s["unique_counts"] for s in summaries])
    top_gaps = np.concatenate([s["top_gaps"] for s in summaries])
    adv_abs_means = np.concatenate([s["adv_abs_group_means"] for s in summaries])
    zero_std_total = sum(int(s["zero_std_groups"]) for s in summaries)
    metrics_sum = np.sum(np.stack([s["metrics_sum"] for s in summaries], axis=0), axis=0)
    metrics_last = summaries[-1]["metrics_last"]
    group_size = total_samples // total_q

    extra: dict[str, Any] = {}
    prefix = "rollout/generative_retriever/group_reward"
    add_distribution_metrics(extra, f"{prefix}/std", stds)
    add_distribution_metrics(extra, f"{prefix}/range", ranges)
    add_distribution_metrics(extra, f"{prefix}/unique_count", unique_counts)
    add_distribution_metrics(extra, f"{prefix}/top1_top2_gap", top_gaps)
    extra[f"{prefix}/group_count"] = total_q
    extra[f"{prefix}/zero_std_count"] = zero_std_total
    extra[f"{prefix}/zero_std_fraction"] = zero_std_total / float(total_q)
    add_distribution_metrics(extra, f"{prefix}/advantage_abs_group_mean", adv_abs_means)
    extra[f"{prefix}/advantage_abs/mean"] = sum(float(s["adv_abs_sum"]) for s in summaries) / float(total_samples)
    extra[f"{prefix}/advantage_abs/max"] = max(float(s["adv_abs_max"]) for s in summaries)
    extra["rollout/generative_retriever/unique_weight"] = float(unique_weight)

    for idx, name in enumerate(IN_PROCESS_METRIC_KEYS):
        if name.endswith("/request_count") or name.startswith(("cache_delta/", "routing_delta/")):
            val = float(metrics_sum[idx])
        elif name.startswith("cache_gauge/"):
            val = float(metrics_last[idx])
        else:
            val = float(metrics_sum[idx]) / float(total_q)
        extra[f"rollout/generative_retriever/{name}"] = val

    extra["rollout/generative_retriever/distinct_docid_ratio"] = sum(
        float(s["distinct_sum"]) for s in summaries
    ) / float(total_q * group_size)
    extra["rollout/generative_retriever/prob_over_quota64_count"] = sum(
        float(s["over_quota_sum"]) for s in summaries
    ) / float(total_q)
    extra["rollout/generative_retriever/top16_prob_mass"] = sum(
        float(s["top16_mass_sum"]) for s in summaries
    ) / float(total_q)
    extra["rollout/generative_retriever/sampled512_prob_mass"] = sum(
        float(s["samp_mass_sum"]) for s in summaries
    ) / float(total_q)
    total_resp_tokens = float(total_samples * RQ_CODE_NUM)
    safe_time = max(1e-6, rollout_time)
    return {
        **extra,
        "rollout/num_training_samples": total_samples,
        "rollout/episode_raw_reward": sum(float(s["raw_sum"]) for s in summaries) / float(total_samples),
        "rollout/episode_response_length/mean": float(RQ_CODE_NUM),
        "rollout/episode_response_length/min": float(RQ_CODE_NUM),
        "rollout/episode_response_length/max": float(RQ_CODE_NUM),
        "rollout/episode_response_length/std": 0.0,
        "rollout/episode_total_response_length/mean": float(RQ_CODE_NUM),
        "rollout/response_len/mean": float(RQ_CODE_NUM),
        "rollout/response_len/min": float(RQ_CODE_NUM),
        "rollout/response_len/max": float(RQ_CODE_NUM),
        "rollout/response_len/std": 0.0,
        "rollout/zero_std/all_zero_percentage": sum(int(s["all_zero_cnt"]) for s in summaries) / float(total_q),
        "rollout/zero_std/all_one_percentage": sum(int(s["all_one_cnt"]) for s in summaries) / float(total_q),
        "rollout/repetition_frac": 0.0,
        "rollout/truncated_ratio": 0.0,
        "perf/rollout_time": safe_time,
        "perf/tokens_per_gpu_per_sec": total_resp_tokens / safe_time / float(max(1, rollout_gpus)),
        "perf/longest_sample_tokens_per_sec": float(RQ_CODE_NUM) / safe_time,
        "perf/effective_tokens_per_gpu_per_sec": total_resp_tokens / safe_time / float(max(1, rollout_gpus)),
        "perf/longest_effective_sample_tokens_per_sec": float(RQ_CODE_NUM) / safe_time,
    }


def discover_worker_urls_by_node(router_url: str, gpus_per_node: int) -> list[tuple[str, list[str]]]:
    """Query SGLang router ``/workers`` and group worker URLs by node IP in placement-group order."""
    resp = httpx.get(f"{router_url}/workers", timeout=30.0)
    resp.raise_for_status()
    info = resp.json()
    by_host: dict[str, list[tuple[int, str]]] = {}
    for worker in info.get("workers", []):
        if not isinstance(worker, dict) or not worker.get("url"):
            continue
        parsed = urlparse(str(worker["url"]).rstrip("/"))
        if parsed.hostname and parsed.port is not None:
            host_ip = socket.gethostbyname(parsed.hostname)
            by_host.setdefault(host_ip, []).append((int(parsed.port), f"http://{host_ip}:{int(parsed.port)}"))
    ordered: list[tuple[str, list[str]]] = []
    for host_ip in sorted(by_host, key=ip_sort_key):
        ports_and_urls = sorted(by_host[host_ip], key=lambda item: item[0])
        if len(ports_and_urls) != gpus_per_node:
            raise RuntimeError(
                f"Expected {gpus_per_node} SGLang workers on {host_ip}, found {len(ports_and_urls)}"
            )
        ordered.append((host_ip, [url for _, url in ports_and_urls]))
    return ordered


@ray.remote(num_cpus=4, num_gpus=0)
class _NodeColocRolloutWorker:
    def __init__(
        self,
        *,
        local_urls: list[str],
        hf_checkpoint: str,
        group_size: int,
        rewards_norm: bool,
        grpo_std_norm: bool,
        anti_collapse_mode: AntiCollapseMode = "none",
    ) -> None:
        self.local_urls = list(local_urls)
        self.gpus_per_node = len(self.local_urls)
        self.hf_checkpoint = hf_checkpoint
        self.group_size = group_size
        self.rewards_norm = rewards_norm
        self.grpo_std_norm = grpo_std_norm
        self.anti_collapse_mode: AntiCollapseMode = anti_collapse_mode
        self.tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
        self.pad_tok_id = int(self.tokenizer.pad_token_id)
        sentinel = "__GR_QUERY_PLACEHOLDER__"
        tmpl = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f'Given the user Query: "{sentinel}", summarize it in one token:'}],
            tokenize=False,
            add_generation_prompt=True,
        )
        p0, p1 = tmpl.split(sentinel)
        self.chat_pair = (p0, p1)
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.gpus_per_node))

    async def _post_batch(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        batch_inputs: list[list[int]],
        expected_width: int,
    ) -> np.ndarray:
        attempt = 1
        while True:
            try:
                async with session.post(
                    f"{base_url}/v1/embeddings",
                    json={"model": self.hf_checkpoint, "input": batch_inputs, "encoding_format": "base64"},
                ) as resp:
                    resp.raise_for_status()
                    payload = await resp.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= _TRANSPORT_ATTEMPTS:
                    raise
                await asyncio.sleep(_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)))
                attempt += 1
        items = payload.get("data")
        if not isinstance(items, list) or len(items) != len(batch_inputs):
            raise RuntimeError(f"Expected {len(batch_inputs)} embeddings, got {type(items)}")
        out = np.empty((len(batch_inputs), expected_width), dtype=np.float32)
        for item in items:
            idx = int(item["index"])
            arr = np.frombuffer(base64.b64decode(item["embedding"]), dtype="<f4")
            if arr.size != expected_width:
                raise RuntimeError(f"Unexpected GR embedding width {arr.size} != {expected_width}")
            out[idx] = arr
        return out

    async def _run_one_local_gpu(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        prompt_ids_slice: list[list[int]],
        base_sample_idx: int,
        batch_sz: int,
        expected_width: int,
        unique_weight: float,
    ) -> tuple[Box, dict[str, Any]]:
        num_q = len(prompt_ids_slice)
        out_mat = np.empty((num_q, expected_width), dtype=np.float32)
        sem = asyncio.Semaphore(2)

        async def _chunk_job(st: int, ed: int) -> None:
            async with sem:
                out_mat[st:ed] = await self._post_batch(session, base_url, prompt_ids_slice[st:ed], expected_width)

        await asyncio.gather(*(_chunk_job(st, min(st + batch_sz, num_q)) for st in range(0, num_q, batch_sz)))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool,
            lambda: build_local_gpu_shard(
                prompt_ids_list=prompt_ids_slice,
                out_mat=out_mat,
                base_sample_idx=base_sample_idx,
                group_size=self.group_size,
                pad_tok_id=self.pad_tok_id,
                unique_weight=unique_weight,
                rewards_norm=self.rewards_norm,
                grpo_std_norm=self.grpo_std_norm,
                anti_collapse_mode=self.anti_collapse_mode,
            ),
        )

    async def run_node_step(
        self,
        queries: list[str],
        node_base_sample_idx: int,
        batch_sz: int,
        unique_weight: float,
    ) -> tuple[list[Box], list[dict[str, Any]]]:
        prompt_ids_all = batch_encode_prompts(self.tokenizer, self.chat_pair, queries)
        n_total = len(prompt_ids_all)
        if n_total % self.gpus_per_node != 0:
            raise ValueError(f"Node query count {n_total} must be divisible by {self.gpus_per_node}")
        per_gpu = n_total // self.gpus_per_node
        expected_width = expected_in_process_embedding_width(self.group_size)

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, keepalive_timeout=600)
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=30.0,
            sock_connect=30.0,
            sock_read=_SOCK_READ_TIMEOUT_SECONDS,
        )
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            results = await asyncio.gather(
                *(
                    self._run_one_local_gpu(
                        session,
                        self.local_urls[g],
                        prompt_ids_all[g * per_gpu : (g + 1) * per_gpu],
                        node_base_sample_idx + g * per_gpu * self.group_size,
                        batch_sz,
                        expected_width,
                        unique_weight,
                    )
                    for g in range(self.gpus_per_node)
                )
            )
        return [r[0] for r in results], [r[1] for r in results]


class GRLocalColocatedRolloutFn(BaseRolloutFn):
    """Decentralized per-node batched rollout producing zero-copy local /dev/shm DP shards."""

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.args = input.args
        self.data_source = cast(GRRolloutDataSource, input.data_source)
        self.gr_config = GRConfig.from_env()
        self._workers: list[Any] = []

    def _ensure_workers(self) -> list[Any]:
        if self._workers:
            return self._workers
        router_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        gpus_per_node = int(self.args.actor_num_gpus_per_node)
        worker_groups = discover_worker_urls_by_node(router_url, gpus_per_node)
        expected_nodes = int(self.args.actor_num_nodes)
        if len(worker_groups) != expected_nodes:
            raise RuntimeError(f"Expected {expected_nodes} SGLang worker nodes, found {len(worker_groups)}")

        node_id_by_ip = {
            socket.gethostbyname(str(n["NodeManagerAddress"])): n["NodeID"]
            for n in ray.nodes()
            if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0.0)) > 0
        }
        grpo_std = self.args.advantage_estimator in ["grpo", "gspo"] and bool(self.args.grpo_std_normalization)

        self._workers = [
            _NodeColocRolloutWorker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id_by_ip[host_ip],
                    soft=False,
                )
            ).remote(
                local_urls=urls,
                hf_checkpoint=str(self.args.hf_checkpoint),
                group_size=int(self.args.n_samples_per_prompt),
                rewards_norm=bool(self.args.rewards_normalization),
                grpo_std_norm=grpo_std,
                anti_collapse_mode=self.gr_config.anti_collapse_mode,
            )
            for host_ip, urls in worker_groups
        ]
        return self._workers

    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if isinstance(input, RolloutFnEvalInput):
            raise NotImplementedError("GR offline evaluation uses run_eval.py rather than inline RolloutFnEvalInput")

        started = time.perf_counter()
        workers = self._ensure_workers()
        n_nodes = len(workers)
        total_q = int(self.args.rollout_batch_size)
        if total_q % n_nodes != 0:
            raise ValueError(f"rollout_batch_size {total_q} must divide evenly across {n_nodes} nodes")
        per_node_q = total_q // n_nodes
        group_size = int(self.args.n_samples_per_prompt)
        unique_weight = self.gr_config.resolve_unique_weight(
            int(input.rollout_id),
            int(self.args.num_rollout or 1),
        )

        prompts, _, start_sample_idx = self.data_source.pop_prompt_batch(total_q)
        futures = [
            workers[k].run_node_step.remote(
                prompts[k * per_node_q : (k + 1) * per_node_q],
                start_sample_idx + k * per_node_q * group_size,
                self.gr_config.local_rollout_batch_size,
                unique_weight,
            )
            for k in range(n_nodes)
        ]
        node_outputs = ray.get(futures)
        elapsed = time.perf_counter() - started

        all_refs: list[Box] = []
        all_summaries: list[dict[str, Any]] = []
        for refs_k, sum_k in node_outputs:
            all_refs.extend(refs_k)
            all_summaries.extend(sum_k)

        metrics = aggregate_cluster_rollout_metrics(
            all_summaries,
            rollout_time=elapsed,
            rollout_gpus=n_nodes * int(self.args.actor_num_gpus_per_node),
            unique_weight=unique_weight,
        )
        return RolloutFnTrainOutput(samples=[], metrics=metrics, presharded_data_refs=all_refs)

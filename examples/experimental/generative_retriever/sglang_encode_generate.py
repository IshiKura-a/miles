"""Miles rollout hook for the SGLang-hosted SLMv3 policy."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, cast
from weakref import WeakKeyDictionary

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.types import Sample
from examples.experimental.generative_retriever.relevance_reward import REWARD_MODES

_REQUESTS_BY_STATE: WeakKeyDictionary[Any, dict[Any, Any]] = WeakKeyDictionary()
_PREPARE_BATCHERS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, float, int], _BaselinePrepareBatcher]
] = WeakKeyDictionary()
_DOCID_SIZE = 4
_DOCID_DIM = 256
_MAX_PREPARE_QUERIES = 512
_PREPARE_BATCH_WAIT_SECONDS = 0.01


@dataclass
class _PendingPrepare:
    query: str
    future: asyncio.Future[dict[str, float]]


class _BaselinePrepareBatcher:
    def __init__(self, url: str, timeout_seconds: float, concurrency: int) -> None:
        if concurrency <= 0:
            raise ValueError("Baseline prepare concurrency must be positive")
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._slots = asyncio.Semaphore(concurrency)
        self._requests: set[asyncio.Task[None]] = set()
        self._pending: deque[_PendingPrepare] = deque()
        self._flush_task: asyncio.Task[None] | None = None

    async def submit(self, query: str) -> dict[str, float]:
        future = asyncio.get_running_loop().create_future()
        self._pending.append(_PendingPrepare(query=query, future=future))
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush())
        return await future

    async def _flush(self) -> None:
        try:
            while self._pending:
                await asyncio.sleep(_PREPARE_BATCH_WAIT_SECONDS)
                await self._slots.acquire()
                requests = [
                    self._pending.popleft()
                    for _ in range(min(len(self._pending), _MAX_PREPARE_QUERIES))
                ]
                task = asyncio.create_task(self._dispatch(requests))
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.create_task(self._flush())

    async def _dispatch(self, requests: list[_PendingPrepare]) -> None:
        try:
            metrics = await self._post(requests)
            for index, request in enumerate(requests):
                if not request.future.done():
                    request.future.set_result(metrics if index == 0 else {})
        except asyncio.CancelledError:
            for request in requests:
                request.future.cancel()
            raise
        except Exception as error:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(error)
        finally:
            self._slots.release()

    async def _post(self, requests: list[_PendingPrepare]) -> dict[str, float]:
        queries = list(dict.fromkeys(request.query for request in requests))
        response = await GeneralHttpClientProvider.client().post(
            self._url,
            json={"queries": queries},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Reward prepare endpoint must return a JSON object")
        if payload.get("prepared_queries") != len(queries):
            raise ValueError("Reward prepare endpoint returned an unexpected query count")
        return _prepare_metrics(payload)


def _get_prepare_batcher(
    url: str, timeout_seconds: float, concurrency: int,
) -> _BaselinePrepareBatcher:
    loop = asyncio.get_running_loop()
    batchers = _PREPARE_BATCHERS.setdefault(loop, {})
    key = (url, timeout_seconds, concurrency)
    batcher = batchers.get(key)
    if batcher is None:
        batcher = _BaselinePrepareBatcher(url, timeout_seconds, concurrency)
        batchers[key] = batcher
    return batcher


def _prepare_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = {}

    def collect(prefix: str, values: Any) -> None:
        if not isinstance(values, dict):
            raise ValueError(f"Reward prepare {prefix} metrics must be a JSON object")
        for name, value in values.items():
            path = f"{prefix}/{name}"
            if isinstance(value, dict):
                collect(path, value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[path] = float(value)

    collect("prepare/cache_gauge", payload.get("cache", {}))
    collect("prepare/cache_delta", payload.get("cache_delta", {}))
    collect("prepare/routing_delta", payload.get("routed_queries", {}))
    latency = payload.get("latency_seconds")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool):
        raise ValueError("Reward prepare endpoint returned invalid latency")
    metrics["prepare/latency_seconds"] = float(latency)
    metrics["prepare/request_count"] = 1.0
    return metrics


def _add_arguments(parser) -> None:
    parser.set_defaults(wandb_always_use_train_step=True)
    parser.add_argument(
        "--generative-retriever-reward-mode", choices=REWARD_MODES,
        default="discrete_relevance_unique",
    )
    parser.add_argument(
        "--generative-retriever-reward-url",
        type=str,
        default=None,
        help="HTTP endpoint returning retrieval rewards for generated docids.",
    )
    parser.add_argument(
        "--generative-retriever-reward-prepare-url",
        type=str,
        default=None,
        help="HTTP endpoint preparing frozen baselines concurrently with rollout.",
    )
    parser.add_argument(
        "--generative-retriever-reward-timeout",
        type=float,
        default=300.0,
        help="Retrieval-reward HTTP request timeout in seconds.",
    )
    parser.add_argument("--generative-retriever-reward-batch-size", type=int, default=2048)
    parser.add_argument("--generative-retriever-reward-concurrency", type=int, default=1)
    parser.add_argument("--generative-retriever-prepare-concurrency", type=int, default=1)


def _parse_embedding(
    output: dict[str, Any],
    docid_size: int,
    docid_dim: int,
    num_return_sequences: int = 1,
    beam_rank: int = 0,
) -> tuple[list[int], list[float]]:
    raw = output.get("embedding")
    expected = num_return_sequences * docid_size * 2
    if not isinstance(raw, list) or len(raw) != expected:
        raise ValueError(
            f"GR SGLang /encode must return {expected} beam values, got {raw!r}"
        )
    if not 0 <= beam_rank < num_return_sequences:
        raise ValueError("GR beam rank is outside the returned beam group")
    start = beam_rank * docid_size * 2
    raw = raw[start:start + docid_size * 2]
    token_ids = [int(value) for value in raw[:docid_size]]
    if any(isinstance(value, bool) or float(value) != token_id
           for value, token_id in zip(raw[:docid_size], token_ids, strict=True)):
        raise ValueError(f"GR SGLang returned non-integral token ids: {raw[:docid_size]!r}")
    for position, token_id in enumerate(token_ids):
        start = 2 + position * docid_dim
        if not start <= token_id < start + docid_dim:
            raise ValueError(
                f"GR SGLang returned token {token_id} outside position {position} codebook"
            )
    logprobs = [float(value) for value in raw[docid_size:]]
    if not all(math.isfinite(logprob) for logprob in logprobs):
        raise ValueError(f"GR SGLang returned non-finite logprobs: {logprobs!r}")
    return token_ids, logprobs


async def _resolve_engine_url(args: Any, sample: Sample) -> str:
    del sample
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}"


async def _request_embedding(args: Any, router_url: str, prompt_ids: list[int]) -> dict[str, Any]:
    response = await GeneralHttpClientProvider.client().post(
        f"{router_url}/v1/embeddings",
        json={
            "model": args.hf_checkpoint,
            "input": prompt_ids,
        },
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("GR router returned an invalid embeddings response")
    embedding = data[0].get("embedding")
    if not isinstance(embedding, list):
        raise ValueError("GR router response is missing an embedding list")
    return {"embedding": embedding}


async def _request_embedding_and_prepare(
    args: Any,
    router_url: str,
    prompt_ids: list[int],
    query: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    prepare_url = getattr(args, "generative_retriever_reward_prepare_url", None)
    if not prepare_url:
        raise ValueError("--generative-retriever-reward-prepare-url is required")
    embedding, prepare_metrics = await asyncio.gather(
        _request_embedding(args, router_url, prompt_ids),
        _get_prepare_batcher(
            prepare_url,
            args.generative_retriever_reward_timeout,
            args.generative_retriever_prepare_concurrency,
        ).submit(query),
    )
    return embedding, prepare_metrics


def _training_prompt_ids(state: Any, sample: Sample) -> list[int]:
    if not isinstance(sample.prompt, str) or not sample.prompt.strip():
        raise ValueError("GR rollout requires a non-empty query string prompt")
    query_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)[:16]
    truncated_query = state.tokenizer.decode(query_ids)
    instruction = f'Given the user Query: "{truncated_query}", summarize it in one token:'
    prompt = state.tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(state.tokenizer.encode(prompt, add_special_tokens=False))


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    if sample.status not in {Sample.Status.PENDING, Sample.Status.ABORTED}:
        raise ValueError(f"generative retriever received sample with status {sample.status!r}")
    if sample.response or sample.response_length:
        raise ValueError("GR SGLang /encode rollout is single-turn and cannot continue a partial response")

    query = sample.prompt
    if not isinstance(query, str):
        raise TypeError("GR rollout query must be a string")
    engine_url = await _resolve_engine_url(args, sample)
    group_size = args.n_samples_per_prompt
    if sample.index is None:
        raise ValueError("Constrained beam rollout requires sample.index")
    if not input.evaluation and sample.group_index is None:
        raise ValueError("Constrained training rollout requires sample.group_index")
    beam_rank = 0 if input.evaluation else sample.index % group_size
    requests = _REQUESTS_BY_STATE.setdefault(input.state, {})
    group_key = sample.index if input.evaluation else sample.group_index
    key = (input.evaluation, group_key, engine_url, query)
    if key not in requests:
        prompt_ids = _training_prompt_ids(input.state, sample)
        requests[key] = (
            asyncio.create_task(
                _request_embedding_and_prepare(
                    args,
                    engine_url,
                    prompt_ids,
                    query,
                )
            ),
            set(),
            prompt_ids,
        )
    task, consumed, prompt_ids = requests[key]
    try:
        output, prepare_metrics = await asyncio.shield(task)
    finally:
        if task.done() and (task.cancelled() or task.exception() is not None):
            requests.pop(key, None)
    consumed.add(beam_rank)
    if input.evaluation or len(consumed) == group_size:
        requests.pop(key, None)
    docid_size = _DOCID_SIZE
    docid_dim = _DOCID_DIM
    token_ids, logprobs = _parse_embedding(output, docid_size, docid_dim, group_size, beam_rank)

    sample.tokens = prompt_ids + token_ids
    sample.response_length = docid_size
    sample.rollout_log_probs = logprobs
    sample.response = ",".join(str(token_id - 2) for token_id in token_ids)
    sample.status = Sample.Status.COMPLETED
    metadata = sample.metadata.setdefault("generative_retriever", {})
    metadata["docid_tokens"] = token_ids
    metadata["beam_rank"] = beam_rank
    metadata["positional_docid"] = [token_id - 2 for token_id in token_ids]
    metadata["rq_codes"] = [
        token_id - 2 - position * docid_dim
        for position, token_id in enumerate(token_ids)
    ]
    if beam_rank == 0 and prepare_metrics:
        metadata["prepare_metrics"] = prepare_metrics
    return GenerateFnOutput(samples=sample)


cast(Any, generate).add_arguments = _add_arguments

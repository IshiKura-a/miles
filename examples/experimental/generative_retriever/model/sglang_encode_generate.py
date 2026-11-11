"""Miles rollout hook for the SGLang-hosted SLMv3 policy."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any
from weakref import WeakKeyDictionary

import httpx

from miles.rollout.base_types import BaseGroupGenerateFn, GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.types import Sample
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_REWARD_BATCH_SIZE,
    DOCID_TOKEN_OFFSET,
    QUERY_MAX_TOKENS,
    REWARD_MODES,
    RQ_CODE_DIM,
    RQ_CODE_NUM,
)
from examples.experimental.generative_retriever.reward.reward_http import (
    AsyncRequestBatcher,
    PendingBatchRequest,
    post_json_async,
)

_PREPARE_BATCHERS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, float, int], BaselinePrepareBatcher]
] = WeakKeyDictionary()
_DOCID_SIZE = RQ_CODE_NUM
_DOCID_DIM = RQ_CODE_DIM
_MAX_PREPARE_QUERIES = 64
_PREPARE_BATCH_WAIT_SECONDS = 0.01
_TRANSPORT_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 0.5

logger = logging.getLogger(__name__)


class BaselinePrepareBatcher(AsyncRequestBatcher[str, dict[str, float]]):
    def __init__(self, url: str, timeout_seconds: float, concurrency: int) -> None:
        super().__init__(
            max_batch_size=_MAX_PREPARE_QUERIES,
            concurrency=concurrency,
            batch_wait_seconds=_PREPARE_BATCH_WAIT_SECONDS,
        )
        self._url = url
        self._timeout_seconds = timeout_seconds

    async def submit(self, query: str) -> dict[str, float]:
        return await self.submit_weighted(query, weight=1)

    async def execute_batch(
        self,
        requests: list[PendingBatchRequest[str, dict[str, float]]],
    ) -> None:
        queries = list(dict.fromkeys(request.payload for request in requests))
        payload = await post_json_async(
            self._url,
            {"queries": queries},
            timeout=self._timeout_seconds,
        )
        if payload.get("prepared_queries") != len(queries):
            raise ValueError("Reward prepare endpoint returned an unexpected query count")
        metrics = parse_prepare_metrics(payload)
        for index, request in enumerate(requests):
            if not request.future.done():
                request.future.set_result(metrics if index == 0 else {})


def get_prepare_batcher(
    url: str, timeout_seconds: float, concurrency: int,
) -> BaselinePrepareBatcher:
    loop = asyncio.get_running_loop()
    batchers = _PREPARE_BATCHERS.setdefault(loop, {})
    key = (url, timeout_seconds, concurrency)
    batcher = batchers.get(key)
    if batcher is None:
        batcher = BaselinePrepareBatcher(url, timeout_seconds, concurrency)
        batchers[key] = batcher
    return batcher


def parse_prepare_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

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


def add_arguments(parser: Any) -> None:
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
    parser.add_argument(
        "--generative-retriever-reward-batch-size",
        type=int,
        default=DEFAULT_REWARD_BATCH_SIZE,
    )
    parser.add_argument("--generative-retriever-reward-concurrency", type=int, default=1)
    parser.add_argument("--generative-retriever-prepare-concurrency", type=int, default=1)


def _validate_token_slice(raw_tokens: list[Any], docid_size: int, docid_dim: int) -> list[int]:
    token_ids = [int(value) for value in raw_tokens]
    if any(isinstance(value, bool) or float(value) != token_id
           for value, token_id in zip(raw_tokens, token_ids, strict=True)):
        raise ValueError(f"GR SGLang returned non-integral token ids: {raw_tokens!r}")
    for position, token_id in enumerate(token_ids):
        low = DOCID_TOKEN_OFFSET + position * docid_dim
        if not low <= token_id < low + docid_dim:
            raise ValueError(
                f"GR SGLang returned token {token_id} outside position {position} codebook"
            )
    return token_ids


def parse_embedding(
    output: dict[str, Any],
    docid_size: int,
    docid_dim: int,
    num_return_sequences: int = 1,
    beam_rank: int = 0,
) -> tuple[list[int], list[float]]:
    raw = output.get("embedding")
    expected_2 = num_return_sequences * docid_size * 2
    expected_3 = num_return_sequences * docid_size * 3
    if not isinstance(raw, list) or len(raw) not in (expected_2, expected_3):
        raise ValueError(
            f"GR SGLang /encode must return {expected_2} or {expected_3} beam values, got {raw!r}"
        )
    if not 0 <= beam_rank < num_return_sequences:
        raise ValueError("GR beam rank is outside the returned beam group")
    stride = docid_size * (3 if len(raw) == expected_3 else 2)
    start = beam_rank * stride
    beam_raw = raw[start : start + docid_size * 2]
    token_ids = _validate_token_slice(beam_raw[:docid_size], docid_size, docid_dim)
    logprobs = [float(value) for value in beam_raw[docid_size:]]
    if not all(math.isfinite(logprob) for logprob in logprobs):
        raise ValueError(f"GR SGLang returned non-finite logprobs: {logprobs!r}")
    return token_ids, logprobs


def parse_baseline_docids(
    output: dict[str, Any],
    docid_size: int,
    docid_dim: int,
    num_return_sequences: int,
) -> list[tuple[int, int, int, int]] | None:
    raw = output.get("embedding")
    expected_3 = num_return_sequences * docid_size * 3
    if not isinstance(raw, list) or len(raw) != expected_3:
        return None
    stride = docid_size * 3
    docids: list[tuple[int, int, int, int]] = []
    for rank in range(num_return_sequences):
        start = rank * stride + docid_size * 2
        token_ids = _validate_token_slice(raw[start : start + docid_size], docid_size, docid_dim)
        docids.append((
            token_ids[0] - DOCID_TOKEN_OFFSET,
            token_ids[1] - DOCID_TOKEN_OFFSET,
            token_ids[2] - DOCID_TOKEN_OFFSET,
            token_ids[3] - DOCID_TOKEN_OFFSET,
        ))
    return docids


def resolve_engine_url(args: Any) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}"


async def request_embedding(args: Any, router_url: str, prompt_ids: list[int]) -> dict[str, Any]:
    attempt = 1
    while True:
        try:
            response = await GeneralHttpClientProvider.client().post(
                f"{router_url}/v1/embeddings",
                json={
                    "model": args.hf_checkpoint,
                    "input": prompt_ids,
                },
            )
            break
        except (httpx.TransportError, httpx.TimeoutException) as error:
            if attempt == _TRANSPORT_ATTEMPTS:
                raise
            logger.warning(
                "Embedding connection %s (attempt %d/%d); retrying before generation",
                type(error).__name__, attempt, _TRANSPORT_ATTEMPTS,
            )
            await asyncio.sleep(_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
            attempt += 1
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("GR router returned an invalid embeddings response")
    embedding = data[0].get("embedding")
    if not isinstance(embedding, list):
        raise ValueError("GR router response is missing an embedding list")
    return {"embedding": embedding}


async def request_embedding_and_prepare(
    args: Any,
    router_url: str,
    prompt_ids: list[int],
    query: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    embedding = await request_embedding(args, router_url, prompt_ids)
    raw = embedding.get("embedding")
    if isinstance(raw, list) and len(raw) == args.n_samples_per_prompt * _DOCID_SIZE * 3:
        return embedding, {}
    prepare_url = args.generative_retriever_reward_prepare_url
    if not prepare_url:
        raise ValueError("--generative-retriever-reward-prepare-url is required")
    prepare_metrics = await get_prepare_batcher(
        prepare_url,
        args.generative_retriever_reward_timeout,
        args.generative_retriever_prepare_concurrency,
    ).submit(query)
    return embedding, prepare_metrics


def encode_training_prompt(state: Any, query: str) -> list[int]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("GR rollout requires a non-empty query string prompt")
    query_ids = state.tokenizer.encode(query, add_special_tokens=False)[:QUERY_MAX_TOKENS]
    truncated_query = state.tokenizer.decode(query_ids)
    instruction = f'Given the user Query: "{truncated_query}", summarize it in one token:'
    prompt = state.tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(state.tokenizer.encode(prompt, add_special_tokens=False))


def populate_beam_sample(
    sample: Sample,
    *,
    prompt_ids: list[int],
    output: dict[str, Any],
    group_size: int,
    beam_rank: int,
    prepare_metrics: dict[str, float],
) -> None:
    token_ids, logprobs = parse_embedding(
        output, _DOCID_SIZE, _DOCID_DIM, group_size, beam_rank,
    )
    positional_docid = [token_id - DOCID_TOKEN_OFFSET for token_id in token_ids]
    sample.tokens = prompt_ids + token_ids
    sample.response_length = _DOCID_SIZE
    sample.rollout_log_probs = logprobs
    sample.response = ",".join(str(code) for code in positional_docid)
    sample.status = Sample.Status.COMPLETED
    metadata = sample.metadata.setdefault("generative_retriever", {})
    metadata["docid_tokens"] = token_ids
    metadata["beam_rank"] = beam_rank
    metadata["positional_docid"] = positional_docid
    metadata["rq_codes"] = [
        code - position * _DOCID_DIM
        for position, code in enumerate(positional_docid)
    ]
    if beam_rank == 0:
        baseline_docids = parse_baseline_docids(output, _DOCID_SIZE, _DOCID_DIM, group_size)
        if baseline_docids is not None:
            metadata["baseline_docids"] = baseline_docids
        if prepare_metrics:
            metadata["prepare_metrics"] = prepare_metrics


def validate_pending_sample(sample: Sample, *, evaluation: bool) -> str:
    if sample.status not in {Sample.Status.PENDING, Sample.Status.ABORTED}:
        raise ValueError(f"generative retriever received sample with status {sample.status!r}")
    if sample.response or sample.response_length:
        raise ValueError("GR SGLang /encode rollout is single-turn and cannot continue a partial response")
    if not isinstance(sample.prompt, str):
        raise TypeError("GR rollout query must be a string")
    if sample.index is None:
        raise ValueError("Constrained beam rollout requires sample.index")
    if not evaluation and sample.group_index is None:
        raise ValueError("Constrained training rollout requires sample.group_index")
    return sample.prompt


async def generate_group_samples(
    state: Any,
    group: list[Sample],
    *,
    evaluation: bool,
) -> list[Sample]:
    args = state.args
    group_size = args.n_samples_per_prompt
    if len(group) != group_size:
        raise ValueError(f"Expected group of {group_size} samples, got {len(group)}")
    first = group[0]
    query = validate_pending_sample(first, evaluation=evaluation)
    for beam_rank, sample in enumerate(group):
        if validate_pending_sample(sample, evaluation=evaluation) != query:
            raise ValueError("All samples in a GR query group must share the same prompt")
        if not evaluation and sample.group_index != first.group_index:
            raise ValueError("All samples in a GR query group must share group_index")
        if not evaluation and sample.index % group_size != beam_rank:
            raise ValueError(f"Expected beam rank {beam_rank}, got sample.index={sample.index}")

    prompt_ids = encode_training_prompt(state, query)
    output, prepare_metrics = await request_embedding_and_prepare(
        args,
        resolve_engine_url(args),
        prompt_ids,
        query,
    )
    for beam_rank, sample in enumerate(group):
        populate_beam_sample(
            sample,
            prompt_ids=prompt_ids,
            output=output,
            group_size=group_size,
            beam_rank=0 if evaluation else beam_rank,
            prepare_metrics=prepare_metrics,
        )
    return group


async def generate_single_sample(
    state: Any,
    sample: Sample,
    *,
    evaluation: bool,
) -> Sample:
    args = state.args
    group_size = args.n_samples_per_prompt
    query = validate_pending_sample(sample, evaluation=evaluation)
    assert sample.index is not None
    beam_rank = 0 if evaluation else sample.index % group_size
    prompt_ids = encode_training_prompt(state, query)
    output, prepare_metrics = await request_embedding_and_prepare(
        args,
        resolve_engine_url(args),
        prompt_ids,
        query,
    )
    populate_beam_sample(
        sample,
        prompt_ids=prompt_ids,
        output=output,
        group_size=group_size,
        beam_rank=beam_rank,
        prepare_metrics=prepare_metrics,
    )
    return sample


class GRGroupGenerateFn(BaseGroupGenerateFn):
    add_arguments = staticmethod(add_arguments)

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        if isinstance(input.sample, list):
            samples = await generate_group_samples(
                input.state,
                input.sample,
                evaluation=input.evaluation,
            )
            return GenerateFnOutput(samples=samples)
        sample = await generate_single_sample(
            input.state,
            input.sample,
            evaluation=input.evaluation,
        )
        return GenerateFnOutput(samples=sample)


generate = GRGroupGenerateFn()

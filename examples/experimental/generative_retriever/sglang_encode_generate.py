"""Miles rollout hook for the SGLang-hosted SLMv3 policy."""

from __future__ import annotations

import math
import asyncio
from typing import Any

from miles.backends.sglang_utils.sglang_router_api_client import use_legacy_router_api
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import GeneralHttpClientProvider, post
from miles.utils.types import Sample
from examples.experimental.generative_retriever.relevance_reward import REWARD_MODES


def _add_arguments(parser) -> None:
    parser.add_argument(
        "--generative-retriever-reward-mode", choices=REWARD_MODES,
        default="continuous_relevance_unique",
    )
    parser.add_argument(
        "--generative-retriever-docid-size",
        type=int,
        default=4,
        help="Number of positional docid tokens returned by the GR policy.",
    )
    parser.add_argument(
        "--generative-retriever-docid-dim",
        type=int,
        default=256,
        help="Number of codes available at each docid position.",
    )
    parser.add_argument(
        "--generative-retriever-reward-url",
        type=str,
        default=None,
        help="HTTP endpoint returning retrieval rewards for generated docids.",
    )
    parser.add_argument(
        "--generative-retriever-reward-timeout",
        type=float,
        default=30.0,
        help="Retrieval-reward HTTP request timeout in seconds.",
    )


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
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    if use_legacy_router_api(args):
        response = await GeneralHttpClientProvider.client().get(f"{router_url}/list_workers")
        response.raise_for_status()
        worker_urls = response.json().get("urls", [])
    else:
        response = await GeneralHttpClientProvider.client().get(f"{router_url}/workers")
        response.raise_for_status()
        worker_urls = sorted(
            worker["url"]
            for worker in response.json().get("workers", [])
            if worker.get("is_healthy") and worker.get("worker_type") == "regular"
        )
    if not worker_urls:
        raise RuntimeError(f"GR rollout router has no healthy regular workers: {router_url}")
    routing_index = sample.group_index if sample.group_index is not None else sample.index
    routing_index = routing_index if routing_index is not None else 0
    return worker_urls[routing_index % len(worker_urls)]


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

    prompt_ids = _training_prompt_ids(input.state, sample)
    engine_url = await _resolve_engine_url(args, sample)
    group_size = args.n_samples_per_prompt
    if sample.index is None:
        raise ValueError("Constrained beam rollout requires sample.index")
    if not input.evaluation and sample.group_index is None:
        raise ValueError("Constrained training rollout requires sample.group_index")
    beam_rank = 0 if input.evaluation else sample.index % group_size
    if not hasattr(input.state, "_gr_beam_requests"):
        input.state._gr_beam_requests = {}
    requests = input.state._gr_beam_requests
    group_key = sample.index if input.evaluation else sample.group_index
    key = (input.evaluation, group_key, engine_url, tuple(prompt_ids))
    if key not in requests:
        requests[key] = (
            asyncio.create_task(post(f"{engine_url}/encode", {"input_ids": prompt_ids})),
            set(),
        )
    task, consumed = requests[key]
    try:
        output = await asyncio.shield(task)
    finally:
        if task.done() and (task.cancelled() or task.exception() is not None):
            requests.pop(key, None)
    consumed.add(beam_rank)
    if input.evaluation or len(consumed) == group_size:
        requests.pop(key, None)
    docid_size = args.generative_retriever_docid_size
    docid_dim = args.generative_retriever_docid_dim
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
    return GenerateFnOutput(samples=sample)


generate.add_arguments = _add_arguments

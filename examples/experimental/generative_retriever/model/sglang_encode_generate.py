"""Miles rollout hook for the SGLang-hosted SLMv3 policy."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
from typing import Any

import httpx
import numpy as np

from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_REWARD_BATCH_SIZE,
    DOCID_TOKEN_OFFSET,
    QUERY_MAX_TOKENS,
    REWARD_MODES,
    RQ_CODE_DIM,
    RQ_CODE_NUM,
)
from examples.experimental.generative_retriever.sglang_models.gr_policy import (
    IN_PROCESS_BEAM_COLS,
    IN_PROCESS_METRIC_KEYS,
)
from miles.rollout.base_types import BaseGroupGenerateFn, GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.types import Sample

_DOCID_SIZE = RQ_CODE_NUM
_DOCID_DIM = RQ_CODE_DIM
_TRANSPORT_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 0.5

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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
        help="Optional legacy HTTP endpoint returning retrieval rewards for generated docids.",
    )
    parser.add_argument(
        "--generative-retriever-reward-prepare-url",
        type=str,
        default=None,
        help="Optional legacy HTTP endpoint preparing frozen baselines.",
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
                    "encoding_format": "base64",
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
    if isinstance(embedding, str):
        arr = np.frombuffer(base64.b64decode(embedding), dtype="<f4")
        return {"embedding_np": arr}
    if not isinstance(embedding, list):
        raise ValueError("GR router response is missing an embedding list")
    return {"embedding": embedding}


def encode_training_prompt(state: Any, query: str) -> list[int]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("GR rollout requires a non-empty query string prompt")
    pair = getattr(state, "_gr_chat_template_pair", None)
    if pair is None:
        sentinel = "__GR_QUERY_PLACEHOLDER__"
        tmpl = state.tokenizer.apply_chat_template(
            [{"role": "user", "content": f'Given the user Query: "{sentinel}", summarize it in one token:'}],
            tokenize=False,
            add_generation_prompt=True,
        )
        pair = tuple(tmpl.split(sentinel))
        setattr(state, "_gr_chat_template_pair", pair)
    query_ids = state.tokenizer.encode(query, add_special_tokens=False)[:QUERY_MAX_TOKENS]
    truncated_query = state.tokenizer.decode(query_ids)
    prompt = f"{pair[0]}{truncated_query}{pair[1]}"
    return list(state.tokenizer.encode(prompt, add_special_tokens=False))


def populate_beam_sample(
    sample: Sample,
    *,
    prompt_ids: list[int],
    output: dict[str, Any],
    group_size: int,
    beam_rank: int,
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

    prompt_ids = encode_training_prompt(state, query)
    output = await request_embedding(args, resolve_engine_url(args), prompt_ids)
    raw_np = output.get("embedding_np")
    expected_in_process = group_size * IN_PROCESS_BEAM_COLS + len(IN_PROCESS_METRIC_KEYS)
    if not evaluation and raw_np is not None and raw_np.size == expected_in_process:
        core = raw_np[: group_size * IN_PROCESS_BEAM_COLS].reshape(group_size, IN_PROCESS_BEAM_COLS)
        tok_mat = core[:, :4].astype(np.int64)
        tok_list = tok_mat.tolist()
        lp_list = core[:, 4:8].astype(np.float64).tolist()
        rewards_list = core[:, 8].astype(np.float64).tolist()
        raw_rewards_list = core[:, 9].astype(np.float64).tolist()
        pos_mat = tok_mat - DOCID_TOKEN_OFFSET
        u32_arr = (
            (pos_mat[:, 0].astype(np.uint32) << np.uint32(24))
            | ((pos_mat[:, 1] - 256).astype(np.uint32) << np.uint32(16))
            | ((pos_mat[:, 2] - 512).astype(np.uint32) << np.uint32(8))
            | (pos_mat[:, 3] - 768).astype(np.uint32)
        )
        for beam_rank, sample in enumerate(group):
            sample.tokens = prompt_ids + tok_list[beam_rank]
            sample.response_length = _DOCID_SIZE
            sample.rollout_log_probs = lp_list[beam_rank]
            sample.reward = rewards_list[beam_rank]
            sample.status = Sample.Status.COMPLETED
        p0, p1, p2, p3 = (int(v) for v in pos_mat[0].tolist())
        first.response = f"{p0},{p1},{p2},{p3}"
        meta0 = first.metadata.setdefault("generative_retriever", {})
        meta0["group_u32"] = u32_arr
        meta0["group_raw_rewards"] = raw_rewards_list
        metrics_vals = raw_np[group_size * IN_PROCESS_BEAM_COLS :].astype(np.float64).tolist()
        meta0["reward_metrics"] = dict(zip(IN_PROCESS_METRIC_KEYS, metrics_vals, strict=True))
        return group

    expected_2 = group_size * _DOCID_SIZE * 2
    expected_3 = group_size * _DOCID_SIZE * 3
    if not evaluation and raw_np is not None and raw_np.size in (expected_2, expected_3):
        cols = 12 if raw_np.size == expected_3 else 8
        mat = raw_np.reshape(group_size, cols)
        tok_mat = mat[:, :4].astype(np.int64).tolist()
        lp_mat = mat[:, 4:8].astype(np.float64).tolist()
        pos_mat = (mat[:, :4].astype(np.int64) - DOCID_TOKEN_OFFSET).tolist()
        baseline_docids_np: list[tuple[int, int, int, int]] | None = None
        if cols == 12:
            b_mat = (mat[:, 8:12].astype(np.int64) - DOCID_TOKEN_OFFSET).tolist()
            baseline_docids_np = [(r[0], r[1], r[2], r[3]) for r in b_mat]
        for beam_rank, sample in enumerate(group):
            token_ids = tok_mat[beam_rank]
            positional_docid = pos_mat[beam_rank]
            sample.tokens = prompt_ids + token_ids
            sample.response_length = _DOCID_SIZE
            sample.rollout_log_probs = lp_mat[beam_rank]
            sample.response = f"{positional_docid[0]},{positional_docid[1]},{positional_docid[2]},{positional_docid[3]}"
            sample.status = Sample.Status.COMPLETED
            metadata = sample.metadata.setdefault("generative_retriever", {})
            metadata["docid_tokens"] = token_ids
            metadata["beam_rank"] = beam_rank
            metadata["positional_docid"] = positional_docid
        if baseline_docids_np is not None:
            group[0].metadata["generative_retriever"]["baseline_docids"] = baseline_docids_np
        return group

    raw = output.get("embedding")
    if not evaluation and isinstance(raw, list) and len(raw) in (expected_2, expected_3):
        stride = _DOCID_SIZE * (3 if len(raw) == expected_3 else 2)
        baseline_docids: list[tuple[int, int, int, int]] | None = [] if len(raw) == expected_3 else None
        for beam_rank, sample in enumerate(group):
            start = beam_rank * stride
            t0 = int(raw[start])
            t1 = int(raw[start + 1])
            t2 = int(raw[start + 2])
            t3 = int(raw[start + 3])
            token_ids = [t0, t1, t2, t3]
            logprobs = [
                float(raw[start + 4]),
                float(raw[start + 5]),
                float(raw[start + 6]),
                float(raw[start + 7]),
            ]
            p0 = t0 - DOCID_TOKEN_OFFSET
            p1 = t1 - DOCID_TOKEN_OFFSET
            p2 = t2 - DOCID_TOKEN_OFFSET
            p3 = t3 - DOCID_TOKEN_OFFSET
            positional_docid = [p0, p1, p2, p3]
            sample.tokens = prompt_ids + token_ids
            sample.response_length = _DOCID_SIZE
            sample.rollout_log_probs = logprobs
            sample.response = f"{p0},{p1},{p2},{p3}"
            sample.status = Sample.Status.COMPLETED
            metadata = sample.metadata.setdefault("generative_retriever", {})
            metadata["docid_tokens"] = token_ids
            metadata["beam_rank"] = beam_rank
            metadata["positional_docid"] = positional_docid
            if baseline_docids is not None:
                b0 = int(raw[start + 8]) - DOCID_TOKEN_OFFSET
                b1 = int(raw[start + 9]) - DOCID_TOKEN_OFFSET
                b2 = int(raw[start + 10]) - DOCID_TOKEN_OFFSET
                b3 = int(raw[start + 11]) - DOCID_TOKEN_OFFSET
                baseline_docids.append((b0, b1, b2, b3))
        if baseline_docids is not None:
            group[0].metadata["generative_retriever"]["baseline_docids"] = baseline_docids
        return group

    for beam_rank, sample in enumerate(group):
        populate_beam_sample(
            sample,
            prompt_ids=prompt_ids,
            output=output,
            group_size=group_size,
            beam_rank=0 if evaluation else beam_rank,
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
    output = await request_embedding(args, resolve_engine_url(args), prompt_ids)
    if "embedding_np" in output and "embedding" not in output:
        output["embedding"] = output["embedding_np"].tolist()
    populate_beam_sample(
        sample,
        prompt_ids=prompt_ids,
        output=output,
        group_size=group_size,
        beam_rank=beam_rank,
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

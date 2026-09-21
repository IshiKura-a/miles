"""SGLang generation hook for a generative retriever.

The SGLang server is expected to load the retriever model and expose its
decoder vocabulary through the normal ``/generate`` endpoint. The hook keeps
the generated docid tokens and rollout log probabilities on ``Sample`` so the
downstream reward and training stages can consume them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample


def _docid_size(args: Any, sampling_params: dict[str, Any]) -> int | None:
    """Return the configured number of generated docid tokens, if configured."""
    sampling_value = sampling_params.pop("docid_size", None)
    configured = getattr(args, "generative_retriever_docid_size", None)
    if configured is None:
        configured = sampling_value
    if configured is None:
        return None
    if not isinstance(configured, int) or configured <= 0:
        raise ValueError("generative_retriever_docid_size must be a positive integer")
    return configured


def _add_arguments(parser) -> None:
    parser.add_argument(
        "--generative-retriever-docid-size",
        type=int,
        default=None,
        help="Number of decoder tokens used to represent one generated docid.",
    )


def _extract_output_token_logprobs(output: dict[str, Any]) -> list[tuple[float, int]]:
    """Extract the token-level signal required by the RL learner.

    A retriever rollout is trained on generated docid tokens, so accepting a
    response without one log probability per token would create a
    success-shaped sample that cannot be used by the policy-loss code.
    """
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("generative retriever response is missing meta_info")

    raw_logprobs = meta_info.get("output_token_logprobs")
    if not isinstance(raw_logprobs, Sequence) or isinstance(raw_logprobs, (str, bytes)):
        raise ValueError("generative retriever requires output_token_logprobs for every generated token")

    result: list[tuple[float, int]] = []
    for item in raw_logprobs:
        if not isinstance(item, Sequence) or len(item) < 2:
            raise ValueError(f"invalid output_token_logprobs item: {item!r}")
        logprob, token_id = item[0], item[1]
        if logprob is None:
            raise ValueError("generative retriever received a token without a rollout log probability")
        result.append((float(logprob), int(token_id)))
    return result


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Generate a docid sequence through SGLang's stateless ``/generate`` API."""
    args = input.args
    sample = input.sample
    sampling_params = dict(input.sampling_params)

    if sample.status not in {Sample.Status.PENDING, Sample.Status.ABORTED}:
        raise ValueError(f"generative retriever received sample with status {sample.status!r}")

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)

    if sample.response:
        input_ids = sample.tokens
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
        if sampling_params["max_new_tokens"] <= 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    docid_size = _docid_size(args, sampling_params)
    if docid_size is not None:
        generated_tokens = len(input_ids) - len(prompt_ids)
        remaining_tokens = docid_size - generated_tokens
        if remaining_tokens <= 0:
            sample.status = Sample.Status.COMPLETED
            return GenerateFnOutput(samples=sample)
        sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"],
            remaining_tokens,
        )
        # A docid is a fixed-length code. Do not let a configured text stop
        # sequence terminate it early.
        sampling_params.pop("stop", None)
        sampling_params.pop("stop_token_ids", None)

    payload, halt_status = compute_request_payload(
        args,
        input_ids=input_ids,
        sampling_params=sampling_params,
        multimodal_inputs=sample.multimodal_inputs,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    output = await post(
        url,
        payload,
        headers=compute_routing_headers(args, sample),
    )
    output_token_logprobs = _extract_output_token_logprobs(output)
    await update_sample_from_response(
        args,
        sample,
        payload=payload,
        output=output,
    )

    new_docid_tokens = [token_id for _, token_id in output_token_logprobs]
    if sample.metadata is None:
        sample.metadata = {}
    retriever_metadata = sample.metadata.setdefault("generative_retriever", {})
    docid_tokens = list(retriever_metadata.get("docid_tokens", []))
    docid_tokens.extend(new_docid_tokens)
    retriever_metadata["docid_tokens"] = docid_tokens
    retriever_metadata["docid_size"] = docid_size
    retriever_metadata["evaluation"] = input.evaluation

    if docid_size is not None:
        if len(docid_tokens) >= docid_size:
            sample.status = Sample.Status.COMPLETED
        else:
            sample.status = Sample.Status.TRUNCATED

    return GenerateFnOutput(samples=sample)


generate.add_arguments = _add_arguments

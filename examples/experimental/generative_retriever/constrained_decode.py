"""SLMGR constrained beam generation over the official serving ID mapping."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable

import torch
from torch import nn
from transformers import T5Config, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from examples.experimental.generative_retriever.model_loading import import_slmv3, require_directory

PrefixConstraint = Callable[[int, torch.Tensor], list[int]]


@lru_cache(maxsize=1)
def load_constraint(slm_root: str, mapping_path: str, code_num: int, code_dim: int) -> PrefixConstraint:
    slmv3 = import_slmv3(require_directory(slm_root, "SLMv3 source"))
    mapping = Path(mapping_path).expanduser().resolve()
    initial = mapping.stat()
    raw_ids: set[bytes] = set()
    scanned_bytes = 0
    with mapping.open("rb", buffering=8 * 1024 * 1024) as stream:
        for line_number, line in enumerate(stream, 1):
            raw_id, separator, _ = line.partition(b"\t")
            if not separator:
                raise ValueError(f"Official mapping row {line_number} has no text column")
            raw_ids.add(raw_id)
            scanned_bytes += len(line)
    final = mapping.stat()
    if (scanned_bytes, final.st_size, final.st_mtime_ns) != (initial.st_size, initial.st_size, initial.st_mtime_ns):
        raise RuntimeError("Official mapping changed while loading decoder constraints")
    docids = []
    roots = set()
    for raw_id in raw_ids:
        values = [int(value) for value in raw_id.split()]
        if len(values) != code_num or any(not i * code_dim <= value < (i + 1) * code_dim
                                          for i, value in enumerate(values)):
            raise ValueError(f"Invalid official positional ID: {raw_id!r}")
        roots.add(values[0])
        docids.append(" ".join(map(str, values)))
    if roots != set(range(code_dim)):
        raise ValueError("Official mapping must cover the initial codebook used by SLMGR")
    del raw_ids
    args = slmv3["RunArguments"](infer_mode=2, code_num=code_num, code_dim=code_dim, uniqueid=1)
    return slmv3["get_restrict_decode_vocab"](args, docid_list=docids, id_bias=2)


def build_generation_model(
    *,
    slm_root: str,
    config: T5Config,
    projection: nn.Module,
    decoder: nn.Module,
    lm_head: nn.Module,
) -> nn.Module:
    """Attach existing policy tensors to the official generation class."""
    slmv3 = import_slmv3(require_directory(slm_root, "SLMv3 source"))
    source_class = slmv3["T5MultiWithGivenQueryDocEncoderDSI"]

    class GenerationHead(source_class):
        def __init__(self) -> None:
            T5PreTrainedModel.__init__(self, config)
            self.docEnc2decoder = projection
            self.decoder = decoder
            self.lm_head = lm_head
            self.running_args = slmv3["RunArguments"]()
            self.margin = self.running_args.margin

    return GenerationHead().eval()


@torch.no_grad()
def generate_beams(
    model: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    constraint: PrefixConstraint,
    *,
    code_num: int,
    num_beams: int,
    num_return_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= num_return_sequences <= num_beams:
        raise ValueError("Require 1 <= num_return_sequences <= num_beams")
    output = model.generate(
        encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=cast(Any, hidden_states),
        ),
        attention_mask=attention_mask,
        max_length=code_num + 1,
        min_length=code_num + 1,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        prefix_allowed_tokens_fn=constraint,
        early_stopping=True,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        decoder_start_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
    )
    tokens = output.sequences[:, 1:]
    scores = model.compute_transition_scores(
        output.sequences, output.scores, getattr(output, "beam_indices", None), normalize_logits=True,
    )
    expected = (hidden_states.shape[0] * num_return_sequences, code_num)
    if tokens.shape != expected or scores.shape != expected or not torch.isfinite(scores).all():
        raise RuntimeError("SLMGR constrained beam output has invalid shape or token logprobs")
    for sequence in output.sequences:
        for position in range(code_num):
            if sequence[position + 1].item() not in constraint(0, sequence[:position + 1]):
                raise RuntimeError("SLMGR beam returned a token outside the official prefix constraint")
    tokens = tokens.reshape(hidden_states.shape[0], num_return_sequences, code_num)
    if any(len(set(map(tuple, rows.tolist()))) != num_return_sequences for rows in tokens):
        raise RuntimeError("SLMGR constrained beam returned duplicate IDs")
    return tokens, scores.reshape_as(tokens)


@torch.no_grad()
def generate_samples(
    model: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    constraint: PrefixConstraint,
    *,
    code_num: int,
    num_return_sequences: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_return_sequences <= 0:
        raise ValueError("num_return_sequences must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    output = model.generate(
        encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=cast(Any, hidden_states),
        ),
        attention_mask=attention_mask,
        max_length=code_num + 1,
        min_length=code_num + 1,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        num_beams=1,
        num_return_sequences=num_return_sequences,
        prefix_allowed_tokens_fn=constraint,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        decoder_start_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
    )
    tokens = output.sequences[:, 1:]
    scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True,
    )
    expected = (hidden_states.shape[0] * num_return_sequences, code_num)
    if tokens.shape != expected or scores.shape != expected or not torch.isfinite(scores).all():
        raise RuntimeError("SLMGR constrained sampling output has invalid shape or token logprobs")
    for sequence in output.sequences:
        for position in range(code_num):
            if sequence[position + 1].item() not in constraint(0, sequence[:position + 1]):
                raise RuntimeError("SLMGR sampling returned a token outside the official prefix constraint")
    return (
        tokens.reshape(hidden_states.shape[0], num_return_sequences, code_num),
        scores.reshape(hidden_states.shape[0], num_return_sequences, code_num),
    )


def _sample_gumbel(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    one_minus_eps = 1.0 - torch.finfo(torch.float32).eps
    uniform = torch.empty(shape, device=device, dtype=torch.float32).uniform_(tiny, one_minus_eps)
    return -torch.log(-torch.log(uniform))


def _truncate_gumbel(gumbels: torch.Tensor, truncation: torch.Tensor) -> torch.Tensor:
    """Condition child Gumbels so their maximum equals the parent beam Gumbel (Kool et al., 2019)."""
    max_gumbels = gumbels.max(dim=-1, keepdim=True).values
    delta = gumbels - max_gumbels
    one_minus_exp_delta = -torch.expm1(delta)
    shifted = -gumbels + torch.log(one_minus_exp_delta)
    truncated = -torch.logaddexp(-truncation.expand_as(gumbels), shifted)
    return torch.where(torch.isfinite(gumbels) & torch.isfinite(max_gumbels), truncated, gumbels)


@torch.no_grad()
def generate_stochastic_beams(
    model: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    constraint: PrefixConstraint,
    *,
    code_num: int,
    num_beams: int,
    num_return_sequences: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample distinct constrained sequences without replacement via Stochastic Beam Search."""
    if not 1 <= num_return_sequences <= num_beams:
        raise ValueError("Require 1 <= num_return_sequences <= num_beams")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    vocab_size = int(model.lm_head.out_features)
    projected = model.docEnc2decoder(hidden_states.to(model.docEnc2decoder.weight.dtype))

    sequences = torch.zeros((batch_size, 1, 1), dtype=torch.long, device=device)
    seq_logprobs = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    beam_gumbels = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    token_logprobs = torch.empty((batch_size, 1, 0), dtype=torch.float32, device=device)

    for step in range(code_num):
        active_beams = sequences.shape[1]
        flat_sequences = sequences.reshape(batch_size * active_beams, step + 1)
        decoder_outputs = model.decoder(
            input_ids=flat_sequences,
            encoder_hidden_states=projected.repeat_interleave(active_beams, dim=0),
            encoder_attention_mask=attention_mask.repeat_interleave(active_beams, dim=0),
            use_cache=False,
            return_dict=True,
        )
        logits = model.lm_head(decoder_outputs.last_hidden_state[:, -1, :]).float()
        masked_logits = torch.full_like(logits, float("-inf"))
        for row in range(batch_size * active_beams):
            allowed = constraint(row // active_beams, flat_sequences[row])
            masked_logits[row, allowed] = logits[row, allowed]

        reshaped_masked = masked_logits.reshape(batch_size, active_beams, vocab_size)
        step_lp = torch.log_softmax(reshaped_masked, dim=-1)
        step_lp = torch.where(
            torch.isfinite(reshaped_masked),
            step_lp,
            torch.full_like(step_lp, float("-inf")),
        )
        candidate_lp = seq_logprobs.unsqueeze(-1) + step_lp
        scaled_lp = candidate_lp / temperature
        noise = _sample_gumbel(scaled_lp.shape, device)
        raw_gumbels = torch.where(torch.isfinite(scaled_lp), scaled_lp + noise, scaled_lp)
        candidate_gumbels = (
            raw_gumbels
            if step == 0
            else _truncate_gumbel(raw_gumbels, beam_gumbels.unsqueeze(-1))
        )

        flat_gumbels = candidate_gumbels.reshape(batch_size, active_beams * vocab_size)
        valid_counts = int(torch.isfinite(flat_gumbels).sum(dim=-1).min().item())
        next_beams = (
            num_return_sequences
            if step == code_num - 1
            else min(num_beams, valid_counts)
        )
        if valid_counts < next_beams:
            raise RuntimeError(
                f"SLMGR stochastic beam has {valid_counts} valid continuations at step {step}, "
                f"need {next_beams}"
            )
        top_gumbels, top_indices = torch.topk(flat_gumbels, k=next_beams, dim=-1)
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        parent_sequences = sequences.gather(
            1, beam_indices.unsqueeze(-1).expand(batch_size, next_beams, step + 1),
        )
        sequences = torch.cat((parent_sequences, token_indices.unsqueeze(-1)), dim=-1)
        chosen_step_lp = step_lp.reshape(batch_size, active_beams * vocab_size).gather(
            1, top_indices,
        )
        parent_token_lp = token_logprobs.gather(
            1, beam_indices.unsqueeze(-1).expand(batch_size, next_beams, step),
        )
        token_logprobs = torch.cat((parent_token_lp, chosen_step_lp.unsqueeze(-1)), dim=-1)
        seq_logprobs = candidate_lp.reshape(batch_size, active_beams * vocab_size).gather(
            1, top_indices,
        )
        beam_gumbels = top_gumbels

    tokens = sequences[:, :, 1:]
    expected = (batch_size, num_return_sequences, code_num)
    if tokens.shape != expected or token_logprobs.shape != expected or not torch.isfinite(token_logprobs).all():
        raise RuntimeError("SLMGR stochastic beam output has invalid shape or token logprobs")
    for batch_index in range(batch_size):
        for sequence in sequences[batch_index]:
            for position in range(code_num):
                if sequence[position + 1].item() not in constraint(batch_index, sequence[:position + 1]):
                    raise RuntimeError("SLMGR stochastic beam returned a token outside the official prefix constraint")
    if any(len(set(map(tuple, rows.tolist()))) != num_return_sequences for rows in tokens):
        raise RuntimeError("SLMGR stochastic beam returned duplicate IDs")
    return tokens, token_logprobs

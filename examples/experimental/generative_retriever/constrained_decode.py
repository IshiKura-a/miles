"""SLMGR constrained beam generation over the official serving ID mapping."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
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
            last_hidden_state=hidden_states, hidden_states=hidden_states, attentions=attention_mask,
        ),
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

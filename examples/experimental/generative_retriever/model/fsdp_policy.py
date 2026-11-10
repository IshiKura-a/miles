"""FSDP learner policy for SLMv3 Stage5-style reinforcement learning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.t5.modeling_t5 import T5Attention
from examples.experimental.generative_retriever.model.constrained_decode import (
    PrefixConstraint,
    SortedU32Constraint,
    load_constraint,
)

from examples.experimental.generative_retriever.model.model_loading import (
    build_run_arguments,
    checkpoint_tensor_shape,
    import_slmv3,
    load_checkpoint,
    require_directory,
    stage_local_model_dir,
)


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"{variable} must be set for the GR FSDP model factory")
    return stage_local_model_dir(value, variable)


class GRStage5Policy(nn.Module):
    """Frozen Qwen encoder plus trainable projection and autoregressive docid decoder."""

    _miles_requires_response_lengths = True
    _miles_fsdp_ignore_frozen_parameters = True
    _miles_fsdp_wrap_root_only = True
    _no_split_modules = ["Qwen3DecoderLayer", "T5Block"]

    def __init__(
        self, source_model: nn.Module, pad_token_id: int, constraint: PrefixConstraint,
    ) -> None:
        super().__init__()
        source: Any = source_model
        self.config = source.encoder_list[0].model.encoder.config
        self.config.tie_word_embeddings = False
        self.pad_token_id = pad_token_id
        self.code_dim: int = source.docid_dim
        self.code_num: int = source.docid_size
        self.constraint = constraint
        self._prefix_allowed_cache: dict[tuple[int, ...], tuple[torch.Tensor, set[int]]] = {}

        self.encoder_list: Any = source.encoder_list
        self.docEnc2decoder: Any = source.docEnc2decoder
        self.decoder: Any = source.decoder
        self.lm_head: Any = source.lm_head

        self.encoder_list.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, T5Attention):
                module.dropout = 0.0
        for module in (
            self.docEnc2decoder,
            self.decoder,
            self.lm_head,
        ):
            module.requires_grad_(True)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        del kwargs
        self.decoder.gradient_checkpointing = True

    def _encode_prompts(
        self,
        sample_tokens: list[torch.Tensor],
        response_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        prompt_lengths = [
            tokens.shape[0] - response_length
            for tokens, response_length in zip(sample_tokens, response_lengths, strict=True)
        ]
        prompts = [
            sample_tokens[index][:prompt_length]
            for index, prompt_length in enumerate(prompt_lengths)
        ]
        padded_prompts = pad_sequence(
            [prompt.flip(0) for prompt in prompts],
            batch_first=True,
            padding_value=self.pad_token_id,
        ).flip(1)
        unique_prompts, inverse_indices = torch.unique(
            padded_prompts,
            dim=0,
            return_inverse=True,
        )
        unique_mask = unique_prompts.ne(self.pad_token_id).long()
        with torch.no_grad():
            causal_lm = self.encoder_list[0].model.encoder
            backbone = getattr(causal_lm, "model", None)
            if backbone is None:
                raise TypeError("GR Qwen encoder must expose its backbone as .model")
            outputs = backbone(
                input_ids=unique_prompts,
                attention_mask=unique_mask,
                return_dict=True,
                use_cache=False,
            )
            unique_hidden_states = outputs.last_hidden_state
        return unique_hidden_states, unique_mask, inverse_indices, prompt_lengths

    def _allowed_for_prefix(self, prefix_tuple: tuple[int, ...]) -> tuple[torch.Tensor, set[int]]:
        cached = self._prefix_allowed_cache.get(prefix_tuple)
        if cached is not None:
            return cached
        allowed_list = self.constraint(0, torch.tensor(prefix_tuple, dtype=torch.long))
        allowed_tensor = torch.tensor(allowed_list, dtype=torch.long)
        entry = (allowed_tensor, set(allowed_list))
        if len(self._prefix_allowed_cache) < 262_144:
            self._prefix_allowed_cache[prefix_tuple] = entry
        return entry

    def _build_allowed_mask(
        self,
        responses: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        if isinstance(self.constraint, SortedU32Constraint):
            mask_gpu = self.constraint.build_response_mask(responses, vocab_size)
            if not bool(mask_gpu.gather(2, responses.unsqueeze(-1)).all()):
                raise ValueError("GR response violates official prefix constraint")
            return mask_gpu
        sample_count = responses.shape[0]
        responses_cpu: list[list[int]] = responses.detach().cpu().tolist()
        mask_cpu = torch.zeros((sample_count, self.code_num, vocab_size), dtype=torch.bool)
        for index, row in enumerate(responses_cpu):
            prefix: tuple[int, ...] = (0,)
            for position in range(self.code_num):
                allowed_tensor, allowed_set = self._allowed_for_prefix(prefix)
                token_id = row[position]
                if not allowed_set or token_id not in allowed_set:
                    raise ValueError(f"GR response violates official prefix constraint at position {position}")
                mask_cpu[index, position, allowed_tensor] = True
                prefix = (*prefix, token_id)
        return mask_cpu.to(device=responses.device, non_blocking=True)

    @staticmethod
    def _unpack_tokens(
        input_ids: torch.Tensor,
        total_lengths: list[int],
    ) -> tuple[list[torch.Tensor], list[int]]:
        if input_ids.ndim != 2:
            raise ValueError(f"GRStage5Policy expected rank-2 input_ids, got {input_ids.shape}")
        if input_ids.shape[0] == len(total_lengths):
            return (
                [
                    input_ids[index, :total_length]
                    for index, total_length in enumerate(total_lengths)
                ],
                [0] * len(total_lengths),
            )
        if input_ids.shape[0] != 1:
            raise ValueError(
                "GRStage5Policy requires Miles thd packed input or one padded row per sample"
            )

        samples = []
        offsets = []
        offset = 0
        for total_length in total_lengths:
            offsets.append(offset)
            samples.append(input_ids[0, offset : offset + total_length])
            offset += total_length
        return samples, offsets

    def forward(
        self,
        input_ids: torch.Tensor,
        response_lengths: list[int] | torch.Tensor,
        total_lengths: list[int] | torch.Tensor,
        **kwargs: Any,
    ) -> CausalLMOutput:
        del kwargs
        response_lengths = [int(length) for length in response_lengths]
        total_lengths = [int(length) for length in total_lengths]
        if any(length != self.code_num for length in response_lengths):
            raise ValueError(f"GR response lengths must all equal {self.code_num}")
        sample_tokens, output_offsets = self._unpack_tokens(input_ids, total_lengths)

        unique_hidden_states, unique_mask, inverse_indices, prompt_lengths = self._encode_prompts(
            sample_tokens,
            response_lengths,
        )
        unique_hidden_states = unique_hidden_states.to(self.docEnc2decoder.weight.dtype)
        unique_projected = self.docEnc2decoder(unique_hidden_states)
        projected_hidden_states = unique_projected.index_select(0, inverse_indices)
        attention_mask = unique_mask.index_select(0, inverse_indices)
        input_batch_size, sequence_length = input_ids.shape
        sample_count = len(sample_tokens)
        vocab_size = self.code_num * self.code_dim + 2
        logits = projected_hidden_states.new_zeros(
            (input_batch_size, sequence_length, vocab_size)
        )

        responses = torch.stack([
            tokens[prompt_length : prompt_length + self.code_num]
            for tokens, prompt_length in zip(sample_tokens, prompt_lengths, strict=True)
        ])
        decoder_input_ids = torch.cat(
            (
                torch.zeros(
                    (sample_count, 1),
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                responses[:, :-1],
            ),
            dim=-1,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=projected_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        response_logits = self.lm_head(decoder_outputs.last_hidden_state)
        invalid_logit = torch.finfo(response_logits.dtype).min
        allowed_mask = self._build_allowed_mask(responses, vocab_size)
        masked_response_logits = response_logits.masked_fill(~allowed_mask, invalid_logit)
        starts = torch.tensor(
            [
                output_offset + prompt_length - 1
                for prompt_length, output_offset in zip(prompt_lengths, output_offsets, strict=True)
            ],
            dtype=torch.long,
            device=input_ids.device,
        )
        positions = starts.unsqueeze(1) + torch.arange(self.code_num, device=input_ids.device).unsqueeze(0)
        if input_batch_size == 1:
            logits[0, positions.reshape(-1)] = masked_response_logits.reshape(-1, vocab_size)
        else:
            row_indices = torch.arange(sample_count, device=input_ids.device).unsqueeze(1).expand_as(positions)
            logits[row_indices, positions] = masked_response_logits
        return CausalLMOutput(logits=logits)


def build_gr_stage5_policy(checkpoint_path: str, args: Any, init_context) -> GRStage5Policy:
    """Miles ``--custom-fsdp-model-factory-path`` entrypoint."""
    del checkpoint_path, args

    checkpoint = _required_path("GR_CHECKPOINT")
    qwen_model = _required_path("GR_QWEN_MODEL")
    slm_root_value = os.environ.get("GR_SLM_ROOT")
    if not slm_root_value:
        raise ValueError("GR_SLM_ROOT must be set for the GR FSDP model factory")
    slm_root = require_directory(slm_root_value, "GR_SLM_ROOT")
    mapping = os.environ.get("GR_ID_MAPPING")
    if not mapping:
        raise ValueError("GR_ID_MAPPING must be set for constrained GR training")
    constraint = load_constraint(mapping, 4, 256)
    slmv3 = import_slmv3(slm_root)
    run_args = build_run_arguments(slmv3["RunArguments"], checkpoint, qwen_model)

    with init_context():
        t5_config = T5Config.from_pretrained(checkpoint)
        t5_config.vocab_size = checkpoint_tensor_shape(
            checkpoint,
            "query_encoder.embed_tokens.weight",
        )[0]
        t5_model = T5ForConditionalGeneration(t5_config)
        qwen_encoder = slmv3["Encoder_QWen1216t"](run_args)
        source_model = slmv3["T5MultiWithGivenQueryDocEncoderDSI"](
            t5_config,
            t5_model,
            encoder=None,
            docencoder=qwen_encoder,
            docid_dim=run_args.code_dim,
            docid_size=run_args.code_num,
            margin=run_args.margin,
            running_args=run_args,
            domain_idx=[0],
        )

    if not any(parameter.is_meta for parameter in source_model.parameters()):
        load_checkpoint(source_model, checkpoint)
    return GRStage5Policy(
        source_model,
        pad_token_id=qwen_encoder.tokenizer.pad_token_id,
        constraint=constraint,
    )

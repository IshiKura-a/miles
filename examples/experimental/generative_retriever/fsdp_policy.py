"""FSDP learner policy for SLMv3 Stage5-style reinforcement learning."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .model_loading import (
    build_run_arguments,
    checkpoint_tensor_shape,
    import_slmv3,
    load_checkpoint,
    require_directory,
)


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"{variable} must be set for the GR FSDP model factory")
    return require_directory(value, variable)


class GRStage5Policy(nn.Module):
    """Frozen Qwen encoder plus trainable projection and autoregressive docid decoder."""

    _miles_requires_response_lengths = True
    _miles_fsdp_ignore_frozen_parameters = True
    _miles_fsdp_wrap_root_only = True
    _no_split_modules = ["Qwen3DecoderLayer", "T5Block"]

    def __init__(self, source_model: nn.Module, pad_token_id: int) -> None:
        super().__init__()
        self.config = source_model.encoder_list[0].model.encoder.config
        self.config.tie_word_embeddings = False
        self.pad_token_id = pad_token_id
        self.code_dim = source_model.docid_dim
        self.code_num = source_model.docid_size

        self.encoder_list = source_model.encoder_list
        self.docEnc2decoder = source_model.docEnc2decoder
        self.decoder = source_model.decoder
        self.lm_head = source_model.lm_head

        self.encoder_list.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
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
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
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
        attention_mask = padded_prompts.ne(self.pad_token_id).long()
        with torch.no_grad():
            outputs = self.encoder_list[0].model.encoder(
                input_ids=padded_prompts,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
        return hidden_states, attention_mask, prompt_lengths

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
    ) -> SimpleNamespace:
        del kwargs
        response_lengths = [int(length) for length in response_lengths]
        total_lengths = [int(length) for length in total_lengths]
        if any(length != self.code_num for length in response_lengths):
            raise ValueError(f"GR response lengths must all equal {self.code_num}")
        sample_tokens, output_offsets = self._unpack_tokens(input_ids, total_lengths)

        hidden_states, attention_mask, prompt_lengths = self._encode_prompts(
            sample_tokens,
            response_lengths,
        )
        hidden_states = hidden_states.to(self.docEnc2decoder.weight.dtype)
        projected_hidden_states = self.docEnc2decoder(hidden_states)
        input_batch_size, sequence_length = input_ids.shape
        sample_count = len(sample_tokens)
        logits = projected_hidden_states.new_zeros(
            (input_batch_size, sequence_length, self.code_num * self.code_dim + 2)
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
        for index, (prompt_length, output_offset) in enumerate(
            zip(prompt_lengths, output_offsets, strict=True)
        ):
            output_row = index if input_ids.shape[0] > 1 else 0
            start = output_offset + prompt_length - 1
            for position in range(self.code_num):
                position_logits = logits[output_row, start + position]
                position_logits.fill_(invalid_logit)
                codebook_start = 2 + position * self.code_dim
                codebook_end = codebook_start + self.code_dim
                position_logits[codebook_start:codebook_end] = response_logits[
                    index,
                    position,
                    codebook_start:codebook_end,
                ]
        return SimpleNamespace(logits=logits)


def build_gr_stage5_policy(checkpoint_path: str, args: Any, init_context) -> GRStage5Policy:
    """Miles ``--custom-fsdp-model-factory-path`` entrypoint."""
    del checkpoint_path, args
    from transformers import T5Config, T5ForConditionalGeneration

    checkpoint = _required_path("GR_CHECKPOINT")
    qwen_model = _required_path("GR_QWEN_MODEL")
    slm_root = _required_path("GR_SLM_ROOT")
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
    )
